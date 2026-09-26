"""
Evaluation harness, first read (phase C, design 6 and 9).

    python -m pipelines.models.harness --features features_p1.parquet --out report/

What it does, for each segment (F5 = innings 1-5, full8 = innings 1-8):
  1. Fold A: fit on 2022, predict 2023. Fold B: fit on 2022-2023, predict
     2024. 2025 and 2026 are never loaded as outcomes (the harness refuses)
  2. Models: B0 league baseline, B1 team-strength baseline, M1 round 1
     features. All three use the same negative binomial machinery, so M1 can
     only win on its information, not on better plumbing (design 6.1)
  3. Score: joint log score of the actual (home, away) score for the
     segment: log P(home runs) + log P(away runs), home and away treated as
     independent given the features (verified for F5, design 2.1; tested for
     full8 here). Higher is better
  4. Intervals: paired differences between models, 95% block bootstrap over
     7-day blocks, 2,000 resamples (design 9.2). Pooled 2023+2024 with fold A
     at weight 0.5 (design 9.1)
  5. Calibration of the home-win probability for the segment (slope of a
     logistic fit on the model's log-odds; 1 = right confidence)

Not in this first read (coming next in phase C): the late-game engine that
turns the 8-inning grid into full-game results, the distribution-shape
calibration, the zero-adjusted F5 model (M1z) and the split model (M1s).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .nb import MAX_RUNS, NBModel, nb_logpmf

FOLDS = {"A": ([2022], 2023), "B": ([2022, 2023], 2024)}
FOLD_WEIGHT = {"A": 0.5, "B": 1.0}
HELD_OUT = {2025, 2026}
SEGMENTS = {"f5": "runs_f5", "full8": "runs_8"}
SFX = {"f5": "f5", "full8": "8"}


def prepare(f: pd.DataFrame) -> pd.DataFrame:
    """Model-ready columns. Logs for multiplicative inputs; flags as 0/1."""
    f = f.copy()
    for s in ("f5", "8"):
        f[f"log_league_env_{s}"] = np.log(f[f"league_env_{s}"])
        f[f"log_team_off_{s}"] = np.log(f[f"team_off_{s}"])
        f[f"log_team_def_{s}"] = np.log(f[f"team_def_{s}"])
    f["log_park"] = np.log(f["park_factor"])
    # Player-quality features RELATIVE to the day's slate: each value minus
    # the average of the same feature over every team-game that day. The
    # league-level offset carries the run environment; these carry how this
    # team compares with the rest of the league today. Without this, M1
    # trained on 2022 kept 2022's level in its player features and
    # under-predicted 2023 (after the rule changes) by about 7% all season.
    # Leak-free: the other games' features that day are pre-game too.
    for c in [c for c in f.columns if c.startswith(("lineup_woba", "lineup_k_bb", "pen_skill", "pitching_composite"))] + ["sp_skill"]:
        f[c] = f[c] - f.groupby("date")[c].transform("mean")
    for c in ("sp_opener", "roof_park", "sp_rookie", "velo_missing", "pen_missing", "temp_missing", "new_park"):
        f[c] = f[c].astype(float)
    f["is_home"] = f["is_home"].astype(float)
    return f


def model_specs(seg: str) -> dict[str, list[str]]:
    s = SFX[seg]
    return {
        "B0": ["is_home"],
        "B1": ["is_home", f"log_team_off_{s}", f"log_team_def_{s}", "log_park"],
        "M1": [f"lineup_woba_{s}", f"lineup_k_bb_{s}", "lineup_platoon",
               "sp_skill", "sp_fb_velo_delta", "sp_exp_bf", "sp_opener", f"sp_tto3_share_{s}",
               f"pen_skill_{s}", f"pitching_composite_{s}",
               "log_park", "temp_f", "roof_park", "is_home",
               "sp_rookie", "velo_missing", "pen_missing"],
    }


def usable(f: pd.DataFrame, target: str) -> pd.DataFrame:
    """Regular season, completed 9+ inning games (drops rain-shortened and
    resumed games, design 5.1), both teams' rows present."""
    g = f[(f.game_type == "R") & f[target].notna() & ~f.resumed]
    g = g[g.game_id.isin(g.groupby("game_id").size().loc[lambda s: s == 2].index)]
    return g


def game_scores(df: pd.DataFrame, target: str, mu, alpha) -> pd.DataFrame:
    """Per game: joint log score and P(home ahead after the segment)."""
    y = df[target].to_numpy(float)
    ll = nb_logpmf(y, mu, alpha)
    ks = np.arange(MAX_RUNS + 1)
    grid = np.exp(nb_logpmf(ks[None, :], mu[:, None], alpha[:, None]))
    grid /= grid.sum(axis=1, keepdims=True)
    t = df[["game_id", "date", "is_home"]].copy()
    t["ll"] = ll
    t["y"] = y
    t["mu"] = mu
    t["grid"] = list(grid)
    h = t[t.is_home == 1].set_index("game_id")
    a = t[t.is_home == 0].set_index("game_id")
    out = pd.DataFrame({"date": h["date"], "ll": h["ll"] + a.loc[h.index, "ll"],
                        "home_y": h["y"], "away_y": a.loc[h.index, "y"],
                        "home_mu": h["mu"], "away_mu": a.loc[h.index, "mu"]})
    ph, pa = np.stack(h["grid"]), np.stack(a.loc[h.index, "grid"])
    cdf_a = np.cumsum(pa, axis=1)
    # P(home > away) = sum_h P(H=h) P(A < h)
    out["p_home_ahead"] = (ph[:, 1:] * cdf_a[:, :-1]).sum(axis=1)
    out["p_tie"] = (ph * pa).sum(axis=1)
    return out


def block_bootstrap_diff(dates: pd.Series, diff: np.ndarray, weights: np.ndarray, n=2000, seed=20260924):
    """Mean weighted difference and 95% interval, resampling 7-day blocks."""
    d = pd.to_datetime(dates.to_numpy())
    block = (d - d.min()).days // 7
    blocks = np.unique(block)
    rng = np.random.default_rng(seed)
    idx_by_block = {b: np.where(block == b)[0] for b in blocks}
    est = np.sum(weights * diff) / np.sum(weights)
    boots = np.empty(n)
    for i in range(n):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        ix = np.concatenate([idx_by_block[b] for b in pick])
        boots[i] = np.sum(weights[ix] * diff[ix]) / np.sum(weights[ix])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def calib_slope(p, outcome):
    """Logistic regression of the outcome on the model's log-odds."""
    import statsmodels.api as sm
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = sm.add_constant(np.log(p / (1 - p)))
    r = sm.Logit(outcome.astype(float), x).fit(disp=0)
    ci = r.conf_int()
    return {"intercept": float(r.params[0]), "slope": float(r.params[1]),
            "slope_lo": float(ci[1][0]), "slope_hi": float(ci[1][1])}


def run(features_path: str, out_dir: str) -> dict:
    f = prepare(pd.read_parquet(features_path))
    if f.loc[f.season.isin(HELD_OUT), ["runs_f5", "runs_8"]].notna().any().any():
        f.loc[f.season.isin(HELD_OUT), ["runs_f5", "runs_8", "runs_total"]] = np.nan  # never read held-out outcomes
    report = {"folds": {}, "pooled": {}, "models": {}}
    preds = {}
    for seg, target in SEGMENTS.items():
        specs = model_specs(seg)
        for fold, (train_seasons, test_season) in FOLDS.items():
            tr = usable(f[f.season.isin(train_seasons)], target)
            te = usable(f[f.season == test_season], target)
            for name, feats in specs.items():
                m = NBModel(feats, offset=f"log_league_env_{SFX[seg]}").fit(tr, tr[target])
                mu, alpha = m.predict(te)
                gs = game_scores(te, target, mu, alpha)
                preds[(seg, fold, name)] = gs
                report["models"][f"{seg}/{fold}/{name}"] = m.to_json()
            base = preds[(seg, fold, "B0")]
            report["folds"][f"{seg}/{fold}"] = {
                "test_season": test_season, "games": int(len(base)),
                "mean_runs_per_team": float((base.home_y.mean() + base.away_y.mean()) / 2),
                **{name: {"log_score_per_game": float(preds[(seg, fold, name)].ll.mean()),
                          "home_win_calibration": calib_slope(
                              preds[(seg, fold, name)].p_home_ahead.to_numpy(),
                              (preds[(seg, fold, name)].home_y > preds[(seg, fold, name)].away_y).to_numpy()),
                          "residual_corr_home_away": float(np.corrcoef(
                              preds[(seg, fold, name)].home_y - preds[(seg, fold, name)].home_mu,
                              preds[(seg, fold, name)].away_y - preds[(seg, fold, name)].away_mu)[0, 1])}
                   for name in specs},
            }
        # pooled paired comparisons, fold A at half weight
        for a_name, b_name in (("B1", "B0"), ("M1", "B0"), ("M1", "B1")):
            parts = []
            for fold in FOLDS:
                x, y = preds[(seg, fold, a_name)], preds[(seg, fold, b_name)]
                parts.append(pd.DataFrame({"date": x.date, "d": x.ll - y.ll, "w": FOLD_WEIGHT[fold], "fold": fold}))
            p = pd.concat(parts)
            est, lo, hi = block_bootstrap_diff(p.date, p.d.to_numpy(), p.w.to_numpy())
            same_dir = {fold: float(p[p.fold == fold].d.mean()) for fold in FOLDS}
            report["pooled"][f"{seg}: {a_name} minus {b_name}"] = {
                "per_game": est, "lo": lo, "hi": hi,
                "fold_A": same_dir["A"], "fold_B": same_dir["B"],
                "passes": bool(lo > 0 and same_dir["A"] > 0 and same_dir["B"] > 0)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.features, a.out)
    for k, v in r["folds"].items():
        print(k, {n: round(v[n]["log_score_per_game"], 4) for n in ("B0", "B1", "M1")})
    for k, v in r["pooled"].items():
        print(f"{k}: {v['per_game']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] A {v['fold_A']:+.4f} B {v['fold_B']:+.4f} pass={v['passes']}")


if __name__ == "__main__":
    main()

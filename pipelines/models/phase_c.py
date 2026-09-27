"""
Phase C, formal read (design 6, 8, 9): score grids, challengers,
calibration, gates G1 and G2.

    python -m pipelines.models.phase_c --features features_p1.parquet --late late_innings.parquet --out report/

Folds as in the first read (harness.py): A fits 2022 and predicts 2023,
B fits 2022-2023 and predicts 2024. 2025 and 2026 outcomes are never read.

Every model hands back, per game, a grid of probabilities for every score:
  F5          26 x 26 grid from the two teams' F5 run distributions
  full game   8-inning distributions -> 8-inning grid -> late-game engine
              (fitted on the fold's training seasons) -> 40 x 40 final grid
Everything after that works on the grids only.

Models
  B0, B1, M1  as in the first read
  M1z  F5 challenger: M1 plus one parameter for the chance of a scoreless
       F5 (nb.py, zero_adj)
  M1s  full-game challenger: the F5 model plus a separate innings 6-8 model,
       added together (convolved) for 8 innings. Assumes a team's F5 runs
       and its innings 6-8 runs are independent given the features; the
       residual correlation on training data is reported

Calibration and the spread fix
  1. Spread (design E4, adopted 26 Sep 2026 after the first G2 read failed
     on totals; every result is also shown without it):
     each team's expected runs = base part (league level, park, temperature,
     roof, home field) + team part (lineup and pitching features). Per game,
     the two teams' team parts are split into their average (moves the
     total) and their half difference (moves the margin). The average is
     scaled toward the day's slate average (every game that day, all
     pre-game), the difference toward its training mean, each by its own
     factor: tau_total and tau_margin. Why: out of sample, differences between
     games in expected TOTAL runs were about twice as large as reality
     (slope of actual on predicted total 0.5-0.7), while margins were close
     (home-win slopes 0.85-0.95). Both teams' quality cancels in the total
     but noise in the features doesn't, so the total direction needs more
     shrinkage than a team-level fit gives it.
     The factors are tuned season to season inside each fold's training
     window (design E4, adopted 26 Sep before its run), by grid search
     0.2-1.2 including 1 (no change): fold A fits 2021 and tunes on 2022;
     fold B fits 2021-2022 and tunes on 2023. Earlier versions tuned on the
     other test season (replaced after the first review) and then within a
     season (replaced: it can't see a problem that shows up between seasons).
     Level fix (design E5): models are fitted against each training season's
     actual scoring level and predict with the forecast level.
  2. Shape (design 8 as written, calibration.py): total and margin
     distribution shapes, cross-fitted between 2023 and 2024, kept only if it
     improves the joint log score in both seasons

Versions reported
  raw      the models as fitted (with the E5 level fix)
  design   raw + shape calibration: the design before E4
  final    spread fix + shape calibration: design v3.3, the version the
           gates are judged on from 26 Sep 2026

Late-game engine (changed 26 Sep 2026 after the review): team scaling is
each team's expected 8-inning runs relative to the TRAINING seasons' average
level (it was relative to the current league level), so a league-wide rise
in scoring also raises 9th-inning and extra-inning scoring (design 6.3: late
scoring = "league runs per late inning" x team terms). Before, 2023's higher
scoring never reached the 9th inning.

Steps
  1. Fit every model per fold; tune the spread factors inside training
  2. Grids (engine for the full game), raw and with the spread fix
  3. Shape calibration, cross-fitted, on both
  4. Challengers replace M1 only if better by more than one standard error
     (design 9.4), on the final scores
  5. G1: champion vs B1, pooled 2023+2024 (fold A weight 0.5), 7-day block
     bootstrap, interval above zero and same direction in both seasons
  6. G2: the champion's grids on the eight fixed events, as designed and
     with the spread fix
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import Layout, cross_fit, gate_g2, gate_g2_v2, log_score, reliability
from .harness import FOLD_WEIGHT, FOLDS, HELD_OUT, model_specs, prepare, usable
from .late_game import GRID, LateGame
from .nb import MAX_RUNS, NBModel

K5 = MAX_RUNS + 1
KS = np.arange(K5)
LAY = {"f5": Layout(K5), "full": Layout(GRID), "full8": Layout(K5)}
MODELS = {"f5": ["B0", "B1", "M1", "M1z", "M1c", "M1cz"], "full8": ["B0", "B1", "M1", "M1s", "M1c", "M1cs"]}
CHALLENGER = {"f5": "M1z", "full": "M1s"}
# round 2 group 1, contact quality (design E7): the contact version of each candidate
CONTACT_OF = {"M1": "M1c", "M1z": "M1cz", "M1s": "M1cs"}
CT_COMMON = ["ct_lu_xw", "ct_lu_brl", "ct_lu_hh", "ct_sp_xw", "ct_sp_brl"]
CT = {"f5": CT_COMMON + ["ct_pen_xw_f5"], "8": CT_COMMON + ["ct_pen_xw_8"]}
SEASON_OF = {fold: test for fold, (_, test) in FOLDS.items()}
OTHER = {"A": "B", "B": "A"}
SHARED = ("log_park", "temp_f", "roof_park", "is_home")   # the base part (plus offset and constant)
TAU_GRID = np.round(np.arange(0.2, 1.2001, 0.05), 2)
MARKET_CLOSE_ML_LOG_LOSS = {2023: 0.676, 2024: 0.674}   # devigged consensus close (design D3)
BOOT_N = 2000
BOOT_SEED = 20260924


# ------------------------------------------------------------------ models
def _nb(feats, offset, tr, target, **kw):
    return NBModel(feats, offset=offset, **kw).fit(tr, tr[target])


def _convolve(P, Q):
    """Row-wise distribution of the sum, truncated at MAX_RUNS."""
    out = np.zeros_like(P)
    for i in range(P.shape[1]):
        out[:, i:] += P[:, i:i + 1] * Q[:, :P.shape[1] - i]
    return out / out.sum(axis=1, keepdims=True)


OFFSETS = {"log_league_env_f5": "log_level_real_f5", "log_league_env_8": "log_level_real_8",
           "log_league_env_68": "log_level_real_68"}


def train_view(tr):
    """Training rows with the level offsets set to each season's ACTUAL
    scoring level (design E5). Models are fitted on this view and predict
    with the forecast level (league_env), so the fitted constant doesn't
    absorb how the forecast lagged during the training seasons."""
    tr = tr.copy()
    for fc, real in OFFSETS.items():
        tr[fc] = tr[real]
    return tr


def fit_components(seg, name, tr):
    """A model = one or more negative binomial components whose run
    distributions are added (convolved). Returns the fitted components and
    diagnostics. Fits on the actual-level view of the training rows."""
    tr = train_view(tr)
    ct = name in ("M1c", "M1cz", "M1cs")
    if seg == "f5":
        feats = model_specs("f5")["M1" if name in ("M1z", "M1c", "M1cz") else name] + (CT["f5"] if ct else [])
        return [_nb(feats, "log_league_env_f5", tr, "runs_f5", zero_adj=(name in ("M1z", "M1cz")))], {}
    if name not in ("M1s", "M1cs"):
        feats = model_specs("full8")["M1" if name == "M1c" else name] + (CT["8"] if ct else [])
        return [_nb(feats, "log_league_env_8", tr, "runs_8")], {}
    m5 = _nb(model_specs("f5")["M1"] + (CT["f5"] if ct else []), "log_league_env_f5", tr, "runs_f5")
    m68 = _nb(model_specs("full8")["M1"] + (CT["8"] if ct else []), "log_league_env_68", tr, "runs_68")
    mu5, mu68 = m5.predict(tr)[0], m68.predict(tr)[0]
    return [m5, m68], {"train_residual_corr_f5_vs_6to8": float(np.corrcoef(tr.runs_f5 - mu5, tr.runs_68 - mu68)[0, 1])}


def _pair_index(df):
    """Row positions of home and away rows, both in game_id order."""
    gid = df.game_id.to_numpy()
    home = (df.is_home == 1).to_numpy()
    hi, ai = np.where(home)[0], np.where(~home)[0]
    hi, ai = hi[np.argsort(gid[hi], kind="stable")], ai[np.argsort(gid[ai], kind="stable")]
    assert (gid[hi] == gid[ai]).all()
    return hi, ai


@dataclass
class GameParts:
    """Per game and component: base and team parts of the linear predictor."""
    bh: list
    ba: list
    th: list
    ta: list
    mbar: list     # per game: the day's slate average of the team parts' game average
    dbar: list     # training mean of their half difference
    day: np.ndarray = None   # per game: day code, for keeping each day's mean expected runs (E8a)


def game_parts(comps, tr, te, slate):
    """slate: every regular-season row of the test season (pre-game features
    only), used for the day's average."""
    hi, ai = _pair_index(te)
    thi, tai = _pair_index(tr)
    shi, sai = _pair_index(slate)
    day = slate.date.to_numpy()[shi]
    gid_te = te.game_id.to_numpy()[hi]
    gp = GameParts([], [], [], [], [], [], pd.factorize(te.date.to_numpy()[hi])[0])
    for m in comps:
        b, t = m.eta_parts(te, SHARED)
        gp.bh.append(b[hi]); gp.ba.append(b[ai]); gp.th.append(t[hi]); gp.ta.append(t[ai])
        _, t_s = m.eta_parts(slate, SHARED)
        mm = pd.Series((t_s[shi] + t_s[sai]) / 2, index=slate.game_id.to_numpy()[shi])
        day_mean = mm.groupby(day).transform("mean")
        gp.mbar.append(day_mean.loc[gid_te].to_numpy())
        _, t_tr = m.eta_parts(tr, SHARED)
        gp.dbar.append(float(((t_tr[thi] - t_tr[tai]) / 2).mean()))
    return gp


def team_pmfs(comps, gp, tau_t=1.0, tau_m=1.0):
    """Home and away run distributions per game, with the spread factors."""
    Ph = Pa = None
    for j, m in enumerate(comps):
        mm, dd = (gp.th[j] + gp.ta[j]) / 2, (gp.th[j] - gp.ta[j]) / 2
        m2 = gp.mbar[j] + tau_t * (mm - gp.mbar[j])
        d2 = gp.dbar[j] + tau_m * (dd - gp.dbar[j])
        eh, ea = gp.bh[j] + m2 + d2, gp.ba[j] + m2 - d2
        if (tau_t, tau_m) != (1.0, 1.0) and gp.day is not None:
            # E8a: shrinking on the log scale lowers average expected runs a
            # little; scale each day's games (per side) back to the unshrunk mean
            for e_new, e_old in ((eh, gp.bh[j] + gp.th[j]), (ea, gp.ba[j] + gp.ta[j])):
                num = np.bincount(gp.day, weights=np.exp(np.clip(e_old, -5, 5)))
                den = np.bincount(gp.day, weights=np.exp(np.clip(e_new, -5, 5)))
                e_new += np.log(num / den)[gp.day]
        ph, pa = m.pmf_eta(eh), m.pmf_eta(ea)
        Ph = ph if Ph is None else _convolve(Ph, ph)
        Pa = pa if Pa is None else _convolve(Pa, pa)
    return Ph, Pa


def _pair_ll(Ph, Pa, yh, ya):
    r = np.arange(len(yh))
    return np.log(np.maximum(Ph[r, np.minimum(yh, K5 - 1)], 1e-300)) + np.log(np.maximum(Pa[r, np.minimum(ya, K5 - 1)], 1e-300))


def fit_spread(comps, gp, yh, ya):
    """Grid search for (tau_total, tau_margin) maximizing the joint log score."""
    best = (-np.inf, 1.0, 1.0)
    for tt in TAU_GRID:
        for tm in TAU_GRID:
            ll = _pair_ll(*team_pmfs(comps, gp, tt, tm), yh, ya).mean()
            if ll > best[0]:
                best = (float(ll), float(tt), float(tm))
    return {"tau_total": best[1], "tau_margin": best[2], "log_score_on_fit_season": best[0]}


# ------------------------------------------------------------------ build
INNER = {  # (fit rows, tuning rows): fit on the seasons before the last training season, tune on it (design E4)
    "A": (lambda f: f.season == 2021, lambda f: f.season == 2022),
    "B": (lambda f: f.season.isin([2021, 2022]), lambda f: f.season == 2023),
}


def _slate(f, mask):
    x = f[mask & (f.game_type == "R")]
    return x[x.game_id.isin(x.groupby("game_id").size().loc[lambda c: c == 2].index)]


def _both_rows(te, col):
    return te[te.game_id.isin(te.groupby("game_id")[col].count().loc[lambda c: c == 2].index)]


def _rel(comps, gp, tau, game_level, level):
    """Engine scaling per game (design E8b): each team's expected 8-inning
    runs over the game's league level (team strength, fitted exponent), and
    the game's league level over the training level (exponent 1)."""
    Ph, Pa = team_pmfs(comps, gp, *tau)
    return Ph @ KS / game_level, Pa @ KS / game_level, game_level / level


def build(fs, late):
    """Fits, spread tuning, grids. Returns preds[(seg, fold, name)] with raw
    and spread grids, plus models, diagnostics and the spread report."""
    fitted, models, diags, spread = {}, {}, {}, {}
    for fold, (train_seasons, test_season) in FOLDS.items():
        f = fs[fold].copy()
        # guard (review 26 Sep): the actual-level columns come from outcomes,
        # so they are blanked for the test season and later; only training
        # rows may ever carry them
        f.loc[f.season >= test_season, list(OFFSETS.values())] = np.nan
        fit_m, tune_m = INNER[fold][0](f), INNER[fold][1](f)
        assert not (f.season[fit_m | tune_m] >= test_season).any()
        for seg, target in (("f5", "runs_f5"), ("full8", "runs_8")):
            tr = usable(f[f.season.isin(train_seasons)], target)
            te = usable(f[f.season == test_season], target)
            itr, iva = usable(f[fit_m], target), usable(f[tune_m], target)
            if seg == "full8":
                te = _both_rows(te, "runs_total")
            slate_te = _slate(f, f.season == test_season)
            slate_tr = _slate(f, f.season.isin(train_seasons))
            hi, ai = _pair_index(te)
            H = te.iloc[hi].reset_index(drop=True)
            A = te.iloc[ai].reset_index(drop=True)
            for name in MODELS[seg]:
                # spread factors, tuned inside the training seasons
                icomps, _ = fit_components(seg, name, itr)
                igp = game_parts(icomps, itr, iva, _slate(f, tune_m))
                ih, ia = _pair_index(iva)
                tune = fit_spread(icomps, igp, iva[target].to_numpy(int)[ih], iva[target].to_numpy(int)[ia])
                tune["log_score_no_change"] = float(_pair_ll(*team_pmfs(icomps, igp), iva[target].to_numpy(int)[ih],
                                                             iva[target].to_numpy(int)[ia]).mean())
                tau = (tune["tau_total"], tune["tau_margin"])
                # the fold's models
                comps, dg = fit_components(seg, name, tr)
                gp = game_parts(comps, tr, te, slate_te)
                item = {"comps": comps, "gp": gp, "H": H, "A": A, "tau": tau,
                        "yh": H[target].to_numpy(int), "ya": A[target].to_numpy(int)}
                spread[f"{seg}/{fold}/{name}"] = {"tuned_in_training": tune}
                if seg == "full8":
                    # engine fitted on the training seasons; team scaling =
                    # expected 8-inning runs over the training seasons'
                    # average level, from this model's own training predictions
                    trv = train_view(tr)
                    level = float(np.exp(trv["log_level_real_8"]).mean())
                    gp_tr = game_parts(comps, trv, trv, train_view(slate_tr))
                    th, ta = _pair_index(tr)
                    gids = tr.game_id.to_numpy()[th]
                    li = late[late.season.isin(train_seasons)]
                    item["level"] = level
                    item["engine"] = {}
                    glev = np.exp(trv["log_level_real_8"].to_numpy()[th])
                    for tag, tt in (("raw", (1.0, 1.0)), ("spread", tau)):
                        rh, ra, lr = _rel(comps, gp_tr, tt, glev, level)
                        rel = {g: (float(x), float(y), float(z)) for g, x, y, z in zip(gids, rh, ra, lr)}
                        item["engine"][tag] = LateGame().fit(li, rel)
                        models[f"late_game/{fold}/{name}/{tag}"] = item["engine"][tag].to_json()
                fitted[(seg, fold, name)] = item
                models[f"{seg}/{fold}/{name}"] = [m.to_json() for m in comps]
                if dg:
                    diags[f"{seg}/{fold}/{name}"] = dg

    # test-season effect of the spread fix (information only: nothing is
    # chosen from it)
    for (seg, fold, name), it in fitted.items():
        before = float(_pair_ll(*team_pmfs(it["comps"], it["gp"]), it["yh"], it["ya"]).mean())
        after = float(_pair_ll(*team_pmfs(it["comps"], it["gp"], *it["tau"]), it["yh"], it["ya"]).mean())
        spread[f"{seg}/{fold}/{name}"].update({"test_season": SEASON_OF[fold], "test_log_score_raw": before,
                                               "test_log_score_spread": after})

    # grids
    preds = {}
    for (seg, fold, name), it in fitted.items():
        out = {}
        for tag, tt in (("raw", (1.0, 1.0)), ("spread", it["tau"])):
            Ph, Pa = team_pmfs(it["comps"], it["gp"], *tt)
            out[f"{tag}:X8"] = (Ph[:, :, None] * Pa[:, None, :]).reshape(len(Ph), -1)
            if seg == "full8":
                eng = it["engine"][tag]
                lg8 = it["H"].league_env_8.to_numpy()
                eh, ea, lr = Ph @ KS / lg8, Pa @ KS / lg8, lg8 / it["level"]
                XF = np.empty((len(Ph), GRID * GRID))
                for i in range(len(Ph)):
                    XF[i] = eng.final_grid(np.outer(Ph[i], Pa[i]), eh[i], ea[i], lr[i]).ravel()
                out[f"{tag}:XF"] = XF
        base = {"game_id": it["H"].game_id.to_numpy(), "date": it["H"].date.to_numpy()}
        if seg == "f5":
            preds[("f5", fold, name)] = {**base, "raw:X": out["raw:X8"], "spread:X": out["spread:X8"],
                                         "h": it["yh"], "a": it["ya"]}
        else:
            preds[("full8", fold, name)] = {**base, "raw:X": out["raw:X8"], "spread:X": out["spread:X8"],
                                            "h": it["yh"], "a": it["ya"]}
            preds[("full", fold, name)] = {**base, "raw:X": out["raw:XF"], "spread:X": out["spread:XF"],
                                           "h": it["H"].runs_total.to_numpy(int), "a": it["A"].runs_total.to_numpy(int)}
        # F5 vs 8-inning consistency (design 6.2), M1 only
        if seg == "full8" and name == "M1":
            both = pd.concat([it["H"], it["A"]])
            p5 = fitted[("f5", fold, "M1")]["comps"][0].pmf_grid(both)
            p8 = it["comps"][0].pmf_grid(both)
            e5, e8 = p5 @ KS, p8 @ KS
            diags[f"consistency/{fold}/M1"] = {
                "rows": int(len(p5)), "mean_runs_6_to_8": float((e8 - e5).mean()),
                "rows_innings_6_8_below_0.5_runs": int(((e8 - e5) < 0.5).sum()),
                "rows_p0_8_above_p0_f5": int((p8[:, 0] > p5[:, 0] + 1e-12).sum())}
    return preds, models, diags, spread


# --------------------------------------------------------------- scoring
def _boot(dates, diff, w, n=BOOT_N, seed=BOOT_SEED):
    """Weighted mean difference, 95% interval and standard error, 7-day blocks."""
    d = pd.to_datetime(pd.Series(dates))
    block = ((d - d.min()).dt.days // 7).to_numpy()
    blocks = np.unique(block)
    idx = {b: np.where(block == b)[0] for b in blocks}
    rng = np.random.default_rng(seed)
    est = float(np.sum(w * diff) / np.sum(w))
    boots = np.empty(n)
    for i in range(n):
        ix = np.concatenate([idx[b] for b in rng.choice(blocks, size=len(blocks), replace=True)])
        boots[i] = np.sum(w[ix] * diff[ix]) / np.sum(w[ix])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi), float(boots.std(ddof=1))


def compare(scores, seg, a_name, b_name, which):
    """Pooled paired difference a - b on the joint log score."""
    parts = []
    for fold in FOLDS:
        x, y = scores[(seg, fold, a_name)], scores[(seg, fold, b_name)]
        assert (x["game_id"] == y["game_id"]).all()
        parts.append(pd.DataFrame({"date": x["date"], "d": x[which] - y[which], "w": FOLD_WEIGHT[fold], "fold": fold}))
    p = pd.concat(parts, ignore_index=True)
    est, lo, hi, se = _boot(p.date, p.d.to_numpy(), p.w.to_numpy())
    by = {fold: float(p[p.fold == fold].d.mean()) for fold in FOLDS}
    mar_apr = (pd.to_datetime(p.date).dt.month <= 4).to_numpy()
    return {"per_game": est, "lo": lo, "hi": hi, "se": se, "fold_A": by["A"], "fold_B": by["B"],
            "passes": bool(lo > 0 and by["A"] > 0 and by["B"] > 0),
            "march_april": float(np.average(p.d[mar_apr], weights=p.w[mar_apr])),
            "may_on": float(np.average(p.d[~mar_apr], weights=p.w[~mar_apr]))}


def secondary(X, lay, h, a):
    """Market-level and secondary metrics from one set of grids."""
    T, M = lay.marginals(X)
    s = np.minimum(h + a, lay.L - 1)
    d = np.clip(h - a, -(lay.K - 1), lay.K - 1) + lay.K - 1
    ot = (np.arange(lay.L)[None, :] >= s[:, None]).astype(float)
    om = (np.arange(lay.L)[None, :] >= d[:, None]).astype(float)
    p_home = X[:, lay.h > lay.a].sum(1)
    hw = h > a
    out = {"rps_total": float(((np.cumsum(T, 1) - ot) ** 2).sum(1).mean()),
           "rps_margin": float(((np.cumsum(M, 1) - om) ** 2).sum(1).mean()),
           "mae_total": float(np.abs(T @ lay.total_k - (h + a)).mean()),
           "mean_total_pred": float((T @ lay.total_k).mean()), "mean_total_actual": float((h + a).mean()),
           "home_win_pred": float(p_home.mean()), "home_win_actual": float(hw.mean()),
           "home_win_log_loss": float(-np.mean(np.where(hw, np.log(p_home), np.log(1 - p_home))))}
    if lay.K == GRID:
        out["home_m1.5_pred"] = float(X[:, (lay.h - lay.a) >= 2].sum(1).mean())
        out["home_m1.5_actual"] = float((h - a >= 2).mean())
        out["over_8.5_pred"] = float(X[:, lay.s >= 9].sum(1).mean())
        out["over_8.5_actual"] = float((h + a >= 9).mean())
    else:
        p_tie = X[:, lay.h == lay.a].sum(1)
        out["tie_pred"] = float(p_tie.mean())
        out["tie_actual"] = float((h == a).mean())
        out["tie_reliability"] = reliability(p_tie, (h == a).astype(float), bins=5)
        out["over_4.5_pred"] = float(X[:, lay.s >= 5].sum(1).mean())
        out["over_4.5_actual"] = float((h + a >= 5).mean())
    return out


def load_features(path, games_csv):
    f = prepare(pd.read_parquet(path))
    f.loc[f.season.isin(HELD_OUT), ["runs_f5", "runs_8", "runs_total"]] = np.nan   # never read held-out outcomes
    # design E6: rain-shortened games (under 9 innings) leave every target;
    # 2021's 7-inning doubleheaders leave the 8-inning and full-game targets
    g = pd.read_csv(games_csv, usecols=["game_id", "season", "innings", "dh"])
    dh = g.dh.astype(str).str.lower().isin(["true", "t", "1"])
    dh21 = set(g.game_id[(g.season == 2021) & dh])
    short = set(g.game_id[(g.innings < 9) & ~((g.season == 2021) & dh)])
    f.loc[f.game_id.isin(short), ["runs_f5", "runs_8", "runs_total"]] = np.nan
    f.loc[f.game_id.isin(dh21), ["runs_8", "runs_total"]] = np.nan
    f["league_env_68"] = f.league_env_8 - f.league_env_f5
    f["log_league_env_68"] = np.log(f.league_env_68)
    f["runs_68"] = f.runs_8 - f.runs_f5
    # each season's actual scoring level per segment (used only when that
    # season is a training season, design E5)
    reg = f[f.game_type == "R"]
    lv = pd.DataFrame({"f5": reg.groupby("season").runs_f5.mean(),
                       "8": reg[reg.runs_8.notna()].groupby("season").runs_8.mean(),
                       "68": reg[reg.runs_8.notna()].groupby("season").runs_68.mean()})
    for k in ("f5", "8", "68"):
        f[f"log_level_real_{k}"] = np.log(f.season.map(lv[k]))
    return f


def run(features: dict, late_path, out_dir, games_csv, point="P1", n_sim=2000):
    """features: {fold: path}. The same file for both folds unless the prior
    layer was tuned per fold (tune_priors.py)."""
    cache, fs = {}, {}
    for fold, path in features.items():
        if path not in cache:
            cache[path] = load_features(path, games_csv)
        fs[fold] = cache[path]
    late = pd.read_parquet(late_path)
    late = late[~late.season.isin(HELD_OUT)]

    preds, models, diags, spread = build(fs, late)
    report = {"point": point, "features": {k: str(v) for k, v in features.items()},
              "models": models, "diagnostics": diags, "spread": spread, "shape": {},
              "segments": {}, "comparisons": {}, "decisions": {}}

    # ---- shape calibration, cross-fitted, on the raw grids ("design") and
    # on the spread grids ("final")
    scores, grids = {}, {}
    for (seg, fold, name), p in preds.items():
        scores[(seg, fold, name)] = {"game_id": p["game_id"], "date": p["date"],
                                     "raw": log_score(p["raw:X"], LAY[seg], p["h"], p["a"]),
                                     "spread": log_score(p["spread:X"], LAY[seg], p["h"], p["a"])}
        grids[(seg, fold, name, "raw")] = p["raw:X"]
    for seg in ("f5", "full"):
        for name in MODELS["f5" if seg == "f5" else "full8"]:
            A, B = preds[(seg, "A", name)], preds[(seg, "B", name)]
            for src, dst in (("raw", "design"), ("spread", "final")):
                out, rep = cross_fit({SEASON_OF["A"]: (A[f"{src}:X"], A["h"], A["a"]),
                                      SEASON_OF["B"]: (B[f"{src}:X"], B["h"], B["a"])}, LAY[seg])
                report["shape"][f"{seg}/{name}/{dst}"] = rep
                for fold, p in (("A", A), ("B", B)):
                    grids[(seg, fold, name, dst)] = out[SEASON_OF[fold]]
                    scores[(seg, fold, name)][dst] = log_score(out[SEASON_OF[fold]], LAY[seg], p["h"], p["a"])
    for name in MODELS["full8"]:
        for fold in FOLDS:
            sc = scores[("full8", fold, name)]
            sc["design"], sc["final"] = sc["raw"], sc["spread"]
    VERSIONS = ("raw", "design", "final")

    # ---- per-season summaries
    for (seg, fold, name), p in preds.items():
        entry = {"games": int(len(p["h"])),
                 **{f"log_score_{w}": float(scores[(seg, fold, name)][w].mean()) for w in ("raw", "spread", "design", "final")}}
        if seg in ("f5", "full"):
            for v in VERSIONS:
                entry[v] = secondary(grids[(seg, fold, name, v)], LAY[seg], p["h"], p["a"])
            if seg == "full":
                entry["market_close_ml_log_loss"] = MARKET_CLOSE_ML_LOG_LOSS[SEASON_OF[fold]]
        report["segments"][f"{seg}/{SEASON_OF[fold]}/{name}"] = entry

    # ---- comparisons
    for seg in ("f5", "full8", "full"):
        chal = "M1z" if seg == "f5" else "M1s"
        for a_name, b_name in [("B1", "B0"), ("M1", "B0"), ("M1", "B1"), (chal, "M1"), (chal, "B1")]:
            for which in VERSIONS:
                report["comparisons"][f"{seg} [{which}]: {a_name} minus {b_name}"] = compare(scores, seg, a_name, b_name, which)

    # ---- champion (one-standard-error rule, per version) and gates
    lays = {"f5": LAY["f5"], "full": LAY["full"]}
    for v in ("design", "final"):
        report["decisions"][v] = {}
        gi, obs = {}, {}
        for seg in ("f5", "full"):
            chal = CHALLENGER[seg]
            c = report["comparisons"][f"{seg} [{v}]: {chal} minus M1"]
            base = chal if c["per_game"] > c["se"] else "M1"
            cv = CONTACT_OF[base]
            cc = compare(scores, seg, cv, base, v)
            report["comparisons"][f"{seg} [{v}]: {cv} minus {base}"] = cc
            champ = cv if (cc["per_game"] > cc["se"] and cc["fold_A"] > 0 and cc["fold_B"] > 0) else base
            if champ not in ("M1", chal):
                report["comparisons"][f"{seg} [{v}]: {champ} minus B1"] = compare(scores, seg, champ, "B1", v)
            g1 = report["comparisons"][f"{seg} [{v}]: {champ} minus B1"]
            report["decisions"][v][seg] = {
                "challenger": chal, "challenger_minus_M1": c["per_game"], "one_se": c["se"],
                "contact_candidate": cv, "contact_minus_base": cc["per_game"], "contact_one_se": cc["se"],
                "contact_fold_A": cc["fold_A"], "contact_fold_B": cc["fold_B"], "champion": champ,
                "shape_kept": report["shape"][f"{seg}/{champ}/{v}"]["kept"],
                "G1": {"passes": g1["passes"], **{k: g1[k] for k in ("per_game", "lo", "hi", "fold_A", "fold_B")}}}
            gi[seg] = np.vstack([grids[(seg, fold, champ, v)] for fold in FOLDS])
            obs[seg] = (np.concatenate([preds[(seg, fold, champ)]["h"] for fold in FOLDS]),
                        np.concatenate([preds[(seg, fold, champ)]["a"] for fold in FOLDS]))
        dts = {seg: np.concatenate([np.asarray(pd.to_datetime(pd.Series(preds[(seg, fold, report["decisions"][v][seg]["champion"])]["date"]))
                                               .dt.strftime("%Y-%m-%d")) for fold in FOLDS]) for seg in ("f5", "full")}
        report[f"G2_{v}"] = gate_g2_v2(gi, lays, obs, dts, n_sim=n_sim)     # the gate (design v3.3)
        report[f"G2_v1_{v}"] = gate_g2(gi, lays, obs, n_sim=n_sim)          # v3.2 gate, kept for the record
        report[f"G2_{v}_by_season"] = {}
        for fold in FOLDS:
            champs = {seg: report["decisions"][v][seg]["champion"] for seg in ("f5", "full")}
            g = gate_g2({seg: grids[(seg, fold, champs[seg], v)] for seg in ("f5", "full")}, lays,
                        {seg: (preds[(seg, fold, champs[seg])]["h"], preds[(seg, fold, champs[seg])]["a"]) for seg in ("f5", "full")},
                        n_sim=200)
            report[f"G2_{v}_by_season"][SEASON_OF[fold]] = {
                k: {x: e[x] for x in ("slope", "slope_lo", "slope_hi", "intercept", "intercept_lo", "intercept_hi", "mean_pred", "actual_rate")}
                for k, e in g["events"].items()}

    # ---- per-game file for review (probabilities for the main markets)
    rows = []
    for (seg, fold, name), p in preds.items():
        if seg == "full8":
            continue
        lay = LAY[seg]
        for tag in VERSIONS:
            X = grids[(seg, fold, name, tag)]
            d = {"segment": seg, "season": SEASON_OF[fold], "model": name, "version": tag,
                 "game_id": p["game_id"], "date": p["date"], "home": p["h"], "away": p["a"],
                 "log_score": scores[(seg, fold, name)][tag], "p_home_win": X[:, lay.h > lay.a].sum(1),
                 "e_total": X @ lay.s.astype(float), "e_margin": X @ (lay.h - lay.a).astype(float)}
            for ln in ((7.5, 8.5, 9.5) if seg == "full" else (3.5, 4.5, 5.5)):
                d[f"p_over_{ln}"] = X[:, lay.s > ln].sum(1)
            if seg == "full":
                d["p_home_m1.5"] = X[:, (lay.h - lay.a) >= 2].sum(1)
            else:
                d["p_tie"] = X[:, lay.h == lay.a].sum(1)
            rows.append(pd.DataFrame(d))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(Path(out_dir) / "game_predictions.parquet")
    (Path(out_dir) / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def summarize(r):
    lines = []
    for v, dec in r["decisions"].items():
        for seg, d in dec.items():
            g = d["G1"]
            lines.append(f"[{v}] {seg}: champion {d['champion']} (challenger {d['challenger']} {d['challenger_minus_M1']:+.4f}, 1 SE {d['one_se']:.4f}; "
                         f"contact {d['contact_candidate']} {d['contact_minus_base']:+.4f}, 1 SE {d['contact_one_se']:.4f}, A {d['contact_fold_A']:+.4f} B {d['contact_fold_B']:+.4f}); "
                         f"shape kept {d['shape_kept']}; G1 {'PASS' if g['passes'] else 'FAIL'} "
                         f"{g['per_game']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}] A {g['fold_A']:+.4f} B {g['fold_B']:+.4f}")
    for k, v in r["spread"].items():
        t = v["tuned_in_training"]
        lines.append(f"  spread {k}: tau_t {t['tau_total']:.2f} tau_m {t['tau_margin']:.2f} "
                     f"(tuning rows {t['log_score_no_change']:.4f}->{t['log_score_on_fit_season']:.4f}); "
                     f"test {v['test_season']} {v['test_log_score_raw']:.4f}->{v['test_log_score_spread']:.4f}")
    for k, v in r["shape"].items():
        lines.append(f"  shape {k}: kept {v['kept']} " + ", ".join(f"{var} " + "/".join(f"{x:.4f}" for x in s.values()) for var, s in v["log_score"].items()))
    for k, v in r["comparisons"].items():
        lines.append(f"  {k}: {v['per_game']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] se {v['se']:.4f} A {v['fold_A']:+.4f} B {v['fold_B']:+.4f}")
    for tag in ("G2_design", "G2_final"):
        g = r[tag]
        lines.append(f"{tag} (v2): passes={g['passes']} perfect-model whole-gate pass rate {g['perfect_model_whole_gate_pass_rate']:.2f} z {g['z_widened']:.2f}")
        for k, v in g["events"].items():
            lines.append(f"  {k}: {v['verdict']} slope {v['slope']:.2f} [{v['slope_lo']:.2f}, {v['slope_hi']:.2f}] icpt {v['intercept']:+.3f} "
                         f"[{v['intercept_lo']:+.3f}, {v['intercept_hi']:+.3f}] ece {v['ece']:.4f} (thr {v['ece_threshold']:.4f}) "
                         f"pred {v['mean_pred']:.3f} act {v['actual_rate']:.3f}")
        lines.append(f"  v1 gate for the record: passes={r['G2_v1_' + tag[3:]]['passes']}")
        for s, ev in r[f"{tag}_by_season"].items():
            lines.append(f"    {s}: " + ", ".join(f"{k} {e['slope']:.2f}/{e['intercept']:+.2f}" for k, e in ev.items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features for both folds")
    ap.add_argument("--features-a", help="fold A features, if tuned separately")
    ap.add_argument("--features-b", help="fold B features, if tuned separately")
    ap.add_argument("--late", required=True)
    ap.add_argument("--games", required=True, help="games.csv from the feature inputs (innings, doubleheaders)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--point", default="P1")
    a = ap.parse_args()
    r = run({"A": a.features_a or a.features, "B": a.features_b or a.features}, a.late, a.out, a.games, a.point)
    print(summarize(r))


if __name__ == "__main__":
    main()

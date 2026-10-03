"""
Phase D market work (design v3.4, E9): full-game moneyline and run line
against the Covers odds, 2022-2024 only.

    python -m pipelines.models.market_d --odds <private dir> --pred-p1 <dir> --pred-p2 <dir> --out <private dir>

Steps, in the order E9 fixes them:
  1. Devig comparison on 2022 only (no model involved): multiplicative, power
     and Shin, scored by log loss of the devigged price against 2022 results,
     per market. The winner is used for everything after.
  2. Fit on 2023 (fold A predictions, model trained on 2022): blend weight w and
     edge threshold tau, grid search, objective = total line movement toward
     the bet (closing-line value in probability points), at least 100 bets.
  3. Dry run on 2024 (fold B predictions) with w and tau frozen.

Bet timing variants, run separately: the 10am p1 price with the P1 model, and
the lineup-time price with the P2 model. Close = Covers close.

Prices (D9): the bet is taken at the best of DraftKings, FanDuel and Caesars;
the market's fair probability is the mean of the three books' devigged
probabilities. Conservative check: fair probability from all books, bet taken
at the all-book median price (no line shopping).

Hard rules: 2025-2026 outcomes are never read (the games file passed in holds
2022-2024 only, and this script refuses anything else). Odds never enter the
model; they only meet its output here. Odds stay in the private folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BOOKS3 = ["DraftKings", "FanDuel", "Caesars"]
SEASONS = (2022, 2023, 2024)
METHODS = ("multiplicative", "power", "shin")
W_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)
TAU_GRID = np.round(np.arange(0.0, 0.0601, 0.0025), 4)
MAX_RAW_GAP = 0.08      # design 13: bigger raw disagreements are treated as a data problem
MIN_BETS = 100
N_BOOT = 2000
SEED = 20261001


# ---------- prices and devig ----------

def to_dec(am):
    am = np.asarray(am, float)
    return np.where(am > 0, 1 + am / 100.0, 1 + 100.0 / np.abs(am))


def _bisect(f, lo, hi, n=60):
    lo = np.full_like(f(lo), lo) if np.isscalar(lo) else lo
    hi = np.full_like(lo, hi) if np.isscalar(hi) else hi
    for _ in range(n):
        mid = (lo + hi) / 2
        pos = f(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return (lo + hi) / 2


def devig_home(a, b, method):
    """a, b: implied probabilities (1 / decimal odds) of home and away."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    s = a + b
    if method == "multiplicative":
        return a / s
    if method == "power":
        # a^k + b^k = 1, k > 1 when the book has a margin
        k = _bisect(lambda k: a ** k + b ** k - 1, np.ones_like(a), 4.0)
        return a ** k
    if method == "shin":
        def p(z, x):
            return (np.sqrt(z * z + 4 * (1 - z) * x * x / s) - z) / (2 * (1 - z))
        z = _bisect(lambda z: p(z, a) + p(z, b) - 1, np.zeros_like(a), 0.4)
        return p(z, a)
    raise ValueError(method)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


# ---------- data ----------

def load(odds_dir: Path):
    games = pd.read_csv(odds_dir / "games_2022_2024.csv")
    if not set(games.season.unique()) <= set(SEASONS):
        raise SystemExit("games file holds seasons outside 2022-2024; refusing")
    games = games[(games.game_type == "R") & (games.status == "completed")].copy()
    games["date"] = pd.to_datetime(games.date)
    ml = pd.read_parquet(odds_dir / "ml.parquet")
    rl = pd.read_parquet(odds_dir / "rl.parquet")
    return games, ml, rl


def book_probs(df, method):
    d = df.copy()
    d["dec_h"] = to_dec(d.home_odds)
    d["dec_a"] = to_dec(d.away_odds)
    d["fair_h"] = devig_home(1 / d.dec_h, 1 / d.dec_a, method)
    return d


def market_table(df, method, market):
    """One row per game and snapshot: fair probability of the home side,
    best prices among the three books, all-book median prices. For the run
    line, the home line is the majority line across books at that snapshot,
    and only rows at that line are used."""
    d = book_probs(df, method)
    if market == "rl":
        maj = (d.groupby(["game_id", "snap"]).line
                .agg(lambda x: x.value_counts().idxmax()).rename("maj"))
        d = d.join(maj, on=["game_id", "snap"])
        d = d[d.line == d.maj].drop(columns="maj")
    keys = ["game_id", "snap"] + (["line"] if market == "rl" else [])
    three = d[d.book.isin(BOOKS3)]
    g3 = three.groupby(keys).agg(q3=("fair_h", "mean"), n3=("fair_h", "size"),
                                 best_h=("dec_h", "max"), best_a=("dec_a", "max"))
    gall = d.groupby(keys).agg(q6=("fair_h", "mean"), n6=("fair_h", "size"),
                               med_h=("dec_h", "median"), med_a=("dec_a", "median"))
    out = g3.join(gall, how="outer").reset_index()
    return out


# ---------- statistics ----------

def blocks_of(dates: pd.Series):
    d0 = dates.min()
    return ((dates - d0).dt.days // 7).to_numpy()


def boot_ci(values, blocks, stat=np.mean, n=N_BOOT, seed=SEED):
    """95% percentile interval, resampling 7-day blocks."""
    values = np.asarray(values, float)
    ub = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in ub}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(ub, size=len(ub), replace=True)
        rows = np.concatenate([idx[b] for b in pick])
        out[i] = stat(values[rows])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), out


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------- step 1: devig on 2022 ----------

def devig_comparison(games, ml, rl):
    g22 = games[games.season == 2022].set_index("game_id")
    res = {}
    for market, df in (("ml", ml), ("rl", rl)):
        df = df[(df.season == 2022) & df.book.isin(BOOKS3)]
        per = {}
        for snap in ("p1", "close"):
            dd = df[df.snap == snap]
            dd = dd[dd.game_id.isin(g22.index)]
            margin = (g22.home_final - g22.away_final).reindex(dd.game_id).to_numpy()
            if market == "ml":
                y = (margin > 0).astype(float)
            else:
                y = np.where(dd.line.to_numpy() < 0, margin >= 2, margin >= -1).astype(float)
            dates = g22.date.reindex(dd.game_id)
            blk = blocks_of(dates.reset_index(drop=True))
            losses = {}
            for m in METHODS:
                p = book_probs(dd, m).fair_h.to_numpy()
                losses[m] = logloss(p, y)
            row = {m: float(losses[m].mean()) for m in METHODS}
            for m in ("power", "shin"):
                diff = losses[m] - losses["multiplicative"]
                lo, hi, _ = boot_ci(diff, blk)
                row[f"{m}_minus_mult"] = [float(diff.mean()), lo, hi]
            row["n_rows"] = int(len(dd))
            per[snap] = row
        best = min(METHODS, key=lambda m: per["close"][m])
        res[market] = {"by_snapshot": per, "chosen": best}
    return res


# ---------- steps 2-3: fit and dry run ----------

def model_side_prob(pred, market, line):
    if market == "ml":
        return pred.p_home_win.to_numpy()
    return np.where(line < 0, pred["p_home_m1.5"].to_numpy(), 1 - pred["p_away_m1.5"].to_numpy())


def outcome(margin, market, line):
    if market == "ml":
        return (margin > 0).astype(float)
    return np.where(line < 0, margin >= 2, margin >= -1).astype(float)


def build_frame(games, mt, pred, market, snap, variant):
    """Join bet-time market, close market, model and outcome for one season."""
    qcol, hcol, acol = (("q3", "best_h", "best_a") if variant == "best3"
                        else ("q6", "med_h", "med_a"))
    need = 2 if variant == "best3" else 3
    ncol = "n3" if variant == "best3" else "n6"
    keys = ["game_id"] + (["line"] if market == "rl" else [])
    t = mt[(mt.snap == snap) & (mt[ncol] >= need)][keys + [qcol, hcol, acol]].rename(
        columns={qcol: "q_t", hcol: "dec_h", acol: "dec_a"})
    c = mt[(mt.snap == "close") & (mt[ncol] >= need)][keys + [qcol]].rename(columns={qcol: "q_c"})
    f = t.merge(c, on=keys, how="inner")          # run line: same line at bet time and close
    f = f.merge(pred[["game_id", "p_home_win", "p_home_m1.5", "p_away_m1.5"]], on="game_id")
    f = f.merge(games[["game_id", "date", "home_final", "away_final"]], on="game_id")
    line = f["line"].to_numpy() if market == "rl" else None
    f["p"] = model_side_prob(f, market, line)
    f["y"] = outcome((f.home_final - f.away_final).to_numpy(), market, line)
    f = f.sort_values(["date", "game_id"]).reset_index(drop=True)
    return f


def bets(f, w, tau):
    b = expit(logit(f.q_t) + w * (logit(f.p) - logit(f.q_t)))
    side_home = f.p > f.q_t
    ev = np.where(side_home, b * f.dec_h - 1, (1 - b) * f.dec_a - 1)
    # a bet needs a real disagreement: with none, a positive EV could only
    # come from line shopping, which says nothing about the model
    take = (ev > tau) & (np.abs(f.p - f.q_t) <= MAX_RAW_GAP) & (f.p != f.q_t)
    s = np.where(side_home, 1.0, -1.0)
    mv = s * (f.q_c - f.q_t)                       # line movement toward the bet (prob points)
    fair_c = np.where(side_home, f.q_c, 1 - f.q_c)
    dec = np.where(side_home, f.dec_h, f.dec_a)
    clv = fair_c * dec - 1                         # value of the price taken, judged by the close
    won = np.where(side_home, f.y, 1 - f.y)
    pnl = np.where(won == 1, dec - 1, -1.0)
    out = pd.DataFrame({"date": f.date, "bet": take, "home": side_home, "mv": mv,
                        "clv": clv, "pnl": pnl, "ev": ev, "gap": f.p - f.q_t})
    return out[out["bet"]].reset_index(drop=True)


def fit(f):
    best = None
    grid = []
    for w in W_GRID:
        for tau in TAU_GRID:
            bb = bets(f, w, tau)
            n = len(bb)
            score = float(bb.mv.sum()) if n >= MIN_BETS else -np.inf
            grid.append((float(w), float(tau), n, score))
            if best is None or score > best[3]:
                best = (float(w), float(tau), n, score)
    return {"w": best[0], "tau": best[1], "n": best[2], "total_mv": best[3]}, grid


def summarize(bb):
    if len(bb) == 0:
        return {"n_bets": 0}
    blk = blocks_of(bb.date)
    out = {"n_bets": int(len(bb)), "share_home": float(bb.home.mean())}
    for col, stat in (("mv", np.mean), ("clv", np.mean), ("pnl", np.mean)):
        lo, hi, dist = boot_ci(bb[col].to_numpy(), blk, stat)
        est = float(stat(bb[col].to_numpy()))
        out[col] = {"est": est, "lo": lo, "hi": hi, "p_one_sided": float((dist <= 0).mean())}
    out["units"] = float(bb.pnl.sum())
    out["share_mv_positive"] = float((bb.mv > 0).mean())
    out["share_mv_zero"] = float((bb.mv == 0).mean())
    return out


def test_a(f):
    """G4 supporting test (a): logit P(y) = alpha + gamma*market + beta*(model - market),
    standard errors clustered by 7-day block; all games, not just bets."""
    lq = logit(f.q_t.to_numpy())
    X = sm.add_constant(np.column_stack([lq, logit(f.p.to_numpy()) - lq]))
    grp = blocks_of(f.date)
    m = sm.GLM(f.y.to_numpy(), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": grp})
    b, se = float(m.params[2]), float(m.bse[2])
    from scipy.stats import norm
    return {"alpha": float(m.params[0]), "gamma": float(m.params[1]), "beta": b, "beta_se": se,
            "beta_lo": b - 1.96 * se, "beta_hi": b + 1.96 * se,
            "p_one_sided": float(1 - norm.cdf(b / se)), "n_games": int(len(f))}


def run(odds_dir: Path, pred_p1: Path, pred_p2: Path, out_dir: Path, model_p1: str = "M1", model_p2: str = "M1u"):
    """model_p1 / model_p2: the full-game champion at each point (M1 until
    the umpire group was adopted at P2 on 2 Oct 2026, design E14: M1u)."""
    games, ml, rl = load(odds_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"devig_2022": devig_comparison(games, ml, rl)}
    preds = {}
    report["models"] = {"P1": model_p1, "P2": model_p2}
    for point, d, mdl in (("P1", pred_p1, model_p1), ("P2", pred_p2, model_p2)):
        p = pd.read_parquet(d / "game_predictions.parquet")
        p = p[(p.segment == "full") & (p.model == mdl) & (p.version == "final")]
        if p.empty:
            raise SystemExit(f"no final full-game predictions for {mdl} in {d}")
        if not set(p.season.unique()) <= {2023, 2024}:
            raise SystemExit("predictions hold seasons outside 2023-2024; refusing")
        preds[point] = p
    timing = {"p1": "P1", "lineup": "P2"}
    report["runs"] = {}
    for market, df in (("ml", ml), ("rl", rl)):
        method = report["devig_2022"][market]["chosen"]
        mt = market_table(df[df.season.isin([2023, 2024])], method, market)
        for snap, point in timing.items():
            for variant in ("best3", "allbook"):
                key = f"{market}/{snap}/{variant}"
                pr = preds[point]
                f23 = build_frame(games, mt, pr[pr.season == 2023], market, snap, variant)
                f24 = build_frame(games, mt, pr[pr.season == 2024], market, snap, variant)
                if variant == "best3":
                    fitted, grid = fit(f23)
                    frozen = fitted
                else:
                    # the check reuses the frozen best3 rule: same w and tau, no line shopping
                    fitted, grid = dict(frozen, note="w and tau from best3, not refitted"), []
                r = {"devig": method, "fit_2023": fitted,
                     "games": {"2023": int(len(f23)), "2024": int(len(f24))},
                     "in_sample_2023": summarize(bets(f23, fitted["w"], fitted["tau"])),
                     "dry_run_2024": summarize(bets(f24, fitted["w"], fitted["tau"])),
                     "test_a_2023": test_a(f23), "test_a_2024": test_a(f24)}
                report["runs"][key] = r
                if grid:
                    pd.DataFrame(grid, columns=["w", "tau", "n", "total_mv"]).to_csv(
                        out_dir / f"grid_{market}_{snap}_{variant}.csv", index=False)
                print(key, json.dumps({"fit": fitted, "2024": r["dry_run_2024"].get("mv")}))
    (out_dir / "market_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", required=True, type=Path)
    ap.add_argument("--pred-p1", required=True, type=Path)
    ap.add_argument("--pred-p2", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-p1", default="M1", help="full-game champion at P1")
    ap.add_argument("--model-p2", default="M1u", help="full-game champion at P2 (M1u since design E14; M1 before)")
    a = ap.parse_args()
    run(a.odds, a.pred_p1, a.pred_p2, a.out, a.model_p1, a.model_p2)


if __name__ == "__main__":
    main()

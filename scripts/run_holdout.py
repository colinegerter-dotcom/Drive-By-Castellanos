#!/usr/bin/env python3
"""
Phase F runner: the pre-registered 2025-2026 test (claude/pre-registration.md,
signed by Colin 4 Oct 2026 at commit 6ccf6af, tag prereg-2025).

The tagged code refuses 2025-2026 on purpose (folds stop at 2024, held-out
outcomes are blanked, the market test rejects other seasons). This script
only ORCHESTRATES that code for the signed test. It changes no model, feature
or gate code: it sets the fold table, the 3-event calibration gate and the
frozen betting rule from the signed document, and enforces its order.

Run one fold at a time, 2025 first. For each fold:

  tune      prior and league-level settings inside the fold's training
            seasons (as tune_priors.py and tune_league.py do); scores are
            written to file, not printed
  features  build P2 and P1 features with the chosen settings; the test
            season's run columns are moved to a separate outcomes file
  late      late-innings table for the fold's training seasons only
  predict   fit on the training seasons, predict the test season, decide
            bets from bet-time prices only; write predictions, bets and a
            manifest with SHA-256 checksums. No test-season outcome or
            closing price is read

The 2026 fold's stages refuse to start until the 2025 fold's manifest exists,
because the 2026 fold trains on 2025 outcomes. Then, once, for all folds:

  score     verify every checksum; only then load outcomes and closing
            prices; G3 (2025), the 2026 confirmation, G4 (2025+2026 pooled)

    python scripts/run_holdout.py --stage tune --fold F25 --mode holdout --inputs inputs_holdout --work holdout/
    ... features, late, predict (with --odds <private odds dir>) for F25, then the same for F26 ...
    python scripts/run_holdout.py --stage score --mode holdout --inputs inputs_holdout --work holdout/ --odds <private odds dir>

--mode dev runs the same machinery on the development folds (A: train 2022,
test 2023; B: train 2022-2023, test 2024) as a parity check against phase C.

Test-season games for prediction are chosen without their scores: regular
season, status completed, 9 or more innings, not resumed, both team rows. Their
run columns are set to 0 as a placeholder for the prediction step only (phase
C's helpers select games by "target present"); no prediction uses a target
value. At scoring, games without both outcome rows are dropped and counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.models import calibration, harness, market_d as md, phase_c, tune_league, tune_priors  # noqa: E402
from pipelines.models.harness import prepare  # noqa: E402

FOLDS = {
    "holdout": {"F25": ([2022, 2023, 2024], 2025), "F26": ([2022, 2023, 2024, 2025], 2026)},
    "dev": {"A": ([2022], 2023), "B": ([2022, 2023], 2024)},
}
FIRST_FEATURE_SEASON = 2021
TARGETS = ["runs_f5", "runs_8", "runs_total"]
# the signed betting rule, frozen from the 2023 fit (E9, rerun with M1u on 2 Oct)
RULE = {"ml": {"devig": "shin", "w": 0.30, "tau": 0.0125},
        "rl": {"devig": "multiplicative", "w": 0.65, "tau": 0.06}}
EVENTS_ALL = list(calibration.EVENTS_V2)
# the signed G2 events: the markets in use
EVENTS_SIGNED = [e for e in EVENTS_ALL if (e[0], e[1]) in
                 {("full", "home_win"), ("full", "home_m1.5"), ("f5", "home_win")}]
FINALISTS = {"full": "M1u", "f5": "M1uz"}         # at P2
REPORT_P1 = {"full": "M1", "f5": "M1z"}          # at P1, reported only
# bet variants decided in predict: (market, snapshot, book variant, point)
VARIANTS = [("ml", "lineup", "best3", "P2"), ("rl", "lineup", "best3", "P2"),       # deciding
            ("ml", "lineup", "allbook", "P2"), ("rl", "lineup", "allbook", "P2"),   # conservative check
            ("ml", "p1", "best3", "P1"), ("rl", "p1", "best3", "P1")]               # 10am version, report only


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seasons_through(s):
    return list(range(FIRST_FEATURE_SEASON, s + 1))


def checks():
    from pipelines.features import rookie, umpire
    assert rookie.ENABLED is False, "rookie translation must stay off (signed)"
    assert tuple(umpire.FIT_SEASONS) == (2021,), "umpire grid must stay frozen on 2021 (signed)"


def require_earlier(work, folds, fold):
    """A later fold trains on the earlier fold's test season: its stages may
    start only after the earlier fold's predictions are locked."""
    order = list(folds)
    for earlier in order[:order.index(fold)]:
        if not (work / f"manifest_{earlier}.json").exists():
            raise SystemExit(f"fold {fold} needs fold {earlier}'s predictions locked first (manifest_{earlier}.json)")


# ----------------------------------------------------------------- tune
def stage_tune(inputs, work, fold, train, test):
    last = max(train)
    build_seasons = seasons_through(last)          # never the test season
    rows = []
    for s in tune_priors.settings():
        p = work / f"tune_{fold}_prior_{tune_priors.tag(s)}.parquet"
        if not p.exists():
            tune_priors.build(inputs, s, build_seasons, str(p))
        f = prepare(pd.read_parquet(p))
        assert f.season.max() <= last
        if len(train) == 1:
            d = pd.to_datetime(f.date)
            fit_m = (f.season == last) & (d < f"{last}-07-01")
            val_m = (f.season == last) & (d >= f"{last}-07-01")
        else:
            fit_m, val_m = f.season.isin(train[:-1]), f.season == last
        rows.append({**s, "tag": tune_priors.tag(s), "score": tune_priors.inner_score(f, fit_m, val_m)})
    r = pd.DataFrame(rows)
    start = r[r.tag == tune_priors.tag(tune_priors.START)].iloc[0]
    best = r.loc[r.score.idxmax()]
    pick = best if best.score > start.score + tune_priors.TIE_MARGIN else start
    prior = {k: (pick[k].item() if hasattr(pick[k], "item") else pick[k]) for k in tune_priors.START}
    lrows = []
    for k in tune_league.GRID:
        p = work / f"tune_{fold}_league_{k}.parquet"
        if not p.exists():
            tune_league.build(inputs, k, build_seasons, str(p))
        f = prepare(pd.read_parquet(p))
        assert f.season.max() <= last
        lrows.append({"k_league": k, "score": tune_league.in_sample(f, train)})
    lr = pd.DataFrame(lrows)
    lstart = float(lr[lr.k_league == tune_league.START].score.iloc[0])
    lbest = lr.loc[lr.score.idxmax()]
    k_league = int(lbest.k_league) if lbest.score > lstart + tune_league.TIE_MARGIN else tune_league.START
    res = {"prior": prior, "k_league": k_league, "prior_grid": rows, "league_grid": lrows}
    (work / f"tuning_{fold}.json").write_text(json.dumps(res, indent=2, default=str))
    print(fold, "chosen:", json.dumps({"prior": prior, "k_league": k_league}))


# ------------------------------------------------------------- features
def _build(inputs, prior, k_league, seasons, out, point):
    from pipelines.features import build as fb
    from pipelines.features import environment, priors
    saved = (dict(priors.K_HIT), dict(priors.PIT_WEIGHTS), dict(priors.K_PIT), environment.K_LEAGUE)
    try:
        priors.K_HIT = {**priors.K_HIT, "woba": prior["woba_k"]}
        priors.PIT_WEIGHTS = dict(tune_priors.PIT_WEIGHTS[prior["pit_weights"]])
        priors.K_PIT = {**priors.K_PIT, "k": prior["pit_k_k"]}
        environment.K_LEAGUE = k_league
        con, _ = fb.build_features(inputs, seasons, point)
        con.execute(f"copy features to '{out}' (format parquet, compression zstd)")
        con.close()
    finally:
        priors.K_HIT, priors.PIT_WEIGHTS, priors.K_PIT, environment.K_LEAGUE = saved


def stage_features(inputs, work, fold, train, test):
    t = json.loads((work / f"tuning_{fold}.json").read_text())
    for point in ("P2", "P1"):
        raw = work / f"raw_{fold}_{point}.parquet"
        _build(inputs, t["prior"], t["k_league"], seasons_through(test), str(raw), point)
        f = pd.read_parquet(raw)
        outc = f.loc[f.season == test, ["game_id", "bat_team"] + TARGETS]
        outc.to_parquet(work / f"outcomes_{fold}_{point}.parquet")      # not read until scoring
        for c in TARGETS:
            f[c] = f[c].astype(float)
        f.loc[f.season == test, TARGETS] = np.nan
        f.to_parquet(work / f"features_{fold}_{point}.parquet")
        raw.unlink()
    print(fold, "features written; test-season runs moved to outcomes files")


def stage_late(inputs, work, fold, train, test):
    from pipelines.models.late_game import build_late_innings
    build_late_innings(inputs, seasons_through(max(train)), str(work / f"late_{fold}.parquet"))
    late = pd.read_parquet(work / f"late_{fold}.parquet")
    assert late.season.max() <= max(train)


# -------------------------------------------------------------- predict
def test_games(games_csv, test):
    g = pd.read_csv(games_csv, usecols=["game_id", "season", "game_type", "status", "innings"])
    g = g[(g.season == test) & (g.game_type == "R") & (g.status == "completed") & (g.innings >= 9)]
    return set(g.game_id)


def set_folds(folds_one, fold_weight=1.0):
    """Point the tagged phase C code at one fold (module tables only)."""
    for mod in (harness, phase_c):
        mod.FOLDS = folds_one
        mod.FOLD_WEIGHT = {k: fold_weight for k in folds_one}
        mod.HELD_OUT = set()
    phase_c.SEASON_OF = {k: v[1] for k, v in folds_one.items()}
    inner = {}
    for k, (train, test) in folds_one.items():
        last = max(train)
        inner[k] = (lambda f, last=last: (f.season >= FIRST_FEATURE_SEASON) & (f.season < last),
                    lambda f, last=last: f.season == last)
    phase_c.INNER = inner


def predict_point(work, inputs, fold, train, test, point, group, late):
    set_folds({fold: (train, test)})           # before load_features: HELD_OUT must be empty
    f = phase_c.load_features(str(work / f"features_{fold}_{point}.parquet"), str(Path(inputs) / "games.csv"))
    assert f.loc[f.season == test, TARGETS].isna().all().all(), "test-season runs present"
    for s in train:
        assert f.loc[f.season == s, "runs_f5"].notna().sum() > 1000, f"training season {s} lacks targets"
    keep = test_games(Path(inputs) / "games.csv", test)
    m = (f.season == test) & f.game_id.isin(keep)
    for c in TARGETS + ["runs_68"]:
        f.loc[m, c] = 0.0          # placeholder so phase C keeps the game; never used for a prediction
    phase_c.set_group(group)
    preds, models, diags, spread = phase_c.build({fold: f}, late)
    out = {}
    for (seg, fd, name), p in preds.items():
        if seg == "full8":
            continue
        X = p["spread:X"]          # the final version: spread fix, no shape step (kept "none" in phase C)
        assert np.isfinite(X).all()
        lay = phase_c.LAY[seg]
        d = pd.DataFrame({"game_id": p["game_id"], "date": p["date"],
                          "p_home_win": X[:, lay.h > lay.a].sum(1)})
        if seg == "full":
            d["p_home_m1.5"] = X[:, (lay.h - lay.a) >= 2].sum(1)
            d["p_away_m1.5"] = X[:, (lay.a - lay.h) >= 2].sum(1)
        np.save(work / f"grid_{fold}_{point}_{seg}_{name}.npy", X)
        d.to_parquet(work / f"pred_{fold}_{point}_{seg}_{name}.parquet")
        out[f"{seg}/{name}"] = int(len(d))
    (work / f"models_{fold}_{point}.json").write_text(json.dumps({"models": models, "spread": {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("test_")} for k, v in spread.items()}}, default=str))
    return out


def _take(f, rule):
    """md.bets' decision, kept with game keys (same formula and filters)."""
    b = md.expit(md.logit(f.q_t) + rule["w"] * (md.logit(f.p) - md.logit(f.q_t)))
    home = f.p > f.q_t
    ev = np.where(home, b * f.dec_h - 1, (1 - b) * f.dec_a - 1)
    take = (ev > rule["tau"]) & (np.abs(f.p - f.q_t) <= md.MAX_RAW_GAP) & (f.p != f.q_t)
    return take, home, ev


def decide_bets(odds_dir, pred_full, market, snap, variant):
    """Bets from bet-time prices only: the close is never loaded here."""
    df = pd.read_parquet(Path(odds_dir) / f"{market}.parquet")
    df = df[df.snap == snap]
    rule = RULE[market]
    mt = md.market_table(df, rule["devig"], market)
    qcol, hcol, acol = (("q3", "best_h", "best_a") if variant == "best3" else ("q6", "med_h", "med_a"))
    ncol, need = ("n3", 2) if variant == "best3" else ("n6", 3)
    keys = ["game_id"] + (["line"] if market == "rl" else [])
    t = mt[(mt.snap == snap) & (mt[ncol] >= need)][keys + [qcol, hcol, acol]].rename(
        columns={qcol: "q_t", hcol: "dec_h", acol: "dec_a"})
    f = t.merge(pred_full, on="game_id")
    line = f["line"].to_numpy() if market == "rl" else None
    f["p"] = md.model_side_prob(f, market, line)
    take, home, ev = _take(f, rule)
    out = f.loc[take, keys + ["date", "q_t", "dec_h", "dec_a", "p"]].copy()
    out["side_home"] = np.asarray(home)[take]
    out["ev"] = ev[take]
    if market == "ml":
        out["line"] = 0.0
    return out.sort_values(["date", "game_id"]).reset_index(drop=True)


def stage_predict(inputs, work, fold, train, test, odds_dir):
    late = pd.read_parquet(work / f"late_{fold}.parquet")
    assert late.season.max() <= max(train)
    counts = {"P2": predict_point(work, inputs, fold, train, test, "P2", "ump", late),
              "P1": predict_point(work, inputs, fold, train, test, "P1", "none", late)}
    if odds_dir is not None:
        for market, snap, variant, point in VARIANTS:
            name = FINALISTS["full"] if point == "P2" else REPORT_P1["full"]
            pf = pd.read_parquet(work / f"pred_{fold}_{point}_full_{name}.parquet")
            b = decide_bets(odds_dir, pf, market, snap, variant)
            b.to_parquet(work / f"bets_{fold}_{market}_{snap}_{variant}.parquet")
            counts[f"bets/{market}/{snap}/{variant}"] = int(len(b))
    files = (sorted(work.glob(f"pred_{fold}_*.parquet")) + sorted(work.glob(f"grid_{fold}_*.npy"))
             + sorted(work.glob(f"bets_{fold}_*.parquet")) + sorted(work.glob(f"features_{fold}_*.parquet"))
             + sorted(work.glob(f"outcomes_{fold}_*.parquet")) + [work / f"tuning_{fold}.json", work / f"late_{fold}.parquet"]
             + sorted(work.glob(f"models_{fold}_*.json")))
    inputs_files = sorted(p for p in Path(inputs).iterdir() if p.is_file())
    odds_files = sorted(Path(odds_dir).glob("*")) if odds_dir else []
    manifest = {"fold": fold, "train": train, "test": test, "counts": counts,
                "runner_sha256": sha(Path(__file__)),
                "outputs_sha256": {p.name: sha(p) for p in files},
                "inputs_sha256": {p.name: sha(p) for p in inputs_files},
                "odds_sha256": {p.name: sha(p) for p in odds_files if p.is_file()}}
    (work / f"manifest_{fold}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(counts, indent=2))
    print(f"fold {fold}: predictions and bets locked in manifest_{fold}.json. No test outcome or close was read.")


# ---------------------------------------------------------------- score
def verify(work, folds, inputs, odds_dir):
    """Every checksum in every manifest: outputs, inputs, odds, the runner
    (the same runner for all folds and now)."""
    runners = set()
    for fold in folds:
        m = json.loads((work / f"manifest_{fold}.json").read_text())
        for name, h in m["outputs_sha256"].items():
            assert sha(work / name) == h, f"{name} changed after it was locked"
        for name, h in m["inputs_sha256"].items():
            assert sha(Path(inputs) / name) == h, f"input {name} changed after fold {fold} was locked"
        for name, h in m["odds_sha256"].items():
            assert odds_dir is not None and sha(Path(odds_dir) / name) == h, f"odds file {name} changed"
        runners.add(m["runner_sha256"])
    runners.add(sha(Path(__file__)))
    assert len(runners) == 1, "the runner changed between folds or since they were locked"
    print("all locked files, inputs, odds and the runner verified against the manifests")


def _scores(work, inputs, fold, point, seg, name):
    X = np.load(work / f"grid_{fold}_{point}_{seg}_{name}.npy")
    p = pd.read_parquet(work / f"pred_{fold}_{point}_{seg}_{name}.parquet")
    o = pd.read_parquet(work / f"outcomes_{fold}_{point}.parquet")
    g = pd.read_csv(Path(inputs) / "games.csv", usecols=["game_id", "home_team"])
    o = o.merge(g, on="game_id")
    col = "runs_f5" if seg == "f5" else "runs_total"
    h = o[o.bat_team == o.home_team].set_index("game_id")[col]
    a = o[o.bat_team != o.home_team].set_index("game_id")[col]
    ok = p.game_id.isin(h.dropna().index) & p.game_id.isin(a.dropna().index)
    X, p = X[ok.to_numpy()], p[ok].reset_index(drop=True)
    hh, aa = h.loc[p.game_id].to_numpy(int), a.loc[p.game_id].to_numpy(int)
    ls = calibration.log_score(X, phase_c.LAY[seg], hh, aa)
    return X, p, hh, aa, ls, int((~ok).sum())


def _boot_draws(dates, diff, n=phase_c.BOOT_N, seed=phase_c.BOOT_SEED):
    """phase_c._boot's resampling (7-day blocks, same seed), returning the draws."""
    d = pd.to_datetime(pd.Series(dates))
    block = ((d - d.min()).dt.days // 7).to_numpy()
    blocks = np.unique(block)
    idx = {b: np.where(block == b)[0] for b in blocks}
    rng = np.random.default_rng(seed)
    w = np.ones(len(diff))
    boots = np.empty(n)
    for i in range(n):
        ix = np.concatenate([idx[b] for b in rng.choice(blocks, size=len(blocks), replace=True)])
        boots[i] = np.sum(w[ix] * diff[ix]) / np.sum(w[ix])
    return boots


def g1(work, inputs, fold, point, seg, name):
    X, p, h, a, ls, dropped = _scores(work, inputs, fold, point, seg, name)
    _, pb, _, _, lsb, _ = _scores(work, inputs, fold, point, seg, "B1")
    assert (p.game_id.to_numpy() == pb.game_id.to_numpy()).all()
    d = ls - lsb
    est, lo, hi, se = phase_c._boot(p.date, d, np.ones(len(d)))
    draws = _boot_draws(p.date, d)
    return {"games": int(len(d)), "dropped_no_outcome": dropped, "per_game": est, "lo95": lo, "hi95": hi,
            "se": se, "_draws": draws}


def holm(results, alpha=0.05):
    """Holm across the finalists on the bootstrap: the stronger result's
    two-sided interval at 1 - alpha/2 (lower bound at the 1.25th percentile
    for two finalists), the other's at 1 - alpha; a finalist passes if its
    Holm-adjusted lower bound is above zero and every stronger one passed.
    The normal-approximation p-values are reported alongside."""
    from scipy.stats import norm
    items = sorted(results.items(), key=lambda kv: np.mean(kv[1]["_draws"] <= 0))   # strongest first
    m = len(items)
    out, ok = {}, True
    for i, (k, r) in enumerate(items):
        a_i = alpha / (m - i)
        lower = float(np.percentile(r["_draws"], 100 * a_i / 2))
        ok = ok and lower > 0
        rr = {kk: v for kk, v in r.items() if kk != "_draws"}
        rr.update({"holm_alpha": a_i, "holm_lower": lower, "passes": bool(ok),
                   "p_two_sided_normal": float(2 * (1 - norm.cdf(abs(r["per_game"]) / r["se"])))})
        out[k] = rr
    return out


def g2(work, inputs, fold, point, events):
    calibration.EVENTS_V2 = events
    calibration.V2_CHECKS = 3 * len(events)
    try:
        grids, lays, obs, dts = {}, {}, {}, {}
        for seg, name in FINALISTS.items():
            X, p, h, a, _, _ = _scores(work, inputs, fold, point, seg, name)
            grids[seg], lays[seg], obs[seg], dts[seg] = X, phase_c.LAY[seg], (h, a), p.date.to_numpy()
        return calibration.gate_g2_v2(grids, lays, obs, dts)
    finally:
        calibration.EVENTS_V2 = EVENTS_ALL
        calibration.V2_CHECKS = 3 * len(EVENTS_ALL)


def g4(odds_dir, work, folds, market, snap, variant, point):
    """Line move toward the bet on all test seasons pooled, frozen rule.
    Scored bets must be among the bets locked before the close was read."""
    rule = RULE[market]
    games = pd.read_csv(Path(odds_dir) / "games_holdout.csv")
    games = games[(games.game_type == "R") & (games.status == "completed")].copy()
    games["date"] = pd.to_datetime(games.date)
    df = pd.read_parquet(Path(odds_dir) / f"{market}.parquet")
    mt = md.market_table(df, rule["devig"], market)
    name = FINALISTS["full"] if point == "P2" else REPORT_P1["full"]
    frames, allb, unscored = [], [], {}
    for fold, (train, test) in folds.items():
        pf = pd.read_parquet(work / f"pred_{fold}_{point}_full_{name}.parquet")
        fr = md.build_frame(games[games.season == test], mt, pf, market, snap, variant)
        bb = md.bets(fr, rule["w"], rule["tau"])
        take, home, _ = _take(fr, rule)
        assert int(take.sum()) == len(bb)
        keys = set(zip(fr.game_id[take], (fr["line"][take] if market == "rl" else pd.Series(0.0, index=fr.index)[take]),
                       np.asarray(home)[take]))
        saved = pd.read_parquet(work / f"bets_{fold}_{market}_{snap}_{variant}.parquet")
        locked = set(zip(saved.game_id, saved.line, saved.side_home))
        assert keys <= locked, "a scored bet was not among the bets locked before the close was read"
        unscored[test] = int(len(locked - keys))           # no close at the same line, or game not completed
        frames.append(fr.assign(season=test))
        allb.append(bb.assign(season=test))
    fr, bb = pd.concat(frames, ignore_index=True), pd.concat(allb, ignore_index=True)
    out = {"rule": rule, "market": market, "snapshot": snap, "books": variant, "model_point": point,
           "bets": int(len(bb)), "locked_but_unscored": unscored,
           "by_season": {int(s): md.summarize(q) for s, q in bb.groupby("season")}}
    if len(bb) == 0:
        out.update({"passes_b": False, "note": "no bets"})
        return out
    s = md.summarize(bb)
    _, _, dist = md.boot_ci(bb.mv.to_numpy(), md.blocks_of(bb.date), np.mean)
    lo = float(np.percentile(dist, 5))
    out.update({"line_move": s["mv"], "line_move_one_sided_lo95": lo, "passes_b": bool(lo > 0),
                "test_a": md.test_a(fr), "reported": {k: s.get(k) for k in ("clv", "pnl", "units", "share_home")}})
    return out


def stage_score(inputs, work, folds, odds_dir, mode):
    verify(work, folds, inputs, odds_dir)
    order = list(folds)
    first, rest = order[0], order[1:]
    rep = {"mode": mode, "read_order": order}
    # G3 on the first test season
    r1 = {seg: g1(work, inputs, first, "P2", seg, name) for seg, name in FINALISTS.items()}
    rep["G3_G1"] = holm(r1)
    rep["G3_G2"] = g2(work, inputs, first, "P2", EVENTS_SIGNED)
    rep["G2_all_events_report"] = g2(work, inputs, first, "P2", EVENTS_ALL)
    rep["P1_report_G1"] = {seg: {k: v for k, v in g1(work, inputs, first, "P1", seg, name).items() if k != "_draws"}
                           for seg, name in REPORT_P1.items()}
    confirms = True
    for fold in rest:
        te2 = folds[fold][1]
        c = {seg: {k: v for k, v in g1(work, inputs, fold, "P2", seg, name).items() if k != "_draws"}
             for seg, name in FINALISTS.items()}
        c2 = g2(work, inputs, fold, "P2", EVENTS_SIGNED)
        both_pos = all(v["per_game"] > 0 for v in c.values())
        ml_ok = c2["events"]["full/home_win"]["verdict"] != "fail"
        # signed text: "both finalists' G1 differences must be positive" and
        # "G2 must not fail on the moneyline event" (one joint condition)
        rep[f"confirm_{te2}"] = {"G1": c, "G2": c2, "both_finalists_positive": both_pos,
                                 "moneyline_g2_not_fail": ml_ok, "confirms": bool(both_pos and ml_ok)}
        rep[f"P1_report_G1_{te2}"] = {seg: {k: v for k, v in g1(work, inputs, fold, "P1", seg, name).items() if k != "_draws"}
                                      for seg, name in REPORT_P1.items()}
        confirms = confirms and both_pos and ml_ok
    ev = rep["G3_G2"]["events"]
    g2_ok = {"full": ev["full/home_win"]["verdict"] != "fail" and ev["full/home_m1.5"]["verdict"] != "fail",
             "f5": ev["f5/home_win"]["verdict"] != "fail"}
    rep["G3_passes"] = {seg: bool(rep["G3_G1"][seg]["passes"] and g2_ok[seg] and confirms) for seg in FINALISTS}
    # G4, pooled; moneyline first, run line only if the moneyline passes
    if odds_dir is not None:
        ml = g4(odds_dir, work, folds, "ml", "lineup", "best3", "P2")
        ml["passes"] = bool(ml.get("passes_b") and ml.get("test_a", {}).get("beta", -1) > 0)
        rep["G4_moneyline"] = ml
        if ml["passes"]:
            rl = g4(odds_dir, work, folds, "rl", "lineup", "best3", "P2")
            rl["passes"] = bool(rl.get("passes_b") and rl.get("test_a", {}).get("beta", -1) > 0)
            rep["G4_runline"] = rl
        else:
            rep["G4_runline"] = "not tested: the moneyline did not pass G4 (fixed order)"
        rep["reported_only"] = {f"{m}/{s}/{v}/{pt}": g4(odds_dir, work, folds, m, s, v, pt)
                                for m, s, v, pt in VARIANTS[2:]}
        if not ml["passes"]:
            rep["reported_only"]["rl/lineup/best3/P2"] = g4(odds_dir, work, folds, "rl", "lineup", "best3", "P2")
    (work / f"score_{mode}.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({"G3_passes": rep["G3_passes"],
                      "G4_moneyline_passes": rep.get("G4_moneyline", {}).get("passes")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "features", "late", "predict", "score"])
    ap.add_argument("--mode", required=True, choices=["holdout", "dev"])
    ap.add_argument("--fold", default=None)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--odds", default=None)
    a = ap.parse_args()
    checks()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    folds = FOLDS[a.mode]
    if a.mode == "dev":
        assert all(t <= 2024 for _, t in folds.values())
    if a.stage == "score":
        stage_score(a.inputs, work, folds, a.odds, a.mode)
        return
    if a.fold not in folds:
        raise SystemExit(f"--fold must be one of {list(folds)}")
    require_earlier(work, folds, a.fold)
    if (work / f"manifest_{a.fold}.json").exists():
        raise SystemExit(f"fold {a.fold} is locked (manifest_{a.fold}.json exists); its stages can't be rerun")
    train, test = folds[a.fold]
    {"tune": stage_tune, "features": stage_features, "late": stage_late}.get(
        a.stage, lambda *x: stage_predict(*x, a.odds))(a.inputs, work, a.fold, train, test)


if __name__ == "__main__":
    main()

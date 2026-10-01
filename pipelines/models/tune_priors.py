"""
Tuning the prior layer inside each fold's training seasons (design 5.4, C9).

    python -m pipelines.models.tune_priors --inputs inputs/ --work tune/

The prior layer (features/priors.py) blends each player's seasons and pulls
small samples toward the league average. Its settings were starting values.
Design 5.4 says they are tuned inside each fold's training seasons only, so
the seasons being scored never help choose them.

Grid (12 settings, fixed before looking):
  hitter wOBA shrinkage k         150, 300 (start), 600 plate appearances
  pitcher season weights          3/2/1 (start) or 5/4/3 for the last 3 seasons
  pitcher strikeout shrinkage k   70 (start) or 140 batters faced

Inner validation, never touching the fold's test season:
  fold A (tests 2023)   fit M1 on 2022 before 1 July, score 2022 from 1 July
  fold B (tests 2024)   fit M1 on 2022, score 2023
Score: M1's joint log score per game, F5 plus 8 innings (one setting serves
both segments, since they share the features). The fold keeps the setting
with the best score; ties within 0.0005 per game go to the starting values.

Tuning builds cover 2021-2023 only, so no 2024 information exists anywhere in
the choice. The chosen fold B setting is then built for 2021-2024.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .harness import model_specs, prepare, usable
from .nb import NBModel, nb_logpmf

HIT_WOBA_K = [150, 300, 600]
PIT_WEIGHTS = {"3/2/1": {1: 0.75, 2: 0.5, 3: 0.25}, "5/4/3": {1: 0.8, 2: 0.64, 3: 0.48}}
PIT_K_K = [70, 140]
START = {"woba_k": 300, "pit_weights": "3/2/1", "pit_k_k": 70}
TIE_MARGIN = 0.0005


def settings():
    for wk, pw, pk in itertools.product(HIT_WOBA_K, PIT_WEIGHTS, PIT_K_K):
        yield {"woba_k": wk, "pit_weights": pw, "pit_k_k": pk}


def tag(s):
    return f"wk{s['woba_k']}_pw{s['pit_weights'].replace('/', '')}_pk{s['pit_k_k']}"


def build(inputs, s, seasons, out, point="P1"):
    """Build features with one prior setting (patches the module constants)."""
    from pipelines.features import build as fb
    from pipelines.features import priors
    saved = (dict(priors.K_HIT), dict(priors.PIT_WEIGHTS), dict(priors.K_PIT))
    try:
        priors.K_HIT = {**priors.K_HIT, "woba": s["woba_k"]}
        priors.PIT_WEIGHTS = dict(PIT_WEIGHTS[s["pit_weights"]])
        priors.K_PIT = {**priors.K_PIT, "k": s["pit_k_k"]}
        con, notes = fb.build_features(inputs, seasons, point)
        con.execute(f"copy features to '{out}' (format parquet, compression zstd)")
        con.close()
    finally:
        priors.K_HIT, priors.PIT_WEIGHTS, priors.K_PIT = saved
    return notes


def inner_score(f, fit_mask, val_mask):
    """M1 joint log score per game on the validation games, F5 + 8 innings."""
    total = 0.0
    for seg, target, off in (("f5", "runs_f5", "log_league_env_f5"), ("full8", "runs_8", "log_league_env_8")):
        tr = usable(f[fit_mask], target)
        va = usable(f[val_mask], target)
        m = NBModel(model_specs(seg)["M1"], offset=off).fit(tr, tr[target])
        mu, al = m.predict(va)
        total += nb_logpmf(va[target].to_numpy(float), mu, al).sum() / (len(va) / 2)
    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    results = []
    for s in settings():
        path = work / f"feat_{tag(s)}_2021_2023.parquet"
        if not path.exists():
            build(a.inputs, s, [2021, 2022, 2023], str(path))
        f = prepare(pd.read_parquet(path))
        assert not (f.season >= 2024).any()
        d = pd.to_datetime(f.date)
        score_a = inner_score(f, (f.season == 2022) & (d < "2022-07-01"), (f.season == 2022) & (d >= "2022-07-01"))
        score_b = inner_score(f, f.season == 2022, f.season == 2023)
        results.append({**s, "tag": tag(s), "fold_A_inner": score_a, "fold_B_inner": score_b})
        print(results[-1], flush=True)
    r = pd.DataFrame(results)
    chosen = {}
    for fold in ("A", "B"):
        col = f"fold_{fold}_inner"
        start = r[r.tag == tag(START)][col].iloc[0]
        best = r.loc[r[col].idxmax()]
        pick = best if best[col] > start + TIE_MARGIN else r[r.tag == tag(START)].iloc[0]
        chosen[fold] = {"tag": pick.tag, "setting": {k: (pick[k].item() if hasattr(pick[k], "item") else pick[k]) for k in START},
                        "inner_score": float(pick[col]), "start_inner_score": float(start),
                        "gain_vs_start": float(pick[col] - start)}
    (work / "tuning.json").write_text(json.dumps({"grid": results, "chosen": chosen}, indent=2))
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()

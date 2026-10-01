"""
Tuning the league-level blend inside each fold's training seasons.

    python -m pipelines.models.tune_league --inputs inputs/ --work tune/

league_env (features/environment.py) = this season's runs so far blended
with K_LEAGUE team-games of last season's average. It sets each prediction's
level (the model's offset). Its starting value (500) was a guess. The first
G2 read showed totals under-predicted early in 2023 (the 2023 rule changes
raised scoring and the level caught up slowly), so the blend is tuned here,
inside the training seasons only, like the prior layer (design 5.4, C9).

Grid (fixed before looking): K_LEAGUE 100, 200, 300, 500 (start), 800.
Score: M1's joint log score, F5 plus 8 innings, fitted and scored on the
fold's training seasons (fold A: 2022; fold B: 2022-2023). Fitting on the same
games is fine here: the level enters as an offset with its coefficient fixed
at 1, so the fit can't bend to a badly timed level path; only the path
differs between settings. Ties within 0.0005 per game go to the start.
Builds cover 2021-2023 only; the chosen fold B setting is rebuilt for
2021-2024 afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .harness import model_specs, prepare, usable
from .nb import NBModel, nb_logpmf

GRID = [100, 200, 300, 500, 800]
START = 500
TIE_MARGIN = 0.0005


def build(inputs, k, seasons, out, point="P1"):
    from pipelines.features import build as fb
    from pipelines.features import environment
    saved = environment.K_LEAGUE
    try:
        environment.K_LEAGUE = k
        con, _ = fb.build_features(inputs, seasons, point)
        con.execute(f"copy features to '{out}' (format parquet, compression zstd)")
        con.close()
    finally:
        environment.K_LEAGUE = saved


def in_sample(f, seasons):
    total = 0.0
    for seg, target, off in (("f5", "runs_f5", "log_league_env_f5"), ("full8", "runs_8", "log_league_env_8")):
        tr = usable(f[f.season.isin(seasons)], target)
        m = NBModel(model_specs(seg)["M1"], offset=off).fit(tr, tr[target])
        mu, al = m.predict(tr)
        total += nb_logpmf(tr[target].to_numpy(float), mu, al).sum() / (len(tr) / 2)
    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in GRID:
        path = work / f"feat_kleague{k}_2021_2023.parquet"
        if not path.exists():
            build(a.inputs, k, [2021, 2022, 2023], str(path))
        f = prepare(pd.read_parquet(path))
        assert not (f.season >= 2024).any()
        rows.append({"k_league": k, "fold_A": in_sample(f, [2022]), "fold_B": in_sample(f, [2022, 2023])})
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    chosen = {}
    for fold in ("A", "B"):
        start = float(r.loc[r.k_league == START, f"fold_{fold}"].iloc[0])
        best = r.loc[r[f"fold_{fold}"].idxmax()]
        k = int(best.k_league) if best[f"fold_{fold}"] > start + TIE_MARGIN else START
        chosen[fold] = {"k_league": k, "score": float(r.loc[r.k_league == k, f"fold_{fold}"].iloc[0]), "start_score": start}
    (work / "tuning_league.json").write_text(json.dumps({"grid": rows, "chosen": chosen}, indent=2))
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()

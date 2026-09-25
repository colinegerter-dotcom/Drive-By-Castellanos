"""
build_features(): one row per team-game, everything the models read.

    python -m pipelines.features.build --inputs <folder> --out features.parquet [--point P1|P2]

Row = the BATTING team in one game: its lineup, against the FIELDING
team's starter and bullpen, in that park and weather. A game has two rows.
Targets (runs_f5, runs_8, runs_total) ride along for convenience; they are
outcomes, never inputs, and every feature column is listed in
FEATURE_COLUMNS so a model can't pick up a target by accident.

Cutoff: a game's features use only games whose data date is before the
game's date (P1, 10am game day, and P2, confirmed lineups). A same-day
doubleheader game 1 is NOT used for game 2 yet; that is allowed at P2 by
the design and can be added later. Being stricter than required is safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from . import environment, events, inputs, lineup, pitching, priors

# Features per segment. The model for a segment uses the matching suffix.
FEATURE_COLUMNS = {
    "f5": ["lineup_woba_f5", "lineup_k_bb_f5", "lineup_platoon",
           "sp_skill", "sp_fb_velo_delta", "sp_exp_bf", "sp_opener", "sp_tto3_share_f5",
           "pen_skill_f5", "pitching_composite_f5",
           "park_factor", "temp_f", "roof_park", "league_env_f5", "is_home"],
    "full8": ["lineup_woba_8", "lineup_k_bb_8", "lineup_platoon",
              "sp_skill", "sp_fb_velo_delta", "sp_exp_bf", "sp_opener", "sp_tto3_share_8",
              "pen_skill_8", "pitching_composite_8",
              "park_factor", "temp_f", "roof_park", "league_env_8", "is_home"],
}
FLAG_COLUMNS = ["new_park", "temp_missing", "sp_rookie", "velo_missing", "pen_missing", "lineup_rookies"]
BASELINE_COLUMNS = ["team_off_f5", "team_off_8", "team_def_f5", "team_def_8"]
TARGET_COLUMNS = ["runs_f5", "runs_8", "runs_total"]


def build_features(folder: str | Path, seasons: list[int], point: str = "P2") -> tuple[duckdb.DuckDBPyConnection, dict]:
    """Returns the DuckDB connection holding table `features`, and build
    notes (fitted coefficients, counts) worth saving next to the output."""
    con = inputs.connect(folder)
    events.build_pa(con)
    events.build_appearances(con)
    priors.build_player_seasons(con)
    lineup.slot_weights(con)
    notes: dict = {"point": point, "seasons": seasons}

    # ---- keys: two rows per game ----
    season_list = ",".join(map(str, seasons))
    con.execute(f"""
        create or replace table tg as
        select g.game_id, g.season, g.date, g.date as cutoff, g.game_type, g.venue_id, g.resumed,
               t.bat_team, t.fld_team, t.is_home, t.sp_id
        from games g,
             lateral (select g.home_team as bat_team, g.away_team as fld_team, 1 as is_home, g.away_sp as sp_id
                      union all
                      select g.away_team, g.home_team, 0, g.home_sp) t
        where g.season in ({season_list})
          and g.game_type in ('R', 'F', 'D', 'L', 'W')
    """)

    # ---- fitted pieces (on the 2021 warm-up season) ----
    skill = pitching.fit_skill(con)
    expbf = pitching.fit_exp_bf(con)
    notes["skill_coef"] = skill
    notes["exp_bf_coef"] = expbf
    seg = con.execute("select sum(w_f5), sum(w_8) from slot_w").fetchone()
    seg_pa = {"f5": float(seg[0]), "8": float(seg[1])}
    notes["segment_pa"] = seg_pa
    notes["slot_weights"] = con.execute("select * from slot_w order by slot").fetchall()

    # ---- starter ----
    con.execute("create or replace table sp_keys as select distinct sp_id as player_id, season, cutoff from tg where sp_id is not null")
    priors.pitcher_rates(con, "sp_keys", "sp_rates")
    pitching.starter_history(con, "sp_keys")
    con.execute("create or replace table hook_keys as select distinct fld_team as team_id, season, cutoff from tg")
    pitching.team_hook(con, "hook_keys")
    sp_df = con.execute(
        pitching.exp_bf_inputs_sql("tg", "sp_id", "fld_team")
        + """ , q.game_id, q.bat_team from tg q
        left join sp_hist h on h.player_id = q.sp_id and h.season = q.season and h.cutoff = q.cutoff
        left join hook k on k.team_id = q.fld_team and k.season = q.season and k.cutoff = q.cutoff"""
    ).fetchdf()
    sp_df["sp_exp_bf"] = pitching.apply_exp_bf(sp_df, expbf)
    # How many of the segment's batters the starter faces, and how many of
    # those are third-time-through, as EXPECTED values over his uncertain
    # outing length (normal, sd = the fit's error). Capping the average
    # outing at the segment length instead overstated the starter's F5
    # share (95.6% modelled vs 90.7% actual, review 25 Sep).
    sd = expbf["_rmse"]
    for sfx, spa in (("f5", seg_pa["f5"]), ("8", seg_pa["8"])):
        sp_df[f"sp_pa_{sfx}"] = pitching.expected_capped(sp_df["sp_exp_bf"], sd, spa)
        sp_df[f"tto3_pa_{sfx}"] = (pitching.expected_excess(sp_df["sp_exp_bf"], sd, 18.0)
                                   - pitching.expected_excess(sp_df["sp_exp_bf"], sd, min(spa, 27.0)))
    con.register("sp_expbf_df", sp_df[["game_id", "bat_team", "sp_exp_bf", "rest_bucket",
                                       "sp_pa_f5", "sp_pa_8", "tto3_pa_f5", "tto3_pa_8"]])

    mu = con.execute("select season + 1 as season, pit_k, pit_bb, pit_gb from league_season").fetchdf()
    con.register("mu_df", mu)
    s_sp = pitching.skill_sql(skill, "r.k_rate", "r.bb_rate", "r.gb_rate")
    s_mu = pitching.skill_sql(skill, "m.pit_k", "m.pit_bb", "m.pit_gb")
    con.execute(f"""
        create or replace table sp_feat as
        select q.game_id, q.bat_team,
               {s_sp} + case when r.rookie then {pitching.ROOKIE_PIT_RUNS} else 0 end as sp_skill,
               r.k_rate as sp_k, r.bb_rate as sp_bb, r.gb_rate as sp_gb,
               r.rookie as sp_rookie,
               case when r.velo_n_now >= 30 and r.velo_prev is not null then r.velo_now - r.velo_prev end as sp_fb_velo_delta,
               e.sp_exp_bf, e.rest_bucket as sp_rest_bucket,
               e.sp_pa_f5, e.sp_pa_8, e.tto3_pa_f5, e.tto3_pa_8,
               -- opener: relieved within the last 4 days, or under 30% of this
               -- season's (else last season's) appearances were starts, or
               -- expected to face fewer than 9 batters
               ((h.last_started = false and h.days_since_last <= 4)
                or (case when h.n_apps > 0 then h.n_starts_all::double / h.n_apps
                         when h.p_apps > 0 then h.p_starts::double / h.p_apps end) < 0.3
                or e.sp_exp_bf < 9) as sp_opener,
               {s_mu} as mu_skill
        from tg q
        left join sp_rates r on r.player_id = q.sp_id and r.season = q.season and r.cutoff = q.cutoff
        left join sp_hist h on h.player_id = q.sp_id and h.season = q.season and h.cutoff = q.cutoff
        left join sp_expbf_df e on e.game_id = q.game_id and e.bat_team = q.bat_team
        left join mu_df m on m.season = q.season
    """)

    # ---- bullpen ----
    con.execute("create or replace table pen_keys_t as select distinct fld_team as team_id, season, cutoff from tg")
    pitching.bullpen(con, "pen_keys_t", skill)

    # ---- environment and team strength ----
    environment.environment(con, "tg")

    # ---- lineup ----
    if point == "P2":
        lineup.actual_lineup(con)
        lineup.lineup_features(con, "lineup_actual", "tg", "lu_feat")
    elif point == "P1":
        lineup.projected_lineup(con, "tg")
        lineup.lineup_features(con, "lineup_proj", "tg", "lu_feat")
    else:
        raise ValueError(f"unknown prediction point {point}")

    # ---- assemble ----
    tto = pitching.TTO3_RUNS_PER_PA
    parts = []
    for sfx, spa in (("f5", seg_pa["f5"]), ("8", seg_pa["8"])):
        parts.append(f"""
            s.sp_pa_{sfx},
            s.tto3_pa_{sfx} / {spa} as sp_tto3_share_{sfx},
            (s.sp_pa_{sfx} * coalesce(s.sp_skill, s.mu_skill)
               + s.tto3_pa_{sfx} * {tto}
               + ({spa} - s.sp_pa_{sfx}) * coalesce(p.pen_skill_{sfx}, s.mu_skill)) / {spa}
              as pitching_composite_{sfx}""")
    con.execute(f"""
        create or replace table features as
        select q.game_id, q.season, q.date, q.game_type, q.bat_team, q.fld_team, q.is_home, q.sp_id, q.resumed,
               '{point}' as point,
               l.lineup_woba_f5, l.lineup_woba_8, l.lineup_k_bb_f5, l.lineup_k_bb_8, l.lineup_platoon,
               l.lineup_rookies, l.lineup_n,
               s.sp_skill, s.sp_k, s.sp_bb, s.sp_gb, s.sp_rookie, s.sp_fb_velo_delta,
               (s.sp_fb_velo_delta is null) as velo_missing,
               s.sp_exp_bf, s.sp_rest_bucket, coalesce(s.sp_opener, false) as sp_opener,
               p.pen_skill_f5, p.pen_skill_8, p.pen_unavail_share_8, p.pen_pool_n,
               (p.pen_skill_8 is null) as pen_missing,
               {','.join(parts)},
               e.park_factor, e.new_park, e.temp_f, e.temp_missing, e.roof_park,
               e.league_env_f5, e.league_env_8,
               e.team_off_f5, e.team_off_8, e.team_def_f5, e.team_def_8,
               t.runs_f5, t.runs_8, t.runs_total
        from tg q
        left join lu_feat l on l.game_id = q.game_id and l.bat_team = q.bat_team
        left join sp_feat s on s.game_id = q.game_id and s.bat_team = q.bat_team
        left join pen p on p.team_id = q.fld_team and p.season = q.season and p.cutoff = q.cutoff
        left join env e on e.game_id = q.game_id and e.bat_team = q.bat_team
        left join team_runs t on t.game_id = q.game_id and t.team_id = q.bat_team
        order by q.game_id, q.is_home
    """)
    notes["rows"] = con.execute("select count(*) from features").fetchone()[0]
    return con, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seasons", default="2021,2022,2023,2024,2025,2026")
    ap.add_argument("--point", default="P2")
    a = ap.parse_args()
    seasons = [int(s) for s in a.seasons.split(",")]
    con, notes = build_features(a.inputs, seasons, a.point)
    con.execute(f"copy features to '{a.out}' (format parquet, compression zstd)")
    Path(a.out).with_suffix(".notes.json").write_text(json.dumps(notes, indent=2, default=str))
    print(json.dumps({k: v for k, v in notes.items() if k != "slot_weights"}, indent=2, default=str))


if __name__ == "__main__":
    main()

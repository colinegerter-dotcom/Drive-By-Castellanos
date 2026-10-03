"""
Round 2 group 5: pitch movement as an in-house "stuff" score (design 5.7
item 5, as built: design E16, drafted 2 Oct 2026 and revised after an
independent review, before the build).

A pitcher's pitch traits (velocity, movement, spin, release) show his quality
faster than his results do. Per pitch, the chance of a swinging strike from
those traits only (not location or count), from four logistic models by pitch
class fitted on FIT_SEASONS (2021) and then frozen:

  class FB (FF, SI)   base + velo^2, ride^2, run^2, velo x ride
  class CT (FC)       base + gaps to primary fastball (velo, ride, run), velo^2
  class BR (SL, ST, SV, CU, KC, CS)
                      base + the three gaps, velo^2, ride^2, run^2, velo gap^2
  class OS (CH, FS, FO, SC)
                      base + the three gaps and each gap squared
  base = velo, ride (induced vertical break, in), run (horizontal break, in,
         mirrored so arm-side run is positive for both hands), spin, extension,
         release height, release side (mirrored)

Primary fastball: the four-seam or sinker the pitcher threw more in that game
(four-seam wins ties); the fit season's league fastball average if none.
Left out: other pitch types, pitches missing movement, extension or release
point, pitchers whose listed position isn't P or TWP. Missing spin takes the
fit season's pitch-type median.

Each pitch is scored against its own group (FF; SI; FC; SL/ST/SV; CU/KC/CS;
CH/SC; FS/FO): the score minus the fit season's average score of that group,
stored as integer millionths so per-pitcher and per-day sums are exact in any
run order (the later weighted blends are doubles, so builds can differ by
about 1e-15).

Pitcher measure: his average group-relative score minus the league's average
over the same span (each prior season's own league average; the current
season's league average over days before the game), pooled over the current
season's days before the game (weight 1) and up to 3 prior seasons (weights
0.25, 0.17, 0.08), shrunk toward 0 with K_SHRINK = 50 pitches, times 100:
extra swinging strikes per 100 pitches from stuff. Regular season only.

Features per team-game (the batting team's view):
  sp_stuff      the opposing starter's measure
  pen_stuff_8   the opposing bullpen's, available relievers weighted by their
                late-inning batters faced (as pen_skill_8 and ct_pen_xw_8)

Every fitted constant comes from fit_stuff() over FIT_SEASONS. The module
switches below exist for the leak test's mutation checks (design E16) and
must stay at their defaults.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

FIT_SEASONS: tuple[int, ...] | None = (2021,)   # None = every loaded season (mutation only)
K_SHRINK = 50.0                                  # pitches
PRIOR_W = {1: 0.25, 2: 0.17, 3: 0.08}            # light: older stuff is in the K rating already
SAME_DAY = False             # True counts the game's own day (mutation only)
LEAGUE_FULL_SEASON = False   # True uses the current season's full-season league average (mutation only)

CLASS_OF = {"FF": "FB", "SI": "FB", "FC": "CT", "SL": "BR", "ST": "BR", "SV": "BR", "CU": "BR",
            "KC": "BR", "CS": "BR", "CH": "OS", "FS": "OS", "FO": "OS", "SC": "OS"}
GROUP_OF = {"FF": "FF", "SI": "SI", "FC": "FC", "SL": "SL", "ST": "SL", "SV": "SL", "CU": "CU",
            "KC": "CU", "CS": "CU", "CH": "CH", "SC": "CH", "FS": "FS", "FO": "FS"}
RAW = ["v", "ivb", "hb", "spin", "ext", "rz", "rxa", "dv", "di", "dh"]
BASE = ["v", "ivb", "hb", "spin", "ext", "rz", "rxa"]
TERMS = {
    "FB": BASE + ["v2", "ivb2", "hb2", "v_ivb"],
    "CT": BASE + ["dv", "di", "dh", "v2"],
    "BR": BASE + ["dv", "di", "dh", "v2", "ivb2", "hb2", "dv2"],
    "OS": BASE + ["dv", "di", "dh", "dv2", "di2", "dh2"],
}
RIDGE = 1e-4
SCALE = 1_000_000   # integer millionths


def load_pitches(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Regular-season pitches of real pitchers with the traits the model uses,
    plus each pitch's primary-fastball reference (null if he threw none)."""
    types = ",".join(f"'{t}'" for t in CLASS_OF)
    df = con.execute(f"""
        with p as (
            select p.game_id, p.season, p.data_date, p.pitcher_id, p.pitch_type as pt,
                   p.release_speed as v, p.pfx_z * 12 as ivb,
                   -- arm-side run positive for both hands (a righty's arm side is negative pfx_x)
                   case when coalesce(pl.throws, 'R') = 'L' then 1 else -1 end * p.pfx_x * 12 as hb,
                   p.spin_rate::double as spin, p.release_extension as ext, p.release_pos_z as rz,
                   case when coalesce(pl.throws, 'R') = 'L' then 1 else -1 end * p.release_pos_x as rxa,
                   (p.pitch_result in ('swinging_strike', 'swinging_strike_blocked', 'foul_tip'))::int as wh
            from pitches p
            join players pl on pl.player_id = p.pitcher_id
            where p.game_type = 'R' and pl.position in ('P', 'TWP')
              and p.pitch_type in ({types})
              and p.release_speed is not null and p.pfx_x is not null and p.pfx_z is not null
              and p.release_extension is not null and p.release_pos_x is not null and p.release_pos_z is not null
        ),
        fb as (
            select game_id, pitcher_id, pt, count(*) as n, avg(v) as fv, avg(ivb) as fi, avg(hb) as fh
            from p where pt in ('FF', 'SI') group by all
        ),
        prim as (
            select * from fb
            qualify row_number() over (partition by game_id, pitcher_id
                                       order by n desc, case when pt = 'FF' then 0 else 1 end) = 1
        )
        select p.*, prim.fv, prim.fi, prim.fh
        from p left join prim using (game_id, pitcher_id)
        order by p.game_id, p.pitcher_id, p.data_date
    """).fetchdf()
    return df


def _design(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    """Fills the reference and spin from the frozen constants, adds gaps."""
    df = df.copy()
    df["spin"] = df["spin"].fillna(df["pt"].map(c["spin_median"]))
    df["spin"] = df["spin"].fillna(c["spin_overall"])
    df["fv"] = df["fv"].fillna(c["fb_ref"]["v"])
    df["fi"] = df["fi"].fillna(c["fb_ref"]["ivb"])
    df["fh"] = df["fh"].fillna(c["fb_ref"]["hb"])
    df["dv"], df["di"], df["dh"] = df.v - df.fv, df.ivb - df.fi, df.hb - df.fh
    df["cl"] = df.pt.map(CLASS_OF)
    df["grp"] = df.pt.map(GROUP_OF)
    return df


def _X(q: pd.DataFrame, cls: str, mu: dict, sd: dict) -> np.ndarray:
    z = {f: (q[f].to_numpy(float) - mu[f]) / sd[f] for f in RAW}
    z["v2"], z["ivb2"], z["hb2"] = z["v"] ** 2, z["ivb"] ** 2, z["hb"] ** 2
    z["v_ivb"] = z["v"] * z["ivb"]
    z["dv2"], z["di2"], z["dh2"] = z["dv"] ** 2, z["di"] ** 2, z["dh"] ** 2
    return np.column_stack([np.ones(len(q))] + [z[t] for t in TERMS[cls]])


def _irls(X: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        w = p * (1 - p)
        H = (X * w[:, None]).T @ X + RIDGE * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - RIDGE * b)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def fit_stuff(raw: pd.DataFrame) -> dict:
    """Every fitted constant, from FIT_SEASONS rows only."""
    fit = raw if FIT_SEASONS is None else raw[raw.season.isin(FIT_SEASONS)]
    if fit.empty:
        raise RuntimeError("no pitches in the fit season(s) for the stuff model")
    c: dict = {"spin_median": fit.groupby("pt").spin.median().to_dict(),
               "spin_overall": float(fit.spin.median())}
    fbp = fit[fit.pt.isin(["FF", "SI"])]
    c["fb_ref"] = {"v": float(fbp.v.mean()), "ivb": float(fbp.ivb.mean()), "hb": float(fbp.hb.mean())}
    fd = _design(fit, c)
    c["classes"], c["group_mean"] = {}, {}
    for cls in TERMS:
        q = fd[fd.cl == cls]
        mu = {f: float(q[f].mean()) for f in RAW}
        sd = {f: float(q[f].std()) or 1.0 for f in RAW}
        b = _irls(_X(q, cls, mu, sd), q.wh.to_numpy(float))
        c["classes"][cls] = {"mu": mu, "sd": sd, "b": b.tolist(), "n": int(len(q))}
    scored = score(fd, c, relative=False)
    c["group_mean"] = scored.groupby("grp").p.mean().to_dict()
    return c


def score(df: pd.DataFrame, c: dict, relative: bool = True) -> pd.DataFrame:
    """Adds p (swinging-strike chance) and, if relative, s_int (score minus
    its group's fit-season mean, in integer millionths)."""
    df = df.copy()
    df["p"] = np.nan
    for cls, m in c["classes"].items():
        idx = df.index[df.cl == cls]
        if len(idx):
            X = _X(df.loc[idx], cls, m["mu"], m["sd"])
            df.loc[idx, "p"] = 1.0 / (1.0 + np.exp(-(X @ np.asarray(m["b"]))))
    if relative:
        rel = (df.p - df.grp.map(c["group_mean"])).fillna(0.0)
        df["s_int"] = np.rint(rel * SCALE).astype("int64")
    return df


def stuff_features(con: duckdb.DuckDBPyConnection, keys: str = "tg") -> dict:
    """Writes table stuff_feat (game_id, bat_team, sp_stuff, pen_stuff_8).
    Needs pen_pool (pitching.bullpen) and the keys table (with sp_id)."""
    raw = load_pitches(con)
    c = fit_stuff(raw)
    sc = score(_design(raw, c), c)
    pg = sc.groupby(["season", "pitcher_id", "game_id", "data_date"], as_index=False).agg(
        s=("s_int", "sum"), n=("s_int", "size"))
    con.register("stuff_pg_df", pg)
    con.execute("create or replace table stuff_pg as select season, pitcher_id, game_id, data_date::date as data_date, "
                "s::hugeint as s, n::bigint as n from stuff_pg_df")
    con.unregister("stuff_pg_df")
    con.execute("create or replace table stuff_lg_day as select season, data_date, sum(s) s, sum(n) n from stuff_pg group by all")
    con.execute("create or replace table stuff_lg_season as select season, sum(s)::double / sum(n) as l from stuff_pg group by all")
    con.execute("""
        create or replace table stuff_season as
        select p.pitcher_id, p.season, sum(p.s)::double - sum(p.n) * any_value(l.l) as rc, sum(p.n) as n
        from stuff_pg p join stuff_lg_season l using (season) group by all
    """)
    day_cmp = "<=" if SAME_DAY else "<"
    w1, w2, w3 = PRIOR_W[1], PRIOR_W[2], PRIOR_W[3]
    lg_cur = ("(select l from stuff_lg_season s where s.season = k.season)" if LEAGUE_FULL_SEASON else
              "(select sum(d.s)::double / nullif(sum(d.n), 0) from stuff_lg_day d "
              f"where d.season = k.season and d.data_date {day_cmp} k.cutoff)")
    con.execute(f"""
        create or replace table stuff_keys as
        select distinct sp_id as player_id, season, cutoff from {keys} where sp_id is not null
        union
        select distinct player_id, season, cutoff from pen_pool
    """)
    con.execute(f"""
        create or replace table stuff_rates as
        with k as (select * from stuff_keys),
        cur as (
            select k.player_id, k.season, k.cutoff, sum(p.s)::double as s, sum(p.n) as n
            from k join stuff_pg p
              on p.pitcher_id = k.player_id and p.season = k.season and p.data_date {day_cmp} k.cutoff
            group by all
        ),
        pri as (
            select k.player_id, k.season, k.cutoff,
                   sum(case k.season - s.season when 1 then {w1} when 2 then {w2} else {w3} end * s.rc) as rc,
                   sum(case k.season - s.season when 1 then {w1} when 2 then {w2} else {w3} end * s.n) as n
            from k join stuff_season s
              on s.pitcher_id = k.player_id and s.season between k.season - 3 and k.season - 1
            group by all
        )
        select k.player_id, k.season, k.cutoff,
               (100.0 * (coalesce(cur.s, 0) - coalesce(cur.n, 0) * coalesce({lg_cur}, 0) + coalesce(pri.rc, 0))
                  / {SCALE} / (coalesce(cur.n, 0) + coalesce(pri.n, 0) + {K_SHRINK}))::double as stuff,
               coalesce(cur.n, 0) as cur_n, coalesce(pri.n, 0) as pri_n
        from k left join cur using (player_id, season, cutoff) left join pri using (player_id, season, cutoff)
    """)
    con.execute("""
        create or replace table stuff_pen as
        with p as (
            select p.*, r.stuff,
                   not ((p.pitched_d1 and p.pitched_d2) or p.pitches_d1 >= 25) as available
            from pen_pool p join stuff_rates r using (player_id, season, cutoff)
        ), t as (
            select team_id, season, cutoff, sum(late_bf) tot_late, sum(bf) tot_bf from p group by 1, 2, 3
        ), wts as (
            select p.*, case when t.tot_late > 0 then p.late_bf / t.tot_late else p.bf / t.tot_bf end as w8
            from p join t using (team_id, season, cutoff)
        )
        select team_id, season, cutoff,
               sum(w8 * stuff) filter (where available) / nullif(sum(w8) filter (where available), 0) as pen_stuff_8
        from wts group by 1, 2, 3
    """)
    con.execute(f"""
        create or replace table stuff_feat as
        select q.game_id, q.bat_team,
               coalesce(sp.stuff, 0.0)::double as sp_stuff,
               coalesce(pn.pen_stuff_8, 0.0)::double as pen_stuff_8
        from {keys} q
        left join stuff_rates sp on sp.player_id = q.sp_id and sp.season = q.season and sp.cutoff = q.cutoff
        left join stuff_pen pn on pn.team_id = q.fld_team and pn.season = q.season and pn.cutoff = q.cutoff
    """)
    notes = {"fit_seasons": list(FIT_SEASONS) if FIT_SEASONS else "all",
             "fit_pitches": {k: v["n"] for k, v in c["classes"].items()},
             "group_mean": {k: round(float(v), 5) for k, v in c["group_mean"].items()},
             "scored_pitches": int(len(sc))}
    sp_mean, sp_sd, pen_mean, pen_sd = con.execute(
        "select avg(sp_stuff), stddev(sp_stuff), avg(pen_stuff_8), stddev(pen_stuff_8) from stuff_feat").fetchone()
    notes.update({"sp_mean": float(sp_mean), "sp_sd": float(sp_sd), "pen_mean": float(pen_mean), "pen_sd": float(pen_sd)})
    return notes

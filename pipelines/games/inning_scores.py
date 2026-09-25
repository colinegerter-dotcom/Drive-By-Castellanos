"""
inning_scores table -- runs scored in every half inning, from the pitch files.

Why (model design item A3, 25 Sep 2026): the full-game models predict runs
through 8 innings and a rules engine plays the 9th and extras, because the
bottom of the 9th is skipped when the home team leads and walk-offs stop at
the winning run. game_results only holds final and first-five scores, so the
8-inning target, and the late-inning rates the engine is fitted on, need
runs by inning.

HOW: every pitch carries the batting team's score BEFORE it is thrown. The
runs a team scores in its half of inning N are its score at the first pitch
of its next half inning minus its score at the first pitch of this one. For
its last half inning, the official final score replaces "next". The sums
therefore telescope to the final score by construction, so the real checks
are elsewhere:
  * every team's first half inning starts at 0 runs
  * no half inning is missing between the 1st and the team's last
  * no half inning comes out negative
  * runs through 5 innings match game_results' first-five score, which MLB
    computes separately from the linescore. Checked 25 Sep 2026 on the
    local pitch files: 14,712 of 14,712 games with a first-five score in
    2021-2026 matched, and through 8 innings home teams scored 3.99 vs 3.66
    in 2022 and 4.28 vs 4.01 in 2023

A game failing a check gets no rows and is listed; it isn't guessed at.
Games that stopped early (rain) simply have fewer innings. A skipped bottom
of the 9th has no row, because it was never played.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

INNING_KEY = ["game_id", "inning", "half"]

# Batting team's score at the first pitch of every half inning. Ordered by
# at-bat then pitch number, so the first pitch really is the first one.
HALF_STARTS_SQL = """
    select game_id, inning, inning_topbot,
           arg_min(case when inning_topbot = 'Top' then away_score else home_score end,
                   at_bat_id * 1000 + pitch_number) as start_score,
           arg_min(outs, at_bat_id * 1000 + pitch_number) as start_outs
    from pitches
    {where}
    group by game_id, inning, inning_topbot
"""


def half_starts(pitches, game_ids: list[int] | None = None) -> list[tuple]:
    """(game_id, inning, 'Top'/'Bot', batting team's starting score, outs at
    the first pitch) from a PitchSource. game_ids limits it to some games
    (the nightly job)."""
    if game_ids is not None:
        if not game_ids:
            return []
        ids = ",".join(str(int(g)) for g in game_ids)
        sql = HALF_STARTS_SQL.format(where=f"where game_id in ({ids})")
    else:
        sql = HALF_STARTS_SQL.format(where="")
    return [tuple(r) for r in pitches.query(sql).fetchall()]


def build_inning_rows(starts: list[tuple], games: dict[int, dict]) -> tuple[list[dict], list[tuple[int, str]]]:
    """Turn half-inning starting scores into one row per half inning.

    games: {game_id: {"home_team", "away_team", "home_final", "away_final",
    optional "innings_played"}} for completed games. Returns (rows, problems);
    a problem game gets no rows at all.

    Checks added after an independent review (25 Sep 2026), because a gap in
    the pitch data can move runs into the wrong half inning while the totals
    still add up:
      * each half inning's first recorded pitch has 0 outs (a missing leadoff
        at-bat shows up as a half starting with 1 or 2 outs)
      * both teams batted: a completed game with no bottom halves is a gap
      * with innings_played known, the away team batted exactly that many
        innings and the home team that many or one fewer (a skipped bottom
        of the 9th)
    Pitchless at-bats (automatic intentional walks) leave harmless gaps in
    at-bat numbers and are not treated as problems.
    """
    by_side: dict[tuple[int, str], list[tuple[int, int, int | None]]] = {}
    for rec in starts:
        game_id, inning, tb, start = rec[:4]
        outs = rec[4] if len(rec) > 4 else 0
        if game_id in games and start is not None:
            by_side.setdefault((game_id, tb), []).append((int(inning), int(start), outs))

    rows: list[dict] = []
    problems: list[tuple[int, str]] = []
    for game_id, g in games.items():
        game_rows: list[dict] = []
        problem = None
        n_halves = {}
        for tb, team, final in (("Top", g["away_team"], g["away_final"]), ("Bot", g["home_team"], g["home_final"])):
            halves = sorted(by_side.get((game_id, tb), []))
            n_halves[tb] = len(halves)
            if not halves:
                problem = "no pitch data" if tb == "Top" else "home team never batted in the pitch data"
                break
            innings = [h[0] for h in halves]
            if innings != list(range(1, len(innings) + 1)):
                problem = f"{tb}: innings {innings[:3]}...{innings[-3:]} not consecutive from 1"
                break
            if halves[0][1] != 0:
                problem = f"{tb}: first half inning starts at {halves[0][1]} runs"
                break
            late = [h[0] for h in halves if h[2] not in (0, None)]
            if late:
                problem = f"{tb}: half inning(s) {late[:3]} start with outs already recorded (missing at-bat)"
                break
            if final is None:
                problem = "no final score"
                break
            for k, (inning, start, _outs) in enumerate(halves):
                end = halves[k + 1][1] if k + 1 < len(halves) else int(final)
                runs = end - start
                if runs < 0:
                    problem = f"{tb} {inning}: {runs} runs"
                    break
                game_rows.append({
                    "game_id": game_id, "inning": inning,
                    "half": "top" if tb == "Top" else "bottom",
                    "batting_team": team, "runs": runs,
                })
            if problem:
                break
        ip = g.get("innings_played")
        if problem is None and ip:
            ip = int(ip)
            if n_halves.get("Top") != ip or n_halves.get("Bot") not in (ip, ip - 1):
                problem = f"half innings (top {n_halves.get('Top')}, bottom {n_halves.get('Bot')}) don't fit {ip} innings played"
        if problem:
            problems.append((game_id, problem))
        else:
            rows.extend(game_rows)
    return rows, problems


def compare_first_five(rows: list[dict], games: dict[int, dict]) -> dict:
    """Runs through 5 innings vs game_results' first-five score."""
    f5: dict[tuple[int, int], int] = {}
    for r in rows:
        if r["inning"] <= 5:
            f5[(r["game_id"], r["batting_team"])] = f5.get((r["game_id"], r["batting_team"]), 0) + r["runs"]
    compared = agree = 0
    misses = []
    for game_id in {r["game_id"] for r in rows}:
        g = games[game_id]
        if g.get("home_f5") is None or g.get("away_f5") is None:
            continue
        compared += 1
        got = (f5.get((game_id, g["away_team"]), 0), f5.get((game_id, g["home_team"]), 0))
        want = (int(g["away_f5"]), int(g["home_f5"]))
        if got == want:
            agree += 1
        else:
            misses.append({"game_id": game_id, "pitch_file": got, "game_results": want})
    return {"compared": compared, "agree": agree, "misses": misses,
            "agreement": agree / compared if compared else 1.0}

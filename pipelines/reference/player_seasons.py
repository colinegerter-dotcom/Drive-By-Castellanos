"""
player_season_stats table and players.birth_date -- the prior layer's inputs.

Why (model design item A5, 25 Sep 2026): player rates are blended with up to
three previous seasons (a Marcel-style prior), and adjusted for age. The
pitch files start in 2021, so a 2022 player's 2019-2020 seasons, and a 2021
player's 2018-2020, can only come from MLB's official season lines. Birth
dates drive the age adjustment; the players table didn't have them.

SOURCE: the Stats API /people endpoint with a yearByYear hydrate, regular
season, MLB level only (sport id 1).

TRADED PLAYERS (found 25 Sep 2026): for a season split across teams, the API
returns one line per team AND a combined line with no team and a
`numTeams` field. Juan Soto 2022: WSH 436 PA, SD 228 PA, combined 664 PA.
Adding every line up double-counts the season. This module keeps the
combined line when there is one, otherwise the single team line.
(mlb_stats_client.get_career_totals_before_season had exactly that
double count; fixed in the same change.)
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

PLAYER_SEASON_KEY = ["player_id", "season", "stat_group"]

# our column -> Stats API stat key
FIELD_MAP = {
    "age": "age",
    "games": "gamesPlayed",
    "games_started": "gamesStarted",
    "plate_appearances": "plateAppearances",
    "at_bats": "atBats",
    "batters_faced": "battersFaced",
    "outs": "outs",
    "hits": "hits",
    "doubles": "doubles",
    "triples": "triples",
    "home_runs": "homeRuns",
    "walks": "baseOnBalls",
    "intentional_walks": "intentionalWalks",
    "strikeouts": "strikeOuts",
    "sac_flies": "sacFlies",
    "sac_bunts": "sacBunts",
    "ground_outs": "groundOuts",
    "air_outs": "airOuts",
    "runs": "runs",
    "earned_runs": "earnedRuns",
    "number_of_pitches": "numberOfPitches",
}
# hit-by-pitch has a different key per group
HBP_KEY = {"hitting": "hitByPitch", "pitching": "hitBatsmen"}


def season_lines(splits: list[dict]) -> dict[int, tuple[dict, int]]:
    """{season: (stat dict, number of teams)} from yearByYear splits, MLB
    level only, one line per season: the combined line if MLB gave one."""
    by_season: dict[int, list[dict]] = {}
    for sp in splits:
        if (sp.get("sport") or {}).get("id") not in (1, None):
            continue
        if sp.get("gameType") not in (None, "R"):
            continue
        try:
            season = int(sp.get("season"))
        except (TypeError, ValueError):
            continue
        by_season.setdefault(season, []).append(sp)
    out: dict[int, tuple[dict, int]] = {}
    for season, sps in by_season.items():
        combined = [s for s in sps if not s.get("team")]
        if combined:
            s = max(combined, key=lambda x: int(x.get("numTeams") or 1))
            out[season] = (s.get("stat") or {}, int(s.get("numTeams") or 1))
        elif len(sps) == 1:
            out[season] = (sps[0].get("stat") or {}, 1)
        else:
            # several team lines and no combined one: add up the counting stats
            total: dict = {}
            for s in sps:
                for k, v in (s.get("stat") or {}).items():
                    if isinstance(v, int):
                        total[k] = total.get(k, 0) + v
            out[season] = (total, len(sps))
    return out


def player_season_rows(person: dict) -> list[dict]:
    """One row per season and stat group for one /people entry."""
    pid = person.get("id")
    rows = []
    for block in person.get("stats") or []:
        group = (block.get("group") or {}).get("displayName")
        if group not in ("hitting", "pitching"):
            continue
        if (block.get("type") or {}).get("displayName") != "yearByYear":
            continue
        for season, (stat, num_teams) in sorted(season_lines(block.get("splits") or []).items()):
            row = {"player_id": pid, "season": season, "stat_group": group, "num_teams": num_teams}
            for col, key in FIELD_MAP.items():
                v = stat.get(key)
                row[col] = v if isinstance(v, int) else None
            v = stat.get(HBP_KEY[group])
            row["hit_by_pitch"] = v if isinstance(v, int) else None
            row["stat_json"] = stat
            rows.append(row)
    return rows


def birth_date(person: dict) -> str | None:
    return person.get("birthDate") or None

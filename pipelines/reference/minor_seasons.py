"""
Minor-league season lines (model design group 8, rookie translations, 3 Oct 2026).

Same shape as player_seasons.py, for Triple-A (sport id 11) and Double-A
(12). The Stats API's /people endpoint takes the level inside the stats
hydrate: stats(group=[hitting,pitching],type=[yearByYear],sportId=11).

The level is read from each split's own `sport.id` and must equal the one
asked for. A split with no sport id is NOT accepted (player_seasons.py accepts
one, because it only ever asks for the majors): if the API ignored the
sportId filter and sent MLB lines, they would otherwise be stored as minor
league lines. In that case no rows come out and the build fails loudly.

Two-team seasons at one level: the API's combined line (no team) when there
is one, else the team lines added up, as in player_seasons.py.
"""
from __future__ import annotations

from pipelines.mlb_stats_client import _get
from pipelines.config import MLB_STATS_API_BASE
from pipelines.reference.player_seasons import FIELD_MAP, HBP_KEY

LEVELS = {"AAA": 11, "AA": 12}
MINOR_KEY = ["player_id", "season", "stat_group", "sport_id"]
TEAM_TOTAL_KEY = ["season", "sport_id", "stat_group", "team_id"]
TEAM_FIELDS = ["games", "plate_appearances", "at_bats", "batters_faced", "hits", "doubles", "triples",
               "home_runs", "walks", "intentional_walks", "strikeouts", "sac_flies", "sac_bunts",
               "ground_outs", "air_outs"]


def minor_season_lines(splits: list[dict], sport_id: int) -> dict[int, tuple[dict, int]]:
    """{season: (stat dict, number of teams)}, one line per season at one level."""
    by_season: dict[int, list[dict]] = {}
    for sp in splits:
        if (sp.get("sport") or {}).get("id") != sport_id:
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
            total: dict = {}
            for s in sps:
                for k, v in (s.get("stat") or {}).items():
                    if isinstance(v, int):
                        total[k] = total.get(k, 0) + v
            out[season] = (total, len(sps))
    return out


def minor_season_rows(person: dict, sport_id: int) -> list[dict]:
    """One row per season and stat group for one /people entry at one level."""
    pid = person.get("id")
    rows = []
    for block in person.get("stats") or []:
        group = (block.get("group") or {}).get("displayName")
        if group not in ("hitting", "pitching"):
            continue
        if (block.get("type") or {}).get("displayName") != "yearByYear":
            continue
        for season, (stat, num_teams) in sorted(minor_season_lines(block.get("splits") or [], sport_id).items()):
            row = {"player_id": pid, "season": season, "stat_group": group, "sport_id": sport_id,
                   "num_teams": num_teams}
            for col, key in FIELD_MAP.items():
                v = stat.get(key)
                row[col] = v if isinstance(v, int) else None
            v = stat.get(HBP_KEY[group])
            row["hit_by_pitch"] = v if isinstance(v, int) else None
            row["stat_json"] = stat
            rows.append(row)
    return rows


def get_people_minor_year_by_year(player_ids: list[int], sport_id: int, chunk: int = 50) -> list[dict]:
    """Raw /people entries with yearByYear hitting and pitching lines at one
    minor-league level. One request per `chunk` players."""
    hydrate = f"stats(group=[hitting,pitching],type=[yearByYear],sportId={sport_id})"
    ids = sorted({int(p) for p in player_ids if p is not None})
    out: list[dict] = []
    for i in range(0, len(ids), chunk):
        data = _get(
            f"{MLB_STATS_API_BASE}/people",
            {"personIds": ",".join(str(p) for p in ids[i:i + chunk]), "hydrate": hydrate},
        )
        out.extend(data.get("people", []))
    return out


def debut_coverage(rows: list[dict], debut: dict, ids: list[int], years=(2021, 2022)) -> tuple[int, int]:
    """(players in `ids` who debuted in MLB in `years`, how many of them have a
    minor-league line from a season before their debut season). Players with
    no debut date (not yet debuted, or missing in mlb.players) are skipped:
    the first full build crashed on one (KeyError 676601, 4 Oct 2026)."""
    idset = set(ids)
    rookies = {p for p, d in debut.items() if d is not None and d.year in years and p in idset}
    have = {r["player_id"] for r in rows
            if r["player_id"] in rookies and r["season"] < debut[r["player_id"]].year}
    return len(rookies), len(have)


def team_total_rows(resp: dict, season: int, sport_id: int, group: str) -> list[dict]:
    """Rows for minor_league_team_totals from one /teams/stats response
    (stats=season, one group, one level). A split whose season or level
    doesn't match what was asked for is dropped, so a response that ignored
    the filters yields nothing and the build's team-count check fails."""
    rows = []
    for block in resp.get("stats") or []:
        for sp in block.get("splits") or []:
            team = sp.get("team") or {}
            if team.get("id") is None:
                continue
            try:
                if int(sp.get("season")) != season:
                    continue
            except (TypeError, ValueError):
                continue
            sid = (sp.get("sport") or {}).get("id")
            if sid is not None and sid != sport_id:
                continue
            stat = sp.get("stat") or {}
            row = {"season": season, "sport_id": sport_id, "stat_group": group,
                   "team_id": team["id"], "team_name": team.get("name")}
            for col in TEAM_FIELDS:
                v = stat.get(FIELD_MAP[col])
                row[col] = v if isinstance(v, int) else None
            v = stat.get(HBP_KEY[group])
            row["hit_by_pitch"] = v if isinstance(v, int) else None
            row["stat_json"] = stat
            rows.append(row)
    return rows


def get_team_totals(season: int, sport_id: int, group: str) -> dict:
    """Raw /teams/stats response: every team's season totals at one level."""
    return _get(f"{MLB_STATS_API_BASE}/teams/stats",
                {"season": season, "sportIds": sport_id, "group": group, "stats": "season", "gameType": "R"})

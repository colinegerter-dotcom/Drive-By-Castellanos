"""teams table -- MLB Stats API, refreshed on every run (cheap: 30 rows)."""
from __future__ import annotations

from pipelines.mlb_stats_client import get_teams


def build_team_rows(season: int) -> list[dict]:
    teams = get_teams(season)
    rows = []
    for t in teams:
        rows.append(
            {
                "team_id": t["id"],
                "team_name": t.get("name"),
                "league": (t.get("league") or {}).get("name"),
                "division": (t.get("division") or {}).get("name"),
            }
        )
    return rows

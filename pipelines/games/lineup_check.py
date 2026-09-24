"""
Cross-check the box-score starting lineups against the pitch files.

Why (24 Sep 2026, model design A1, "option C"): the lineup table held
end-of-game lineups for years without anything noticing. The fixed parser
(pipelines/games/lineup.py) is the source of truth, and this is the
independent second opinion that would have caught the old bug on day one.

The pitch-file version of a starting lineup: for each team in a game, the
first 9 distinct batters in order of their first plate appearance. It is
right almost always, and wrong in one known way: a starter replaced before
he ever came to bat (hurt in the field in the 1st, pinch-hit for in his
first turn) never appears, so his substitute takes the slot. Those are
labelled "starter never batted" and are expected. Anything else is a real
disagreement worth reading.

Pass mark per season: at least 99.5% of compared slots agree.
"""
from __future__ import annotations

MIN_AGREEMENT = 0.995

_PITCH_LINEUP_SQL = """
    with first_pa as (
        select game_id, inning_topbot, batter_id, min(at_bat_id) as first_ab
        from pitches
        group by game_id, inning_topbot, batter_id
    ),
    ordered as (
        select *, row_number() over (
            partition by game_id, inning_topbot order by first_ab, batter_id
        ) as slot
        from first_pa
    )
    select game_id, inning_topbot, slot, batter_id from ordered where slot <= 9
"""

_BATTED_SQL = "select distinct game_id, batter_id from pitches"


def pitch_lineups(pitches, home_away: dict[int, tuple[int, int]]):
    """Pitch-file starting lineups for every game in the source.

    pitches: a PitchSource (pipelines/pitch_store.py).
    home_away: {game_id: (home_team_id, away_team_id)} from mlb.games.
    Returns ({(game_id, team_id): {slot: batter_id}}, {game_id: set(batter_ids)}).
    Top of an inning is the away team batting.
    """
    slots: dict[tuple[int, int], dict[int, int]] = {}
    for game_id, topbot, slot, batter_id in pitches.query(_PITCH_LINEUP_SQL).fetchall():
        teams = home_away.get(game_id)
        if teams is None:
            continue
        team_id = teams[1] if topbot == "Top" else teams[0]
        slots.setdefault((game_id, team_id), {})[int(slot)] = batter_id
    batted: dict[int, set[int]] = {}
    for game_id, batter_id in pitches.query(_BATTED_SQL).fetchall():
        batted.setdefault(game_id, set()).add(batter_id)
    return slots, batted


def compare_lineups(box_rows: list[dict], pitch_slots: dict, batted: dict) -> dict:
    """Compare box-score starters with pitch-file starters slot by slot.

    Only team-games present in both sources are compared; the rest are
    counted as skipped (e.g. a game with no pitch data).
    """
    box: dict[tuple[int, int], dict[int, int]] = {}
    for r in box_rows:
        box.setdefault((r["game_id"], r["team_id"]), {})[r["batting_order_slot"]] = r["player_id"]

    compared = agree = skipped = 0
    mismatches = []
    for key, box_slots in box.items():
        p = pitch_slots.get(key)
        if p is None:
            skipped += 1
            continue
        game_batters = batted.get(key[0], set())
        for slot, box_player in sorted(box_slots.items()):
            compared += 1
            pitch_player = p.get(slot)
            if pitch_player == box_player:
                agree += 1
                continue
            reason = "starter never batted" if box_player not in game_batters else "other"
            mismatches.append(
                {"game_id": key[0], "team_id": key[1], "slot": slot,
                 "box_score": box_player, "pitch_file": pitch_player, "reason": reason}
            )
    share = agree / compared if compared else 1.0
    return {
        "team_games_compared": len(box) - skipped,
        "team_games_skipped": skipped,
        "slots_compared": compared,
        "slots_agree": agree,
        "agreement": share,
        "never_batted": sum(1 for m in mismatches if m["reason"] == "starter never batted"),
        "other": [m for m in mismatches if m["reason"] == "other"],
        "passed": share >= MIN_AGREEMENT,
    }

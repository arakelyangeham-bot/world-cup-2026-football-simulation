#smoke_test_competition_schema.py

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.competition_schema import (
    Competition,
    CompetitionRoster,
    TeamEntry,
)


def main() -> None:
    competition = Competition(
        competition_id="wc_2026",
        name="FIFA World Cup",
        season="2026",
        competition_type="international",
    )

    team = TeamEntry(
        team_id="germany",
        name="Germany",
        competition_id=competition.competition_id,
        country="Germany",
    )

    roster = CompetitionRoster(
        competition_id=competition.competition_id,
        team_id=team.team_id,
        team_name=team.name,
        player_ids=(
            "test_player_001",
            "test_player_002",
            "test_player_003",
        ),
    )

    print("Competition Schema Smoke Test")
    print("-----------------------------")
    print(f"Competition: {competition.name} {competition.season}")
    print(f"Team: {team.name}")
    print(f"Roster players: {len(roster.player_ids)}")
    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
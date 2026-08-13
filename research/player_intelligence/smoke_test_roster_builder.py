#smoke_test_roster_builder.py

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.roster_builder import RosterBuilder


def main() -> None:
    builder = RosterBuilder()

    teams = builder.list_teams()

    print("Roster Builder Smoke Test")
    print("-------------------------")
    print(f"Teams found: {len(teams)}")
    print()
    print("First 20 teams")
    for team in teams[:20]:
        print(f"  {team}")

    assert len(teams) > 0

    test_team = teams[0]
    squad = builder.get_squad(test_team)

    print()
    print(f"Test squad: {test_team}")
    print(f"Players: {len(squad.players)}")

    for player in squad.players[:10]:
        print(
            f"  {player.identity.name} | "
            f"{player.identity.primary_position} | "
            f"rating={player.ratings.overall}"
        )

    assert len(squad.players) > 0

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
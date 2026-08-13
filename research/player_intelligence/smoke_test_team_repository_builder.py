#smoke_test_team_repository_builder.py

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.roster_builder import RosterBuilder
from research.player_intelligence.team_representation_builder import (
    build_team_representation_from_squad,
)
from research.player_intelligence.team_repository_builder import (
    project_representation_to_repository_entry,
    repository_entry_to_dict,
)


def main() -> None:
    roster_builder = RosterBuilder()

    teams = roster_builder.list_teams()
    assert len(teams) > 0

    test_team = teams[0]
    squad = roster_builder.get_squad(test_team)

    representation = build_team_representation_from_squad(squad)

    entry = project_representation_to_repository_entry(
        representation=representation,
        fifa_points=None,
    )

    entry_dict = repository_entry_to_dict(entry)

    print("Team Repository Builder Smoke Test")
    print("----------------------------------")
    print(f"Team: {entry.team}")
    print(f"Players: {len(squad.players)}")
    print()
    print("Representation")
    print(f"  attack: {representation.attack:.3f}")
    print(f"  midfield: {representation.midfield:.3f}")
    print(f"  defense: {representation.defense:.3f}")
    print(f"  goalkeeper: {representation.goalkeeper:.3f}")
    print(f"  squad_quality: {representation.squad_quality:.3f}")
    print(f"  evidence_score: {representation.evidence_score:.3f}")
    print()
    print("Repository entry")
    for key, value in entry_dict.items():
        print(f"  {key}: {value}")

    assert len(squad.players) > 0
    assert entry.team == test_team

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
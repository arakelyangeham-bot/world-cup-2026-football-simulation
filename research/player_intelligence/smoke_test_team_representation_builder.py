#smoke_test_team_representation_builder

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.player_repository import PlayerRepository
from research.player_intelligence.team_representation_builder import (
    build_team_representation_from_squad,
)


TEST_TEAM = "Germany"


def main() -> None:
    repository = PlayerRepository()
    squad = repository.get_squad(TEST_TEAM)

    representation = build_team_representation_from_squad(squad)

    print("Team Representation Builder Smoke Test")
    print("--------------------------------------")
    print(f"Team: {representation.national_team}")
    print(f"Attack: {representation.attack:.3f}")
    print(f"Midfield: {representation.midfield:.3f}")
    print(f"Defense: {representation.defense:.3f}")
    print(f"Goalkeeper: {representation.goalkeeper:.3f}")
    print(f"Attack depth: {representation.attack_depth:.3f}")
    print(f"Midfield depth: {representation.midfield_depth:.3f}")
    print(f"Defense depth: {representation.defense_depth:.3f}")
    print(f"Squad quality: {representation.squad_quality:.3f}")
    print(f"Evidence score: {representation.evidence_score:.3f}")

    assert representation.national_team == TEST_TEAM
    assert len(squad.players) > 0

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
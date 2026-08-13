#smoke_test_player_representation.py

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.player_evidence_repository import (
    PlayerEvidenceRepository,
)
from research.player_intelligence.player_repository import PlayerRepository
from research.player_intelligence.player_representation_engine import (
    build_player_representation,
)


def main() -> None:
    evidence_repository = PlayerEvidenceRepository()

    player_repository = PlayerRepository(
        evidence_repository=evidence_repository,
    )

    players = player_repository.load_players()

    print("Player Representation Smoke Test")
    print("--------------------------------")
    print(f"Players loaded: {len(players)}")
    print()

    for player in players[:10]:
        representation = build_player_representation(player)

        print(
            f"{representation.player_name} | "
            f"ability={representation.current_ability:.3f} | "
            f"confidence={representation.evidence_confidence:.3f} | "
            f"minutes={representation.total_minutes:.1f} | "
            f"competitions={representation.competition_count} | "
            f"seasons={representation.season_count} | "
            f"latest={representation.latest_season} | "
            f"recency={representation.recency_share:.3f}"
        )

    assert len(players) > 0

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
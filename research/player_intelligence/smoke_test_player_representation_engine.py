#smoke_test_player_representation_engine.py

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
    build_player_representation_diagnostics,
)


def main() -> None:
    evidence_repository = PlayerEvidenceRepository()

    player_repository = PlayerRepository(
        evidence_repository=evidence_repository,
    )

    players = player_repository.load_players()

    players_with_history = [
        player
        for player in players
        if player.evidence_history is not None
    ]

    assert len(players_with_history) > 0

    print("Player Representation Engine Smoke Test")
    print("---------------------------------------")
    print(f"Players loaded: {len(players)}")
    print(f"Players with evidence history: {len(players_with_history)}")
    print()

    for player in players_with_history[:10]:
        diagnostics = build_player_representation_diagnostics(player)

        print(
            f"{diagnostics.player_name} | "
            f"minutes={diagnostics.total_minutes:.1f} | "
            f"competitions={diagnostics.competition_count} | "
            f"seasons={diagnostics.season_count} | "
            f"latest={diagnostics.latest_season} | "
            f"recent_share={diagnostics.recency_share:.3f}"
        )

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
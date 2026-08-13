#smoke_test_player_with_evidence_history.py

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.player_evidence_repository import (
    PlayerEvidenceRepository,
)
from research.player_intelligence.player_repository import PlayerRepository


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

    print("Player With Evidence History Smoke Test")
    print("---------------------------------------")
    print(f"Players loaded: {len(players)}")
    print(f"Players with evidence history: {len(players_with_history)}")

    assert len(players) > 0

    if players_with_history:
        player = players_with_history[0]
        history = player.evidence_history

        print()
        print(f"Player: {player.identity.name}")
        print(f"Team: {player.identity.national_team}")
        print(f"Evidence summary minutes: {player.evidence.minutes_played}")
        print(f"Evidence history entries: {len(history.entries)}")
        print(f"Evidence history total minutes: {history.total_minutes}")

        print()
        print("First evidence entries")
        for entry in history.entries[:5]:
            print(
                f"  {entry.competition} | "
                f"{entry.season_year} | "
                f"{entry.team} | "
                f"minutes={entry.minutes_played} | "
                f"rating={entry.rating}"
            )

    assert len(players_with_history) > 0

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
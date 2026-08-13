#smoke_test_player_repository.py

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.player_repository import PlayerRepository


TEST_TEAM = "Germany"


def main() -> None:
    repository = PlayerRepository()

    players = repository.load_players()
    squad = repository.get_squad(TEST_TEAM)

    position_counts = Counter(
        player.identity.primary_position
        for player in squad.players
    )

    ratings = [
        player.ratings.overall
        for player in squad.players
    ]

    avg_rating = (
        sum(ratings) / len(ratings)
        if ratings
        else float("nan")
    )

    print("Player Repository Smoke Test")
    print("----------------------------")
    print(f"Total players loaded: {len(players)}")
    print(f"{TEST_TEAM} players loaded: {len(squad.players)}")
    print(f"{TEST_TEAM} average rating: {avg_rating:.2f}")
    print()
    print("Position counts")
    for position, count in sorted(position_counts.items()):
        print(f"  {position}: {count}")

    print()
    print("First 10 players")
    for player in squad.players[:10]:
        print(
            f"  {player.identity.name} | "
            f"{player.identity.primary_position} | "
            f"rating={player.ratings.overall}"
        )

    assert len(players) > 0
    assert len(squad.players) > 0

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
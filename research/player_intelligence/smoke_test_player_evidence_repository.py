#smoke_test_player_evidence_repository.py

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.player_evidence_repository import (
    PlayerEvidenceRepository,
)


def main() -> None:
    repository = PlayerEvidenceRepository()
    histories = repository.load_histories()

    print("Player Evidence Repository Smoke Test")
    print("-------------------------------------")
    print(f"Players with evidence history: {len(histories)}")

    assert len(histories) > 0

    sample_player_id = sorted(histories.keys())[0]
    history = histories[sample_player_id]

    print()
    print(f"Sample player ID: {history.player_id}")
    print(f"Evidence entries: {len(history.entries)}")
    print(f"Total minutes: {history.total_minutes}")
    print(f"Competition count: {history.competition_count}")
    print(f"Season count: {history.season_count}")

    print()
    print("First 10 evidence entries")
    for entry in history.entries[:10]:
        print(
            f"  {entry.player_name} | "
            f"{entry.competition} | "
            f"{entry.season_year} | "
            f"{entry.team} | "
            f"minutes={entry.minutes_played} | "
            f"rating={entry.rating}"
        )

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
#smoke_test_player_validation.py

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.player_repository import PlayerRepository
from research.player_intelligence.player_validation import (
    validate_players,
    validation_report_to_dataframe,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "player_intelligence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "player_validation_report.csv"


def main() -> None:
    repository = PlayerRepository()
    players = repository.load_players()

    report = validate_players(players)
    report_df = validation_report_to_dataframe(report)

    report_df.to_csv(OUTPUT_PATH, index=False)

    print("Player Validation Smoke Test")
    print("----------------------------")
    print(report_df.to_string(index=False))
    print()
    print(f"Wrote validation report -> {OUTPUT_PATH}")

    assert report.total_players > 0

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
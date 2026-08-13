#smoke_test_player_sample_quality.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.player_sample_quality import (
    add_sample_quality,
    build_sample_quality_report,
    sample_quality_by_team,
    sample_quality_report_to_dataframe,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wc_2026_model_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "player_intelligence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_OUTPUT_PATH = OUTPUT_DIR / "player_sample_quality_report.csv"
TEAM_OUTPUT_PATH = OUTPUT_DIR / "player_sample_quality_by_team.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df = add_sample_quality(df)

    report = build_sample_quality_report(df)
    report_df = sample_quality_report_to_dataframe(report)
    team_df = sample_quality_by_team(df)

    report_df.to_csv(OVERALL_OUTPUT_PATH, index=False)
    team_df.to_csv(TEAM_OUTPUT_PATH, index=False)

    print("Player Sample Quality Smoke Test")
    print("--------------------------------")
    print()
    print("Overall report")
    print(report_df.to_string(index=False))
    print()
    print("Lowest observed-share teams")
    print(team_df.head(15).round(4).to_string(index=False))
    print()
    print(f"Wrote overall report -> {OVERALL_OUTPUT_PATH}")
    print(f"Wrote team report    -> {TEAM_OUTPUT_PATH}")

    assert report.total_players > 0

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
#smoke_test_observatory_schema.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.football_observatory.observatory_schema import (
    match_observation_from_row,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    observations = [
        match_observation_from_row(row)
        for _, row in df.iterrows()
    ]

    print("Football Observatory Schema Smoke Test")
    print("--------------------------------------")
    print(f"Rows loaded: {len(df)}")
    print(f"Observations created: {len(observations)}")

    first = observations[0]

    print()
    print("First observation")
    print("-----------------")
    print("Home team:", first.prematch.home_team)
    print("Away team:", first.prematch.away_team)
    print("Scoreline:", first.outcome.scoreline)
    print("Result:", first.outcome.result)
    print("Total goals:", first.outcome.total_goals)
    print("One-goal match:", first.outcome.is_one_goal_match)
    print("Both teams scored:", first.outcome.both_teams_scored)
    print("Clean sheet:", first.outcome.is_clean_sheet)


if __name__ == "__main__":
    main()
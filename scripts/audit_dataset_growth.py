#audit_dataset_growth.py

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)


def main():
    df = pd.read_csv(DATASET_PATH)

    print("Historical Dataset Growth")
    print("-------------------------")

    print(f"Rows: {len(df)}")
    print(f"Unique tournaments: {df['dataset_id'].nunique()}")
    print(f"Unique competitions: {df['competition'].nunique()}")

    print()

    summary = (
        df.groupby("competition")
        .agg(
            tournaments=("dataset_id", "nunique"),
            matches=("event_id", "count"),
            teams=("home_team", "nunique"),
        )
        .sort_values("matches", ascending=False)
    )

    print(summary)

    print()

    print("Tournament contribution")
    print("-----------------------")

    contribution = (
        df["dataset_id"]
        .value_counts()
        .sort_index()
    )

    print(contribution)


if __name__ == "__main__":
    main()
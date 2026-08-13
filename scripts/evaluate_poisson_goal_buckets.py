#evaluate_poisson_goal_buckets.py

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "poisson_goal_model_predictions.csv"
)


def print_actual_goal_buckets(df, actual_col, pred_col, title):
    print()
    print(title)
    print("-" * len(title))

    grouped = (
        df.groupby(actual_col)
        .agg(
            matches=(actual_col, "count"),
            avg_predicted=(pred_col, "mean"),
        )
        .reset_index()
    )

    print(grouped.to_string(index=False))


def print_prediction_buckets(df, actual_col, pred_col, title):
    print()
    print(title)
    print("-" * len(title))

    buckets = pd.cut(
        df[pred_col],
        bins=[0, 0.75, 1.25, 1.75, 2.25, 3, 10],
        include_lowest=True,
    )

    grouped = (
        df.assign(bucket=buckets)
        .groupby("bucket")
        .agg(
            matches=(actual_col, "count"),
            avg_actual=(actual_col, "mean"),
            avg_predicted=(pred_col, "mean"),
        )
    )

    print(grouped.to_string())


def main():
    df = pd.read_csv(PREDICTIONS_FILE)

    print(f"Matches: {len(df)}")

    print_actual_goal_buckets(
        df,
        "home_score",
        "pred_home_goals",
        "Home goal calibration (actual → predicted)",
    )

    print_actual_goal_buckets(
        df,
        "away_score",
        "pred_away_goals",
        "Away goal calibration (actual → predicted)",
    )

    print_prediction_buckets(
        df,
        "home_score",
        "pred_home_goals",
        "Home goal calibration (predicted → actual)",
    )

    print_prediction_buckets(
        df,
        "away_score",
        "pred_away_goals",
        "Away goal calibration (predicted → actual)",
    )


if __name__ == "__main__":
    main()
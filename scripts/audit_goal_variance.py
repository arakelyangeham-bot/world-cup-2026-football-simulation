#audit_goal_variance.py

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "poisson_goal_model_predictions.csv"
)


def summarize(name, actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    print()
    print(name)
    print("-" * len(name))

    print(f"Actual mean        : {actual.mean():.3f}")
    print(f"Predicted mean     : {predicted.mean():.3f}")

    print(f"Actual variance    : {actual.var(ddof=1):.3f}")
    print(f"Predicted variance : {predicted.var(ddof=1):.3f}")

    print(f"Variance ratio     : {predicted.var(ddof=1) / actual.var(ddof=1):.3f}")

    print(f"Actual std         : {actual.std(ddof=1):.3f}")
    print(f"Predicted std      : {predicted.std(ddof=1):.3f}")


def main():
    df = pd.read_csv(PREDICTIONS_FILE)

    summarize(
        "Home Goals",
        df["home_score"],
        df["pred_home_goals"],
    )

    summarize(
        "Away Goals",
        df["away_score"],
        df["pred_away_goals"],
    )

    summarize(
        "Total Goals",
        df["total_goals"],
        df["pred_total_goals"],
    )

    summarize(
        "Goal Difference",
        df["goal_diff"],
        df["pred_goal_diff"],
    )


if __name__ == "__main__":
    main()
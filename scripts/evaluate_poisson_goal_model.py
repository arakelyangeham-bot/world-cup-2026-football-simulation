#evaluate_poisson_goal_model.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "poisson_goal_model_predictions.csv"
)


def correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


def summarize(target, prediction):
    mae = mean_absolute_error(target, prediction)
    rmse = mean_squared_error(target, prediction) ** 0.5
    corr = correlation(target, prediction)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Correlation": corr,
        "Mean Actual": np.mean(target),
        "Mean Predicted": np.mean(prediction),
    }


def residual_summary(name, residuals):
    print()
    print(name)
    print("-" * len(name))
    print(f"Mean residual: {np.mean(residuals):.3f}")
    print(f"Std residual : {np.std(residuals):.3f}")
    print(f"Min residual : {np.min(residuals):.3f}")
    print(f"Max residual : {np.max(residuals):.3f}")


def main():
    df = pd.read_csv(PREDICTIONS_FILE)

    print(f"Matches: {len(df)}")

    evaluations = {
        "Home Goals": summarize(
            df["home_score"],
            df["pred_home_goals"],
        ),
        "Away Goals": summarize(
            df["away_score"],
            df["pred_away_goals"],
        ),
        "Total Goals": summarize(
            df["total_goals"],
            df["pred_total_goals"],
        ),
        "Goal Difference": summarize(
            df["goal_diff"],
            df["pred_goal_diff"],
        ),
    }

    for model_name, metrics in evaluations.items():
        print()
        print(model_name)
        print("-" * len(model_name))

        for metric, value in metrics.items():
            print(f"{metric:<18} {value:.3f}")

    home_residual = (
        df["home_score"]
        - df["pred_home_goals"]
    )

    away_residual = (
        df["away_score"]
        - df["pred_away_goals"]
    )

    total_residual = (
        df["total_goals"]
        - df["pred_total_goals"]
    )

    residual_summary("Home Goal Residuals", home_residual)
    residual_summary("Away Goal Residuals", away_residual)
    residual_summary("Total Goal Residuals", total_residual)

    print()
    print("Largest prediction errors")
    print("-------------------------")

    df["absolute_total_error"] = abs(total_residual)

    largest = (
        df.sort_values(
            "absolute_total_error",
            ascending=False,
        )
        .head(15)
    )

    print(
        largest[
            [
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "pred_home_goals",
                "pred_away_goals",
                "absolute_total_error",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
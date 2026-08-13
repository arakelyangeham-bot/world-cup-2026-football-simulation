#fit_poisson_goal_model.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COEFFICIENTS_FILE = OUTPUT_DIR / "poisson_goal_model_coefficients.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "poisson_goal_model_predictions.csv"


HOME_FEATURES = [
    "home_poisson_attack",
    "away_poisson_defense",
    "fifa_points_diff",
]

AWAY_FEATURES = [
    "away_poisson_attack",
    "home_poisson_defense",
    "fifa_points_diff",
]


def fit_model(df: pd.DataFrame, features: list[str], target: str):
    model_df = df[features + [target]].dropna()

    x = model_df[features]
    y = model_df[target]

    model = PoissonRegressor(
        alpha=0.0,
        max_iter=1000,
    )

    model.fit(x, y)

    predictions = model.predict(x)

    return model, model_df, predictions


def summarize_model(
    name: str,
    model,
    features: list[str],
    y_true,
    y_pred,
) -> list[dict]:
    print()
    print(name)
    print("-" * len(name))

    print(f"Rows: {len(y_true)}")
    print(f"Intercept: {model.intercept_:.6f}")

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Mean actual goals: {np.mean(y_true):.3f}")
    print(f"Mean predicted goals: {np.mean(y_pred):.3f}")

    rows = []

    for feature, coef in zip(features, model.coef_):
        print(f"{feature:<30} {coef:>10.6f}")

        rows.append(
            {
                "model": name,
                "feature": feature,
                "coefficient": coef,
            }
        )

    rows.append(
        {
            "model": name,
            "feature": "intercept",
            "coefficient": model.intercept_,
        }
    )

    return rows


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    print(f"Loaded rows: {len(df)}")

    home_model, home_df, home_pred = fit_model(
        df,
        HOME_FEATURES,
        "home_score",
    )

    away_model, away_df, away_pred = fit_model(
        df,
        AWAY_FEATURES,
        "away_score",
    )

    coefficient_rows = []

    coefficient_rows.extend(
        summarize_model(
            "home_goal_model",
            home_model,
            HOME_FEATURES,
            home_df["home_score"],
            home_pred,
        )
    )

    coefficient_rows.extend(
        summarize_model(
            "away_goal_model",
            away_model,
            AWAY_FEATURES,
            away_df["away_score"],
            away_pred,
        )
    )

    pd.DataFrame(coefficient_rows).to_csv(
        COEFFICIENTS_FILE,
        index=False,
    )

    predictions = df.copy()

    predictions["pred_home_goals"] = home_model.predict(
        predictions[HOME_FEATURES]
    )

    predictions["pred_away_goals"] = away_model.predict(
        predictions[AWAY_FEATURES]
    )

    predictions["pred_total_goals"] = (
        predictions["pred_home_goals"]
        + predictions["pred_away_goals"]
    )

    predictions["pred_goal_diff"] = (
        predictions["pred_home_goals"]
        - predictions["pred_away_goals"]
    )

    predictions.to_csv(PREDICTIONS_FILE, index=False)

    print()
    print(f"Wrote coefficients -> {COEFFICIENTS_FILE}")
    print(f"Wrote predictions  -> {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
#compare_poisson_feature_sets.py

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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "poisson_feature_model_comparison.csv"
)


MODEL_SPECS = {
    "poisson_attack_only": {
        "home": ["home_poisson_attack"],
        "away": ["away_poisson_attack"],
    },
    "poisson_attack_defense": {
        "home": ["home_poisson_attack", "away_poisson_defense"],
        "away": ["away_poisson_attack", "home_poisson_defense"],
    },
    "poisson_attack_defense_fifa": {
        "home": ["home_poisson_attack", "away_poisson_defense", "fifa_points_diff"],
        "away": ["away_poisson_attack", "home_poisson_defense", "fifa_points_diff"],
    },
    "composite_attack_defense_fifa": {
        "home": ["home_attack", "away_defense", "fifa_points_diff"],
        "away": ["away_attack", "home_defense", "fifa_points_diff"],
    },
    "full_composite": {
        "home": [
            "home_attack",
            "home_midfield",
            "home_gk",
            "away_defense",
            "away_gk",
            "fifa_points_diff",
        ],
        "away": [
            "away_attack",
            "away_midfield",
            "away_gk",
            "home_defense",
            "home_gk",
            "fifa_points_diff",
        ],
    },
}


def correlation(actual, predicted) -> float:
    if len(actual) < 2:
        return float("nan")

    return float(np.corrcoef(actual, predicted)[0, 1])


def fit_poisson_model(
    df: pd.DataFrame,
    features: list[str],
    target: str,
):
    model_df = df[features + [target]].dropna()

    x = model_df[features]
    y = model_df[target]

    model = PoissonRegressor(
        alpha=0.0,
        max_iter=1000,
    )

    model.fit(x, y)
    prediction = model.predict(x)

    return model, model_df, prediction


def evaluate_model(
    df: pd.DataFrame,
    model_name: str,
    spec: dict,
) -> dict:
    home_model, home_df, pred_home = fit_poisson_model(
        df,
        spec["home"],
        "home_score",
    )

    away_model, away_df, pred_away = fit_poisson_model(
        df,
        spec["away"],
        "away_score",
    )

    evaluation = df.copy()
    evaluation["pred_home"] = pred_home
    evaluation["pred_away"] = pred_away
    evaluation["pred_total"] = (
        evaluation["pred_home"]
        + evaluation["pred_away"]
    )
    evaluation["pred_goal_diff"] = (
        evaluation["pred_home"]
        - evaluation["pred_away"]
    )

    return {
        "model": model_name,
        "home_features": ", ".join(spec["home"]),
        "away_features": ", ".join(spec["away"]),
        "home_mae": mean_absolute_error(
            evaluation["home_score"],
            evaluation["pred_home"],
        ),
        "away_mae": mean_absolute_error(
            evaluation["away_score"],
            evaluation["pred_away"],
        ),
        "home_rmse": mean_squared_error(
            evaluation["home_score"],
            evaluation["pred_home"],
        ) ** 0.5,
        "away_rmse": mean_squared_error(
            evaluation["away_score"],
            evaluation["pred_away"],
        ) ** 0.5,
        "home_corr": correlation(
            evaluation["home_score"],
            evaluation["pred_home"],
        ),
        "away_corr": correlation(
            evaluation["away_score"],
            evaluation["pred_away"],
        ),
        "total_mae": mean_absolute_error(
            evaluation["total_goals"],
            evaluation["pred_total"],
        ),
        "goal_diff_mae": mean_absolute_error(
            evaluation["goal_diff"],
            evaluation["pred_goal_diff"],
        ),
        "home_intercept": home_model.intercept_,
        "away_intercept": away_model.intercept_,
    }


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    print(f"Rows: {len(df)}")

    results = [
        evaluate_model(df, model_name, spec)
        for model_name, spec in MODEL_SPECS.items()
    ]

    results_df = pd.DataFrame(results).sort_values("goal_diff_mae")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    display_cols = [
        "model",
        "home_mae",
        "away_mae",
        "home_rmse",
        "away_rmse",
        "home_corr",
        "away_corr",
        "total_mae",
        "goal_diff_mae",
    ]

    print()
    print(results_df[display_cols].round(3).to_string(index=False))

    print()
    print(f"Wrote comparison -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
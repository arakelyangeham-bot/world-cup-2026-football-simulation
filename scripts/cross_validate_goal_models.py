#cross_validate_goal_models.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from simulation.goal_models import PoissonGoalModel


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
    / "goal_model_cross_validation.csv"
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


def evaluate_predictions(
    model_name: str,
    held_out_dataset: str,
    test_df: pd.DataFrame,
    prediction,
) -> dict:
    return {
        "model": model_name,
        "held_out_dataset": held_out_dataset,
        "matches": len(test_df),
        "home_mae": mean_absolute_error(
            test_df["home_score"],
            prediction.pred_home_goals,
        ),
        "away_mae": mean_absolute_error(
            test_df["away_score"],
            prediction.pred_away_goals,
        ),
        "home_rmse": mean_squared_error(
            test_df["home_score"],
            prediction.pred_home_goals,
        ) ** 0.5,
        "away_rmse": mean_squared_error(
            test_df["away_score"],
            prediction.pred_away_goals,
        ) ** 0.5,
        "home_corr": correlation(
            test_df["home_score"],
            prediction.pred_home_goals,
        ),
        "away_corr": correlation(
            test_df["away_score"],
            prediction.pred_away_goals,
        ),
        "total_mae": mean_absolute_error(
            test_df["total_goals"],
            prediction.pred_total_goals,
        ),
        "goal_diff_mae": mean_absolute_error(
            test_df["goal_diff"],
            prediction.pred_goal_diff,
        ),
    }


def build_model(model_name: str, spec: dict) -> PoissonGoalModel:
    return PoissonGoalModel(
        name=model_name,
        home_features=spec["home"],
        away_features=spec["away"],
    )


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    if "dataset_id" not in df.columns:
        raise ValueError("historical dataset must contain dataset_id column")

    dataset_ids = sorted(df["dataset_id"].dropna().unique())

    print(f"Rows: {len(df)}")
    print(f"Datasets: {len(dataset_ids)}")

    rows = []

    for model_name, spec in MODEL_SPECS.items():
        print()
        print(f"Model: {model_name}")
        print("-" * (7 + len(model_name)))

        for held_out_dataset in dataset_ids:
            train_df = df[df["dataset_id"] != held_out_dataset].copy()
            test_df = df[df["dataset_id"] == held_out_dataset].copy()

            if test_df.empty or train_df.empty:
                continue

            model = build_model(model_name, spec)
            model.fit(train_df)

            prediction = model.predict(test_df)

            result = evaluate_predictions(
                model_name,
                held_out_dataset,
                test_df,
                prediction,
            )

            rows.append(result)

            print(
                f"  held out {held_out_dataset:<18} "
                f"matches={len(test_df):>3} "
                f"goal_diff_mae={result['goal_diff_mae']:.3f}"
            )

    results_df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    summary = (
        results_df
        .groupby("model")
        .agg(
            folds=("held_out_dataset", "count"),
            matches=("matches", "sum"),
            home_mae=("home_mae", "mean"),
            away_mae=("away_mae", "mean"),
            home_rmse=("home_rmse", "mean"),
            away_rmse=("away_rmse", "mean"),
            home_corr=("home_corr", "mean"),
            away_corr=("away_corr", "mean"),
            total_mae=("total_mae", "mean"),
            goal_diff_mae=("goal_diff_mae", "mean"),
        )
        .reset_index()
        .sort_values("goal_diff_mae")
    )

    print()
    print("Cross-validation summary")
    print("------------------------")
    print(summary.round(3).to_string(index=False))

    print()
    print(f"Wrote fold results -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
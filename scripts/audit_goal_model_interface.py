#audit_goal_model_interface.py

from pathlib import Path

import pandas as pd

from simulation.goal_models import PoissonGoalModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    model = PoissonGoalModel(
        name="poisson_attack_defense_fifa",
        home_features=[
            "home_poisson_attack",
            "away_poisson_defense",
            "fifa_points_diff",
        ],
        away_features=[
            "away_poisson_attack",
            "home_poisson_defense",
            "fifa_points_diff",
        ],
    )

    model.fit(df)

    prediction = model.predict(df)

    print(f"Rows: {len(df)}")
    print(f"Model: {model.name}")
    print(f"Pred home goals mean: {prediction.pred_home_goals.mean():.3f}")
    print(f"Pred away goals mean: {prediction.pred_away_goals.mean():.3f}")
    print(f"Pred total goals mean: {prediction.pred_total_goals.mean():.3f}")
    print(f"Pred goal diff mean: {prediction.pred_goal_diff.mean():.3f}")


if __name__ == "__main__":
    main()
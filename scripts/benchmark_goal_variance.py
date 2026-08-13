#benchmark_goal_variance.py

from pathlib import Path

import numpy as np
import pandas as pd

from simulation.lambda_models import expected_goals


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

N = 1000
LAMBDA_MODEL = "calibrated"


def row_to_team_dict(row, prefix):
    return {
        "attack": row[f"{prefix}_attack"],
        "defense": row[f"{prefix}_defense"],
        "poisson_attack": row[f"{prefix}_poisson_attack"],
        "poisson_defense": row[f"{prefix}_poisson_defense"],
        "fifa_points": row[f"{prefix}_fifa_points"],
    }


def summarize(label, home_goals, away_goals):
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)

    total_goals = home_goals + away_goals
    goal_diff = home_goals - away_goals

    return {
        "label": label,
        "home_var": home_goals.var(ddof=1),
        "away_var": away_goals.var(ddof=1),
        "total_var": total_goals.var(ddof=1),
        "goal_diff_var": goal_diff.var(ddof=1),
    }


def simulate_calibrated_scores(df):
    home_scores = []
    away_scores = []

    for _, row in df.iterrows():
        home = row_to_team_dict(row, "home")
        away = row_to_team_dict(row, "away")

        lambda_home, lambda_away = expected_goals(
            home,
            away,
            lambda_model=LAMBDA_MODEL,
        )

        for _ in range(N):
            home_scores.append(
                int(np.random.poisson(lambda_home))
            )
            away_scores.append(
                int(np.random.poisson(lambda_away))
            )

    return home_scores, away_scores


def print_variance_comparison(historical, model):
    print()
    print("Goal Variance Benchmark")
    print("-----------------------")
    print(f"{'Metric':20}{'Historical':>14}{'Model':>14}{'Error':>14}{'Ratio':>14}")
    print("-" * 76)

    for key, label in [
        ("home_var", "Home variance"),
        ("away_var", "Away variance"),
        ("total_var", "Total variance"),
        ("goal_diff_var", "Goal diff variance"),
    ]:
        hist_value = historical[key]
        model_value = model[key]
        error = model_value - hist_value
        ratio = model_value / hist_value if hist_value != 0 else float("nan")

        print(
            f"{label:<20}"
            f"{hist_value:>14.3f}"
            f"{model_value:>14.3f}"
            f"{error:>14.3f}"
            f"{ratio:>14.3f}"
        )


def main() -> None:
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)

    historical = summarize(
        "historical",
        df["home_score"],
        df["away_score"],
    )

    model_home, model_away = simulate_calibrated_scores(df)

    calibrated = summarize(
        LAMBDA_MODEL,
        model_home,
        model_away,
    )

    print(f"Historical matches: {len(df)}")
    print(f"Samples per historical match: {N}")
    print_variance_comparison(historical, calibrated)


if __name__ == "__main__":
    main()
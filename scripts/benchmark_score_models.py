#benchmark_score_models.py

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from scripts.team_strength_loader import load_team_repository
from scripts.match_engine import simulate_poisson_score
from shared.team_name_normalizer import normalize_team_name
from simulation.lambda_models import expected_goals


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

N = 1000

LAMBDA_MODELS = [
    "heuristic",
    "calibrated",
]


def row_to_team_dict(row, prefix):
    return {
        "attack": row[f"{prefix}_attack"],
        "defense": row[f"{prefix}_defense"],
        "poisson_attack": row[f"{prefix}_poisson_attack"],
        "poisson_defense": row[f"{prefix}_poisson_defense"],
        "fifa_points": row[f"{prefix}_fifa_points"],
    }


def summarize_scores(label, scores):
    home_goals = np.array([s[0] for s in scores])
    away_goals = np.array([s[1] for s in scores])
    total_goals = home_goals + away_goals
    goal_diff = home_goals - away_goals

    score_counter = Counter(scores)

    return {
        "model": label,
        "matches": len(scores),
        "avg_home_goals": home_goals.mean(),
        "avg_away_goals": away_goals.mean(),
        "avg_total_goals": total_goals.mean(),
        "var_total_goals": total_goals.var(ddof=1),
        "avg_goal_diff": goal_diff.mean(),
        "draw_rate": np.mean(home_goals == away_goals),
        "home_win_rate": np.mean(home_goals > away_goals),
        "away_win_rate": np.mean(away_goals > home_goals),
        "clean_sheet_rate": np.mean((home_goals == 0) | (away_goals == 0)),
        "five_plus_total_rate": np.mean(total_goals >= 5),
        "six_plus_total_rate": np.mean(total_goals >= 6),
        "top_scores": score_counter.most_common(10),
    }


def print_summary(summary):
    print()
    print(summary["model"])
    print("-" * len(summary["model"]))

    for key in [
        "matches",
        "avg_home_goals",
        "avg_away_goals",
        "avg_total_goals",
        "var_total_goals",
        "avg_goal_diff",
        "draw_rate",
        "home_win_rate",
        "away_win_rate",
        "clean_sheet_rate",
        "five_plus_total_rate",
        "six_plus_total_rate",
    ]:
        value = summary[key]

        if isinstance(value, float):
            print(f"{key:<24} {value:.3f}")
        else:
            print(f"{key:<24} {value}")

    print()
    print("Top scorelines")
    print("--------------")

    for (home, away), count in summary["top_scores"]:
        print(f"{home}-{away:<5} {count}")


def simulate_scores_for_lambda_model(df, lambda_model):
    scores = []

    for _, row in df.iterrows():
        home = row_to_team_dict(row, "home")
        away = row_to_team_dict(row, "away")

        for _ in range(N):
            lambda_home, lambda_away = expected_goals(
                home,
                away,
                lambda_model=lambda_model,
            )

            goals_home = np.random.poisson(lambda_home)
            goals_away = np.random.poisson(lambda_away)

            scores.append((int(goals_home), int(goals_away)))

    return scores


def historical_scores(df):
    return list(
        zip(
            df["home_score"].astype(int),
            df["away_score"].astype(int),
        )
    )


def main() -> None:
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)

    print("Score Model Benchmark")
    print("---------------------")
    print(f"Historical matches: {len(df)}")
    print(f"Samples per historical match: {N}")

    summaries = []

    summaries.append(
        summarize_scores(
            "historical",
            historical_scores(df),
        )
    )

    for lambda_model in LAMBDA_MODELS:
        scores = simulate_scores_for_lambda_model(df, lambda_model)
        summaries.append(
            summarize_scores(
                lambda_model,
                scores,
            )
        )

    for summary in summaries:
        print_summary(summary)


if __name__ == "__main__":
    main()
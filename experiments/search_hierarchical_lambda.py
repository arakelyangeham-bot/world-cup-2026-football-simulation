#search_hierarchical_lambda.py

from pathlib import Path

import numpy as np
import pandas as pd

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import (
    poisson_sampler,
    hierarchical_stochastic_lambda_sampler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "model_training" / "hierarchical_lambda_search.csv"

N = 500
LAMBDA_MODEL = "calibrated"

TEMPO_CVS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
TEAM_CVS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def row_to_team_dict(row, prefix):
    return {
        "attack": row[f"{prefix}_attack"],
        "defense": row[f"{prefix}_defense"],
        "poisson_attack": row[f"{prefix}_poisson_attack"],
        "poisson_defense": row[f"{prefix}_poisson_defense"],
        "fifa_points": row[f"{prefix}_fifa_points"],
    }


def summarize_scores(home_goals, away_goals):
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)

    total_goals = home_goals + away_goals
    goal_diff = home_goals - away_goals

    return {
        "avg_total_goals": total_goals.mean(),
        "total_var": total_goals.var(ddof=1),
        "goal_diff_var": goal_diff.var(ddof=1),
        "draw_rate": np.mean(home_goals == away_goals),
        "clean_sheet_rate": np.mean((home_goals == 0) | (away_goals == 0)),
        "five_plus_total_rate": np.mean(total_goals >= 5),
        "six_plus_total_rate": np.mean(total_goals >= 6),
    }


def simulate_hierarchical(df, tempo_cv, team_cv):
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
            goals_home, goals_away = hierarchical_stochastic_lambda_sampler(
                lambda_home,
                lambda_away,
                tempo_cv,
                team_cv,
            )

            home_scores.append(goals_home)
            away_scores.append(goals_away)

    return summarize_scores(home_scores, away_scores)


def normalized_abs_error(model_value, historical_value):
    if historical_value == 0:
        return np.nan

    return abs(model_value - historical_value) / abs(historical_value)


def fitness_score(summary, historical):
    metrics = [
        "avg_total_goals",
        "total_var",
        "goal_diff_var",
        "draw_rate",
        "clean_sheet_rate",
        "five_plus_total_rate",
        "six_plus_total_rate",
    ]

    errors = [
        normalized_abs_error(summary[metric], historical[metric])
        for metric in metrics
    ]

    return float(np.nanmean(errors))


def main() -> None:
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)

    historical = summarize_scores(
        df["home_score"],
        df["away_score"],
    )

    rows = []

    print("Hierarchical Lambda Grid Search")
    print("-------------------------------")
    print(f"Historical matches: {len(df)}")
    print(f"Samples per match: {N}")
    print(f"Candidates: {len(TEMPO_CVS) * len(TEAM_CVS)}")
    print()

    for tempo_cv in TEMPO_CVS:
        for team_cv in TEAM_CVS:
            summary = simulate_hierarchical(
                df,
                tempo_cv=tempo_cv,
                team_cv=team_cv,
            )

            score = fitness_score(summary, historical)

            row = {
                "tempo_cv": tempo_cv,
                "team_cv": team_cv,
                "fitness": score,
            }

            row.update(summary)
            rows.append(row)

            print(
                f"tempo={tempo_cv:.2f} "
                f"team={team_cv:.2f} "
                f"fitness={score:.4f} "
                f"total_var={summary['total_var']:.3f} "
                f"draw={summary['draw_rate']:.3f}"
            )

    results = pd.DataFrame(rows).sort_values("fitness")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Top candidates")
    print("--------------")
    print(results.head(10).round(4).to_string(index=False))

    print()
    print(f"Wrote -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
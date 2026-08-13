#benchmark_negative_binomial_sampler.py

from pathlib import Path

import numpy as np
import pandas as pd

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import poisson_sampler, negative_binomial_sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv"

N = 1000
LAMBDA_MODEL = "calibrated"

K_VALUES = [
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
    2.75,
    3.00,
    3.50,
    4.00,
]


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

    return {
        "model": label,
        "avg_total_goals": total_goals.mean(),
        "total_var": total_goals.var(ddof=1),
        "draw_rate": np.mean(home_goals == away_goals),
        "clean_sheet_rate": np.mean((home_goals == 0) | (away_goals == 0)),
        "five_plus_total_rate": np.mean(total_goals >= 5),
        "six_plus_total_rate": np.mean(total_goals >= 6),
    }


def simulate_with_sampler(df, sampler_name, k=None):
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
            if sampler_name == "poisson":
                home_scores.append(poisson_sampler(lambda_home))
                away_scores.append(poisson_sampler(lambda_away))
            elif sampler_name == "negative_binomial":
                home_scores.append(negative_binomial_sampler(lambda_home, k))
                away_scores.append(negative_binomial_sampler(lambda_away, k))
            else:
                raise ValueError(f"Unknown sampler: {sampler_name}")

    return home_scores, away_scores


def main() -> None:
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)

    historical = summarize(
        "historical",
        df["home_score"],
        df["away_score"],
    )

    rows = [historical]

    home_scores, away_scores = simulate_with_sampler(df, "poisson")
    rows.append(summarize("poisson", home_scores, away_scores))

    for k in K_VALUES:
        home_scores, away_scores = simulate_with_sampler(
            df,
            "negative_binomial",
            k=k,
        )

        rows.append(
            summarize(
                f"negative_binomial_k={k}",
                home_scores,
                away_scores,
            )
        )

    results = pd.DataFrame(rows)

    print("Negative Binomial Sampler Benchmark")
    print("-----------------------------------")
    print(f"Historical matches: {len(df)}")
    print(f"Samples per match: {N}")
    print()
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
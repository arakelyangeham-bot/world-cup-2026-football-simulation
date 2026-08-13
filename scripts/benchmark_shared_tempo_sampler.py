#benchmark_shared_tempo_sampler.py

from pathlib import Path

import numpy as np
import pandas as pd

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import poisson_sampler, shared_tempo_poisson_sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv"

N = 1000
LAMBDA_MODEL = "calibrated"

TEMPO_VARIANCES = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
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


def simulate(df, mode, tempo_variance=None):
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
            if mode == "poisson":
                goals_home = poisson_sampler(lambda_home)
                goals_away = poisson_sampler(lambda_away)

            elif mode == "shared_tempo":
                goals_home, goals_away = shared_tempo_poisson_sampler(
                    lambda_home,
                    lambda_away,
                    tempo_variance,
                )

            else:
                raise ValueError(f"Unknown mode: {mode}")

            home_scores.append(goals_home)
            away_scores.append(goals_away)

    return home_scores, away_scores


def main() -> None:
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)

    rows = []

    rows.append(
        summarize(
            "historical",
            df["home_score"],
            df["away_score"],
        )
    )

    home_scores, away_scores = simulate(df, "poisson")
    rows.append(summarize("poisson", home_scores, away_scores))

    for tempo_variance in TEMPO_VARIANCES:
        home_scores, away_scores = simulate(
            df,
            "shared_tempo",
            tempo_variance=tempo_variance,
        )

        rows.append(
            summarize(
                f"shared_tempo_var={tempo_variance}",
                home_scores,
                away_scores,
            )
        )

    results = pd.DataFrame(rows)

    print("Shared Tempo Goal Sampler Benchmark")
    print("-----------------------------------")
    print(f"Historical matches: {len(df)}")
    print(f"Samples per match: {N}")
    print()
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
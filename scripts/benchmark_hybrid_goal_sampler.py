#benchmark_hybrid_goal_sampler.py

from pathlib import Path

import numpy as np
import pandas as pd

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import poisson_sampler, hybrid_volatility_sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv"

N = 1000
LAMBDA_MODEL = "calibrated"

CONFIGS = [
    (0.05, 1.5),
    (0.10, 1.5),
    (0.10, 2.0),
    (0.15, 1.5),
    (0.15, 2.0),
    (0.20, 1.5),
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


def simulate(df, mode, volatility_probability=None, volatility_multiplier=None):
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
                home_scores.append(poisson_sampler(lambda_home))
                away_scores.append(poisson_sampler(lambda_away))
            elif mode == "hybrid":
                home_scores.append(
                    hybrid_volatility_sampler(
                        lambda_home,
                        volatility_probability,
                        volatility_multiplier,
                    )
                )
                away_scores.append(
                    hybrid_volatility_sampler(
                        lambda_away,
                        volatility_probability,
                        volatility_multiplier,
                    )
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

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

    for probability, multiplier in CONFIGS:
        home_scores, away_scores = simulate(
            df,
            "hybrid",
            volatility_probability=probability,
            volatility_multiplier=multiplier,
        )

        rows.append(
            summarize(
                f"hybrid_p={probability}_m={multiplier}",
                home_scores,
                away_scores,
            )
        )

    results = pd.DataFrame(rows)

    print("Hybrid Goal Sampler Benchmark")
    print("-----------------------------")
    print(f"Historical matches: {len(df)}")
    print(f"Samples per match: {N}")
    print()
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
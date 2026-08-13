#benchmark_scoreline_distribution.py

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import (
    poisson_sampler,
    hierarchical_stochastic_lambda_sampler,
    zero_zero_deflated_hierarchical_sampler,
    dixon_coles_hierarchical_sampler,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "scoreline_distribution_benchmark.csv"
COMPARISON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "benchmarks"
    / "scoreline_frequency_comparison.csv"
)

N = 1000
LAMBDA_MODEL = "calibrated"
PRODUCTION_MODEL = "dixon_coles_rho=0.30"

MAX_GOALS = 6
OTHER_BUCKET = "other"

ZERO_ZERO_RESAMPLE_PROBS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
]

DIXON_COLES_RHOS = [
    -0.30,
    -0.20,
    -0.10,
    0.00,
    0.10,
    0.20,
    0.30,
]

def row_to_team_dict(row, prefix):
    return {
        "attack": row[f"{prefix}_attack"],
        "defense": row[f"{prefix}_defense"],
        "poisson_attack": row[f"{prefix}_poisson_attack"],
        "poisson_defense": row[f"{prefix}_poisson_defense"],
        "fifa_points": row[f"{prefix}_fifa_points"],
    }


def bucket_scoreline(home_goals, away_goals):
    if home_goals > MAX_GOALS or away_goals > MAX_GOALS:
        return OTHER_BUCKET

    return f"{home_goals}-{away_goals}"


def scoreline_distribution(home_goals, away_goals):
    counts = Counter(
        bucket_scoreline(h, a)
        for h, a in zip(home_goals, away_goals)
    )

    total = sum(counts.values())

    return {
        scoreline: count / total
        for scoreline, count in counts.items()
    }


def total_variation_distance(p, q):
    keys = set(p) | set(q)

    return 0.5 * sum(
        abs(p.get(key, 0.0) - q.get(key, 0.0))
        for key in keys
    )


def simulate(
    df,
    mode,
    zero_zero_resample_prob=None,
    rho=None,
):
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

            elif mode == "hierarchical":
                goals_home, goals_away = hierarchical_stochastic_lambda_sampler(
                    lambda_home,
                    lambda_away,
                    tempo_cv=0.60,
                    team_cv=0.10,
                )
            elif mode == "zero_zero_deflated":
                goals_home, goals_away = zero_zero_deflated_hierarchical_sampler(
                    lambda_home,
                    lambda_away,
                    tempo_cv=0.60,
                    team_cv=0.10,
                    zero_zero_resample_prob=zero_zero_resample_prob,
                )
            elif mode == "dixon_coles":
                goals_home, goals_away = dixon_coles_hierarchical_sampler(
                    lambda_home,
                    lambda_away,
                    tempo_cv=0.60,
                    team_cv=0.10,
                    rho=rho,
                    max_goals=10,
                )

            else:
                raise ValueError(f"Unknown mode: {mode}")

            home_scores.append(goals_home)
            away_scores.append(goals_away)

    return home_scores, away_scores


def main():
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)

    distributions = {}

    historical_home = df["home_score"].tolist()
    historical_away = df["away_score"].tolist()
    historical_dist = scoreline_distribution(historical_home, historical_away)

    distributions["historical"] = historical_dist

    rows = []

    rows.append(
        {
            "model": "historical",
            "total_variation_distance": 0.0,
            "is_production": False,
        }
    )

    for mode in ["poisson", "hierarchical"]:
        home_scores, away_scores = simulate(df, mode)
        model_dist = scoreline_distribution(home_scores, away_scores)
        distributions[mode] = model_dist

        rows.append(
            {
                "model": mode,
                "total_variation_distance": total_variation_distance(
                    historical_dist,
                    model_dist,
                ),
                "is_production": mode == PRODUCTION_MODEL,
            }
        )

    for prob in ZERO_ZERO_RESAMPLE_PROBS:
        model_name = f"zero_zero_deflated_{prob:.2f}"

        home_scores, away_scores = simulate(
            df,
            "zero_zero_deflated",
            zero_zero_resample_prob=prob,
        )

        model_dist = scoreline_distribution(home_scores, away_scores)
        distributions[model_name] = model_dist

        rows.append(
            {
                "model": model_name,
                "total_variation_distance": total_variation_distance(
                    historical_dist,
                    model_dist,
                ),
                "is_production": model_name == PRODUCTION_MODEL
            }
        )
    
    for rho in DIXON_COLES_RHOS:
        model_name = f"dixon_coles_rho={rho:.2f}"

        home_scores, away_scores = simulate(
            df,
            "dixon_coles",
            rho=rho,
        )

        model_dist = scoreline_distribution(home_scores, away_scores)
        distributions[model_name] = model_dist

        rows.append(
            {
                "model": model_name,
                "total_variation_distance": total_variation_distance(
                    historical_dist,
                    model_dist,
                ),
                "is_production": model_name == PRODUCTION_MODEL
            }
        )

    results = pd.DataFrame(rows).sort_values("total_variation_distance")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    all_scorelines = sorted(
        set().union(*[set(dist.keys()) for dist in distributions.values()])
    )

    comparison_rows = []

    for scoreline in all_scorelines:
        historical_value = distributions["historical"].get(scoreline, 0.0)

        row = {
            "scoreline": scoreline,
            "historical": historical_value,
        }

        for model_name, dist in distributions.items():
            if model_name == "historical":
                continue

            model_value = dist.get(scoreline, 0.0)

            row[model_name] = model_value
            row[f"{model_name}_error"] = model_value - historical_value
            row[f"{model_name}_abs_error"] = abs(model_value - historical_value)

        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)

    comparison["hierarchical_error_share"] = (
        comparison["hierarchical_abs_error"]
        / comparison["hierarchical_abs_error"].sum()
    )

    comparison["direction"] = np.where(
        comparison["hierarchical_error"] > 0,
        "overproduced",
        "underproduced",
    )

    comparison["improvement"] = (
        comparison["poisson_abs_error"]
        - comparison["hierarchical_abs_error"]
    )

    comparison = comparison.sort_values(
        "improvement",
        ascending=False,
    )



    comparison.to_csv(COMPARISON_OUTPUT_PATH, index=False)

    print("Scoreline Distribution Benchmark")
    print("--------------------------------")
    print(f"Historical matches: {len(df)}")
    print(f"Samples per match: {N}")
    print(f"Max explicit goals bucket: {MAX_GOALS}")
    print()
    print(results.round(4).to_string(index=False))
    print()
    print(f"Wrote benchmark to {OUTPUT_PATH}")
    print(f"Wrote frequency comparison to {COMPARISON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
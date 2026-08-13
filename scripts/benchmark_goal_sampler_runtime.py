#benchmark_goal_sampler_runtime.py

from pathlib import Path
import time

import numpy as np
import pandas as pd

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import (
    hierarchical_stochastic_lambda_sampler,
    dixon_coles_hierarchical_sampler,
    dixon_coles_hierarchical_sampler_fast,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv"

N_PER_MATCH = 250

LAMBDA_MODEL = "calibrated"
TEMPO_CV = 0.60
TEAM_CV = 0.10
DIXON_COLES_RHO = 0.30


def row_to_team_dict(row, prefix):
    return {
        "attack": row[f"{prefix}_attack"],
        "defense": row[f"{prefix}_defense"],
        "poisson_attack": row[f"{prefix}_poisson_attack"],
        "poisson_defense": row[f"{prefix}_poisson_defense"],
        "fifa_points": row[f"{prefix}_fifa_points"],
    }


def precompute_lambdas(df):
    lambdas = []

    for _, row in df.iterrows():
        home = row_to_team_dict(row, "home")
        away = row_to_team_dict(row, "away")

        lambdas.append(
            expected_goals(
                home,
                away,
                lambda_model=LAMBDA_MODEL,
            )
        )

    return lambdas


def benchmark_sampler(label, lambdas, sampler_fn):
    total_samples = len(lambdas) * N_PER_MATCH

    start = time.perf_counter()

    for lambda_home, lambda_away in lambdas:
        for _ in range(N_PER_MATCH):
            sampler_fn(lambda_home, lambda_away)

    elapsed = time.perf_counter() - start

    return {
        "sampler": label,
        "total_samples": total_samples,
        "elapsed_seconds": elapsed,
        "samples_per_second": total_samples / elapsed,
        "milliseconds_per_sample": 1000 * elapsed / total_samples,
    }


def main():
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)
    lambdas = precompute_lambdas(df)

    rows = []

    rows.append(
        benchmark_sampler(
            "hierarchical",
            lambdas,
            lambda lh, la: hierarchical_stochastic_lambda_sampler(
                lh,
                la,
                tempo_cv=TEMPO_CV,
                team_cv=TEAM_CV,
            ),
        )
    )

    rows.append(
        benchmark_sampler(
            "dixon_coles_rho=0.30",
            lambdas,
            lambda lh, la: dixon_coles_hierarchical_sampler(
                lh,
                la,
                tempo_cv=TEMPO_CV,
                team_cv=TEAM_CV,
                rho=DIXON_COLES_RHO,
                max_goals=10,
            ),
        )
    )

    rows.append(
        benchmark_sampler(
            "dixon_coles_fast_rho=0.30",
            lambdas,
            lambda lh, la: dixon_coles_hierarchical_sampler_fast(
                lh,
                la,
                tempo_cv=TEMPO_CV,
                team_cv=TEAM_CV,
                rho=DIXON_COLES_RHO,
            ),
        )
    )

    results = pd.DataFrame(rows)

    print("Goal Sampler Runtime Benchmark")
    print("------------------------------")
    print(f"Historical matches: {len(df)}")
    print(f"Samples per match: {N_PER_MATCH}")
    print()
    print(results.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
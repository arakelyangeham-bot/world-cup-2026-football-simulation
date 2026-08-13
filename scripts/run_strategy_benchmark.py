#run_strategy_benchmark.py

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_scoreline_first_production import run_benchmark
from scripts.team_strength_loader import load_team_repository


REPOSITORY_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_011_team_representation_calibration"
    / "repositories"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "study_011_team_representation_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "strategy_scoreline_benchmark_summary.csv"


REPOSITORIES = {
    "legacy": None,
    "dimension_specific": REPOSITORY_DIR / "dimension_specific_team_repository.csv",
    "starter_plus_depth": REPOSITORY_DIR / "starter_plus_depth_team_repository.csv",
    "top_11_mean": REPOSITORY_DIR / "top_11_mean_team_repository.csv",
    "top_5_mean": REPOSITORY_DIR / "top_5_mean_team_repository.csv",
    "star_weighted": REPOSITORY_DIR / "star_weighted_team_repository.csv",
}


def main() -> None:
    rows = []

    for label, path in REPOSITORIES.items():
        print(f"Running scoreline benchmark for: {label}")

        np.random.seed(42)

        repository = load_team_repository(path=path)

        result = run_benchmark(
            team_repository=repository,
            label=label,
        )

        generated = result[result["model"] == label].copy()
        rows.extend(generated.to_dict("records"))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Strategy Scoreline Benchmark Summary")
    print("------------------------------------")
    print(summary.round(6).to_string(index=False))
    print()
    print(f"Wrote -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
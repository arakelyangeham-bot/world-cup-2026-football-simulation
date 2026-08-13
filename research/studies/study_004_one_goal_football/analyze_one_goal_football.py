from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.football_observatory.observatory_schema import (
    match_observation_from_row,
)
from research.football_observatory.football_population import (
    CORE_POPULATIONS,
    analyze_population,
    population_profiles_to_dataframe,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "studies"
    / "study_004_one_goal_football"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_OUTPUT_PATH = OUTPUT_DIR / "one_goal_profile_summary.csv"
COMPARISON_OUTPUT_PATH = OUTPUT_DIR / "one_goal_profile_comparison.csv"


def get_population(name: str):
    for population in CORE_POPULATIONS:
        if population.name == name:
            return population

    raise ValueError(f"Unknown population: {name}")


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    observations = [
        match_observation_from_row(row)
        for _, row in df.iterrows()
    ]

    population = get_population("one_goal_matches")

    analysis = analyze_population(
        observations=observations,
        population=population,
    )

    profile_summary = population_profiles_to_dataframe(analysis)
    profile_comparison = analysis.comparison

    profile_summary.to_csv(PROFILE_OUTPUT_PATH, index=False)
    profile_comparison.to_csv(COMPARISON_OUTPUT_PATH, index=False)

    print("Study 004 — One-Goal Football")
    print("-----------------------------")
    print()
    print("Profile summary")
    print(profile_summary.round(4).to_string(index=False))
    print()
    print("Largest differences")
    print(
        profile_comparison
        .assign(abs_difference=lambda x: x["difference"].abs())
        .sort_values("abs_difference", ascending=False)
        .head(12)
        .round(4)
        .to_string(index=False)
    )
    print()
    print(f"Wrote profile summary    -> {PROFILE_OUTPUT_PATH}")
    print(f"Wrote profile comparison -> {COMPARISON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
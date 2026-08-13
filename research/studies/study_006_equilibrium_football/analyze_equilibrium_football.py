#analyze_equilibrium_football.py

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
    / "study_006_equilibrium_football"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_OUTPUT_PATH = OUTPUT_DIR / "equilibrium_profile_summary.csv"
SCORELESS_COMPARISON_PATH = OUTPUT_DIR / "scoreless_equilibrium_comparison.csv"
SCORING_COMPARISON_PATH = OUTPUT_DIR / "scoring_equilibrium_comparison.csv"


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

    population_names = [
        "draws",
        "scoreless_equilibrium",
        "scoring_equilibrium",
    ]

    analyses = {
        name: analyze_population(
            observations=observations,
            population=get_population(name),
        )
        for name in population_names
    }

    profile_summary = pd.concat(
        [
            population_profiles_to_dataframe(analysis).iloc[[1]]
            for analysis in analyses.values()
        ],
        ignore_index=True,
    )

    baseline_profile = population_profiles_to_dataframe(
        analyses["draws"]
    ).iloc[[0]]

    profile_summary = pd.concat(
        [baseline_profile, profile_summary],
        ignore_index=True,
    )

    profile_summary.to_csv(PROFILE_OUTPUT_PATH, index=False)

    analyses["scoreless_equilibrium"].comparison.to_csv(
        SCORELESS_COMPARISON_PATH,
        index=False,
    )
    analyses["scoring_equilibrium"].comparison.to_csv(
        SCORING_COMPARISON_PATH,
        index=False,
    )

    print("Study 006 — Equilibrium Football")
    print("--------------------------------")
    print()
    print("Profile summary")
    print(profile_summary.round(4).to_string(index=False))
    print()
    print("Scoreless equilibrium largest differences")
    print(
        analyses["scoreless_equilibrium"]
        .comparison
        .assign(abs_difference=lambda x: x["difference"].abs())
        .sort_values("abs_difference", ascending=False)
        .head(10)
        .round(4)
        .to_string(index=False)
    )
    print()
    print("Scoring equilibrium largest differences")
    print(
        analyses["scoring_equilibrium"]
        .comparison
        .assign(abs_difference=lambda x: x["difference"].abs())
        .sort_values("abs_difference", ascending=False)
        .head(10)
        .round(4)
        .to_string(index=False)
    )
    print()
    print(f"Wrote profile summary -> {PROFILE_OUTPUT_PATH}")
    print(f"Wrote scoreless comparison -> {SCORELESS_COMPARISON_PATH}")
    print(f"Wrote scoring comparison -> {SCORING_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
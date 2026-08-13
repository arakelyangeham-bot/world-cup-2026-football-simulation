#compare_equilibrium_populations.py

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

COMPARISON_OUTPUT_PATH = OUTPUT_DIR / "scoreless_vs_scoring_equilibrium.csv"


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

    scoreless = analyze_population(
        observations=observations,
        population=get_population("scoreless_equilibrium"),
    ).population_profile

    scoring = analyze_population(
        observations=observations,
        population=get_population("scoring_equilibrium"),
    ).population_profile

    rows = []

    for metric, scoreless_value in scoreless.__dict__.items():
        if metric in {"label", "matches"}:
            continue

        scoring_value = getattr(scoring, metric)

        rows.append(
            {
                "metric": metric,
                "scoreless_equilibrium": scoreless_value,
                "scoring_equilibrium": scoring_value,
                "difference_scoring_minus_scoreless": (
                    scoring_value - scoreless_value
                ),
            }
        )

    comparison = pd.DataFrame(rows)

    comparison.to_csv(COMPARISON_OUTPUT_PATH, index=False)

    print("Study 006 — Scoreless vs Scoring Equilibrium")
    print("--------------------------------------------")
    print()
    print(f"Scoreless matches: {scoreless.matches}")
    print(f"Scoring equilibrium matches: {scoring.matches}")
    print()
    print(
        comparison
        .assign(
            abs_difference=lambda x: x[
                "difference_scoring_minus_scoreless"
            ].abs()
        )
        .sort_values("abs_difference", ascending=False)
        .head(15)
        .round(4)
        .to_string(index=False)
    )
    print()
    print(f"Wrote comparison -> {COMPARISON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
#analyze_relationship.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.football_observatory.observatory_schema import (
    match_observation_from_row,
)
from research.football_observatory.observables import CORE_OBSERVABLES
from research.football_observatory.relationship import FootballRelationship
from research.football_observatory.relationship_analyzer import analyze_relationship
from research.football_observatory.binning import BinningStrategy


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research" / "football_observatory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "relationship_fifa_points_diff_draw_quantile.csv"


def get_observable(name: str):
    for observable in CORE_OBSERVABLES:
        if observable.name == name:
            return observable

    raise ValueError(f"Unknown observable: {name}")


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    observations = [
        match_observation_from_row(row)
        for _, row in df.iterrows()
    ]

    relationship = FootballRelationship(
        name="fifa_points_diff_to_draw_quantile",
        description="Draw rate by FIFA points difference using quantile bins.",
        independent_variable="fifa_points_diff",
        observable=get_observable("draw"),
        binning=BinningStrategy(
            mode="quantile",
            n_bins=10,
        ),
    )

    result = analyze_relationship(
        observations=observations,
        relationship=relationship,
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print("Football Relationship Analysis")
    print("------------------------------")
    print(relationship.description)
    print()
    print(result.round(4).to_string(index=False))
    print()
    print(f"Wrote relationship analysis -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
#analyze_competitive_balance.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.football_observatory.binning import BinningStrategy
from research.football_observatory.observatory_schema import (
    match_observation_from_row,
)
from research.football_observatory.observables import CORE_OBSERVABLES
from research.football_observatory.relationship import FootballRelationship
from research.football_observatory.relationship_analyzer import analyze_relationship


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
    / "study_002_competitive_balance"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "competitive_balance_response_curves.csv"


def get_observable(name: str):
    for observable in CORE_OBSERVABLES:
        if observable.name == name:
            return observable

    raise ValueError(f"Unknown observable: {name}")


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    # Study 002 defines competitive balance as the absolute FIFA-points gap.
    df["competitive_balance"] = df["fifa_points_diff"].abs()

    observations = [
        match_observation_from_row(row)
        for _, row in df.iterrows()
    ]

    observable_names = [
        "draw",
        "one_goal_match",
        "clean_sheet",
        "both_teams_scored",
        "high_scoring",
        "blowout",
        "zero_zero",
        "one_one",
        "two_one",
        "one_two",
    ]

    frames = []

    for observable_name in observable_names:
        relationship = FootballRelationship(
            name=f"competitive_balance_to_{observable_name}",
            description=(
                f"{observable_name} rate by competitive balance "
                f"(absolute FIFA-points difference)."
            ),
            independent_variable="competitive_balance",
            observable=get_observable(observable_name),
            binning=BinningStrategy(
                mode="quantile",
                n_bins=10,
            ),
        )

        result = analyze_relationship(
            observations=observations,
            relationship=relationship,
        )

        result["observable"] = observable_name
        frames.append(result)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print("Study 002 — Competitive Balance Response Curves")
    print("-----------------------------------------------")
    print(f"Rows written: {len(combined)}")
    print(f"Observables: {len(observable_names)}")
    print()
    print(combined.head(20).round(4).to_string(index=False))
    print()
    print(f"Wrote response curves -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
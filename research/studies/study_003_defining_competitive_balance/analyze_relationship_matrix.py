#analyze_relationship_matrix.py

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
    / "study_003_defining_competitive_balance"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_OUTPUT_PATH = OUTPUT_DIR / "relationship_matrix_detail.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "relationship_matrix_summary.csv"


REPRESENTATIONS = {
    "fifa_points_gap": "fifa_points_gap",
    "attack_gap": "attack_gap",
    "midfield_gap": "midfield_gap",
    "defense_gap": "defense_gap",
    "gk_gap": "gk_gap",
    "poisson_attack_gap": "poisson_attack_gap",
    "poisson_defense_gap": "poisson_defense_gap",
}

OBSERVABLE_NAMES = [
    "draw",
    "one_goal_match",
    "clean_sheet",
    "both_teams_scored",
    "high_scoring",
]


def get_observable(name: str):
    for observable in CORE_OBSERVABLES:
        if observable.name == name:
            return observable

    raise ValueError(f"Unknown observable: {name}")


def add_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["fifa_points_gap"] = df["fifa_points_diff"].abs()
    df["attack_gap"] = df["attack_diff"].abs()
    df["midfield_gap"] = df["midfield_diff"].abs()
    df["defense_gap"] = df["defense_diff"].abs()
    df["gk_gap"] = df["gk_diff"].abs()
    df["poisson_attack_gap"] = df["poisson_attack_diff"].abs()
    df["poisson_defense_gap"] = df["poisson_defense_diff"].abs()

    return df


def summarize_relationship(result: pd.DataFrame) -> dict:
    valid = result.dropna(subset=["observable_rate"]).copy()

    if valid.empty:
        return {
            "bins": 0,
            "total_matches": 0,
            "min_rate": float("nan"),
            "max_rate": float("nan"),
            "rate_range": float("nan"),
            "weighted_mean_rate": float("nan"),
            "mean_ci_width": float("nan"),
        }

    total_matches = valid["matches"].sum()

    weighted_mean_rate = (
        (valid["observable_rate"] * valid["matches"]).sum()
        / total_matches
    )

    return {
        "bins": len(valid),
        "total_matches": total_matches,
        "min_rate": valid["observable_rate"].min(),
        "max_rate": valid["observable_rate"].max(),
        "rate_range": (
            valid["observable_rate"].max()
            - valid["observable_rate"].min()
        ),
        "weighted_mean_rate": weighted_mean_rate,
        "mean_ci_width": (valid["ci_upper"] - valid["ci_lower"]).mean(),
    }


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    df = add_gap_columns(df)

    base_observations = [
        match_observation_from_row(row)
        for _, row in df.iterrows()
    ]

    observations = []

    for observation, (_, row) in zip(base_observations, df.iterrows()):
        observations.append(
            type(observation)(
                prematch=observation.prematch,
                outcome=observation.outcome,
                events=observation.events,
                derived_prematch={
                    "fifa_points_gap": float(row["fifa_points_gap"]),
                    "attack_gap": float(row["attack_gap"]),
                    "midfield_gap": float(row["midfield_gap"]),
                    "defense_gap": float(row["defense_gap"]),
                    "gk_gap": float(row["gk_gap"]),
                    "poisson_attack_gap": float(row["poisson_attack_gap"]),
                    "poisson_defense_gap": float(row["poisson_defense_gap"]),
                },
            )
        )

    detail_frames = []
    summary_rows = []

    for representation_name, variable_name in REPRESENTATIONS.items():
        for observable_name in OBSERVABLE_NAMES:
            relationship = FootballRelationship(
                name=f"{representation_name}_to_{observable_name}",
                description=(
                    f"{observable_name} response curve by "
                    f"{representation_name}."
                ),
                independent_variable=variable_name,
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

            result["representation"] = representation_name
            result["observable"] = observable_name
            detail_frames.append(result)

            summary = summarize_relationship(result)
            summary["representation"] = representation_name
            summary["observable"] = observable_name
            summary["relationship"] = relationship.name
            summary_rows.append(summary)

    detail = pd.concat(detail_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    detail.to_csv(DETAIL_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("Study 003 — Relationship Matrix")
    print("-------------------------------")
    print(f"Representations: {len(REPRESENTATIONS)}")
    print(f"Observables: {len(OBSERVABLE_NAMES)}")
    print(f"Relationships: {len(summary)}")
    print()
    print(
        summary.sort_values(
            ["observable", "rate_range"],
            ascending=[True, False],
        )
        .round(4)
        .to_string(index=False)
    )
    print()
    print(f"Wrote detail  -> {DETAIL_OUTPUT_PATH}")
    print(f"Wrote summary -> {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
#build_population_training_dataset.py

from __future__ import annotations

"""
Builds supervised-learning datasets for Football Intelligence models.

Unlike the production outcome model, these datasets predict football
phenomena rather than match outcomes.

This script intentionally reuses the production feature set and the
Football Observatory to ensure methodological consistency.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.football_observatory.observatory_schema import (
    match_observation_from_row,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "football_intelligence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "football_population_training_dataset.csv"


FEATURE_COLUMNS = [
    "home_attack",
    "home_midfield",
    "home_defense",
    "home_gk",
    "away_attack",
    "away_midfield",
    "away_defense",
    "away_gk",
    "attack_diff",
    "midfield_diff",
    "defense_diff",
    "gk_diff",
    "home_poisson_attack",
    "home_poisson_defense",
    "away_poisson_attack",
    "away_poisson_defense",
    "poisson_attack_diff",
    "poisson_defense_diff",
    "home_rating_prior",
    "away_rating_prior",
    "rating_prior_diff",
]


TARGET_COLUMNS = [
    "is_draw",
    "is_one_goal_match",
    "is_clean_sheet",
    "both_teams_scored",
    "is_high_scoring",
    "is_blowout",
]


def build_targets(row: pd.Series) -> dict[str, int]:
    observation = match_observation_from_row(row)
    outcome = observation.outcome

    return {
        "is_draw": int(outcome.is_draw),
        "is_one_goal_match": int(outcome.is_one_goal_match),
        "is_clean_sheet": int(outcome.is_clean_sheet),
        "both_teams_scored": int(outcome.both_teams_scored),
        "is_high_scoring": int(outcome.is_high_scoring),
        "is_blowout": int(outcome.is_blowout),
    }

def ensure_generic_rating_prior_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add canonical rating-prior columns to a legacy national dataset.

    Existing generic columns are preserved. When both canonical and
    legacy columns exist, their numerical equivalence is validated.
    """

    result = dataframe.copy()

    column_pairs = {
        "home_rating_prior": (
            "home_fifa_points"
        ),
        "away_rating_prior": (
            "away_fifa_points"
        ),
        "rating_prior_diff": (
            "fifa_points_diff"
        ),
    }

    for canonical, legacy in (
        column_pairs.items()
    ):
        has_canonical = (
            canonical in result.columns
        )
        has_legacy = (
            legacy in result.columns
        )

        if not has_canonical and not has_legacy:
            raise ValueError(
                "Dataset is missing both canonical "
                "and legacy rating-prior columns: "
                f"{canonical!r}, {legacy!r}."
            )

        if not has_canonical:
            result[canonical] = result[legacy]

        elif has_legacy:
            canonical_values = pd.to_numeric(
                result[canonical],
                errors="raise",
            )

            legacy_values = pd.to_numeric(
                result[legacy],
                errors="raise",
            )

            unequal = (
                canonical_values
                .sub(legacy_values)
                .abs()
                .gt(1e-12)
            )

            if unequal.any():
                raise ValueError(
                    "Canonical and legacy rating-prior "
                    f"columns disagree: {canonical!r} "
                    f"vs {legacy!r}."
                )

    return result

def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    missing_features = [
        column for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(missing_features)
        )

    features = df[FEATURE_COLUMNS].copy()

    targets = pd.DataFrame(
        [
            build_targets(row)
            for _, row in df.iterrows()
        ]
    )

    output = pd.concat(
        [
            features.reset_index(drop=True),
            targets.reset_index(drop=True),
        ],
        axis=1,
    )

    output.to_csv(OUTPUT_PATH, index=False)

    print("Football Population Training Dataset")
    print("------------------------------------")
    print(f"Rows: {len(output)}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)}")
    print(f"Target columns: {len(TARGET_COLUMNS)}")
    print()
    print("Target rates")
    print(output[TARGET_COLUMNS].mean().round(4).to_string())
    print()
    print(f"Wrote dataset -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
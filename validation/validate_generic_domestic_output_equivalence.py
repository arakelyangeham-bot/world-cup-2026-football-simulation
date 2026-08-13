#validate_generic_domestic_output_equivalence

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LEGACY_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "historical_matches"
    / "validator_comparison_backup"
)

DEFAULT_GENERIC_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "historical_matches"
    / "premier_league"
)

COMPARISON_COLUMNS = [
    "event_id",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "outcome",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Comparison dataset not found: {path}"
        )

    dataframe = pd.read_csv(path)

    missing_columns = (
        set(COMPARISON_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["event_id"] = pd.to_numeric(
        dataframe["event_id"],
        errors="raise",
    ).astype(int)

    dataframe["home_score"] = pd.to_numeric(
        dataframe["home_score"],
        errors="raise",
    ).astype(int)

    dataframe["away_score"] = pd.to_numeric(
        dataframe["away_score"],
        errors="raise",
    ).astype(int)

    return (
        dataframe[COMPARISON_COLUMNS]
        .sort_values("event_id")
        .reset_index(drop=True)
    )


def compare_season(
    year: int,
    legacy_directory: Path,
    generic_directory: Path,
) -> None:
    legacy_path = (
        legacy_directory
        / f"premier_league_{year}_legacy.csv"
    )

    generic_path = (
        generic_directory
        / f"premier_league_{year}_completed_matches.csv"
    )

    legacy = load_dataset(legacy_path)
    generic = load_dataset(generic_path)

    print(f"Season: {year}–{str(year + 1)[-2:]}")
    print(f"Legacy rows: {len(legacy)}")
    print(f"Generic rows: {len(generic)}")

    if len(legacy) != len(generic):
        raise ValueError(
            f"Row-count mismatch for season {year}: "
            f"{len(legacy)} versus {len(generic)}."
        )

    legacy_event_ids = set(
        legacy["event_id"].tolist()
    )

    generic_event_ids = set(
        generic["event_id"].tolist()
    )

    missing_from_generic = sorted(
        legacy_event_ids - generic_event_ids
    )

    extra_in_generic = sorted(
        generic_event_ids - legacy_event_ids
    )

    if missing_from_generic:
        raise ValueError(
            "Generic output is missing event IDs: "
            f"{missing_from_generic[:20]}"
        )

    if extra_in_generic:
        raise ValueError(
            "Generic output contains extra event IDs: "
            f"{extra_in_generic[:20]}"
        )

    comparison = legacy.merge(
        generic,
        on="event_id",
        how="outer",
        suffixes=("_legacy", "_generic"),
        validate="one_to_one",
    )

    mismatch_conditions = []

    for column in [
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "outcome",
    ]:
        mismatch_conditions.append(
            comparison[f"{column}_legacy"]
            != comparison[f"{column}_generic"]
        )

    mismatch_mask = mismatch_conditions[0]

    for condition in mismatch_conditions[1:]:
        mismatch_mask = (
            mismatch_mask | condition
        )

    mismatches = comparison[
        mismatch_mask
    ]

    if not mismatches.empty:
        raise ValueError(
            f"Content mismatches found for season {year}: "
            f"{mismatches.head(20).to_dict('records')}"
        )

    print("Event IDs: identical")
    print("Teams: identical")
    print("Scores: identical")
    print("Outcomes: identical")
    print("Status: PASS")
    print()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy Premier League canonical outputs "
            "against the generic domestic-league validator."
        )
    )

    parser.add_argument(
        "--legacy-directory",
        type=Path,
        default=DEFAULT_LEGACY_DIRECTORY,
    )

    parser.add_argument(
        "--generic-directory",
        type=Path,
        default=DEFAULT_GENERIC_DIRECTORY,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    print("Domestic Validator Output Equivalence")
    print("=====================================")
    print()

    for year in [2023, 2024]:
        compare_season(
            year=year,
            legacy_directory=(
                arguments.legacy_directory
            ),
            generic_directory=(
                arguments.generic_directory
            ),
        )

    print("All equivalence checks passed.")
    print(
        "The generic domestic-league validator reproduces "
        "the known-good Premier League datasets."
    )


if __name__ == "__main__":
    main()
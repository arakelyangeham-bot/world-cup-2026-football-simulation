#build_league_season_repository

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_FINGERPRINT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_042_cross_league_opta_prior_calibration"
    / "outputs"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "datasets"
    / "league_season_repository"
)

FINGERPRINT_PATTERN = "league_fingerprints_*.csv"

REQUIRED_COLUMNS = {
    "competition_key",
    "competition_name",
    "season_start_year",
    "matches",
    "clubs",
    "total_goals",
    "goals_per_match",
    "home_goals_per_match",
    "away_goals_per_match",
    "mean_home_goal_difference",
    "home_wins",
    "draws",
    "away_wins",
    "home_win_rate",
    "draw_rate",
    "away_win_rate",
    "both_teams_to_score_rate",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    "zero_goal_match_rate",
    "one_goal_match_rate",
    "two_goal_match_rate",
    "three_goal_match_rate",
    "four_plus_goal_match_rate",
    "one_goal_margin_rate",
    "three_plus_goal_margin_rate",
    "home_points_per_match",
    "away_points_per_match",
}

IDENTITY_COLUMNS = [
    "study_id",
    "dataset_name",
    "competition_key",
    "competition_name",
    "season_start_year",
    "source_file",
    "repository_created_utc",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical league-season research repository "
            "from Study 042 league fingerprint datasets."
        )
    )

    parser.add_argument(
        "--fingerprint-directory",
        type=Path,
        default=DEFAULT_FINGERPRINT_DIRECTORY,
        help=(
            "Directory containing league_fingerprints_*.csv files."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory for the canonical repository and metadata."
        ),
    )

    return parser.parse_args()


def discover_fingerprint_files(
    fingerprint_directory: Path,
) -> list[Path]:
    if not fingerprint_directory.exists():
        raise FileNotFoundError(
            "Fingerprint directory was not found:\n"
            f"{fingerprint_directory}"
        )

    files = sorted(
        fingerprint_directory.glob(
            FINGERPRINT_PATTERN
        )
    )

    if not files:
        raise FileNotFoundError(
            "No league fingerprint files matched "
            f"{FINGERPRINT_PATTERN!r} in:\n"
            f"{fingerprint_directory}"
        )

    return files


def load_fingerprint_file(
    input_path: Path,
    repository_created_utc: str,
) -> pd.DataFrame:
    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError(
            f"Fingerprint dataset is empty: {input_path}"
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{input_path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["competition_key"] = (
        dataframe["competition_key"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe["competition_name"] = (
        dataframe["competition_name"]
        .astype(str)
        .str.strip()
    )

    dataframe["season_start_year"] = (
        pd.to_numeric(
            dataframe["season_start_year"],
            errors="raise",
        )
        .astype(int)
    )

    if dataframe["competition_key"].eq("").any():
        raise ValueError(
            f"{input_path.name} contains an empty competition key."
        )

    duplicate_keys = dataframe[
        dataframe.duplicated(
            subset=[
                "competition_key",
                "season_start_year",
            ],
            keep=False,
        )
    ]

    if not duplicate_keys.empty:
        preview = (
            duplicate_keys[
                [
                    "competition_key",
                    "season_start_year",
                ]
            ]
            .to_dict("records")
        )

        raise ValueError(
            f"{input_path.name} contains duplicate "
            f"league-season rows: {preview}"
        )

    dataframe.insert(
        0,
        "study_id",
        "study_042",
    )

    dataframe.insert(
        1,
        "dataset_name",
        "league_season_repository",
    )

    dataframe["source_file"] = (
        input_path.name
    )

    dataframe[
        "repository_created_utc"
    ] = repository_created_utc

    ordered_columns = (
        IDENTITY_COLUMNS
        + [
            column
            for column in dataframe.columns
            if column not in IDENTITY_COLUMNS
        ]
    )

    return dataframe[ordered_columns]


def validate_repository(
    repository: pd.DataFrame,
) -> None:
    if repository.empty:
        raise ValueError(
            "League-season repository contains no rows."
        )

    duplicate_keys = repository[
        repository.duplicated(
            subset=[
                "competition_key",
                "season_start_year",
            ],
            keep=False,
        )
    ]

    if not duplicate_keys.empty:
        preview = (
            duplicate_keys[
                [
                    "competition_key",
                    "season_start_year",
                    "source_file",
                ]
            ]
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate league-season observations found "
            f"across source files: {preview}"
        )

    expected_study_ids = set(
        repository["study_id"].unique()
    )

    if expected_study_ids != {
        "study_042"
    }:
        raise ValueError(
            "Unexpected study identifiers found: "
            f"{sorted(expected_study_ids)}"
        )

    if repository[
        "matches"
    ].isna().any():
        raise ValueError(
            "Repository contains missing match counts."
        )

    if (
        repository["matches"] <= 0
    ).any():
        raise ValueError(
            "Repository contains non-positive match counts."
        )

    rate_columns = [
        column
        for column in repository.columns
        if column.endswith("_rate")
    ]

    for column in rate_columns:
        invalid = repository[
            ~repository[column].between(
                0.0,
                1.0,
                inclusive="both",
            )
        ]

        if not invalid.empty:
            raise ValueError(
                f"Repository contains invalid rate values "
                f"in {column!r}."
            )


def build_metadata(
    repository: pd.DataFrame,
    source_files: list[Path],
    repository_path: Path,
    created_utc: str,
) -> dict[str, object]:
    competition_keys = sorted(
        repository[
            "competition_key"
        ].unique().tolist()
    )

    seasons = sorted(
        repository[
            "season_start_year"
        ].astype(int)
        .unique()
        .tolist()
    )

    rows_by_season = (
        repository.groupby(
            "season_start_year"
        )
        .size()
        .astype(int)
        .to_dict()
    )

    rows_by_competition = (
        repository.groupby(
            "competition_key"
        )
        .size()
        .astype(int)
        .to_dict()
    )

    return {
        "study_id": "study_042",
        "dataset_name": (
            "league_season_repository"
        ),
        "description": (
            "Canonical research dataset containing one "
            "league-level fingerprint observation per "
            "competition-season."
        ),
        "created_utc": created_utc,
        "repository_path": str(
            repository_path
        ),
        "row_count": len(repository),
        "competition_count": (
            len(competition_keys)
        ),
        "season_count": len(seasons),
        "competition_keys": (
            competition_keys
        ),
        "season_start_years": seasons,
        "rows_by_season": {
            str(key): value
            for key, value
            in rows_by_season.items()
        },
        "rows_by_competition": (
            rows_by_competition
        ),
        "source_files": [
            str(path)
            for path in source_files
        ],
        "primary_key": [
            "competition_key",
            "season_start_year",
        ],
        "validation_status": "passed",
    }


def main() -> None:
    arguments = parse_arguments()

    created_utc = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    source_files = (
        discover_fingerprint_files(
            arguments.fingerprint_directory
        )
    )

    print(
        "Study 042 — League-Season Repository Builder"
    )
    print(
        "============================================"
    )
    print(
        f"Fingerprint directory: "
        f"{arguments.fingerprint_directory}"
    )
    print(
        f"Files discovered: "
        f"{len(source_files)}"
    )
    print()

    repository_parts: list[
        pd.DataFrame
    ] = []

    for source_file in source_files:
        dataframe = load_fingerprint_file(
            input_path=source_file,
            repository_created_utc=(
                created_utc
            ),
        )

        repository_parts.append(
            dataframe
        )

        print(
            f"Loaded {source_file.name}: "
            f"{len(dataframe)} rows"
        )

    repository = pd.concat(
        repository_parts,
        ignore_index=True,
    )

    repository = (
        repository.sort_values(
            [
                "competition_key",
                "season_start_year",
            ]
        )
        .reset_index(drop=True)
    )

    validate_repository(
        repository
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    repository_path = (
        arguments.output_directory
        / "league_season_repository.csv"
    )

    metadata_path = (
        arguments.output_directory
        / "metadata.json"
    )

    repository.to_csv(
        repository_path,
        index=False,
        encoding="utf-8",
    )

    metadata = build_metadata(
        repository=repository,
        source_files=source_files,
        repository_path=repository_path,
        created_utc=created_utc,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Repository Summary")
    print("------------------")
    print(
        f"Rows written: "
        f"{len(repository)}"
    )
    print(
        "Competitions represented: "
        f"{repository['competition_key'].nunique()}"
    )
    print(
        "Seasons represented: "
        f"{repository['season_start_year'].nunique()}"
    )
    print(
        "Season start years: "
        f"{sorted(repository['season_start_year'].unique())}"
    )
    print()
    print(
        f"Repository: {repository_path}"
    )
    print(
        f"Metadata: {metadata_path}"
    )
    print()
    print("Validation Result")
    print("-----------------")
    print("PASSED")
    print(
        "Canonical league-season research "
        "repository written successfully."
    )


if __name__ == "__main__":
    main()
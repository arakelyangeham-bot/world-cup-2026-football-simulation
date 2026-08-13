#investigate_fixture_events

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared.competition_registry import (
    get_competition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
)

DISPLAY_COLUMNS = [
    "event_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "status_code",
    "status_desc",
    "stage",
    "round",
    "round_number",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect all raw Sofascore events associated with "
            "a domestic-league fixture or event ID."
        )
    )

    parser.add_argument(
        "--competition",
        required=True,
        help=(
            "Registered competition key, such as bundesliga "
            "or ligue_1."
        ),
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help=(
            "Season start year. For example, 2021 means "
            "the 2021–22 season."
        ),
    )

    parser.add_argument(
        "--home",
        default=None,
        help=(
            "Home club name. Use with --away to search a "
            "fixture pairing."
        ),
    )

    parser.add_argument(
        "--away",
        default=None,
        help=(
            "Away club name. Use with --home to search a "
            "fixture pairing."
        ),
    )

    parser.add_argument(
        "--event-id",
        type=int,
        default=None,
        help=(
            "Optional Sofascore event ID to inspect directly."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional raw CSV path. When omitted, the registered "
            "competition filename convention is used."
        ),
    )

    return parser.parse_args()


def normalize_team_name(
    value: object,
) -> str:
    return str(value).strip().casefold()


def load_raw_dataset(
    input_path: Path,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            "Raw Sofascore dataset not found: "
            f"{input_path}"
        )

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError(
            "Raw Sofascore dataset is empty."
        )

    required_columns = {
        "event_id",
        "home_team",
        "away_team",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Raw Sofascore dataset is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["event_id"] = pd.to_numeric(
        dataframe["event_id"],
        errors="raise",
    ).astype(int)

    if "date" in dataframe.columns:
        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="coerce",
            utc=True,
        )

    return dataframe


def find_pairing_events(
    dataframe: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    normalized_home = normalize_team_name(
        home_team
    )

    normalized_away = normalize_team_name(
        away_team
    )

    source_home = (
        dataframe["home_team"]
        .fillna("")
        .map(normalize_team_name)
    )

    source_away = (
        dataframe["away_team"]
        .fillna("")
        .map(normalize_team_name)
    )

    fixture_mask = (
        (
            source_home.eq(normalized_home)
            & source_away.eq(normalized_away)
        )
        |
        (
            source_home.eq(normalized_away)
            & source_away.eq(normalized_home)
        )
    )

    return dataframe.loc[
        fixture_mask
    ].copy()


def find_event_by_id(
    dataframe: pd.DataFrame,
    event_id: int,
) -> pd.DataFrame:
    return dataframe.loc[
        dataframe["event_id"].eq(event_id)
    ].copy()


def select_display_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    available_columns = [
        column
        for column in DISPLAY_COLUMNS
        if column in dataframe.columns
    ]

    selected = dataframe[
        available_columns
    ].copy()

    sort_columns = [
        column
        for column in [
            "date",
            "event_id",
        ]
        if column in selected.columns
    ]

    if sort_columns:
        selected = selected.sort_values(
            sort_columns,
            na_position="last",
        )

    return selected.reset_index(
        drop=True
    )


def print_fixture_history(
    competition_name: str,
    season_start_year: int,
    source_path: Path,
    results: pd.DataFrame,
    home_team: str | None,
    away_team: str | None,
    event_id: int | None,
) -> None:
    print("Fixture Event Investigation")
    print("===========================")
    print(f"Competition: {competition_name}")
    print(
        f"Season: {season_start_year}–"
        f"{str(season_start_year + 1)[-2:]}"
    )
    print(f"Source: {source_path}")

    if event_id is not None:
        print(f"Event ID query: {event_id}")

    if (
        home_team is not None
        and away_team is not None
    ):
        print(
            f"Pairing query: "
            f"{home_team} ↔ {away_team}"
        )

    print()
    print(f"Matching source events: {len(results)}")
    print()

    if results.empty:
        print(
            "No matching raw source events were found."
        )
        return

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print("Status Summary")
    print("--------------")

    if "status_desc" in results.columns:
        status_counts = (
            results["status_desc"]
            .fillna("<missing>")
            .astype(str)
            .value_counts()
        )

        for status, count in (
            status_counts.items()
        ):
            print(f"{status}: {count}")

    else:
        print(
            "No status_desc column was available."
        )


def main() -> None:
    arguments = parse_arguments()

    if arguments.event_id is None:
        if (
            arguments.home is None
            or arguments.away is None
        ):
            raise ValueError(
                "Provide either --event-id, or both "
                "--home and --away."
            )

    competition = get_competition(
        arguments.competition
    )

    if arguments.input is None:
        raw_filename = (
            competition
            .filename_pattern
            .format(year=arguments.year)
        )

        input_path = (
            DEFAULT_RAW_DIRECTORY
            / raw_filename
        )
    else:
        input_path = arguments.input

    raw = load_raw_dataset(
        input_path
    )

    if arguments.event_id is not None:
        matches = find_event_by_id(
            dataframe=raw,
            event_id=arguments.event_id,
        )
    else:
        matches = find_pairing_events(
            dataframe=raw,
            home_team=arguments.home,
            away_team=arguments.away,
        )

    display_results = select_display_columns(
        matches
    )

    print_fixture_history(
        competition_name=(
            competition.display_name
        ),
        season_start_year=arguments.year,
        source_path=input_path,
        results=display_results,
        home_team=arguments.home,
        away_team=arguments.away,
        event_id=arguments.event_id,
    )


if __name__ == "__main__":
    main()
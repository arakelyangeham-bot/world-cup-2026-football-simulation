#inspect_duplicate_pairings

from __future__ import annotations

import argparse
import re
from collections import Counter
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

REQUIRED_COLUMNS = {
    "event_id",
    "date",
    "stage",
    "round",
    "round_number",
    "home_team",
    "home_team_id",
    "away_team",
    "away_team_id",
    "home_score",
    "away_score",
    "status_code",
    "status_desc",
    "winner",
}


def normalize_stage_label(
    value: object,
) -> str:
    """
    Normalize stage labels so formatting differences such as
    'La Liga' and 'LaLiga' compare equally.
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def load_raw_dataset(
    input_path: Path,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {input_path}"
        )

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError(
            "Raw dataset is empty."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Raw dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["event_id"] = pd.to_numeric(
        dataframe["event_id"],
        errors="raise",
    ).astype(int)

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
        utc=True,
    )

    dataframe["home_score"] = pd.to_numeric(
        dataframe["home_score"],
        errors="coerce",
    )

    dataframe["away_score"] = pd.to_numeric(
        dataframe["away_score"],
        errors="coerce",
    )

    return dataframe


def select_completed_regular_season_matches(
    dataframe: pd.DataFrame,
    competition_name: str,
) -> pd.DataFrame:
    completed_mask = (
        dataframe["home_score"].notna()
        & dataframe["away_score"].notna()
    )

    expected_stage = normalize_stage_label(
        competition_name
    )

    observed_stages = (
        dataframe["stage"]
        .fillna("")
        .map(normalize_stage_label)
    )

    regular_stage_mask = (
        observed_stages == expected_stage
    )

    completed_regular = dataframe.loc[
        completed_mask & regular_stage_mask
    ].copy()

    if completed_regular.empty:
        available_stages = sorted(
            dataframe["stage"]
            .fillna("<missing>")
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "No completed regular-season matches "
            f"matched {competition_name!r}. "
            f"Available stages: {available_stages}"
        )

    return completed_regular


def find_pairing_anomalies(
    completed: pd.DataFrame,
) -> tuple[
    Counter[tuple[str, str]],
    Counter[tuple[str, str]],
]:
    unordered_pairs: Counter[
        tuple[str, str]
    ] = Counter()

    directed_pairs: Counter[
        tuple[str, str]
    ] = Counter()

    for row in completed.itertuples(
        index=False
    ):
        home = str(row.home_team)
        away = str(row.away_team)

        unordered_pairs[
            tuple(sorted((home, away)))
        ] += 1

        directed_pairs[
            (home, away)
        ] += 1

    anomalous_unordered = Counter(
        {
            pair: count
            for pair, count
            in unordered_pairs.items()
            if count != 2
        }
    )

    anomalous_directed = Counter(
        {
            pair: count
            for pair, count
            in directed_pairs.items()
            if count != 1
        }
    )

    return (
        anomalous_unordered,
        anomalous_directed,
    )


def print_match_details(
    matches: pd.DataFrame,
) -> None:
    display_columns = [
        "event_id",
        "date",
        "round",
        "round_number",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "status_code",
        "status_desc",
        "stage",
        "winner",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in matches.columns
    ]

    ordered = matches.sort_values(
        ["date", "event_id"],
        na_position="last",
    )

    for row in ordered[
        available_columns
    ].itertuples(
        index=False,
        name=None,
    ):
        record = dict(
            zip(
                available_columns,
                row,
            )
        )

        print()
        print(
            f"Event ID: "
            f"{record.get('event_id')}"
        )
        print(
            f"Date: "
            f"{record.get('date')}"
        )
        print(
            f"Round: "
            f"{record.get('round')}"
        )
        print(
            f"Round number: "
            f"{record.get('round_number')}"
        )
        print(
            f"Fixture: "
            f"{record.get('home_team')} vs "
            f"{record.get('away_team')}"
        )
        print(
            f"Score: "
            f"{record.get('home_score')}–"
            f"{record.get('away_score')}"
        )
        print(
            f"Status code: "
            f"{record.get('status_code')}"
        )
        print(
            f"Status: "
            f"{record.get('status_desc')}"
        )
        print(
            f"Stage: "
            f"{record.get('stage')}"
        )
        print(
            f"Winner: "
            f"{record.get('winner')}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect anomalous club pairings in a raw "
            "domestic-league Sofascore dataset."
        )
    )

    parser.add_argument(
        "--competition",
        required=True,
        help=(
            "Registered domestic competition key, "
            "such as ligue_1."
        ),
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help=(
            "Season start year. "
            "For example, 2023 means 2023–24."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional raw CSV path. When omitted, "
            "the registered filename convention is used."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    competition = get_competition(
        arguments.competition
    )

    if competition.category != "domestic_league":
        raise ValueError(
            f"{arguments.competition!r} is not "
            "registered as a domestic league."
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

    completed = (
        select_completed_regular_season_matches(
            dataframe=raw,
            competition_name=(
                competition.display_name
            ),
        )
    )

    (
        anomalous_unordered,
        anomalous_directed,
    ) = find_pairing_anomalies(
        completed
    )

    print("Domestic League Pairing Diagnostic")
    print("==================================")
    print(
        f"Competition: "
        f"{competition.display_name}"
    )
    print(
        f"Competition key: "
        f"{competition.key}"
    )
    print(
        f"Season start year: "
        f"{arguments.year}"
    )
    print(f"Input: {input_path}")
    print(f"Raw rows: {len(raw)}")
    print(
        "Completed regular-season rows: "
        f"{len(completed)}"
    )
    print()

    if not anomalous_unordered:
        print(
            "No unordered pairing anomalies found."
        )

    else:
        print("Pairing Anomalies")
        print("-----------------")

        for pair, count in sorted(
            anomalous_unordered.items()
        ):
            team_one, team_two = pair

            print()
            print(
                f"{team_one} — {team_two}"
            )
            print(
                f"Total occurrences: {count}"
            )

            forward_count = (
                anomalous_directed.get(
                    (team_one, team_two),
                    0,
                )
            )

            reverse_count = (
                anomalous_directed.get(
                    (team_two, team_one),
                    0,
                )
            )

            print("Directed fixtures:")
            print(
                f"  {team_one} vs {team_two}: "
                f"{forward_count}"
            )
            print(
                f"  {team_two} vs {team_one}: "
                f"{reverse_count}"
            )

            affected_matches = completed[
                (
                    completed["home_team"]
                    .isin(pair)
                )
                & (
                    completed["away_team"]
                    .isin(pair)
                )
            ].copy()

            print("Match details")
            print("-------------")

            print_match_details(
                affected_matches
            )

    print()
    print("Diagnostic Result")
    print("-----------------")

    if anomalous_unordered:
        print(
            f"FOUND {len(anomalous_unordered)} "
            "anomalous club pairing(s)."
        )

    else:
        print("PASSED")
        print(
            "Every club pairing occurred exactly twice."
        )


if __name__ == "__main__":
    main()
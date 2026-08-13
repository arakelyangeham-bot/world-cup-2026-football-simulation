#build_league_fingerprints

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared.competition_registry import (
    get_competition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "historical_matches"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_042_cross_league_opta_prior_calibration"
    / "outputs"
)

DEFAULT_COMPETITIONS = [
    "premier_league",
    "la_liga",
    "serie_a",
    "bundesliga",
    "ligue_1",
]

REQUIRED_COLUMNS = {
    "competition_key",
    "season_start_year",
    "event_id",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "goal_difference",
    "total_goals",
    "outcome",
    "completed",
}


def parse_competition_keys(
    value: str,
) -> list[str]:
    competition_keys = [
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    ]

    if not competition_keys:
        raise argparse.ArgumentTypeError(
            "At least one competition key is required."
        )

    return competition_keys


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build descriptive league fingerprints from "
            "canonical domestic-league historical datasets."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help=(
            "Season start year. "
            "The default is 2023, representing 2023–24."
        ),
    )

    parser.add_argument(
        "--competitions",
        type=parse_competition_keys,
        default=DEFAULT_COMPETITIONS,
        help=(
            "Comma-separated competition keys. "
            "Defaults to the five supported major leagues."
        ),
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Root directory containing canonical "
            "historical-match datasets."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for Study 042 output files.",
    )

    return parser.parse_args()


def build_input_path(
    input_root: Path,
    competition_key: str,
    season_start_year: int,
) -> Path:
    filename = (
        f"{competition_key}_"
        f"{season_start_year}_"
        "completed_matches.csv"
    )

    return (
        input_root
        / competition_key
        / filename
    )


def load_canonical_dataset(
    input_path: Path,
    competition_key: str,
    season_start_year: int,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            "Canonical historical-match dataset "
            f"was not found:\n{input_path}"
        )

    dataframe = pd.read_csv(
        input_path
    )

    if dataframe.empty:
        raise ValueError(
            f"Canonical dataset is empty: {input_path}"
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{input_path.name} is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["home_score"] = pd.to_numeric(
        dataframe["home_score"],
        errors="raise",
    ).astype(int)

    dataframe["away_score"] = pd.to_numeric(
        dataframe["away_score"],
        errors="raise",
    ).astype(int)

    dataframe["total_goals"] = pd.to_numeric(
        dataframe["total_goals"],
        errors="raise",
    ).astype(int)

    dataframe["goal_difference"] = pd.to_numeric(
        dataframe["goal_difference"],
        errors="raise",
    ).astype(int)

    observed_competitions = set(
        dataframe["competition_key"]
        .astype(str)
        .unique()
    )

    if observed_competitions != {
        competition_key
    }:
        raise ValueError(
            f"{input_path.name} contains unexpected "
            "competition keys: "
            f"{sorted(observed_competitions)}"
        )

    observed_years = set(
        pd.to_numeric(
            dataframe["season_start_year"],
            errors="raise",
        )
        .astype(int)
        .unique()
    )

    if observed_years != {
        season_start_year
    }:
        raise ValueError(
            f"{input_path.name} contains unexpected "
            f"season years: {sorted(observed_years)}"
        )

    if dataframe["event_id"].duplicated().any():
        duplicate_ids = (
            dataframe.loc[
                dataframe["event_id"].duplicated(
                    keep=False
                ),
                "event_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Duplicate event IDs found in "
            f"{input_path.name}: {duplicate_ids[:20]}"
        )

    return dataframe


def calculate_points(
    dataframe: pd.DataFrame,
) -> tuple[int, int]:
    home_points = (
        int(
            (
                dataframe["home_score"]
                > dataframe["away_score"]
            ).sum()
        )
        * 3
        + int(
            (
                dataframe["home_score"]
                == dataframe["away_score"]
            ).sum()
        )
    )

    away_points = (
        int(
            (
                dataframe["away_score"]
                > dataframe["home_score"]
            ).sum()
        )
        * 3
        + int(
            (
                dataframe["home_score"]
                == dataframe["away_score"]
            ).sum()
        )
    )

    return home_points, away_points


def build_league_fingerprint(
    dataframe: pd.DataFrame,
    competition_key: str,
    season_start_year: int,
) -> dict[str, object]:
    competition = get_competition(
        competition_key
    )

    match_count = len(dataframe)

    clubs = (
        set(dataframe["home_team"])
        | set(dataframe["away_team"])
    )

    total_home_goals = int(
        dataframe["home_score"].sum()
    )

    total_away_goals = int(
        dataframe["away_score"].sum()
    )

    total_goals = (
        total_home_goals
        + total_away_goals
    )

    home_wins = int(
        (
            dataframe["home_score"]
            > dataframe["away_score"]
        ).sum()
    )

    draws = int(
        (
            dataframe["home_score"]
            == dataframe["away_score"]
        ).sum()
    )

    away_wins = int(
        (
            dataframe["home_score"]
            < dataframe["away_score"]
        ).sum()
    )

    both_teams_scored = int(
        (
            (dataframe["home_score"] > 0)
            & (dataframe["away_score"] > 0)
        ).sum()
    )

    home_clean_sheets = int(
        (
            dataframe["away_score"] == 0
        ).sum()
    )

    away_clean_sheets = int(
        (
            dataframe["home_score"] == 0
        ).sum()
    )

    scoreless_matches = int(
        (
            dataframe["total_goals"] == 0
        ).sum()
    )

    one_goal_matches = int(
        (
            dataframe["total_goals"] == 1
        ).sum()
    )

    two_goal_matches = int(
        (
            dataframe["total_goals"] == 2
        ).sum()
    )

    three_goal_matches = int(
        (
            dataframe["total_goals"] == 3
        ).sum()
    )

    four_plus_goal_matches = int(
        (
            dataframe["total_goals"] >= 4
        ).sum()
    )

    one_goal_margin_matches = int(
        (
            dataframe["goal_difference"]
            .abs()
            == 1
        ).sum()
    )

    three_plus_goal_margin_matches = int(
        (
            dataframe["goal_difference"]
            .abs()
            >= 3
        ).sum()
    )

    home_points, away_points = (
        calculate_points(
            dataframe
        )
    )

    return {
        "competition_key": competition_key,
        "competition_name": (
            competition.display_name
        ),
        "season_start_year": (
            season_start_year
        ),
        "matches": match_count,
        "clubs": len(clubs),
        "total_goals": total_goals,
        "goals_per_match": (
            total_goals / match_count
        ),
        "home_goals_per_match": (
            total_home_goals / match_count
        ),
        "away_goals_per_match": (
            total_away_goals / match_count
        ),
        "mean_home_goal_difference": (
            dataframe[
                "goal_difference"
            ].mean()
        ),
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "home_win_rate": (
            home_wins / match_count
        ),
        "draw_rate": (
            draws / match_count
        ),
        "away_win_rate": (
            away_wins / match_count
        ),
        "both_teams_to_score_rate": (
            both_teams_scored
            / match_count
        ),
        "home_clean_sheet_rate": (
            home_clean_sheets
            / match_count
        ),
        "away_clean_sheet_rate": (
            away_clean_sheets
            / match_count
        ),
        "zero_goal_match_rate": (
            scoreless_matches
            / match_count
        ),
        "one_goal_match_rate": (
            one_goal_matches
            / match_count
        ),
        "two_goal_match_rate": (
            two_goal_matches
            / match_count
        ),
        "three_goal_match_rate": (
            three_goal_matches
            / match_count
        ),
        "four_plus_goal_match_rate": (
            four_plus_goal_matches
            / match_count
        ),
        "one_goal_margin_rate": (
            one_goal_margin_matches
            / match_count
        ),
        "three_plus_goal_margin_rate": (
            three_plus_goal_margin_matches
            / match_count
        ),
        "home_points_per_match": (
            home_points / match_count
        ),
        "away_points_per_match": (
            away_points / match_count
        ),
    }


def format_console_table(
    fingerprints: pd.DataFrame,
) -> pd.DataFrame:
    display = fingerprints[
        [
            "competition_name",
            "matches",
            "clubs",
            "goals_per_match",
            "home_win_rate",
            "draw_rate",
            "away_win_rate",
            "both_teams_to_score_rate",
            "mean_home_goal_difference",
            "three_plus_goal_margin_rate",
        ]
    ].copy()

    display = display.rename(
        columns={
            "competition_name": "League",
            "matches": "Matches",
            "clubs": "Clubs",
            "goals_per_match": "Goals/Match",
            "home_win_rate": "Home Win %",
            "draw_rate": "Draw %",
            "away_win_rate": "Away Win %",
            "both_teams_to_score_rate": "BTTS %",
            "mean_home_goal_difference": "Mean Home GD",
            "three_plus_goal_margin_rate": "3+ GD %",
        }
    )

    percentage_columns = [
        "Home Win %",
        "Draw %",
        "Away Win %",
        "BTTS %",
        "3+ GD %",
    ]

    for column in percentage_columns:
        display[column] = (
            display[column]
            * 100
        ).round(2)

    display["Goals/Match"] = (
        display["Goals/Match"]
        .round(3)
    )

    display["Mean Home GD"] = (
        display["Mean Home GD"]
        .round(3)
    )

    return display


def main() -> None:
    arguments = parse_arguments()

    fingerprints: list[
        dict[str, object]
    ] = []

    print("Study 042 — League Fingerprints")
    print("===============================")
    print(
        f"Season start year: "
        f"{arguments.year}"
    )
    print(
        "Competitions: "
        f"{arguments.competitions}"
    )
    print()

    for competition_key in (
        arguments.competitions
    ):
        competition = get_competition(
            competition_key
        )

        if (
            competition.category
            != "domestic_league"
        ):
            raise ValueError(
                f"{competition_key!r} is not "
                "registered as a domestic league."
            )

        input_path = build_input_path(
            input_root=arguments.input_root,
            competition_key=competition_key,
            season_start_year=arguments.year,
        )

        dataframe = load_canonical_dataset(
            input_path=input_path,
            competition_key=competition_key,
            season_start_year=arguments.year,
        )

        fingerprint = build_league_fingerprint(
            dataframe=dataframe,
            competition_key=competition_key,
            season_start_year=arguments.year,
        )

        fingerprints.append(
            fingerprint
        )

        print(
            f"Loaded {competition.display_name}: "
            f"{len(dataframe)} matches"
        )

    fingerprint_dataframe = pd.DataFrame(
        fingerprints
    )

    fingerprint_dataframe = (
        fingerprint_dataframe
        .sort_values(
            "competition_name"
        )
        .reset_index(drop=True)
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        arguments.output_directory
        / (
            f"league_fingerprints_"
            f"{arguments.year}.csv"
        )
    )

    fingerprint_dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    display_table = format_console_table(
        fingerprint_dataframe
    )

    print()
    print("League Fingerprint Summary")
    print("--------------------------")
    print(
        display_table.to_string(
            index=False
        )
    )

    print()
    print(f"Output: {output_path}")
    print()
    print("Study Result")
    print("------------")
    print("PASSED")
    print(
        "League fingerprint dataset "
        "written successfully."
    )


if __name__ == "__main__":
    main()
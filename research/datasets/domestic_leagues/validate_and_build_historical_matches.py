from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from shared.competition_registry import (
    get_competition,
)
from shared.domestic_league_rules import (
    DomesticLeagueRules,
    get_domestic_league_rules,
)

from research.datasets.domestic_leagues.fixture_integrity import (
    analyze_fixture_integrity,
    print_fixture_integrity_report,
    print_missing_fixture_source_events
)

from research.datasets.domestic_leagues.known_source_anomalies import (
    KnownSourceAnomaly,
    get_known_source_anomaly,
    validate_known_source_anomaly,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "historical_matches"
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

FINAL_STATUS_CODES = {
    100,
}

FINAL_STATUS_DESCRIPTIONS = {
    "ended",
}

def load_raw_dataset(
    input_path: Path,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            "Raw domestic-league dataset not found: "
            f"{input_path}"
        )

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError(
            "Raw domestic-league dataset is empty."
        )

    missing_columns = (
        REQUIRED_COLUMNS - set(dataframe.columns)
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


def classify_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed_mask = (
        dataframe["home_score"].notna()
        & dataframe["away_score"].notna()
    )

    completed = dataframe[
        completed_mask
    ].copy()

    incomplete = dataframe[
        ~completed_mask
    ].copy()

    return completed, incomplete

def normalize_status_description(
    value: object,
) -> str:
    return str(value).strip().lower()


def filter_officially_completed_matches(
    score_complete: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate officially completed matches from events that have
    scores but do not have a final match status.

    Examples of excluded non-final events include abandoned,
    cancelled, postponed, suspended, or interrupted fixtures.
    """

    numeric_status_codes = pd.to_numeric(
        score_complete["status_code"],
        errors="coerce",
    )

    normalized_descriptions = (
        score_complete["status_desc"]
        .fillna("")
        .map(normalize_status_description)
    )

    final_status_mask = (
        numeric_status_codes.isin(
            FINAL_STATUS_CODES
        )
        | normalized_descriptions.isin(
            FINAL_STATUS_DESCRIPTIONS
        )
    )

    officially_completed = score_complete.loc[
        final_status_mask
    ].copy()

    excluded_non_final = score_complete.loc[
        ~final_status_mask
    ].copy()

    return (
        officially_completed,
        excluded_non_final,
    )

def normalize_stage_label(
    value: object,
) -> str:
    """
    Normalize stage labels for comparison.

    Examples:
        "La Liga" -> "laliga"
        "LaLiga" -> "laliga"
        "Bundesliga, Relegation/Promotion Playoffs"
            -> "bundesligarelegationpromotionplayoffs"
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def filter_regular_season_matches(
    completed: pd.DataFrame,
    competition_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate completed regular-season league matches from other
    completed events associated with the same Sofascore season.
    """

    expected_stage = normalize_stage_label(
        competition_name
    )

    observed_stages = (
        completed["stage"]
        .fillna("")
        .map(normalize_stage_label)
    )

    regular_mask = (
        observed_stages == expected_stage
    )

    regular = completed.loc[
        regular_mask
    ].copy()

    excluded = completed.loc[
        ~regular_mask
    ].copy()

    if regular.empty:
        available_stages = sorted(
            completed["stage"]
            .fillna("<missing>")
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "No completed events matched the registered "
            f"competition stage {competition_name!r}. "
            f"Available source stages: {available_stages}"
        )

    return regular, excluded

def validate_unique_event_ids(
    dataframe: pd.DataFrame,
) -> None:
    duplicates = dataframe[
        dataframe["event_id"].duplicated(
            keep=False
        )
    ]

    if not duplicates.empty:
        duplicate_ids = sorted(
            duplicates["event_id"]
            .astype(int)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate event IDs found: "
            f"{duplicate_ids[:20]}"
        )


def validate_required_completed_values(
    completed: pd.DataFrame,
) -> None:
    required_columns = [
        "event_id",
        "date",
        "home_team",
        "home_team_id",
        "away_team",
        "away_team_id",
        "home_score",
        "away_score",
    ]

    for column in required_columns:
        missing_count = int(
            completed[column].isna().sum()
        )

        if missing_count:
            raise ValueError(
                f"Completed rows contain "
                f"{missing_count} missing value(s) "
                f"in {column!r}."
            )

    same_team_rows = completed[
        completed["home_team"]
        .astype(str)
        .eq(
            completed["away_team"]
            .astype(str)
        )
    ]

    if not same_team_rows.empty:
        raise ValueError(
            "One or more completed matches contain "
            "the same home and away club."
        )


def validate_scores(
    completed: pd.DataFrame,
) -> None:
    home_scores = completed["home_score"]
    away_scores = completed["away_score"]

    if (home_scores < 0).any():
        raise ValueError(
            "Negative home scores found."
        )

    if (away_scores < 0).any():
        raise ValueError(
            "Negative away scores found."
        )

    if (home_scores % 1 != 0).any():
        raise ValueError(
            "Non-integer home scores found."
        )

    if (away_scores % 1 != 0).any():
        raise ValueError(
            "Non-integer away scores found."
        )


def validate_duplicate_completed_fixtures(
    completed: pd.DataFrame,
) -> None:
    duplicates = completed[
        completed.duplicated(
            subset=[
                "date",
                "home_team_id",
                "away_team_id",
            ],
            keep=False,
        )
    ]

    if not duplicates.empty:
        preview = (
            duplicates[
                [
                    "event_id",
                    "date",
                    "home_team",
                    "away_team",
                ]
            ]
            .head(20)
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate completed fixtures found: "
            f"{preview}"
        )


def build_team_match_summary(
    completed: pd.DataFrame,
) -> pd.DataFrame:
    home_counts = (
        completed["home_team"]
        .value_counts()
        .rename("home_matches")
    )

    away_counts = (
        completed["away_team"]
        .value_counts()
        .rename("away_matches")
    )

    teams = sorted(
        set(completed["home_team"])
        | set(completed["away_team"])
    )

    summary = pd.DataFrame(
        {"team": teams}
    )

    summary = summary.merge(
        home_counts,
        left_on="team",
        right_index=True,
        how="left",
    )

    summary = summary.merge(
        away_counts,
        left_on="team",
        right_index=True,
        how="left",
    )

    summary[
        ["home_matches", "away_matches"]
    ] = (
        summary[
            ["home_matches", "away_matches"]
        ]
        .fillna(0)
        .astype(int)
    )

    summary["total_matches"] = (
        summary["home_matches"]
        + summary["away_matches"]
    )

    return summary.sort_values(
        "team"
    ).reset_index(drop=True)


def validate_league_structure(
    completed: pd.DataFrame,
    team_summary: pd.DataFrame,
    rules: DomesticLeagueRules,
) -> None:
    if (
        len(completed)
        != rules.completed_match_count
    ):
        raise ValueError(
            f"Expected {rules.completed_match_count} "
            "completed matches, but found "
            f"{len(completed)}."
        )

    if len(team_summary) != rules.team_count:
        raise ValueError(
            f"Expected {rules.team_count} clubs, "
            f"but found {len(team_summary)}."
        )

    invalid_totals = team_summary[
        team_summary["total_matches"]
        != rules.matches_per_team
    ]

    if not invalid_totals.empty:
        raise ValueError(
            "One or more clubs have an invalid "
            "total match count: "
            f"{invalid_totals.to_dict('records')}"
        )

    invalid_home = team_summary[
        team_summary["home_matches"]
        != rules.home_matches_per_team
    ]

    if not invalid_home.empty:
        raise ValueError(
            "One or more clubs have an invalid "
            "home match count: "
            f"{invalid_home.to_dict('records')}"
        )

    invalid_away = team_summary[
        team_summary["away_matches"]
        != rules.away_matches_per_team
    ]

    if not invalid_away.empty:
        raise ValueError(
            "One or more clubs have an invalid "
            "away match count: "
            f"{invalid_away.to_dict('records')}"
        )


def validate_pairing_structure(
    completed: pd.DataFrame,
    rules: DomesticLeagueRules,
) -> None:
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

    if (
        len(unordered_pairs)
        != rules.unique_pairing_count
    ):
        raise ValueError(
            f"Expected "
            f"{rules.unique_pairing_count} unique "
            "club pairings, but found "
            f"{len(unordered_pairs)}."
        )

    invalid_pairs = {
        pair: count
        for pair, count in unordered_pairs.items()
        if count != 2
    }

    if invalid_pairs:
        raise ValueError(
            "Each club pairing should occur twice. "
            f"Invalid pairings: "
            f"{list(invalid_pairs.items())[:20]}"
        )

    invalid_orientations = {
        pair: count
        for pair, count in directed_pairs.items()
        if count != 1
    }

    if invalid_orientations:
        raise ValueError(
            "Each directed home-away pairing should "
            "occur exactly once. Invalid entries: "
            f"{list(invalid_orientations.items())[:20]}"
        )


def build_canonical_dataset(
    completed: pd.DataFrame,
    competition_key: str,
    season_start_year: int,
) -> pd.DataFrame:
    canonical = completed.copy()

    canonical["home_score"] = (
        canonical["home_score"].astype(int)
    )

    canonical["away_score"] = (
        canonical["away_score"].astype(int)
    )

    canonical["goal_difference"] = (
        canonical["home_score"]
        - canonical["away_score"]
    )

    canonical["total_goals"] = (
        canonical["home_score"]
        + canonical["away_score"]
    )

    canonical["outcome"] = "draw"

    canonical.loc[
        canonical["home_score"]
        > canonical["away_score"],
        "outcome",
    ] = "home_win"

    canonical.loc[
        canonical["home_score"]
        < canonical["away_score"],
        "outcome",
    ] = "away_win"

    canonical["completed"] = True
    canonical["competition_key"] = (
        competition_key
    )
    canonical["season_start_year"] = (
        season_start_year
    )

    canonical = canonical.sort_values(
        ["date", "event_id"]
    ).reset_index(drop=True)

    output_columns = [
        "competition_key",
        "season_start_year",
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
        "goal_difference",
        "total_goals",
        "outcome",
        "status_code",
        "status_desc",
        "winner",
        "completed",
    ]

    return canonical[output_columns]


def write_outputs(
    canonical_completed: pd.DataFrame,
    incomplete: pd.DataFrame,
    excluded_completed: pd.DataFrame,
    team_summary: pd.DataFrame,
    competition_key: str,
    competition_name: str,
    season_start_year: int,
    input_path: Path,
    output_root: Path,
    excluded_non_final_count: int,
    excluded_non_regular_stage_count: int,
    accepted_anomaly: KnownSourceAnomaly | None,
) -> dict[str, Path]:
    output_directory = (
        output_root / competition_key
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename_prefix = (
        f"{competition_key}_"
        f"{season_start_year}"
    )

    completed_path = (
        output_directory
        / f"{filename_prefix}_completed_matches.csv"
    )

    incomplete_path = (
        output_directory
        / f"{filename_prefix}_incomplete_events.csv"
    )

    excluded_completed_path = (
        output_directory
        / (
            f"{filename_prefix}_"
            "excluded_completed_events.csv"
        )
    )

    team_summary_path = (
        output_directory
        / f"{filename_prefix}_team_match_summary.csv"
    )

    metadata_path = (
        output_directory
        / f"{filename_prefix}_dataset_metadata.json"
    )

    canonical_completed.to_csv(
        completed_path,
        index=False,
        encoding="utf-8",
    )

    incomplete.to_csv(
        incomplete_path,
        index=False,
        encoding="utf-8",
    )

    excluded_completed.to_csv(
        excluded_completed_path,
        index=False,
        encoding="utf-8",
    )

    team_summary.to_csv(
        team_summary_path,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "competition_key": competition_key,
        "competition_name": competition_name,
        "season_start_year": season_start_year,
        "input_path": str(input_path),
        "raw_row_count": (
            len(canonical_completed)
            + len(incomplete)
            + len(excluded_completed)
        ),
        "completed_match_count": (
            len(canonical_completed)
        ),
        "incomplete_event_count": (
            len(incomplete)
        ),
        "excluded_completed_event_count": (
            len(excluded_completed)
        ),
        "excluded_non_final_event_count":
            excluded_non_final_count,
        "excluded_non_regular_stage_event_count":
            excluded_non_regular_stage_count,
        "club_count": len(team_summary),
        "canonical_output": str(
            completed_path
        ),
        "validation_status": (
            "passed"
            if accepted_anomaly is None
            else "accepted_with_known_source_anomaly"
        ),
        "known_source_anomaly": (
            None
            if accepted_anomaly is None
            else {
                "missing_fixtures": [
                    {
                        "home_team": home_team,
                        "away_team": away_team,
                    }
                    for home_team, away_team
                    in accepted_anomaly.missing_fixtures
                ],
                "source_event_ids": list(
                    accepted_anomaly.source_event_ids
                ),
                "description": (
                    accepted_anomaly.description
                ),
            }
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "completed": completed_path,
        "incomplete": incomplete_path,
        "excluded_completed": (
            excluded_completed_path
        ),
        "team_summary": team_summary_path,
        "metadata": metadata_path,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and canonicalize a completed "
            "domestic-league Sofascore dataset."
        )
    )

    parser.add_argument(
        "--competition",
        required=True,
        help=(
            "Registered domestic competition key, "
            "such as premier_league or la_liga."
        ),
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help=(
            "Season start year. "
            "For example, 2024 means 2024–25."
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

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory for canonical "
            "historical-match datasets."
        ),
    )

    return parser.parse_args()

def resolve_fixture_integrity(
    competition_key: str,
    season_start_year: int,
    fixture_report,
) -> KnownSourceAnomaly | None:
    """
    Require a perfect fixture report unless the league-season has
    an explicitly registered and exactly matching source anomaly.
    """

    if fixture_report.passed:
        return None

    anomaly = get_known_source_anomaly(
        competition_key=competition_key,
        season_start_year=season_start_year,
    )

    if anomaly is None:
        raise ValueError(
            "Fixture integrity failed and no known source anomaly "
            "is registered for "
            f"{competition_key!r}, season "
            f"{season_start_year}."
        )

    validate_known_source_anomaly(
        anomaly=anomaly,
        missing_fixtures=(
            fixture_report.missing_fixtures
        ),
        unexpected_fixtures=(
            fixture_report.unexpected_fixtures
        ),
        duplicate_fixtures=(
            fixture_report.duplicate_fixtures
        ),
    )

    expected_observed_count = (
        fixture_report.expected_fixture_count
        - anomaly.expected_missing_match_count
    )

    if (
        fixture_report.observed_match_count
        != expected_observed_count
    ):
        raise ValueError(
            "The observed match count does not agree with the "
            "registered anomaly. Expected "
            f"{expected_observed_count} available matches after "
            "accounting for the documented missing fixture(s), "
            f"but found {fixture_report.observed_match_count}."
        )

    return anomaly

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

    rules = get_domestic_league_rules(
        competition_key=arguments.competition,
        season_start_year=arguments.year,
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

    validate_unique_event_ids(raw)

    score_complete, incomplete = classify_rows(
        raw
    )

    (
        officially_completed,
        excluded_non_final,
    ) = filter_officially_completed_matches(
        score_complete=score_complete
    )

    excluded_non_final = (
        excluded_non_final.copy()
    )

    excluded_non_final[
        "exclusion_reason"
    ] = "non_final_status"

    (
        completed,
        excluded_non_regular_stage,
    ) = filter_regular_season_matches(
        completed=officially_completed,
        competition_name=(
            competition.display_name
        ),
    )

    excluded_non_regular_stage = (
        excluded_non_regular_stage.copy()
    )

    excluded_non_regular_stage[
        "exclusion_reason"
    ] = "non_regular_season_stage"

    excluded_completed = pd.concat(
        [
            excluded_non_final,
            excluded_non_regular_stage,
        ],
        ignore_index=True,
    )

    excluded_completed = (
        excluded_completed
        .sort_values(
            ["date", "event_id"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


    validate_required_completed_values(
        completed
    )

    validate_scores(
        completed
    )

    validate_duplicate_completed_fixtures(
        completed
    )

    team_summary = build_team_match_summary(
        completed
    )

    print()
    print("Pre-validation diagnostics")
    print("--------------------------")
    print(f"Score-complete rows: {len(score_complete)}")
    print(f"Officially completed rows: {len(officially_completed)}")
    print(f"Regular-season completed rows: {len(completed)}")
    print(f"Incomplete rows: {len(incomplete)}")
    print(f"Excluded non-final rows: {len(excluded_non_final)}")
    print(
        "Excluded non-regular-stage rows: "
        f"{len(excluded_non_regular_stage)}"
    )

    if not incomplete.empty:
        print()
        print("Incomplete events:")
        print(
            incomplete[
                [
                    "event_id",
                    "date",
                    "home_team",
                    "away_team",
                    "status_code",
                    "status_desc",
                    "stage",
                ]
            ].to_string(index=False)
        )

    if not excluded_non_final.empty:
        print()
        print("Excluded non-final events:")
        print(
            excluded_non_final[
                [
                    "event_id",
                    "date",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                    "status_code",
                    "status_desc",
                    "stage",
                ]
            ].to_string(index=False)
        )

    if not excluded_non_regular_stage.empty:
        print()
        print("Excluded non-regular-stage events:")
        print(
            excluded_non_regular_stage[
                [
                    "event_id",
                    "date",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                    "status_code",
                    "status_desc",
                    "stage",
                ]
            ].to_string(index=False)
        )

    fixture_report = analyze_fixture_integrity(
        completed_matches=completed,
    )

    print()
    print_fixture_integrity_report(
        fixture_report
    )
    print()

    accepted_anomaly = resolve_fixture_integrity(
        competition_key=arguments.competition,
        season_start_year=arguments.year,
        fixture_report=fixture_report,
    )

    print_missing_fixture_source_events(
        raw=raw,
        missing_fixtures=(
            fixture_report.missing_fixtures
        ),
    )

    if accepted_anomaly is None:
        validate_league_structure(
            completed=completed,
            team_summary=team_summary,
            rules=rules,
        )

        validate_pairing_structure(
            completed=completed,
            rules=rules,
        )

    else:
        if len(team_summary) != rules.team_count:
            raise ValueError(
                f"Expected {rules.team_count} clubs, "
                f"but found {len(team_summary)}."
            )

        print("Known Source Anomaly")
        print("--------------------")
        print(
            f"Competition: "
            f"{accepted_anomaly.competition_key}"
        )
        print(
            f"Season start year: "
            f"{accepted_anomaly.season_start_year}"
        )
        print(
            "Missing fixtures:"
        )

        for home_team, away_team in (
            accepted_anomaly.missing_fixtures
        ):
            print(
                f"  {home_team} vs {away_team}"
            )

        print(
            "Source event IDs: "
            f"{list(accepted_anomaly.source_event_ids)}"
        )
        print(
            f"Explanation: "
            f"{accepted_anomaly.description}"
        )
        print()

    canonical_completed = (
        build_canonical_dataset(
            completed=completed,
            competition_key=(
                arguments.competition
            ),
            season_start_year=(
                arguments.year
            ),
        )
    )

    output_paths = write_outputs(
        canonical_completed=(
            canonical_completed
        ),
        incomplete=incomplete,
        excluded_completed=(
            excluded_completed
        ),
        team_summary=team_summary,
        competition_key=(
            arguments.competition
        ),
        competition_name=(
            competition.display_name
        ),
        season_start_year=(
            arguments.year
        ),
        input_path=input_path,
        output_root=arguments.output_root,
        excluded_non_final_count=len(
            excluded_non_final
        ),
        excluded_non_regular_stage_count=len(
            excluded_non_regular_stage
        ),
        accepted_anomaly=accepted_anomaly,
    )

    print(
        "Domestic League Historical "
        "Dataset Validation"
    )
    print(
        "====================================="
    )
    print(
        f"Competition: "
        f"{competition.display_name}"
    )
    print(
        f"Season start year: "
        f"{arguments.year}"
    )
    print(f"Input: {input_path}")
    print(f"Raw rows: {len(raw)}")
    print(
        f"Completed matches: "
        f"{len(canonical_completed)}"
    )
    print(
        "Excluded completed events: "
        f"{len(excluded_completed)}"
    )
    print(
        f"Incomplete events: "
        f"{len(incomplete)}"
    )
    print(
        f"Clubs: {len(team_summary)}"
    )

    print()

    print("League Structure")
    print("----------------")
    print(
        "Expected completed matches: "
        f"{rules.completed_match_count}"
    )
    print(
        "Observed completed matches: "
        f"{len(canonical_completed)}"
    )
    print(
        f"Expected clubs: "
        f"{rules.team_count}"
    )
    print(
        f"Observed clubs: "
        f"{len(team_summary)}"
    )
    print(
        "Matches per club: "
        f"{rules.matches_per_team}"
    )
    print(
        "Home-away split: "
        f"{rules.home_matches_per_team}/"
        f"{rules.away_matches_per_team}"
    )
    if accepted_anomaly is None:
        print(
            "Every club pairing occurred twice "
            "with opposite orientations."
        )
    else:
        print(
            "Fixture structure matches the documented "
            "known source anomaly."
        )
    print()
    
    if not excluded_completed.empty:
        print("Excluded Completed Events")
        print("-------------------------")

        for row in excluded_completed.itertuples(
            index=False
        ):
            print(
                f"Event {row.event_id}: "
                f"{row.home_team} vs "
                f"{row.away_team} — "
                f"{row.status_desc} — "
                f"{row.stage} — "
                f"{row.exclusion_reason}"
            )

        print()
    
    if not incomplete.empty:
        print("Incomplete Source Events")
        print("------------------------")

        for row in incomplete.itertuples(
            index=False
        ):
            print(
                f"Event {row.event_id}: "
                f"{row.home_team} vs "
                f"{row.away_team} — "
                f"{row.status_desc}"
            )

        print()

    print("Outputs")
    print("-------")

    for label, path in output_paths.items():
        print(f"{label}: {path}")

    print()
    print("Validation Result")
    print("-----------------")
    if accepted_anomaly is None:
        print("PASSED")
        print(
            "Canonical domestic-league dataset "
            "accepted for downstream research."
        )
    else:
        print(
            "ACCEPTED WITH KNOWN SOURCE ANOMALY"
        )
        print(
            "The canonical dataset is incomplete by the exact "
            "documented fixture(s). Downstream research must retain "
            "the accompanying anomaly metadata."
        )
        print(
            "Canonical domestic-league dataset "
            "accepted for downstream research."
        )


if __name__ == "__main__":
    main()
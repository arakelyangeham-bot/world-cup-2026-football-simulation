#build_bundesliga_player_usage_features

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PLAYER_STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_stats.csv"
)

PLAYER_MEMBERSHIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_players.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_087_usage_informed_expected_xi"
)

PLAYER_USAGE_PATH = (
    OUTPUT_DIRECTORY
    / "bundesliga_player_usage_features.csv"
)

CLUB_COVERAGE_PATH = (
    OUTPUT_DIRECTORY
    / "club_usage_coverage.csv"
)

DATA_QUALITY_PATH = (
    OUTPUT_DIRECTORY
    / "player_usage_data_quality.csv"
)

SUSPICIOUS_ROWS_PATH = (
    OUTPUT_DIRECTORY
    / "suspicious_player_usage_rows.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


COMPETITION_ID = 35
SEASON_ID = 63516
EXPECTED_CLUB_COUNT = 18

MINUTES_PER_FULL_MATCH = 90.0


IDENTITY_COLUMNS = [
    "competition",
    "competition_type",
    "competition_id",
    "season_id",
    "season_year",
    "player_id",
    "player",
    "player_slug",
    "team_id",
    "team",
    "team_slug",
]


REQUIRED_STATS_COLUMNS = {
    *IDENTITY_COLUMNS,
    "appearances",
    "matchesStarted",
    "minutesPlayed",
    "rating",
}


REQUIRED_MEMBERSHIP_COLUMNS = {
    *IDENTITY_COLUMNS,
}


USAGE_COLUMNS = [
    "appearances",
    "matchesStarted",
    "minutesPlayed",
    "rating",
    "substitutionsIn",
    "substitutionsOut",
]


def load_player_statistics() -> pd.DataFrame:
    if not PLAYER_STATS_PATH.exists():
        raise FileNotFoundError(
            "Player-stat dataset does not exist: "
            f"{PLAYER_STATS_PATH}"
        )

    dataframe = pd.read_csv(
        PLAYER_STATS_PATH,
        dtype={
            "player_id": str,
            "season_id": str,
        },
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Player-stat dataset is empty."
        )

    missing = (
        REQUIRED_STATS_COLUMNS
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Player-stat dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    return dataframe


def load_memberships() -> pd.DataFrame:
    if not PLAYER_MEMBERSHIP_PATH.exists():
        raise FileNotFoundError(
            "Player-membership dataset does not exist: "
            f"{PLAYER_MEMBERSHIP_PATH}"
        )

    dataframe = pd.read_csv(
        PLAYER_MEMBERSHIP_PATH,
        dtype={
            "player_id": str,
            "season_id": str,
        },
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Player-membership dataset is empty."
        )

    missing = (
        REQUIRED_MEMBERSHIP_COLUMNS
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Player-membership dataset is missing columns: "
            f"{sorted(missing)}"
        )

    return dataframe


def select_bundesliga_season(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    competition_ids = pd.to_numeric(
        dataframe["competition_id"],
        errors="raise",
    )

    season_ids = pd.to_numeric(
        dataframe["season_id"],
        errors="raise",
    )

    selected = dataframe.loc[
        competition_ids.eq(COMPETITION_ID)
        & season_ids.eq(SEASON_ID)
    ].copy()

    if selected.empty:
        raise ValueError(
            "No Bundesliga 2024/25 rows were found for "
            f"competition_id={COMPETITION_ID}, "
            f"season_id={SEASON_ID}."
        )

    return selected.reset_index(drop=True)


def normalize_identity_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["player_id"] = (
        output["player_id"]
        .astype(str)
        .str.strip()
    )

    output["team_id"] = pd.to_numeric(
        output["team_id"],
        errors="raise",
    ).astype(int)

    output["competition_id"] = pd.to_numeric(
        output["competition_id"],
        errors="raise",
    ).astype(int)

    output["season_id"] = pd.to_numeric(
        output["season_id"],
        errors="raise",
    ).astype(int)

    for column in (
        "player",
        "team",
        "competition",
    ):
        output[column] = (
            output[column]
            .astype(str)
            .str.strip()
        )

    if output["player_id"].eq("").any():
        raise ValueError(
            "One or more player IDs are empty."
        )

    if output["player"].eq("").any():
        raise ValueError(
            "One or more player names are empty."
        )

    if output["team"].eq("").any():
        raise ValueError(
            "One or more team names are empty."
        )

    return output


def validate_unique_keys(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
) -> None:
    key_columns = [
        "competition_id",
        "season_id",
        "team_id",
        "player_id",
    ]

    duplicate_mask = dataframe.duplicated(
        subset=key_columns,
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            dataframe.loc[
                duplicate_mask,
                key_columns + [
                    "team",
                    "player",
                ],
            ]
            .sort_values(key_columns)
        )

        raise ValueError(
            f"{dataset_name} contains duplicate "
            "competition-season-team-player keys. "
            f"Duplicate rows: {len(duplicates)}"
        )


def prepare_usage_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    for column in USAGE_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan

        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    output["appearances_raw"] = (
        output["appearances"]
    )

    output["matches_started_raw"] = (
        output["matchesStarted"]
    )

    output["minutes_played_raw"] = (
        output["minutesPlayed"]
    )

    output["appearances_missing"] = (
        output["appearances"].isna()
    )

    output["matches_started_missing"] = (
        output["matchesStarted"].isna()
    )

    output["minutes_played_missing"] = (
        output["minutesPlayed"].isna()
    )

    output["negative_appearances"] = (
        output["appearances"].fillna(0.0).lt(0.0)
    )

    output["negative_matches_started"] = (
        output["matchesStarted"]
        .fillna(0.0)
        .lt(0.0)
    )

    output["negative_minutes_played"] = (
        output["minutesPlayed"]
        .fillna(0.0)
        .lt(0.0)
    )

    output["starts_exceed_appearances"] = (
        output["matchesStarted"].notna()
        & output["appearances"].notna()
        & output["matchesStarted"].gt(
            output["appearances"]
        )
    )

    output["minutes_below_start_minimum"] = (
        output["minutesPlayed"].notna()
        & output["matchesStarted"].notna()
        & output["minutesPlayed"].lt(
            output["matchesStarted"]
        )
    )

    output["minutes_exceed_appearance_capacity"] = (
        output["minutesPlayed"].notna()
        & output["appearances"].notna()
        & output["minutesPlayed"].gt(
            output["appearances"]
            * 130.0
        )
    )

    return output


def derive_usage_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["validated_appearances"] = np.maximum(
        output["appearances"].fillna(0.0),
        output["matchesStarted"].fillna(0.0),
    )

    output["validated_matches_started"] = (
        output["matchesStarted"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    output["validated_minutes_played"] = (
        output["minutesPlayed"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    output["start_rate"] = np.where(
        output["validated_appearances"] > 0.0,
        (
            output["validated_matches_started"]
            / output["validated_appearances"]
        ),
        0.0,
    )

    output["start_rate"] = (
        output["start_rate"]
        .clip(lower=0.0, upper=1.0)
    )

    output["minutes_per_appearance"] = np.where(
        output["validated_appearances"] > 0.0,
        (
            output["validated_minutes_played"]
            / output["validated_appearances"]
        ),
        0.0,
    )

    output["minutes_per_start"] = np.where(
        output["validated_matches_started"] > 0.0,
        (
            output["validated_minutes_played"]
            / output["validated_matches_started"]
        ),
        0.0,
    )

    output["estimated_full_match_equivalents"] = (
        output["validated_minutes_played"]
        / MINUTES_PER_FULL_MATCH
    )

    club_max_minutes = (
        output
        .groupby(
            [
                "competition_id",
                "season_id",
                "team_id",
            ]
        )["validated_minutes_played"]
        .transform("max")
    )

    output["minutes_relative_to_club_max"] = np.where(
        club_max_minutes > 0.0,
        (
            output["validated_minutes_played"]
            / club_max_minutes
        ),
        0.0,
    )

    club_total_minutes = (
        output
        .groupby(
            [
                "competition_id",
                "season_id",
                "team_id",
            ]
        )["validated_minutes_played"]
        .transform("sum")
    )

    output["club_minutes_share"] = np.where(
        club_total_minutes > 0.0,
        (
            output["validated_minutes_played"]
            / club_total_minutes
        ),
        0.0,
    )

    club_max_starts = (
        output
        .groupby(
            [
                "competition_id",
                "season_id",
                "team_id",
            ]
        )["validated_matches_started"]
        .transform("max")
    )

    output["starts_relative_to_club_max"] = np.where(
        club_max_starts > 0.0,
        (
            output["validated_matches_started"]
            / club_max_starts
        ),
        0.0,
    )

    output["usage_evidence_score"] = (
        0.50
        * output["starts_relative_to_club_max"]
        + 0.35
        * output["minutes_relative_to_club_max"]
        + 0.15
        * output["start_rate"]
    )

    output["usage_evidence_score"] = (
        output["usage_evidence_score"]
        .clip(lower=0.0, upper=1.0)
    )

    output["regular_starter_flag"] = (
        output["start_rate"].ge(0.60)
        & output[
            "validated_matches_started"
        ].ge(5.0)
    )

    output["high_usage_flag"] = (
        output[
            "minutes_relative_to_club_max"
        ].ge(0.60)
    )

    output["low_evidence_flag"] = (
        output["validated_appearances"].lt(3.0)
    )

    return output


def merge_membership_validation(
    statistics: pd.DataFrame,
    memberships: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = [
        "competition_id",
        "season_id",
        "team_id",
        "player_id",
    ]

    membership_keys = (
        memberships[
            key_columns
        ]
        .drop_duplicates()
        .assign(
            membership_join_pass=True
        )
    )

    merged = statistics.merge(
        membership_keys,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    merged["membership_join_pass"] = (
        merged["membership_join_pass"]
        .fillna(False)
        .astype(bool)
    )

    return merged


def validate_final_dataset(
    dataframe: pd.DataFrame,
) -> None:
    if dataframe.empty:
        raise ValueError(
            "Final Bundesliga usage dataset is empty."
        )

    clubs = dataframe[
        [
            "team_id",
            "team",
        ]
    ].drop_duplicates()

    if len(clubs) != EXPECTED_CLUB_COUNT:
        raise ValueError(
            "Unexpected Bundesliga club count. "
            f"Expected {EXPECTED_CLUB_COUNT}, "
            f"received {len(clubs)}."
        )

    key_columns = [
        "competition_id",
        "season_id",
        "team_id",
        "player_id",
    ]

    if dataframe.duplicated(
        subset=key_columns
    ).any():
        raise ValueError(
            "Final usage dataset contains duplicate keys."
        )

    required_numeric_columns = [
        "validated_appearances",
        "validated_matches_started",
        "validated_minutes_played",
        "start_rate",
        "minutes_per_appearance",
        "estimated_full_match_equivalents",
        "minutes_relative_to_club_max",
        "club_minutes_share",
        "starts_relative_to_club_max",
        "usage_evidence_score",
    ]

    numeric_values = dataframe[
        required_numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Final usage dataset contains non-finite "
            "derived values."
        )

    bounded_columns = [
        "start_rate",
        "minutes_relative_to_club_max",
        "club_minutes_share",
        "starts_relative_to_club_max",
        "usage_evidence_score",
    ]

    for column in bounded_columns:
        if (
            dataframe[column].lt(0.0).any()
            or dataframe[column].gt(1.0).any()
        ):
            raise ValueError(
                f"Derived column {column!r} lies outside "
                "[0, 1]."
            )


def build_club_coverage(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for (
        team_id,
        team,
    ), group in dataframe.groupby(
        [
            "team_id",
            "team",
        ],
        sort=True,
    ):
        records.append(
            {
                "team_id": int(team_id),
                "team": str(team),
                "player_count": int(
                    len(group)
                ),
                "players_with_appearances": int(
                    group[
                        "validated_appearances"
                    ].gt(0.0).sum()
                ),
                "players_with_starts": int(
                    group[
                        "validated_matches_started"
                    ].gt(0.0).sum()
                ),
                "players_with_minutes": int(
                    group[
                        "validated_minutes_played"
                    ].gt(0.0).sum()
                ),
                "regular_starter_count": int(
                    group[
                        "regular_starter_flag"
                    ].sum()
                ),
                "high_usage_player_count": int(
                    group[
                        "high_usage_flag"
                    ].sum()
                ),
                "low_evidence_player_count": int(
                    group[
                        "low_evidence_flag"
                    ].sum()
                ),
                "missing_appearance_count": int(
                    group[
                        "appearances_missing"
                    ].sum()
                ),
                "missing_start_count": int(
                    group[
                        "matches_started_missing"
                    ].sum()
                ),
                "missing_minutes_count": int(
                    group[
                        "minutes_played_missing"
                    ].sum()
                ),
                "starts_exceed_appearances_count": int(
                    group[
                        "starts_exceed_appearances"
                    ].sum()
                ),
                "membership_join_failure_count": int(
                    (
                        ~group[
                            "membership_join_pass"
                        ]
                    ).sum()
                ),
                "total_player_minutes": float(
                    group[
                        "validated_minutes_played"
                    ].sum()
                ),
                "maximum_player_minutes": float(
                    group[
                        "validated_minutes_played"
                    ].max()
                ),
                "median_start_rate": float(
                    group["start_rate"].median()
                ),
                "mean_usage_evidence_score": float(
                    group[
                        "usage_evidence_score"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(records)


def build_data_quality_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        (
            "missing_appearances",
            dataframe[
                "appearances_missing"
            ],
        ),
        (
            "missing_matches_started",
            dataframe[
                "matches_started_missing"
            ],
        ),
        (
            "missing_minutes_played",
            dataframe[
                "minutes_played_missing"
            ],
        ),
        (
            "negative_appearances",
            dataframe[
                "negative_appearances"
            ],
        ),
        (
            "negative_matches_started",
            dataframe[
                "negative_matches_started"
            ],
        ),
        (
            "negative_minutes_played",
            dataframe[
                "negative_minutes_played"
            ],
        ),
        (
            "starts_exceed_appearances",
            dataframe[
                "starts_exceed_appearances"
            ],
        ),
        (
            "minutes_below_start_minimum",
            dataframe[
                "minutes_below_start_minimum"
            ],
        ),
        (
            "minutes_exceed_appearance_capacity",
            dataframe[
                "minutes_exceed_appearance_capacity"
            ],
        ),
        (
            "membership_join_failure",
            ~dataframe[
                "membership_join_pass"
            ],
        ),
    ]

    records: list[dict[str, object]] = []

    for check_name, mask in checks:
        count = int(mask.sum())

        records.append(
            {
                "check": check_name,
                "affected_row_count": count,
                "affected_row_share": (
                    count / len(dataframe)
                ),
                "status": (
                    "PASS"
                    if count == 0
                    else "REVIEW"
                ),
            }
        )

    return pd.DataFrame(records)


def build_suspicious_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    suspicious_mask = (
        dataframe["starts_exceed_appearances"]
        | dataframe[
            "negative_appearances"
        ]
        | dataframe[
            "negative_matches_started"
        ]
        | dataframe[
            "negative_minutes_played"
        ]
        | dataframe[
            "minutes_below_start_minimum"
        ]
        | dataframe[
            "minutes_exceed_appearance_capacity"
        ]
        | (
            ~dataframe[
                "membership_join_pass"
            ]
        )
    )

    columns = [
        "competition_id",
        "season_id",
        "team_id",
        "team",
        "player_id",
        "player",
        "appearances_raw",
        "matches_started_raw",
        "minutes_played_raw",
        "validated_appearances",
        "validated_matches_started",
        "validated_minutes_played",
        "start_rate",
        "minutes_per_appearance",
        "starts_exceed_appearances",
        "negative_appearances",
        "negative_matches_started",
        "negative_minutes_played",
        "minutes_below_start_minimum",
        "minutes_exceed_appearance_capacity",
        "membership_join_pass",
    ]

    return (
        dataframe.loc[
            suspicious_mask,
            columns,
        ]
        .sort_values(
            [
                "team",
                "player",
            ]
        )
        .reset_index(drop=True)
    )


def write_report(
    *,
    player_usage: pd.DataFrame,
    club_coverage: pd.DataFrame,
    data_quality: pd.DataFrame,
    suspicious_rows: pd.DataFrame,
) -> None:
    club_count = int(
        player_usage["team_id"].nunique()
    )

    player_count = len(player_usage)

    players_with_starts = int(
        player_usage[
            "validated_matches_started"
        ].gt(0.0).sum()
    )

    players_with_minutes = int(
        player_usage[
            "validated_minutes_played"
        ].gt(0.0).sum()
    )

    membership_failures = int(
        (
            ~player_usage[
                "membership_join_pass"
            ]
        ).sum()
    )

    starts_exceeding_appearances = int(
        player_usage[
            "starts_exceed_appearances"
        ].sum()
    )

    report = f"""# Study 087A — Bundesliga Player-Usage Feature Construction

## Purpose

Construct a validated Bundesliga 2024/25 player-usage dataset for
future usage-informed expected-lineup experiments.

## Methodological boundary

This phase does not:

- select expected lineups;
- fit a start-probability model;
- alter the production repository;
- alter the production goal model;
- rerun any fixture predictions.

The source statistics are season-level aggregates and are not
prediction-date valid for historical replay.

## Population

- Competition ID: {COMPETITION_ID}
- Season ID: {SEASON_ID}
- Clubs: {club_count}
- Player-season rows: {player_count}
- Players with at least one start: {players_with_starts}
- Players with positive minutes: {players_with_minutes}

## Derived usage features

The study constructs:

- validated appearances;
- validated starts;
- validated minutes;
- start rate;
- minutes per appearance;
- minutes per start;
- full-match equivalents;
- minutes relative to club maximum;
- club minutes share;
- starts relative to club maximum;
- composite usage-evidence score;
- regular-starter and high-usage flags.

## Data-quality handling

Rows where starts exceed appearances are retained and explicitly
flagged.

For derived-rate purposes:

`validated_appearances = max(appearances, matchesStarted)`

This prevents impossible start rates above one without deleting or
silently overwriting the raw values.

## Quality findings

- Rows where starts exceed appearances:
  {starts_exceeding_appearances}
- Membership join failures:
  {membership_failures}
- Total suspicious rows:
  {len(suspicious_rows)}

Suspicious rows remain available in a dedicated audit artifact.

## Interpretation boundary

The derived usage signals estimate season-level player importance.

They must not be interpreted as historically available evidence for
early-season fixtures in the same season.

## Result

**OVERALL RESULT: PASS**

The Bundesliga player-usage feature layer was constructed without
modifying the prediction runtime.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def build_metadata(
    *,
    player_usage: pd.DataFrame,
    club_coverage: pd.DataFrame,
    suspicious_rows: pd.DataFrame,
) -> dict[str, object]:
    return {
        "study_id": "087A",
        "study_name": (
            "Bundesliga Player-Usage Feature Construction"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "competition_id": COMPETITION_ID,
        "season_id": SEASON_ID,
        "competition": "Bundesliga",
        "season": "2024/25",
        "player_stats_source": str(
            PLAYER_STATS_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "membership_source": str(
            PLAYER_MEMBERSHIP_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "player_row_count": len(
            player_usage
        ),
        "club_count": len(
            club_coverage
        ),
        "suspicious_row_count": len(
            suspicious_rows
        ),
        "representation_changed": False,
        "lineups_selected": False,
        "goal_model_changed": False,
        "production_repository_changed": False,
        "prediction_date_valid": False,
        "interpretation_boundary": (
            "Usage features are derived from full-season "
            "aggregates and are suitable for retrospective "
            "representation experiments only."
        ),
        "outputs": [
            PLAYER_USAGE_PATH.name,
            CLUB_COVERAGE_PATH.name,
            DATA_QUALITY_PATH.name,
            SUSPICIOUS_ROWS_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 087A — BUNDESLIGA PLAYER-USAGE "
        "FEATURE CONSTRUCTION"
    )
    print("=" * 88)

    statistics = load_player_statistics()
    memberships = load_memberships()

    statistics = select_bundesliga_season(
        statistics
    )

    memberships = select_bundesliga_season(
        memberships
    )

    statistics = normalize_identity_columns(
        statistics
    )

    memberships = normalize_identity_columns(
        memberships
    )

    validate_unique_keys(
        statistics,
        dataset_name="Player statistics",
    )

    validate_unique_keys(
        memberships,
        dataset_name="Player memberships",
    )

    usage = prepare_usage_values(
        statistics
    )

    usage = derive_usage_features(
        usage
    )

    usage = merge_membership_validation(
        usage,
        memberships,
    )

    validate_final_dataset(
        usage
    )

    club_coverage = build_club_coverage(
        usage
    )

    data_quality = build_data_quality_summary(
        usage
    )

    suspicious_rows = build_suspicious_rows(
        usage
    )

    output_columns = [
        *IDENTITY_COLUMNS,
        "rating",
        "appearances_raw",
        "matches_started_raw",
        "minutes_played_raw",
        "validated_appearances",
        "validated_matches_started",
        "validated_minutes_played",
        "start_rate",
        "minutes_per_appearance",
        "minutes_per_start",
        "estimated_full_match_equivalents",
        "minutes_relative_to_club_max",
        "club_minutes_share",
        "starts_relative_to_club_max",
        "usage_evidence_score",
        "regular_starter_flag",
        "high_usage_flag",
        "low_evidence_flag",
        "appearances_missing",
        "matches_started_missing",
        "minutes_played_missing",
        "starts_exceed_appearances",
        "minutes_below_start_minimum",
        "minutes_exceed_appearance_capacity",
        "membership_join_pass",
    ]

    player_usage = (
        usage[
            output_columns
        ]
        .sort_values(
            [
                "team",
                "usage_evidence_score",
                "player",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    metadata = build_metadata(
        player_usage=player_usage,
        club_coverage=club_coverage,
        suspicious_rows=suspicious_rows,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    player_usage.to_csv(
        PLAYER_USAGE_PATH,
        index=False,
    )

    club_coverage.to_csv(
        CLUB_COVERAGE_PATH,
        index=False,
    )

    data_quality.to_csv(
        DATA_QUALITY_PATH,
        index=False,
    )

    suspicious_rows.to_csv(
        SUSPICIOUS_ROWS_PATH,
        index=False,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        player_usage=player_usage,
        club_coverage=club_coverage,
        data_quality=data_quality,
        suspicious_rows=suspicious_rows,
    )

    print()
    print("Population")
    print("-" * 88)
    print(
        f"  Player rows: {len(player_usage)}"
    )
    print(
        f"  Clubs: {len(club_coverage)}"
    )
    print(
        "  Players with positive minutes: "
        f"{player_usage['validated_minutes_played'].gt(0.0).sum()}"
    )
    print(
        "  Players with at least one start: "
        f"{player_usage['validated_matches_started'].gt(0.0).sum()}"
    )

    print()
    print("Data-quality summary")
    print("-" * 88)
    print(
        data_quality.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Club coverage")
    print("-" * 88)
    print(
        club_coverage[
            [
                "team",
                "player_count",
                "players_with_starts",
                "players_with_minutes",
                "regular_starter_count",
                "starts_exceed_appearances_count",
                "membership_join_failure_count",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Player-stat loading: PASS")
    print("  Membership loading: PASS")
    print("  Competition-season filter: PASS")
    print("  Unique player keys: PASS")
    print("  Usage feature derivation: PASS")
    print("  Finite derived values: PASS")
    print("  Bounded usage rates: PASS")
    print("  Club-count validation: PASS")
    print("  Suspicious-row preservation: PASS")
    print("  Production mutation: NONE")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
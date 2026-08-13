#analyze_representation_sensitivity

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.player_intelligence.player_schema import (
    RoleRatings,
)
from research.player_intelligence.role_projection import (
    project_attack,
    project_defense,
    project_goalkeeper,
    project_midfield,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

JOINED_PLAYER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_087_usage_informed_expected_xi"
    / "bundesliga_usage_rating_join.csv"
)

LINEUP_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_087_usage_informed_expected_xi"
    / "player_selection_policy_lineups.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_087_usage_informed_expected_xi"
)

REPRESENTATION_PATH = (
    OUTPUT_DIRECTORY
    / "selection_policy_team_representations.csv"
)

DELTA_PATH = (
    OUTPUT_DIRECTORY
    / "representation_delta_from_rating_only.csv"
)

FULL_SQUAD_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "representation_comparison_to_full_squad.csv"
)

SPECIFICATION_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "representation_specification_summary.csv"
)

CLUB_SENSITIVITY_PATH = (
    OUTPUT_DIRECTORY
    / "club_representation_sensitivity.csv"
)

FEATURE_RANGE_PATH = (
    OUTPUT_DIRECTORY
    / "representation_feature_ranges.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_087c_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_087c_report.md"
)

NON_FINITE_ROLE_RATINGS_PATH = (
    OUTPUT_DIRECTORY
    / "non_finite_role_rating_audit.csv"
)

EXPECTED_CLUB_COUNT = 18
EXPECTED_LINEUP_SIZE = 11

RATING_ONLY_SPECIFICATION = "rating_only"
FULL_SQUAD_SPECIFICATION = "study_full_squad_reconstruction"

ROLE_COLUMNS = (
    "rating_GK",
    "rating_CB",
    "rating_FB",
    "rating_DM",
    "rating_CM",
    "rating_AM",
    "rating_WM",
    "rating_W",
    "rating_ST",
)

REPRESENTATION_FEATURES = (
    "attack",
    "midfield",
    "defense",
    "goalkeeper",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
)

REQUIRED_PLAYER_COLUMNS = {
    "team_id",
    "team",
    "player_id",
    "player",
    "start_rate",
    "minutes_relative_to_club_max",
    "usage_evidence_score",
    *ROLE_COLUMNS,
}

REQUIRED_LINEUP_COLUMNS = {
    "specification",
    "team_id",
    "team",
    "player_id",
    "player",
    "formation",
    "slot",
    "role",
    "exact_role_match",
    "selection_role_suitability",
}


@dataclass(frozen=True)
class DiagnosticTeamRepresentation:
    specification: str

    team_id: int
    team: str

    representation_type: str
    formation: str | None

    attack: float
    midfield: float
    defense: float
    goalkeeper: float

    attack_depth: float
    midfield_depth: float
    defense_depth: float

    player_count: int
    exact_role_match_count: int
    fallback_role_count: int

    mean_start_rate: float
    mean_relative_minutes: float
    mean_usage_evidence_score: float

    mean_maximum_role_rating: float

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def load_csv(
    path: Path,
    *,
    dataset_name: str,
    required_columns: set[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} does not exist: {path}"
        )

    dataframe = pd.read_csv(
        path,
        dtype={
            "player_id": str,
        },
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"{dataset_name} is empty."
        )

    missing = required_columns - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe["player_id"] = (
        dataframe["player_id"]
        .astype(str)
        .str.strip()
    )

    dataframe["team"] = (
        dataframe["team"]
        .astype(str)
        .str.strip()
    )

    dataframe["player"] = (
        dataframe["player"]
        .astype(str)
        .str.strip()
    )

    if dataframe["player_id"].eq("").any():
        raise ValueError(
            f"{dataset_name} contains empty player IDs."
        )

    return dataframe


def parse_boolean_series(
    values: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    if values.dtype == bool:
        return values.copy()

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    allowed = {
        "true",
        "false",
    }

    unexpected = (
        set(normalized.unique())
        - allowed
    )

    if unexpected:
        raise ValueError(
            f"{column_name!r} contains unexpected boolean "
            f"values: {sorted(unexpected)}"
        )

    return normalized.eq("true")

def build_non_finite_role_rating_audit(
    players: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for _, row in players.iterrows():
        for column in ROLE_COLUMNS:
            raw_value = row[column]

            if pd.isna(raw_value):
                continue

            try:
                numeric_value = float(
                    raw_value
                )
            except (
                TypeError,
                ValueError,
            ):
                records.append(
                    {
                        "team_id": row["team_id"],
                        "team": row["team"],
                        "player_id": row["player_id"],
                        "player": row["player"],
                        "role_rating_column": column,
                        "raw_value": raw_value,
                        "issue": "non_numeric",
                    }
                )
                continue

            if not math.isfinite(
                numeric_value
            ):
                records.append(
                    {
                        "team_id": row["team_id"],
                        "team": row["team"],
                        "player_id": row["player_id"],
                        "player": row["player"],
                        "role_rating_column": column,
                        "raw_value": raw_value,
                        "issue": "non_finite",
                    }
                )

    return pd.DataFrame(
        records,
        columns=[
            "team_id",
            "team",
            "player_id",
            "player",
            "role_rating_column",
            "raw_value",
            "issue",
        ],
    )

def prepare_players(
    players: pd.DataFrame,
) -> pd.DataFrame:
    output = players.copy()

    numeric_columns = [
        "team_id",
        "start_rate",
        "minutes_relative_to_club_max",
        "usage_evidence_score",
        *ROLE_COLUMNS,
    ]

    for column in numeric_columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    bounded_columns = (
        "start_rate",
        "minutes_relative_to_club_max",
        "usage_evidence_score",
    )

    for column in bounded_columns:
        if output[column].isna().any():
            raise ValueError(
                f"Player dataset contains missing "
                f"{column!r} values."
            )

        if (
            output[column].lt(0.0).any()
            or output[column].gt(1.0).any()
        ):
            raise ValueError(
                f"Player dataset contains {column!r} "
                "values outside [0, 1]."
            )

    key_columns = [
        "team_id",
        "player_id",
    ]

    if output.duplicated(
        subset=key_columns
    ).any():
        raise ValueError(
            "Player dataset contains duplicate "
            "team-player keys."
        )

    club_count = output["team_id"].nunique()

    if club_count != EXPECTED_CLUB_COUNT:
        raise ValueError(
            "Unexpected player-dataset club count. "
            f"Expected {EXPECTED_CLUB_COUNT}, "
            f"received {club_count}."
        )

    return output


def prepare_lineups(
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    output = lineups.copy()

    output["team_id"] = pd.to_numeric(
        output["team_id"],
        errors="raise",
    ).astype(int)

    output["exact_role_match"] = (
        parse_boolean_series(
            output["exact_role_match"],
            column_name="exact_role_match",
        )
    )

    output[
        "selection_role_suitability"
    ] = pd.to_numeric(
        output[
            "selection_role_suitability"
        ],
        errors="raise",
    )

    if (
        output[
            "selection_role_suitability"
        ].le(0.0).any()
        or output[
            "selection_role_suitability"
        ].gt(1.0).any()
    ):
        raise ValueError(
            "Role-suitability values must lie in (0, 1]."
        )

    grouping_columns = [
        "specification",
        "team_id",
    ]

    lineup_sizes = (
        output
        .groupby(grouping_columns)
        .size()
    )

    if not lineup_sizes.eq(
        EXPECTED_LINEUP_SIZE
    ).all():
        invalid = lineup_sizes.loc[
            ~lineup_sizes.eq(
                EXPECTED_LINEUP_SIZE
            )
        ]

        raise ValueError(
            "One or more projected lineups do not contain "
            f"{EXPECTED_LINEUP_SIZE} rows: "
            f"{invalid.to_dict()}"
        )

    duplicate_players = output.duplicated(
        subset=[
            "specification",
            "team_id",
            "player_id",
        ],
        keep=False,
    )

    if duplicate_players.any():
        raise ValueError(
            "Projected lineups contain duplicate players."
        )

    if (
        RATING_ONLY_SPECIFICATION
        not in set(output["specification"])
    ):
        raise ValueError(
            "Rating-only baseline is missing."
        )

    return output


def role_ratings_from_row(
    row: pd.Series,
) -> RoleRatings:
    values: dict[str, float | None] = {}

    for role in (
        "GK",
        "CB",
        "FB",
        "DM",
        "CM",
        "AM",
        "WM",
        "W",
        "ST",
    ):
        raw_value = row[
            f"rating_{role}"
        ]

        if pd.isna(raw_value):
            values[role] = None
            continue

        numeric_value = float(
            raw_value
        )

        if not math.isfinite(
            numeric_value
        ):
            values[role] = None
            continue

        values[role] = numeric_value

    return RoleRatings(**values)


def player_projection_record(
    row: pd.Series,
) -> dict[str, float]:
    role_ratings = role_ratings_from_row(
        row
    )

    available_role_values = [
        float(value)
        for value in asdict(
            role_ratings
        ).values()
        if value is not None
        and math.isfinite(float(value))
    ]

    maximum_role_rating = (
        max(available_role_values)
        if available_role_values
        else 0.0
    )

    raw_projections = {
        "projected_attack": project_attack(
            role_ratings
        ),
        "projected_midfield": project_midfield(
            role_ratings
        ),
        "projected_defense": project_defense(
            role_ratings
        ),
        "projected_goalkeeper": project_goalkeeper(
            role_ratings
        ),
        "projected_maximum_role_rating": maximum_role_rating,
    }

    cleaned_projections: dict[str, float] = {}

    for column, raw_value in raw_projections.items():
        numeric_value = float(
            raw_value
        )

        cleaned_projections[column] = (
            numeric_value
            if math.isfinite(numeric_value)
            else 0.0
        )

    return cleaned_projections

def add_player_projections(
    players: pd.DataFrame,
) -> pd.DataFrame:
    projection_rows = [
        player_projection_record(row)
        for _, row in players.iterrows()
    ]

    projections = pd.DataFrame(
        projection_rows,
        index=players.index,
    )

    output = pd.concat(
        [
            players,
            projections,
        ],
        axis=1,
    )

    projection_columns = [
        "projected_attack",
        "projected_midfield",
        "projected_defense",
        "projected_goalkeeper",
        "projected_maximum_role_rating",
    ]

    projection_values = output[
        projection_columns
    ].to_numpy(
        dtype=float
    )

    finite_mask = np.isfinite(
        projection_values
    )

    if not finite_mask.all():
        invalid_row_mask = (
            ~finite_mask
        ).any(axis=1)

        invalid_rows = output.loc[
            invalid_row_mask,
            [
                "team_id",
                "team",
                "player_id",
                "player",
                *ROLE_COLUMNS,
                *projection_columns,
            ],
        ]

        print()
        print(
            "Non-finite player projection rows"
        )
        print("-" * 88)
        print(
            invalid_rows.to_string(
                index=False
            )
        )

        raise ValueError(
            "Player projection calculation produced "
            f"non-finite values for {len(invalid_rows)} "
            "player rows."
        )

    return output


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def top_n_mean(
    values: list[float],
    *,
    n: int,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(
        values,
        reverse=True,
    )

    return mean(
        ordered[:n]
    )


def build_representation(
    *,
    specification: str,
    team_id: int,
    team: str,
    formation: str | None,
    player_rows: pd.DataFrame,
    representation_type: str,
    exact_role_match_count: int,
    fallback_role_count: int,
) -> DiagnosticTeamRepresentation:
    if player_rows.empty:
        raise ValueError(
            "Cannot build a representation from an empty "
            "player population."
        )

    attack_values = (
        player_rows[
            "projected_attack"
        ].astype(float).tolist()
    )

    midfield_values = (
        player_rows[
            "projected_midfield"
        ].astype(float).tolist()
    )

    defense_values = (
        player_rows[
            "projected_defense"
        ].astype(float).tolist()
    )

    goalkeeper_values = (
        player_rows[
            "projected_goalkeeper"
        ].astype(float).tolist()
    )

    representation = DiagnosticTeamRepresentation(
        specification=specification,
        team_id=int(team_id),
        team=str(team),
        representation_type=(
            representation_type
        ),
        formation=formation,
        attack=top_n_mean(
            attack_values,
            n=5,
        ),
        midfield=top_n_mean(
            midfield_values,
            n=5,
        ),
        defense=top_n_mean(
            defense_values,
            n=5,
        ),
        goalkeeper=(
            max(goalkeeper_values)
            if goalkeeper_values
            else 0.0
        ),
        attack_depth=mean(
            attack_values
        ),
        midfield_depth=mean(
            midfield_values
        ),
        defense_depth=mean(
            defense_values
        ),
        player_count=len(
            player_rows
        ),
        exact_role_match_count=int(
            exact_role_match_count
        ),
        fallback_role_count=int(
            fallback_role_count
        ),
        mean_start_rate=float(
            player_rows[
                "start_rate"
            ].mean()
        ),
        mean_relative_minutes=float(
            player_rows[
                "minutes_relative_to_club_max"
            ].mean()
        ),
        mean_usage_evidence_score=float(
            player_rows[
                "usage_evidence_score"
            ].mean()
        ),
        mean_maximum_role_rating=float(
            player_rows[
                "projected_maximum_role_rating"
            ].mean()
        ),
    )

    values = [
        getattr(
            representation,
            feature,
        )
        for feature in REPRESENTATION_FEATURES
    ]

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "Team representation contains non-finite "
            "feature values."
        )

    return representation


def build_full_squad_representations(
    players: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for (
        team_id,
        team,
    ), club_players in players.groupby(
        [
            "team_id",
            "team",
        ],
        sort=True,
    ):
        representation = build_representation(
            specification=(
                FULL_SQUAD_SPECIFICATION
            ),
            team_id=int(team_id),
            team=str(team),
            formation=None,
            player_rows=club_players,
            representation_type=(
                "full_squad_reconstruction"
            ),
            exact_role_match_count=0,
            fallback_role_count=0,
        )

        records.append(
            representation.to_record()
        )

    output = pd.DataFrame(records)

    if len(output) != EXPECTED_CLUB_COUNT:
        raise AssertionError(
            "Unexpected full-squad representation count."
        )

    return output


def build_lineup_representations(
    *,
    players: pd.DataFrame,
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    player_projection = players[
        [
            "team_id",
            "team",
            "player_id",
            "projected_attack",
            "projected_midfield",
            "projected_defense",
            "projected_goalkeeper",
            "projected_maximum_role_rating",
        ]
    ].copy()

    joined = lineups.merge(
        player_projection,
        on=[
            "team_id",
            "team",
            "player_id",
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not joined["_merge"].eq("both").all():
        failures = joined.loc[
            ~joined["_merge"].eq("both"),
            [
                "specification",
                "team",
                "player_id",
                "player",
            ],
        ]

        raise ValueError(
            "One or more selected players could not be "
            "joined to projection features: "
            f"{failures.to_dict('records')[:20]}"
        )

    joined = joined.drop(
        columns=["_merge"]
    )

    required_joined_columns = {
        "start_rate",
        "minutes_relative_to_club_max",
        "usage_evidence_score",
        "projected_attack",
        "projected_midfield",
        "projected_defense",
        "projected_goalkeeper",
        "projected_maximum_role_rating",
    }

    missing_joined_columns = (
        required_joined_columns
        - set(joined.columns)
    )

    if missing_joined_columns:
        raise ValueError(
            "Lineup-projection join is missing required columns: "
            f"{sorted(missing_joined_columns)}"
        )

    records: list[dict[str, object]] = []

    for (
        specification,
        team_id,
        team,
        formation,
    ), group in joined.groupby(
        [
            "specification",
            "team_id",
            "team",
            "formation",
        ],
        sort=True,
    ):
        exact_count = int(
            group[
                "exact_role_match"
            ].sum()
        )

        fallback_count = (
            EXPECTED_LINEUP_SIZE
            - exact_count
        )

        representation = build_representation(
            specification=str(
                specification
            ),
            team_id=int(
                team_id
            ),
            team=str(
                team
            ),
            formation=str(
                formation
            ),
            player_rows=group,
            representation_type=(
                "projected_starting_xi"
            ),
            exact_role_match_count=(
                exact_count
            ),
            fallback_role_count=(
                fallback_count
            ),
        )

        records.append(
            representation.to_record()
        )

    output = pd.DataFrame(records)

    expected_rows = (
        lineups["specification"].nunique()
        * EXPECTED_CLUB_COUNT
    )

    if len(output) != expected_rows:
        raise AssertionError(
            "Unexpected projected-XI representation count. "
            f"Expected {expected_rows}, "
            f"received {len(output)}."
        )

    return output


def build_delta_from_rating_only(
    representations: pd.DataFrame,
) -> pd.DataFrame:
    baseline = representations.loc[
        representations[
            "specification"
        ].eq(
            RATING_ONLY_SPECIFICATION
        ),
        [
            "team_id",
            "team",
            *REPRESENTATION_FEATURES,
        ],
    ].copy()

    baseline = baseline.rename(
        columns={
            feature: (
                f"baseline_{feature}"
            )
            for feature
            in REPRESENTATION_FEATURES
        }
    )

    comparison = representations.merge(
        baseline,
        on=[
            "team_id",
            "team",
        ],
        how="left",
        validate="many_to_one",
    )

    for feature in REPRESENTATION_FEATURES:
        comparison[
            f"{feature}_delta"
        ] = (
            comparison[feature]
            - comparison[
                f"baseline_{feature}"
            ]
        )

        comparison[
            f"{feature}_absolute_delta"
        ] = comparison[
            f"{feature}_delta"
        ].abs()

    delta_columns = [
        f"{feature}_delta"
        for feature in REPRESENTATION_FEATURES
    ]

    comparison[
        "mean_absolute_feature_delta"
    ] = (
        comparison[
            [
                f"{feature}_absolute_delta"
                for feature
                in REPRESENTATION_FEATURES
            ]
        ]
        .mean(axis=1)
    )

    comparison[
        "maximum_absolute_feature_delta"
    ] = (
        comparison[
            [
                f"{feature}_absolute_delta"
                for feature
                in REPRESENTATION_FEATURES
            ]
        ]
        .max(axis=1)
    )

    comparison[
        "euclidean_feature_distance"
    ] = np.sqrt(
        (
            comparison[
                delta_columns
            ]
            ** 2
        ).sum(axis=1)
    )

    return comparison


def build_full_squad_comparison(
    *,
    lineup_representations: pd.DataFrame,
    full_squad_representations: pd.DataFrame,
) -> pd.DataFrame:
    full_squad = (
        full_squad_representations[
            [
                "team_id",
                "team",
                *REPRESENTATION_FEATURES,
            ]
        ]
        .rename(
            columns={
                feature: (
                    f"full_squad_{feature}"
                )
                for feature
                in REPRESENTATION_FEATURES
            }
        )
    )

    comparison = lineup_representations.merge(
        full_squad,
        on=[
            "team_id",
            "team",
        ],
        how="left",
        validate="many_to_one",
    )

    absolute_delta_columns: list[str] = []

    for feature in REPRESENTATION_FEATURES:
        delta_column = (
            f"{feature}_minus_full_squad"
        )

        absolute_column = (
            f"{feature}_absolute_delta_"
            "from_full_squad"
        )

        comparison[delta_column] = (
            comparison[feature]
            - comparison[
                f"full_squad_{feature}"
            ]
        )

        comparison[absolute_column] = (
            comparison[
                delta_column
            ].abs()
        )

        absolute_delta_columns.append(
            absolute_column
        )

    comparison[
        "mean_absolute_delta_from_full_squad"
    ] = comparison[
        absolute_delta_columns
    ].mean(axis=1)

    return comparison


def build_specification_summary(
    delta: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for specification, group in delta.groupby(
        "specification",
        sort=True,
    ):
        record: dict[str, object] = {
            "specification": specification,
            "club_count": len(group),
            "mean_absolute_feature_delta": float(
                group[
                    "mean_absolute_feature_delta"
                ].mean()
            ),
            "maximum_club_mean_absolute_feature_delta":
                float(
                    group[
                        "mean_absolute_feature_delta"
                    ].max()
                ),
            "mean_euclidean_feature_distance":
                float(
                    group[
                        "euclidean_feature_distance"
                    ].mean()
                ),
            "maximum_euclidean_feature_distance":
                float(
                    group[
                        "euclidean_feature_distance"
                    ].max()
                ),
            "unchanged_representation_count": int(
                group[
                    "maximum_absolute_feature_delta"
                ]
                .le(1e-12)
                .sum()
            ),
            "mean_fallback_role_count": float(
                group[
                    "fallback_role_count"
                ].mean()
            ),
            "mean_start_rate": float(
                group[
                    "mean_start_rate"
                ].mean()
            ),
            "mean_relative_minutes": float(
                group[
                    "mean_relative_minutes"
                ].mean()
            ),
            "mean_usage_evidence_score": float(
                group[
                    "mean_usage_evidence_score"
                ].mean()
            ),
        }

        for feature in REPRESENTATION_FEATURES:
            record[
                f"mean_{feature}_delta"
            ] = float(
                group[
                    f"{feature}_delta"
                ].mean()
            )

            record[
                f"mean_absolute_{feature}_delta"
            ] = float(
                group[
                    f"{feature}_absolute_delta"
                ].mean()
            )

            record[
                f"maximum_absolute_{feature}_delta"
            ] = float(
                group[
                    f"{feature}_absolute_delta"
                ].max()
            )

        records.append(record)

    return (
        pd.DataFrame(records)
        .sort_values(
            "mean_absolute_feature_delta"
        )
        .reset_index(drop=True)
    )


def build_club_sensitivity(
    delta: pd.DataFrame,
) -> pd.DataFrame:
    non_baseline = delta.loc[
        ~delta[
            "specification"
        ].eq(
            RATING_ONLY_SPECIFICATION
        )
    ].copy()

    records: list[dict[str, object]] = []

    for (
        team_id,
        team,
    ), group in non_baseline.groupby(
        [
            "team_id",
            "team",
        ],
        sort=True,
    ):
        record: dict[str, object] = {
            "team_id": int(
                team_id
            ),
            "team": str(
                team
            ),
            "specification_count": len(
                group
            ),
            "mean_absolute_feature_delta": float(
                group[
                    "mean_absolute_feature_delta"
                ].mean()
            ),
            "maximum_mean_absolute_feature_delta":
                float(
                    group[
                        "mean_absolute_feature_delta"
                    ].max()
                ),
            "mean_euclidean_feature_distance":
                float(
                    group[
                        "euclidean_feature_distance"
                    ].mean()
                ),
            "maximum_euclidean_feature_distance":
                float(
                    group[
                        "euclidean_feature_distance"
                    ].max()
                ),
            "most_disruptive_specification": str(
                group.sort_values(
                    "euclidean_feature_distance",
                    ascending=False,
                )
                .iloc[0][
                    "specification"
                ]
            ),
        }

        for feature in REPRESENTATION_FEATURES:
            record[
                f"maximum_absolute_{feature}_delta"
            ] = float(
                group[
                    f"{feature}_absolute_delta"
                ].max()
            )

        records.append(record)

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "mean_euclidean_feature_distance",
                "maximum_euclidean_feature_distance",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def build_feature_ranges(
    representations: pd.DataFrame,
) -> pd.DataFrame:
    non_full_squad = representations.loc[
        ~representations[
            "specification"
        ].eq(
            FULL_SQUAD_SPECIFICATION
        )
    ].copy()

    records: list[dict[str, object]] = []

    for (
        team_id,
        team,
    ), group in non_full_squad.groupby(
        [
            "team_id",
            "team",
        ],
        sort=True,
    ):
        for feature in REPRESENTATION_FEATURES:
            minimum = float(
                group[feature].min()
            )

            maximum = float(
                group[feature].max()
            )

            records.append(
                {
                    "team_id": int(
                        team_id
                    ),
                    "team": str(
                        team
                    ),
                    "feature": feature,
                    "minimum_value":
                        minimum,
                    "maximum_value":
                        maximum,
                    "feature_range":
                        maximum
                        - minimum,
                    "rating_only_value":
                        float(
                            group.loc[
                                group[
                                    "specification"
                                ].eq(
                                    RATING_ONLY_SPECIFICATION
                                ),
                                feature,
                            ].iloc[0]
                        ),
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "feature_range",
                "team",
                "feature",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_metadata(
    *,
    representations: pd.DataFrame,
    lineup_representations: pd.DataFrame,
) -> dict[str, object]:
    return {
        "study_id": "087C",
        "study_name": (
            "Team Representation Sensitivity Analysis"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "competition": "Bundesliga",
        "season": "2024/25",
        "club_count": int(
            representations[
                "team_id"
            ].nunique()
        ),
        "selection_specification_count": int(
            lineup_representations[
                "specification"
            ].nunique()
        ),
        "lineup_representation_count": len(
            lineup_representations
        ),
        "full_squad_reconstruction_count": int(
            representations[
                "specification"
            ].eq(
                FULL_SQUAD_SPECIFICATION
            )
            .sum()
        ),
        "representation_features": list(
            REPRESENTATION_FEATURES
        ),
        "aggregation_profile": (
            "legacy_top_5-compatible diagnostic aggregation"
        ),
        "full_squad_control_status": (
            "Study-local reconstruction from the same joined "
            "Bundesliga player population; not promoted as the "
            "authoritative production repository."
        ),
        "prediction_date_valid": False,
        "goal_model_fitted": False,
        "production_repository_changed": False,
        "production_runtime_changed": False,
        "interpretation_boundary": (
            "This study measures representation sensitivity to "
            "season-level selection policies. It does not assess "
            "fixture-specific lineup accuracy or downstream "
            "predictive performance."
        ),
        "outputs": [
            REPRESENTATION_PATH.name,
            DELTA_PATH.name,
            FULL_SQUAD_COMPARISON_PATH.name,
            SPECIFICATION_SUMMARY_PATH.name,
            CLUB_SENSITIVITY_PATH.name,
            FEATURE_RANGE_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
            NON_FINITE_ROLE_RATINGS_PATH.name,
        ],
    }


def write_report(
    *,
    delta: pd.DataFrame,
    specification_summary: pd.DataFrame,
    club_sensitivity: pd.DataFrame,
    full_squad_comparison: pd.DataFrame,
) -> None:
    non_baseline = (
        specification_summary.loc[
            ~specification_summary[
                "specification"
            ].eq(
                RATING_ONLY_SPECIFICATION
            )
        ]
    )

    most_stable = (
        non_baseline.sort_values(
            "mean_absolute_feature_delta"
        )
        .iloc[0]
    )

    most_disruptive = (
        non_baseline.sort_values(
            "mean_absolute_feature_delta",
            ascending=False,
        )
        .iloc[0]
    )

    most_sensitive_club = (
        club_sensitivity.iloc[0]
    )

    least_sensitive_club = (
        club_sensitivity.iloc[-1]
    )

    rating_only_full_squad = (
        full_squad_comparison.loc[
            full_squad_comparison[
                "specification"
            ].eq(
                RATING_ONLY_SPECIFICATION
            )
        ]
    )

    report = f"""# Study 087C — Team Representation Sensitivity Analysis

## Purpose

Measure how much Bundesliga team-strength representations change
when season-level usage evidence alters the projected starting XI.

## Methodological boundary

This study:

- consumes frozen Study 087B lineups;
- changes no player-selection policy;
- uses the existing role-projection functions;
- mirrors the legacy top-five aggregation profile;
- fits no goal model;
- changes no production artifact.

The selected lineups are retrospective and are not prediction-date
valid.

## Population

- Clubs: {EXPECTED_CLUB_COUNT}
- Selection specifications:
  {delta["specification"].nunique()}
- Projected-XI representations:
  {len(delta)}
- Representation features:
  {len(REPRESENTATION_FEATURES)}

## Most stable non-baseline specification

- Specification:
  `{most_stable["specification"]}`
- Mean absolute feature delta:
  {most_stable["mean_absolute_feature_delta"]:.8f}
- Mean Euclidean feature distance:
  {most_stable["mean_euclidean_feature_distance"]:.8f}
- Unchanged club representations:
  {int(most_stable["unchanged_representation_count"])}

## Most disruptive specification

- Specification:
  `{most_disruptive["specification"]}`
- Mean absolute feature delta:
  {most_disruptive["mean_absolute_feature_delta"]:.8f}
- Maximum club mean absolute feature delta:
  {most_disruptive["maximum_club_mean_absolute_feature_delta"]:.8f}
- Mean Euclidean feature distance:
  {most_disruptive["mean_euclidean_feature_distance"]:.8f}

## Club sensitivity

Most sensitive club:

- Club:
  `{most_sensitive_club["team"]}`
- Mean Euclidean feature distance:
  {most_sensitive_club["mean_euclidean_feature_distance"]:.8f}
- Maximum Euclidean feature distance:
  {most_sensitive_club["maximum_euclidean_feature_distance"]:.8f}
- Most disruptive specification:
  `{most_sensitive_club["most_disruptive_specification"]}`

Least sensitive club:

- Club:
  `{least_sensitive_club["team"]}`
- Mean Euclidean feature distance:
  {least_sensitive_club["mean_euclidean_feature_distance"]:.8f}
- Maximum Euclidean feature distance:
  {least_sensitive_club["maximum_euclidean_feature_distance"]:.8f}

## Rating-only XI versus study-local full squad

Mean absolute representation difference:

{rating_only_full_squad["mean_absolute_delta_from_full_squad"].mean():.8f}

This full-squad control is reconstructed from the same joined
Bundesliga player population. It is diagnostic and is not the
authoritative production repository.

## Interpretation

If usage-informed policies produce only small representation
changes, then the selection heuristic is unlikely to be the primary
source of the remaining prediction error.

Larger representation changes would justify a downstream goal-model
benchmark before introducing more complex matchday information.

## Result

**OVERALL RESULT: PASS**

Representation sensitivity was measured without refitting or
modifying the production model.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 087C — TEAM REPRESENTATION "
        "SENSITIVITY ANALYSIS"
    )
    print("=" * 88)

    players = load_csv(
        JOINED_PLAYER_PATH,
        dataset_name=(
            "Bundesliga usage-rating join"
        ),
        required_columns=(
            REQUIRED_PLAYER_COLUMNS
        ),
    )

    lineups = load_csv(
        LINEUP_PATH,
        dataset_name=(
            "Study 087B policy lineups"
        ),
        required_columns=(
            REQUIRED_LINEUP_COLUMNS
        ),
    )

    players = prepare_players(
        players
    )

    non_finite_role_rating_audit = (
        build_non_finite_role_rating_audit(
            players
        )
    )

    lineups = prepare_lineups(
        lineups
    )

    players = add_player_projections(
        players
    )

    full_squad_representations = (
        build_full_squad_representations(
            players
        )
    )

    lineup_representations = (
        build_lineup_representations(
            players=players,
            lineups=lineups,
        )
    )

    representations = pd.concat(
        [
            full_squad_representations,
            lineup_representations,
        ],
        ignore_index=True,
    )

    delta = build_delta_from_rating_only(
        lineup_representations
    )

    full_squad_comparison = (
        build_full_squad_comparison(
            lineup_representations=(
                lineup_representations
            ),
            full_squad_representations=(
                full_squad_representations
            ),
        )
    )

    specification_summary = (
        build_specification_summary(
            delta
        )
    )

    club_sensitivity = (
        build_club_sensitivity(
            delta
        )
    )

    feature_ranges = build_feature_ranges(
        representations
    )

    metadata = build_metadata(
        representations=representations,
        lineup_representations=(
            lineup_representations
        ),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    representations.to_csv(
        REPRESENTATION_PATH,
        index=False,
    )

    delta.to_csv(
        DELTA_PATH,
        index=False,
    )

    full_squad_comparison.to_csv(
        FULL_SQUAD_COMPARISON_PATH,
        index=False,
    )

    specification_summary.to_csv(
        SPECIFICATION_SUMMARY_PATH,
        index=False,
    )

    club_sensitivity.to_csv(
        CLUB_SENSITIVITY_PATH,
        index=False,
    )

    feature_ranges.to_csv(
        FEATURE_RANGE_PATH,
        index=False,
    )

    non_finite_role_rating_audit.to_csv(
        NON_FINITE_ROLE_RATINGS_PATH,
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
        delta=delta,
        specification_summary=(
            specification_summary
        ),
        club_sensitivity=(
            club_sensitivity
        ),
        full_squad_comparison=(
            full_squad_comparison
        ),
    )

    print()
    print("Population")
    print("-" * 88)
    print(
        "  Selection specifications: "
        f"{lineup_representations['specification'].nunique()}"
    )
    print(
        "  Projected-XI representations: "
        f"{len(lineup_representations)}"
    )
    print(
        "  Full-squad reconstructions: "
        f"{len(full_squad_representations)}"
    )

    print()
    print("Specification sensitivity")
    print("-" * 88)

    display_columns = [
        "specification",
        "mean_absolute_feature_delta",
        "maximum_club_mean_absolute_feature_delta",
        "mean_euclidean_feature_distance",
        "maximum_euclidean_feature_distance",
        "unchanged_representation_count",
        "mean_attack_delta",
        "mean_defense_delta",
        "mean_attack_depth_delta",
    ]

    print(
        specification_summary[
            display_columns
        ]
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )

    print()
    print("Most representation-sensitive clubs")
    print("-" * 88)

    print(
        club_sensitivity.head(10).to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Player artifact loading: PASS")
    print("  Lineup artifact loading: PASS")
    print("  Role projection: PASS")
    print("  Full-squad reconstruction: PASS")
    print("  Projected-XI aggregation: PASS")
    print("  Rating-only delta baseline: PASS")
    print("  Club sensitivity analysis: PASS")
    print("  Full-squad comparison: PASS")
    print("  Goal-model fitting: NONE")
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
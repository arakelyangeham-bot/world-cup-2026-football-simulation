#benchmark_player_selection_policies

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.player_intelligence.player_selection_scoring import (
    RATING_ONLY_SPECIFICATION,
    PlayerSelectionSpecification,
    generate_weight_grid,
    rank_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

JOINED_PLAYER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_087_usage_informed_expected_xi"
    / "bundesliga_usage_rating_join.csv"
)

FORMATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "formation_manifest.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_087_usage_informed_expected_xi"
)

SPECIFICATION_PATH = (
    OUTPUT_DIRECTORY
    / "selection_specifications.csv"
)

LINEUP_PATH = (
    OUTPUT_DIRECTORY
    / "player_selection_policy_lineups.csv"
)

CLUB_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "club_lineup_comparison.csv"
)

SPECIFICATION_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "selection_specification_summary.csv"
)

PLAYER_CHANGE_PATH = (
    OUTPUT_DIRECTORY
    / "player_selection_changes.csv"
)

RANKING_STABILITY_PATH = (
    OUTPUT_DIRECTORY
    / "ranking_stability_summary.csv"
)

FORMATION_COVERAGE_PATH = (
    OUTPUT_DIRECTORY
    / "formation_slot_coverage.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_087b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_087b_report.md"
)

ROLE_FALLBACK_PATH = (
    OUTPUT_DIRECTORY
    / "role_fallback_usage.csv"
)

FORMATION = "4-3-3"
EXPECTED_CLUB_COUNT = 18
EXPECTED_LINEUP_SIZE = 11

ROLE_COLUMNS = {
    "GK": "rating_GK",
    "CB": "rating_CB",
    "FB": "rating_FB",
    "DM": "rating_DM",
    "CM": "rating_CM",
    "AM": "rating_AM",
    "WM": "rating_WM",
    "W": "rating_W",
    "ST": "rating_ST",
}

ROLE_COMPATIBILITY = {
    "GK": (
        ("GK", 1.00),
    ),
    "CB": (
        ("CB", 1.00),
        ("DM", 0.95),
        ("FB", 0.90),
    ),
    "FB": (
        ("FB", 1.00),
        ("WM", 0.95),
        ("CB", 0.90),
    ),
    "DM": (
        ("DM", 1.00),
        ("CM", 0.97),
        ("CB", 0.92),
    ),
    "CM": (
        ("CM", 1.00),
        ("DM", 0.97),
        ("AM", 0.95),
        ("WM", 0.92),
    ),
    "AM": (
        ("AM", 1.00),
        ("CM", 0.95),
        ("W", 0.95),
        ("WM", 0.93),
        ("ST", 0.90),
    ),
    "WM": (
        ("WM", 1.00),
        ("W", 0.98),
        ("AM", 0.95),
        ("CM", 0.92),
        ("FB", 0.90),
    ),
    "W": (
        ("W", 1.00),
        ("WM", 0.98),
        ("AM", 0.94),
        ("ST", 0.90),
    ),
    "ST": (
        ("ST", 1.00),
        ("W", 0.94),
        ("AM", 0.90),
    ),
}

EFFECTIVE_ROLE_RATING_COLUMN = (
    "effective_role_rating"
)

REQUIRED_PLAYER_COLUMNS = {
    "competition_id",
    "season_id",
    "team_id",
    "team",
    "player_id",
    "player",
    "start_rate",
    "minutes_relative_to_club_max",
    "usage_evidence_score",
    "lineup_eligible",
    *ROLE_COLUMNS.values(),
}

REQUIRED_FORMATION_COLUMNS = {
    "formation",
    "slot",
    "role",
}


def load_joined_players() -> pd.DataFrame:
    if not JOINED_PLAYER_PATH.exists():
        raise FileNotFoundError(
            "Study 087B0 joined-player artifact does not exist: "
            f"{JOINED_PLAYER_PATH}"
        )

    dataframe = pd.read_csv(
        JOINED_PLAYER_PATH,
        dtype={
            "player_id": str,
        },
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Study 087B0 joined-player artifact is empty."
        )

    missing = REQUIRED_PLAYER_COLUMNS - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Joined-player artifact is missing required columns: "
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

    dataframe["lineup_eligible"] = (
        dataframe["lineup_eligible"]
        .astype(bool)
    )

    numeric_columns = [
        "team_id",
        "start_rate",
        "minutes_relative_to_club_max",
        "usage_evidence_score",
        *ROLE_COLUMNS.values(),
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    bounded_columns = [
        "start_rate",
        "minutes_relative_to_club_max",
        "usage_evidence_score",
    ]

    for column in bounded_columns:
        values = dataframe[column]

        if values.isna().any():
            raise ValueError(
                f"Joined-player artifact contains missing "
                f"{column!r} values."
            )

        if (
            values.lt(0.0).any()
            or values.gt(1.0).any()
        ):
            raise ValueError(
                f"Joined-player artifact contains {column!r} "
                "values outside [0, 1]."
            )

    if dataframe["player_id"].duplicated().any():
        raise ValueError(
            "Joined-player artifact contains duplicate player IDs."
        )

    club_count = dataframe["team_id"].nunique()

    if club_count != EXPECTED_CLUB_COUNT:
        raise ValueError(
            "Unexpected club count. "
            f"Expected {EXPECTED_CLUB_COUNT}, "
            f"received {club_count}."
        )

    return dataframe


def load_formation() -> pd.DataFrame:
    if not FORMATION_PATH.exists():
        raise FileNotFoundError(
            "Formation manifest does not exist: "
            f"{FORMATION_PATH}"
        )

    dataframe = pd.read_csv(
        FORMATION_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Formation manifest is empty."
        )

    missing = REQUIRED_FORMATION_COLUMNS - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Formation manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    selected = dataframe.loc[
        dataframe["formation"]
        .astype(str)
        .eq(FORMATION)
    ].copy()

    if selected.empty:
        raise KeyError(
            f"Formation {FORMATION!r} was not found."
        )

    selected["slot"] = (
        selected["slot"]
        .astype(str)
        .str.strip()
    )

    selected["role"] = (
        selected["role"]
        .astype(str)
        .str.strip()
    )

    if selected["slot"].duplicated().any():
        raise ValueError(
            f"Formation {FORMATION!r} contains duplicate slots."
        )

    unknown_roles = (
        set(selected["role"])
        - set(ROLE_COLUMNS)
    )

    if unknown_roles:
        raise ValueError(
            "Formation contains unsupported roles: "
            f"{sorted(unknown_roles)}"
        )

    if len(selected) != EXPECTED_LINEUP_SIZE:
        raise ValueError(
            "Unexpected formation slot count. "
            f"Expected {EXPECTED_LINEUP_SIZE}, "
            f"received {len(selected)}."
        )

    return selected.reset_index(drop=True)


def build_specifications(
) -> tuple[PlayerSelectionSpecification, ...]:
    generated = generate_weight_grid(
        role_weights=(
            0.50,
            0.60,
            0.70,
            0.80,
        ),
        start_weights=(
            0.00,
            0.10,
            0.20,
            0.30,
        ),
    )

    specifications = [
        RATING_ONLY_SPECIFICATION,
        *generated,
    ]

    unique: dict[
        tuple[float, float, float],
        PlayerSelectionSpecification,
    ] = {}

    for specification in specifications:
        specification.validate()

        key = (
            round(
                specification.role_rating_weight,
                12,
            ),
            round(
                specification.start_rate_weight,
                12,
            ),
            round(
                specification.minutes_weight,
                12,
            ),
        )

        unique.setdefault(
            key,
            specification,
        )

    return tuple(
        unique.values()
    )

def build_compatible_role_candidates(
    *,
    players: pd.DataFrame,
    target_role: str,
) -> pd.DataFrame:
    """
    Build candidates for one formation role using a fixed
    role-compatibility hierarchy.

    Each player receives the greatest suitability-adjusted rating
    available across the compatible source roles.

    Exact-role ratings always retain their full value. Fallback
    ratings receive a transparent suitability multiplier.
    """

    try:
        compatibility = ROLE_COMPATIBILITY[
            target_role
        ]
    except KeyError as error:
        raise KeyError(
            "No compatibility definition exists for "
            f"target role {target_role!r}."
        ) from error

    working = players.copy()

    adjusted_columns: list[str] = []

    source_role_columns: list[str] = []

    for source_role, suitability in compatibility:
        source_column = ROLE_COLUMNS[
            source_role
        ]

        adjusted_column = (
            f"_adjusted_{target_role}_{source_role}"
        )

        source_role_column = (
            f"_source_{target_role}_{source_role}"
        )

        numeric_rating = pd.to_numeric(
            working[source_column],
            errors="coerce",
        )

        working[adjusted_column] = (
            numeric_rating
            * suitability
        )

        working[source_role_column] = np.where(
            numeric_rating.notna(),
            source_role,
            None,
        )

        adjusted_columns.append(
            adjusted_column
        )

        source_role_columns.append(
            source_role_column
        )

    working[
        EFFECTIVE_ROLE_RATING_COLUMN
    ] = working[
        adjusted_columns
    ].max(
        axis=1,
        skipna=True,
    )

    def resolve_source_role(
        row: pd.Series,
    ) -> str | None:
        effective_rating = row[
            EFFECTIVE_ROLE_RATING_COLUMN
        ]

        if pd.isna(effective_rating):
            return None

        for (
            source_role,
            suitability,
        ), adjusted_column in zip(
            compatibility,
            adjusted_columns,
        ):
            candidate_value = row[
                adjusted_column
            ]

            if (
                pd.notna(candidate_value)
                and math.isclose(
                    float(candidate_value),
                    float(effective_rating),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                return source_role

        raise AssertionError(
            "Could not resolve the effective role-rating source."
        )

    working[
        "selection_source_role"
    ] = working.apply(
        resolve_source_role,
        axis=1,
    )

    suitability_lookup = {
        source_role: suitability
        for source_role, suitability
        in compatibility
    }

    working[
        "selection_role_suitability"
    ] = (
        working[
            "selection_source_role"
        ]
        .map(
            suitability_lookup
        )
    )

    output = working.loc[
        working[
            EFFECTIVE_ROLE_RATING_COLUMN
        ].notna()
    ].copy()

    return output.drop(
        columns=[
            *adjusted_columns,
            *source_role_columns,
        ]
    )

def build_lineup_for_club(
    *,
    club_players: pd.DataFrame,
    formation: pd.DataFrame,
    specification: PlayerSelectionSpecification,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Greedily fill the fixed formation using role-local rankings.

    This deliberately preserves the current StartingXIBuilder
    behavior:

    - slots are filled in formation-manifest order;
    - players cannot be selected twice;
    - the highest-ranked remaining role-eligible player is selected.
    """

    selected_player_ids: set[str] = set()

    lineup_records: list[dict[str, object]] = []
    coverage_records: list[dict[str, object]] = []

    for slot_order, slot_row in formation.iterrows():
        slot = str(
            slot_row["slot"]
        )

        role = str(
            slot_row["role"]
        )

        role_rating_column = ROLE_COLUMNS[
            role
        ]

        unselected = club_players.loc[
            ~club_players[
                "player_id"
            ].isin(
                selected_player_ids
            )
        ].copy()

        candidates = (
            build_compatible_role_candidates(
                players=unselected,
                target_role=role,
            )
        )

        coverage_records.append(
            {
                "specification":
                    specification.name,
                "team_id": int(
                    club_players[
                        "team_id"
                    ].iloc[0]
                ),
                "team": str(
                    club_players[
                        "team"
                    ].iloc[0]
                ),
                "slot_order": int(
                    slot_order
                ),
                "slot": slot,
                "role": role,
                "role_rating_column":
                    role_rating_column,
                "available_candidate_count": int(
                    len(candidates)
                ),
                "already_selected_player_count": int(
                    len(selected_player_ids)
                ),
                "slot_fill_pass": (
                    not candidates.empty
                ),
                "exact_role_candidate_count": int(
                    unselected[
                        role_rating_column
                    ].notna().sum()
                ),
                "compatible_role_candidate_count": int(
                    len(candidates)
                ),
                "fallback_required": bool(
                    unselected[
                        role_rating_column
                    ].notna().sum()
                    == 0
                ),
            }
        )

        if candidates.empty:
            raise ValueError(
                "Could not fill formation slot. "
                f"Club={club_players['team'].iloc[0]!r}, "
                f"specification={specification.name!r}, "
                f"slot={slot!r}, role={role!r}."
            )

        ranked = rank_candidates(
            candidates,
            role_rating_column=(
                EFFECTIVE_ROLE_RATING_COLUMN
            ),
            specification=specification,
        )

        chosen = ranked.iloc[0]

        player_id = str(
            chosen["player_id"]
        )

        if player_id in selected_player_ids:
            raise AssertionError(
                "Lineup construction selected a duplicate player."
            )

        selected_player_ids.add(
            player_id
        )

        lineup_records.append(
            {
                "specification":
                    specification.name,
                "role_rating_weight":
                    specification.role_rating_weight,
                "start_rate_weight":
                    specification.start_rate_weight,
                "minutes_weight":
                    specification.minutes_weight,
                "team_id": int(
                    chosen["team_id"]
                ),
                "team": str(
                    chosen["team"]
                ),
                "formation": FORMATION,
                "slot_order": int(
                    slot_order
                ),
                "slot": slot,
                "role": role,
                "role_rating_column":
                    role_rating_column,
                "player_id": player_id,
                "player": str(
                    chosen["player"]
                ),
                "raw_role_rating": float(
                    chosen[
                        EFFECTIVE_ROLE_RATING_COLUMN
                    ]
                ),
                "selection_source_role": str(
                    chosen[
                        "selection_source_role"
                    ]
                ),
                "selection_role_suitability": float(
                    chosen[
                        "selection_role_suitability"
                    ]
                ),
                "exact_role_match": bool(
                    chosen[
                        "selection_source_role"
                    ]
                    == role
                ),
                "normalized_role_rating":
                    float(
                        chosen[
                            "normalized_role_rating"
                        ]
                    ),
                "start_rate": float(
                    chosen[
                        "start_rate"
                    ]
                ),
                "minutes_relative_to_club_max":
                    float(
                        chosen[
                            "minutes_relative_to_club_max"
                        ]
                    ),
                "usage_evidence_score":
                    float(
                        chosen[
                            "usage_evidence_score"
                        ]
                    ),
                "selection_score": float(
                    chosen[
                        "selection_score"
                    ]
                ),
                "role_rating_component":
                    float(
                        chosen[
                            "role_rating_component"
                        ]
                    ),
                "start_rate_component":
                    float(
                        chosen[
                            "start_rate_component"
                        ]
                    ),
                "minutes_component":
                    float(
                        chosen[
                            "minutes_component"
                        ]
                    ),
                "role_candidate_count": int(
                    len(ranked)
                ),
            }
        )

    lineup = pd.DataFrame(
        lineup_records
    )

    coverage = pd.DataFrame(
        coverage_records
    )

    validate_lineup(
        lineup,
        expected_team=str(
            club_players["team"].iloc[0]
        ),
        expected_specification=(
            specification.name
        ),
    )

    return lineup, coverage


def validate_lineup(
    lineup: pd.DataFrame,
    *,
    expected_team: str,
    expected_specification: str,
) -> None:
    if len(lineup) != EXPECTED_LINEUP_SIZE:
        raise AssertionError(
            "Lineup does not contain exactly 11 players. "
            f"Team={expected_team!r}, "
            f"specification={expected_specification!r}, "
            f"count={len(lineup)}."
        )

    if lineup["player_id"].duplicated().any():
        raise AssertionError(
            "Lineup contains duplicate players. "
            f"Team={expected_team!r}, "
            f"specification={expected_specification!r}."
        )

    if lineup["slot"].duplicated().any():
        raise AssertionError(
            "Lineup contains duplicate formation slots."
        )

    numeric_columns = [
        "raw_role_rating",
        "normalized_role_rating",
        "start_rate",
        "minutes_relative_to_club_max",
        "usage_evidence_score",
        "selection_score",
    ]

    values = lineup[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise AssertionError(
            "Lineup contains non-finite scoring values."
        )


def build_all_lineups(
    *,
    players: pd.DataFrame,
    formation: pd.DataFrame,
    specifications: tuple[
        PlayerSelectionSpecification,
        ...,
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lineup_frames: list[
        pd.DataFrame
    ] = []

    coverage_frames: list[
        pd.DataFrame
    ] = []

    eligible_players = players.loc[
        players["lineup_eligible"]
    ].copy()

    for specification in specifications:
        for (
            team_id,
            team,
        ), club_players in eligible_players.groupby(
            [
                "team_id",
                "team",
            ],
            sort=True,
        ):
            lineup, coverage = (
                build_lineup_for_club(
                    club_players=(
                        club_players
                    ),
                    formation=formation,
                    specification=(
                        specification
                    ),
                )
            )

            lineup_frames.append(
                lineup
            )

            coverage_frames.append(
                coverage
            )

    lineups = pd.concat(
        lineup_frames,
        ignore_index=True,
    )

    coverage = pd.concat(
        coverage_frames,
        ignore_index=True,
    )

    expected_rows = (
        len(specifications)
        * EXPECTED_CLUB_COUNT
        * EXPECTED_LINEUP_SIZE
    )

    if len(lineups) != expected_rows:
        raise AssertionError(
            "Unexpected total lineup-row count. "
            f"Expected {expected_rows}, "
            f"received {len(lineups)}."
        )

    return lineups, coverage


def build_club_comparison(
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    baseline = lineups.loc[
        lineups[
            "specification"
        ].eq(
            RATING_ONLY_SPECIFICATION.name
        )
    ].copy()

    baseline_sets = {
        (
            int(team_id),
            str(team),
        ): set(
            group[
                "player_id"
            ].astype(str)
        )
        for (
            team_id,
            team,
        ), group in baseline.groupby(
            [
                "team_id",
                "team",
            ],
            sort=True,
        )
    }

    baseline_slot_players = {
        (
            int(row.team_id),
            str(row.team),
            str(row.slot),
        ): str(
            row.player_id
        )
        for row in baseline.itertuples(
            index=False
        )
    }

    baseline_goalkeepers = {
        (
            int(row.team_id),
            str(row.team),
        ): str(
            row.player_id
        )
        for row in baseline.loc[
            baseline["role"].eq("GK")
        ].itertuples(
            index=False
        )
    }

    records: list[
        dict[str, object]
    ] = []

    for (
        specification,
        team_id,
        team,
    ), group in lineups.groupby(
        [
            "specification",
            "team_id",
            "team",
        ],
        sort=True,
    ):
        selected_set = set(
            group[
                "player_id"
            ].astype(str)
        )

        baseline_set = baseline_sets[
            (
                int(team_id),
                str(team),
            )
        ]

        overlap = len(
            selected_set
            & baseline_set
        )

        added_players = sorted(
            selected_set
            - baseline_set
        )

        removed_players = sorted(
            baseline_set
            - selected_set
        )

        same_slot_count = 0

        for row in group.itertuples(
            index=False
        ):
            baseline_player = (
                baseline_slot_players[
                    (
                        int(team_id),
                        str(team),
                        str(row.slot),
                    )
                ]
            )

            if str(row.player_id) == (
                baseline_player
            ):
                same_slot_count += 1

        goalkeeper_row = (
            group.loc[
                group["role"].eq("GK")
            ]
            .iloc[0]
        )

        goalkeeper_changed = (
            str(
                goalkeeper_row[
                    "player_id"
                ]
            )
            != baseline_goalkeepers[
                (
                    int(team_id),
                    str(team),
                )
            ]
        )

        records.append(
            {
                "specification":
                    specification,
                "team_id": int(
                    team_id
                ),
                "team": str(
                    team
                ),
                "lineup_size": len(
                    group
                ),
                "baseline_player_overlap_count":
                    overlap,
                "baseline_player_overlap_share":
                    overlap
                    / EXPECTED_LINEUP_SIZE,
                "players_changed_from_baseline":
                    EXPECTED_LINEUP_SIZE
                    - overlap,
                "same_slot_player_count":
                    same_slot_count,
                "same_slot_player_share":
                    same_slot_count
                    / EXPECTED_LINEUP_SIZE,
                "goalkeeper_changed":
                    goalkeeper_changed,
                "added_player_ids":
                    ", ".join(
                        added_players
                    ),
                "removed_player_ids":
                    ", ".join(
                        removed_players
                    ),
                "mean_raw_role_rating":
                    float(
                        group[
                            "raw_role_rating"
                        ].mean()
                    ),
                "mean_normalized_role_rating":
                    float(
                        group[
                            "normalized_role_rating"
                        ].mean()
                    ),
                "mean_start_rate":
                    float(
                        group[
                            "start_rate"
                        ].mean()
                    ),
                "mean_relative_minutes":
                    float(
                        group[
                            "minutes_relative_to_club_max"
                        ].mean()
                    ),
                "mean_usage_evidence_score":
                    float(
                        group[
                            "usage_evidence_score"
                        ].mean()
                    ),
                "mean_selection_score":
                    float(
                        group[
                            "selection_score"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def build_specification_summary(
    club_comparison: pd.DataFrame,
    specifications: tuple[
        PlayerSelectionSpecification,
        ...,
    ],
) -> pd.DataFrame:
    weight_lookup = {
        specification.name:
            specification
        for specification in specifications
    }

    records: list[
        dict[str, object]
    ] = []

    for specification, group in (
        club_comparison.groupby(
            "specification",
            sort=True,
        )
    ):
        weights = weight_lookup[
            specification
        ]

        records.append(
            {
                "specification":
                    specification,
                "role_rating_weight":
                    weights.role_rating_weight,
                "start_rate_weight":
                    weights.start_rate_weight,
                "minutes_weight":
                    weights.minutes_weight,
                "club_count": len(
                    group
                ),
                "mean_player_overlap_share":
                    float(
                        group[
                            "baseline_player_overlap_share"
                        ].mean()
                    ),
                "minimum_player_overlap_share":
                    float(
                        group[
                            "baseline_player_overlap_share"
                        ].min()
                    ),
                "mean_players_changed":
                    float(
                        group[
                            "players_changed_from_baseline"
                        ].mean()
                    ),
                "maximum_players_changed":
                    int(
                        group[
                            "players_changed_from_baseline"
                        ].max()
                    ),
                "unchanged_club_count":
                    int(
                        group[
                            "players_changed_from_baseline"
                        ].eq(0).sum()
                    ),
                "clubs_with_three_or_more_changes":
                    int(
                        group[
                            "players_changed_from_baseline"
                        ].ge(3).sum()
                    ),
                "goalkeeper_change_count":
                    int(
                        group[
                            "goalkeeper_changed"
                        ].sum()
                    ),
                "mean_same_slot_share":
                    float(
                        group[
                            "same_slot_player_share"
                        ].mean()
                    ),
                "mean_raw_role_rating":
                    float(
                        group[
                            "mean_raw_role_rating"
                        ].mean()
                    ),
                "mean_start_rate":
                    float(
                        group[
                            "mean_start_rate"
                        ].mean()
                    ),
                "mean_relative_minutes":
                    float(
                        group[
                            "mean_relative_minutes"
                        ].mean()
                    ),
                "mean_usage_evidence_score":
                    float(
                        group[
                            "mean_usage_evidence_score"
                        ].mean()
                    ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "role_rating_weight",
                "start_rate_weight",
                "minutes_weight",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_player_changes(
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    baseline = lineups.loc[
        lineups[
            "specification"
        ].eq(
            RATING_ONLY_SPECIFICATION.name
        ),
        [
            "team_id",
            "team",
            "slot",
            "role",
            "player_id",
            "player",
            "raw_role_rating",
            "start_rate",
            "minutes_relative_to_club_max",
            "usage_evidence_score",
        ],
    ].copy()

    baseline = baseline.rename(
        columns={
            "player_id":
                "baseline_player_id",
            "player":
                "baseline_player",
            "raw_role_rating":
                "baseline_role_rating",
            "start_rate":
                "baseline_start_rate",
            "minutes_relative_to_club_max":
                "baseline_relative_minutes",
            "usage_evidence_score":
                "baseline_usage_evidence_score",
        }
    )

    candidates = lineups.loc[
        ~lineups[
            "specification"
        ].eq(
            RATING_ONLY_SPECIFICATION.name
        )
    ].copy()

    comparison = candidates.merge(
        baseline,
        on=[
            "team_id",
            "team",
            "slot",
            "role",
        ],
        how="left",
        validate="many_to_one",
    )

    comparison["slot_player_changed"] = (
        comparison["player_id"]
        .astype(str)
        .ne(
            comparison[
                "baseline_player_id"
            ].astype(str)
        )
    )

    comparison[
        "role_rating_difference"
    ] = (
        comparison[
            "raw_role_rating"
        ]
        - comparison[
            "baseline_role_rating"
        ]
    )

    comparison[
        "start_rate_difference"
    ] = (
        comparison[
            "start_rate"
        ]
        - comparison[
            "baseline_start_rate"
        ]
    )

    comparison[
        "relative_minutes_difference"
    ] = (
        comparison[
            "minutes_relative_to_club_max"
        ]
        - comparison[
            "baseline_relative_minutes"
        ]
    )

    comparison[
        "usage_evidence_difference"
    ] = (
        comparison[
            "usage_evidence_score"
        ]
        - comparison[
            "baseline_usage_evidence_score"
        ]
    )

    return (
        comparison.loc[
            comparison[
                "slot_player_changed"
            ]
        ]
        .sort_values(
            [
                "specification",
                "team",
                "slot_order",
            ]
        )
        .reset_index(drop=True)
    )


def build_ranking_stability(
    *,
    players: pd.DataFrame,
    specifications: tuple[
        PlayerSelectionSpecification,
        ...,
    ],
) -> pd.DataFrame:
    """
    Compare role-level top candidate rankings before formation
    constraints are applied.
    """

    eligible_players = players.loc[
        players["lineup_eligible"]
    ].copy()

    baseline_top: dict[
        tuple[int, str, str],
        str,
    ] = {}

    for (
        team_id,
        team,
    ), club_players in eligible_players.groupby(
        [
            "team_id",
            "team",
        ],
        sort=True,
    ):
        for role, rating_column in (
            ROLE_COLUMNS.items()
        ):
            role_candidates = club_players.loc[
                club_players[
                    rating_column
                ].notna()
            ].copy()

            if role_candidates.empty:
                continue

            ranked = rank_candidates(
                role_candidates,
                role_rating_column=(
                    rating_column
                ),
                specification=(
                    RATING_ONLY_SPECIFICATION
                ),
            )

            baseline_top[
                (
                    int(team_id),
                    str(team),
                    role,
                )
            ] = str(
                ranked.iloc[0][
                    "player_id"
                ]
            )

    records: list[
        dict[str, object]
    ] = []

    for specification in specifications:
        top_candidate_matches: list[
            bool
        ] = []

        evaluated_roles = 0

        for (
            team_id,
            team,
        ), club_players in eligible_players.groupby(
            [
                "team_id",
                "team",
            ],
            sort=True,
        ):
            for role, rating_column in (
                ROLE_COLUMNS.items()
            ):
                role_candidates = (
                    club_players.loc[
                        club_players[
                            rating_column
                        ].notna()
                    ].copy()
                )

                if role_candidates.empty:
                    continue

                ranked = rank_candidates(
                    role_candidates,
                    role_rating_column=(
                        rating_column
                    ),
                    specification=(
                        specification
                    ),
                )

                key = (
                    int(team_id),
                    str(team),
                    role,
                )

                top_candidate_matches.append(
                    str(
                        ranked.iloc[0][
                            "player_id"
                        ]
                    )
                    == baseline_top[key]
                )

                evaluated_roles += 1

        records.append(
            {
                "specification":
                    specification.name,
                "role_rating_weight":
                    specification.role_rating_weight,
                "start_rate_weight":
                    specification.start_rate_weight,
                "minutes_weight":
                    specification.minutes_weight,
                "club_role_populations_evaluated":
                    evaluated_roles,
                "baseline_top_candidate_match_count":
                    int(
                        sum(
                            top_candidate_matches
                        )
                    ),
                "baseline_top_candidate_match_share":
                    float(
                        np.mean(
                            top_candidate_matches
                        )
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def build_metadata(
    *,
    specifications: tuple[
        PlayerSelectionSpecification,
        ...,
    ],
    lineups: pd.DataFrame,
    player_changes: pd.DataFrame,
) -> dict[str, object]:
    return {
        "study_id": "087B",
        "study_name": (
            "Player Selection Policy Benchmark"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "competition": "Bundesliga",
        "season": "2024/25",
        "formation": FORMATION,
        "club_count": int(
            lineups["team_id"].nunique()
        ),
        "specification_count": len(
            specifications
        ),
        "lineup_row_count": len(
            lineups
        ),
        "changed_slot_row_count": len(
            player_changes
        ),
        "baseline_specification": (
            RATING_ONLY_SPECIFICATION.name
        ),
        "usage_source": str(
            JOINED_PLAYER_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "formation_source": str(
            FORMATION_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "selection_algorithm": (
            "Greedy formation-slot filling using role-local "
            "candidate rankings and no duplicate players."
        ),
        "lineups_prediction_date_valid": False,
        "representations_built": False,
        "goal_model_changed": False,
        "production_changed": False,
        "interpretation_boundary": (
            "Usage variables are full-season aggregates. "
            "This study characterizes retrospective lineup "
            "sensitivity and does not estimate historical "
            "pre-match lineup accuracy."
        ),
        "outputs": [
            SPECIFICATION_PATH.name,
            LINEUP_PATH.name,
            CLUB_COMPARISON_PATH.name,
            SPECIFICATION_SUMMARY_PATH.name,
            PLAYER_CHANGE_PATH.name,
            RANKING_STABILITY_PATH.name,
            FORMATION_COVERAGE_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
            ROLE_FALLBACK_PATH.name,
        ],
    }


def write_report(
    *,
    specification_summary: pd.DataFrame,
    club_comparison: pd.DataFrame,
    player_changes: pd.DataFrame,
    specifications: tuple[
        PlayerSelectionSpecification,
        ...,
    ],
) -> None:
    non_baseline = (
        specification_summary.loc[
            ~specification_summary[
                "specification"
            ].eq(
                RATING_ONLY_SPECIFICATION.name
            )
        ]
    )

    most_stable = (
        non_baseline.sort_values(
            [
                "mean_players_changed",
                "mean_raw_role_rating",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .iloc[0]
    )

    most_disruptive = (
        non_baseline.sort_values(
            [
                "mean_players_changed",
                "mean_start_rate",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
    )

    highest_usage = (
        non_baseline.sort_values(
            [
                "mean_usage_evidence_score",
                "mean_raw_role_rating",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
    )

    club_sensitivity = (
        club_comparison.loc[
            ~club_comparison[
                "specification"
            ].eq(
                RATING_ONLY_SPECIFICATION.name
            )
        ]
        .groupby(
            [
                "team_id",
                "team",
            ],
            as_index=False,
        )
        .agg(
            mean_players_changed=(
                "players_changed_from_baseline",
                "mean",
            ),
            maximum_players_changed=(
                "players_changed_from_baseline",
                "max",
            ),
            mean_overlap_share=(
                "baseline_player_overlap_share",
                "mean",
            ),
        )
        .sort_values(
            [
                "mean_players_changed",
                "maximum_players_changed",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    most_sensitive_club = (
        club_sensitivity.iloc[0]
    )

    least_sensitive_club = (
        club_sensitivity.iloc[-1]
    )

    report = f"""# Study 087B — Player Selection Policy Benchmark

## Purpose

Measure how Bundesliga 2024/25 projected lineups change when
season-level player usage evidence is introduced into role-specific
player selection.

## Methodological boundary

This study:

- uses a fixed 4-3-3 formation;
- preserves greedy slot-by-slot lineup construction;
- changes only the role-local player ranking specification;
- builds no team representations;
- fits no goal model;
- changes no production artifact.

The usage inputs are full-season aggregates and are not
prediction-date valid.

## Population

- Clubs: {EXPECTED_CLUB_COUNT}
- Formation slots per club: {EXPECTED_LINEUP_SIZE}
- Selection specifications: {len(specifications)}
- Lineup rows: {
    len(specifications)
    * EXPECTED_CLUB_COUNT
    * EXPECTED_LINEUP_SIZE
}
- Slot-level changes from rating-only baseline:
  {len(player_changes)}

## Baseline

The control specification is:

`rating_only`

with:

- role-rating weight: 1.00
- start-rate weight: 0.00
- relative-minutes weight: 0.00

## Most stable non-baseline specification

- Specification:
  `{most_stable["specification"]}`
- Mean players changed per club:
  {most_stable["mean_players_changed"]:.6f}
- Mean raw role rating:
  {most_stable["mean_raw_role_rating"]:.6f}
- Mean start rate:
  {most_stable["mean_start_rate"]:.6f}

## Most disruptive specification

- Specification:
  `{most_disruptive["specification"]}`
- Mean players changed per club:
  {most_disruptive["mean_players_changed"]:.6f}
- Maximum players changed for one club:
  {int(most_disruptive["maximum_players_changed"])}
- Mean start rate:
  {most_disruptive["mean_start_rate"]:.6f}

## Highest-usage selected lineups

- Specification:
  `{highest_usage["specification"]}`
- Mean usage-evidence score:
  {highest_usage["mean_usage_evidence_score"]:.6f}
- Mean raw role rating:
  {highest_usage["mean_raw_role_rating"]:.6f}

## Club sensitivity

Most sensitive club across non-baseline policies:

- Club:
  `{most_sensitive_club["team"]}`
- Mean players changed:
  {most_sensitive_club["mean_players_changed"]:.6f}
- Maximum players changed:
  {int(most_sensitive_club["maximum_players_changed"])}

Least sensitive club:

- Club:
  `{least_sensitive_club["team"]}`
- Mean players changed:
  {least_sensitive_club["mean_players_changed"]:.6f}
- Maximum players changed:
  {int(least_sensitive_club["maximum_players_changed"])}

## Interpretation

This study does not identify a winning production policy.

It establishes:

- whether usage materially changes projected XIs;
- the quality-versus-usage tradeoff;
- which clubs are sensitive to policy weights;
- which specifications remain close to the rating-only control.

A representation benchmark is required before any policy can be
judged by downstream predictive value.

## Result

**OVERALL RESULT: PASS**

Player-selection sensitivity was measured without modifying team
representations or the prediction runtime.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 087B — PLAYER SELECTION POLICY BENCHMARK"
    )
    print("=" * 88)

    players = load_joined_players()
    formation = load_formation()
    specifications = build_specifications()

    lineups, formation_coverage = (
        build_all_lineups(
            players=players,
            formation=formation,
            specifications=specifications,
        )
    )

    role_fallback_usage = (
        lineups.loc[
            ~lineups[
                "exact_role_match"
            ],
            [
                "specification",
                "team_id",
                "team",
                "slot_order",
                "slot",
                "role",
                "player_id",
                "player",
                "selection_source_role",
                "selection_role_suitability",
                "raw_role_rating",
                "selection_score",
            ],
        ]
        .sort_values(
            [
                "specification",
                "team",
                "slot_order",
            ]
        )
        .reset_index(drop=True)
    )

    club_comparison = build_club_comparison(
        lineups
    )
    club_comparison = build_club_comparison(
        lineups
    )

    specification_summary = (
        build_specification_summary(
            club_comparison,
            specifications,
        )
    )

    player_changes = build_player_changes(
        lineups
    )

    ranking_stability = (
        build_ranking_stability(
            players=players,
            specifications=specifications,
        )
    )

    specification_records = pd.DataFrame(
        [
            specification.to_record()
            for specification in specifications
        ]
    )

    metadata = build_metadata(
        specifications=specifications,
        lineups=lineups,
        player_changes=player_changes,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    specification_records.to_csv(
        SPECIFICATION_PATH,
        index=False,
    )

    lineups.to_csv(
        LINEUP_PATH,
        index=False,
    )

    club_comparison.to_csv(
        CLUB_COMPARISON_PATH,
        index=False,
    )

    specification_summary.to_csv(
        SPECIFICATION_SUMMARY_PATH,
        index=False,
    )

    player_changes.to_csv(
        PLAYER_CHANGE_PATH,
        index=False,
    )

    ranking_stability.to_csv(
        RANKING_STABILITY_PATH,
        index=False,
    )

    formation_coverage.to_csv(
        FORMATION_COVERAGE_PATH,
        index=False,
    )

    role_fallback_usage.to_csv(
        ROLE_FALLBACK_PATH,
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
        specification_summary=(
            specification_summary
        ),
        club_comparison=club_comparison,
        player_changes=player_changes,
        specifications=specifications,
    )

    print()
    print("Benchmark population")
    print("-" * 88)
    print(
        f"  Clubs: {EXPECTED_CLUB_COUNT}"
    )
    print(
        f"  Formation: {FORMATION}"
    )
    print(
        "  Selection specifications: "
        f"{len(specifications)}"
    )
    print(
        f"  Lineup rows: {len(lineups)}"
    )
    print(
        "  Slot-level changes from baseline: "
        f"{len(player_changes)}"
    )

    print()
    print("Specification summary")
    print("-" * 88)

    display_columns = [
        "specification",
        "role_rating_weight",
        "start_rate_weight",
        "minutes_weight",
        "mean_players_changed",
        "maximum_players_changed",
        "unchanged_club_count",
        "goalkeeper_change_count",
        "mean_raw_role_rating",
        "mean_start_rate",
        "mean_relative_minutes",
        "mean_usage_evidence_score",
    ]

    print(
        specification_summary[
            display_columns
        ]
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Most policy-sensitive clubs")
    print("-" * 88)

    club_sensitivity = (
        club_comparison.loc[
            ~club_comparison[
                "specification"
            ].eq(
                RATING_ONLY_SPECIFICATION.name
            )
        ]
        .groupby(
            "team",
            as_index=False,
        )
        .agg(
            mean_players_changed=(
                "players_changed_from_baseline",
                "mean",
            ),
            maximum_players_changed=(
                "players_changed_from_baseline",
                "max",
            ),
            mean_overlap_share=(
                "baseline_player_overlap_share",
                "mean",
            ),
        )
        .sort_values(
            [
                "mean_players_changed",
                "maximum_players_changed",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    print(
        club_sensitivity.head(10).to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Joined-player loading: PASS")
    print("  Formation loading: PASS")
    print("  Specification validation: PASS")
    print("  Role-local ranking: PASS")
    print("  Greedy formation filling: PASS")
    print("  Eleven-player lineup contract: PASS")
    print("  No duplicate players: PASS")
    print("  All 18 clubs completed: PASS")
    print("  Baseline comparison: PASS")
    print("  Ranking-stability analysis: PASS")
    print("  Team representation construction: NONE")
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
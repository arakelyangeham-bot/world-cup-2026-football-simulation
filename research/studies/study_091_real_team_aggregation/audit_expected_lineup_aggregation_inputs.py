# audit_expected_lineup_aggregation_inputs.py

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.player_intelligence.aggregation_adapter import (
    aggregate_dimension_values,
    minimum_required_population,
)

from research.studies.study_089_aggregation_mathematics.aggregation_specifications import (
    AggregationSpecification,
    build_aggregation_specifications,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_LINEUPS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expected_lineups.csv"
)

PLAYER_RATINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_ratings.csv"
)

LINEUP_COVERAGE_PATH = (
    PROJECT_ROOT/
    "outputs"
    / "lineup_coverage.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_091_real_team_aggregation"
)

PLAYER_POPULATION_PATH = (
    OUTPUT_DIRECTORY
    / "expected_lineup_player_population.csv"
)

TEAM_COVERAGE_PATH = (
    OUTPUT_DIRECTORY
    / "expected_lineup_team_coverage.csv"
)

COMPATIBILITY_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_real_scale_compatibility.csv"
)

SPECIFICATION_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_real_scale_specification_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_091a_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_091a_report.md"
)


SHORTLISTED_SPECIFICATION_IDS = (
    "top5_arithmetic",
    "top5_power_1_50",
    "top5_rank_moderate",
    "top5_star_alpha_0_20",
    "top5_softmax_beta_3",
    "ability_power_gamma_2",
    "arithmetic_all",
)


DIMENSION_COLUMNS = {
    "attack": "projected_attack",
    "midfield": "projected_midfield",
    "defense": "projected_defense",
}


REQUIRED_LINEUP_COLUMNS = {
    "slot",
    "role",
    "player_id",
    "player",
    "rating",
    "country",
    "formation",
}


REQUIRED_RATING_COLUMNS = {
    "player_id",
    "player",
    "country",
    "attribute_overall",
    "evidence_confidence",
    "minutesPlayed",
    "rating_GK",
    "rating_CB",
    "rating_FB",
    "rating_DM",
    "rating_CM",
    "rating_AM",
    "rating_WM",
    "rating_W",
    "rating_ST",
}


REQUIRED_COVERAGE_COLUMNS = {
    "country",
    "rated_players",
    "selected_players",
    "missing_slots",
}


PROJECTION_WEIGHTS = {
    "projected_attack": {
        "rating_ST": 0.40,
        "rating_W": 0.25,
        "rating_AM": 0.20,
        "rating_CM": 0.10,
        "rating_FB": 0.05,
    },
    "projected_midfield": {
        "rating_CM": 0.35,
        "rating_DM": 0.25,
        "rating_AM": 0.20,
        "rating_WM": 0.10,
        "rating_FB": 0.10,
    },
    "projected_defense": {
        "rating_CB": 0.40,
        "rating_FB": 0.25,
        "rating_DM": 0.25,
        "rating_GK": 0.10,
    },
    "projected_goalkeeper": {
        "rating_GK": 1.00,
    },
}


def require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(
        required_columns - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: {missing}"
        )


def normalize_player_ids(
    values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    return numeric.astype("Int64")


def weighted_projection(
    frame: pd.DataFrame,
    *,
    weights: dict[str, float],
) -> pd.Series:
    total_weight = float(
        sum(weights.values())
    )

    if total_weight <= 0.0:
        raise ValueError(
            "Projection weights must sum to a positive value."
        )

    result = pd.Series(
        0.0,
        index=frame.index,
        dtype=float,
    )

    for column, weight in weights.items():
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(0.0)

        result = (
            result
            + values.astype(float)
            * float(weight)
        )

    return result / total_weight


def selected_specifications(
) -> tuple[AggregationSpecification, ...]:
    registry = build_aggregation_specifications()

    by_id = {
        specification.specification_id:
            specification
        for specification in registry
    }

    missing = [
        specification_id
        for specification_id
        in SHORTLISTED_SPECIFICATION_IDS
        if specification_id not in by_id
    ]

    if missing:
        raise KeyError(
            "Missing shortlisted aggregation specifications: "
            f"{missing}"
        )

    return tuple(
        by_id[specification_id]
        for specification_id
        in SHORTLISTED_SPECIFICATION_IDS
    )


def load_inputs(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    expected_lineups = pd.read_csv(
        EXPECTED_LINEUPS_PATH
    )

    player_ratings = pd.read_csv(
        PLAYER_RATINGS_PATH
    )

    lineup_coverage = pd.read_csv(
        LINEUP_COVERAGE_PATH
    )

    require_columns(
        expected_lineups,
        REQUIRED_LINEUP_COLUMNS,
        frame_name="Expected lineups",
    )

    require_columns(
        player_ratings,
        REQUIRED_RATING_COLUMNS,
        frame_name="Player ratings",
    )

    require_columns(
        lineup_coverage,
        REQUIRED_COVERAGE_COLUMNS,
        frame_name="Lineup coverage",
    )

    expected_lineups = expected_lineups.copy()
    player_ratings = player_ratings.copy()
    lineup_coverage = lineup_coverage.copy()

    expected_lineups["player_id"] = (
        normalize_player_ids(
            expected_lineups["player_id"]
        )
    )

    player_ratings["player_id"] = (
        normalize_player_ids(
            player_ratings["player_id"]
        )
    )

    if player_ratings[
        "player_id"
    ].isna().any():
        raise ValueError(
            "Player ratings contain missing or invalid player IDs."
        )

    if player_ratings[
        "player_id"
    ].duplicated().any():
        duplicate_ids = (
            player_ratings.loc[
                player_ratings[
                    "player_id"
                ].duplicated(
                    keep=False
                ),
                "player_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Player ratings contain duplicate player IDs: "
            f"{duplicate_ids[:20]}"
        )

    return (
        expected_lineups,
        player_ratings,
        lineup_coverage,
    )


def build_player_population(
    *,
    expected_lineups: pd.DataFrame,
    player_ratings: pd.DataFrame,
) -> pd.DataFrame:
    selected = expected_lineups.loc[
        expected_lineups[
            "player_id"
        ].notna()
    ].copy()

    selected["player_id"] = (
        selected["player_id"]
        .astype("int64")
    )

    if selected[
        [
            "country",
            "player_id",
        ]
    ].duplicated().any():
        raise ValueError(
            "Expected lineups contain duplicate selected player IDs "
            "within one country."
        )

    rating_columns = [
        column
        for column in player_ratings.columns
        if column != "country"
    ]

    joined = selected.merge(
        player_ratings[
            rating_columns
        ],
        on="player_id",
        how="left",
        validate="many_to_one",
        suffixes=(
            "_lineup",
            "_ratings",
        ),
        indicator=True,
    )

    unresolved = joined.loc[
        joined["_merge"].ne("both")
    ]

    if not unresolved.empty:
        unresolved_ids = (
            unresolved[
                "player_id"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Expected-lineup players could not be resolved in "
            f"player ratings: {unresolved_ids[:20]}"
        )

    joined = joined.drop(
        columns="_merge"
    )

    for output_column, weights in (
        PROJECTION_WEIGHTS.items()
    ):
        joined[output_column] = (
            weighted_projection(
                joined,
                weights=weights,
            )
        )

    projection_columns = list(
        PROJECTION_WEIGHTS
    )

    projection_array = joined[
        projection_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        projection_array
    ).all():
        raise ValueError(
            "Player dimension projection produced non-finite values."
        )

    joined[
        "any_negative_role_rating"
    ] = (
        joined[
            [
                "rating_GK",
                "rating_CB",
                "rating_FB",
                "rating_DM",
                "rating_CM",
                "rating_AM",
                "rating_WM",
                "rating_W",
                "rating_ST",
            ]
        ]
        .lt(0.0)
        .any(axis=1)
    )

    joined[
        "any_negative_dimension_projection"
    ] = (
        joined[
            projection_columns
        ]
        .lt(0.0)
        .any(axis=1)
    )

    joined[
        "lineup_player_name_matches_rating_name"
    ] = (
        joined[
            "player_lineup"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(
            joined[
                "player_ratings"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    )

    preferred_columns = [
        "country",
        "formation",
        "slot",
        "role",
        "player_id",
        "player_lineup",
        "player_ratings",
        "rating",
        "current_team",
        "position",
        "eligible_roles",
        "minutesPlayed",
        "evidence_confidence",
        "attribute_overall",
        "projected_attack",
        "projected_midfield",
        "projected_defense",
        "projected_goalkeeper",
        "any_negative_role_rating",
        "any_negative_dimension_projection",
        "lineup_player_name_matches_rating_name",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in joined.columns
    ]

    remaining_columns = [
        column
        for column in joined.columns
        if column not in available_columns
    ]

    return (
        joined[
            available_columns
            + remaining_columns
        ]
        .sort_values(
            [
                "country",
                "slot",
            ]
        )
        .reset_index(drop=True)
    )


def build_team_coverage(
    *,
    expected_lineups: pd.DataFrame,
    player_population: pd.DataFrame,
    lineup_coverage: pd.DataFrame,
) -> pd.DataFrame:
    slot_summary = (
        expected_lineups
        .groupby(
            "country",
            as_index=False,
        )
        .agg(
            expected_slot_rows=(
                "slot",
                "size",
            ),
            selected_player_rows=(
                "player_id",
                lambda values: int(
                    values.notna().sum()
                ),
            ),
            formation_count=(
                "formation",
                "nunique",
            ),
        )
    )

    population_summary = (
        player_population
        .groupby(
            "country",
            as_index=False,
        )
        .agg(
            resolved_player_count=(
                "player_id",
                "size",
            ),
            unique_player_count=(
                "player_id",
                "nunique",
            ),
            negative_role_rating_players=(
                "any_negative_role_rating",
                "sum",
            ),
            negative_projection_players=(
                "any_negative_dimension_projection",
                "sum",
            ),
            player_name_mismatch_count=(
                "lineup_player_name_matches_rating_name",
                lambda values: int(
                    (~values.astype(bool)).sum()
                ),
            ),
            minimum_attack_projection=(
                "projected_attack",
                "min",
            ),
            maximum_attack_projection=(
                "projected_attack",
                "max",
            ),
            minimum_midfield_projection=(
                "projected_midfield",
                "min",
            ),
            maximum_midfield_projection=(
                "projected_midfield",
                "max",
            ),
            minimum_defense_projection=(
                "projected_defense",
                "min",
            ),
            maximum_defense_projection=(
                "projected_defense",
                "max",
            ),
        )
    )

    coverage = (
        lineup_coverage
        .merge(
            slot_summary,
            on="country",
            how="outer",
            validate="one_to_one",
        )
        .merge(
            population_summary,
            on="country",
            how="left",
            validate="one_to_one",
        )
    )

    integer_columns = [
        "rated_players",
        "selected_players",
        "missing_slots",
        "expected_slot_rows",
        "selected_player_rows",
        "formation_count",
        "resolved_player_count",
        "unique_player_count",
        "negative_role_rating_players",
        "negative_projection_players",
        "player_name_mismatch_count",
    ]

    for column in integer_columns:
        if column in coverage.columns:
            coverage[column] = (
                pd.to_numeric(
                    coverage[column],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

    coverage[
        "complete_expected_xi"
    ] = (
        coverage[
            "selected_player_rows"
        ].eq(11)
        & coverage[
            "missing_slots"
        ].eq(0)
        & coverage[
            "resolved_player_count"
        ].eq(11)
        & coverage[
            "unique_player_count"
        ].eq(11)
    )

    coverage[
        "eligible_for_real_scale_audit"
    ] = coverage[
        "complete_expected_xi"
    ]

    return (
        coverage
        .sort_values(
            [
                "complete_expected_xi",
                "selected_player_rows",
                "country",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def evaluate_dimension(
    values: tuple[float, ...],
    *,
    specification: AggregationSpecification,
) -> dict[str, object]:
    minimum_size = minimum_required_population(
        specification
    )

    negative_count = sum(
        value < 0.0
        for value in values
    )

    zero_count = sum(
        math.isclose(
            value,
            0.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for value in values
    )

    try:
        result = aggregate_dimension_values(
            values,
            specification=specification,
        )

    except Exception as error:
        return {
            "status": "failed",
            "aggregated_value": None,
            "error_type":
                type(error).__name__,
            "error_message":
                str(error),
            "minimum_required_population":
                minimum_size,
            "observed_population_size":
                len(values),
            "negative_value_count":
                negative_count,
            "zero_value_count":
                zero_count,
            "minimum_input_value":
                min(values),
            "maximum_input_value":
                max(values),
            "mean_input_value":
                float(
                    sum(values)
                    / len(values)
                ),
        }

    numeric_result = float(result)

    if not math.isfinite(
        numeric_result
    ):
        return {
            "status": "failed",
            "aggregated_value": None,
            "error_type":
                "NonFiniteAggregation",
            "error_message": (
                "Aggregation returned a non-finite value."
            ),
            "minimum_required_population":
                minimum_size,
            "observed_population_size":
                len(values),
            "negative_value_count":
                negative_count,
            "zero_value_count":
                zero_count,
            "minimum_input_value":
                min(values),
            "maximum_input_value":
                max(values),
            "mean_input_value":
                float(
                    sum(values)
                    / len(values)
                ),
        }

    return {
        "status": "evaluated",
        "aggregated_value":
            numeric_result,
        "error_type":
            None,
        "error_message":
            None,
        "minimum_required_population":
            minimum_size,
        "observed_population_size":
            len(values),
        "negative_value_count":
            negative_count,
        "zero_value_count":
            zero_count,
        "minimum_input_value":
            min(values),
        "maximum_input_value":
            max(values),
        "mean_input_value":
            float(
                sum(values)
                / len(values)
            ),
    }


def build_compatibility_results(
    *,
    player_population: pd.DataFrame,
    team_coverage: pd.DataFrame,
    specifications: tuple[
        AggregationSpecification,
        ...,
    ],
) -> pd.DataFrame:
    eligible_teams = set(
        team_coverage.loc[
            team_coverage[
                "eligible_for_real_scale_audit"
            ],
            "country",
        ]
    )

    records: list[
        dict[str, object]
    ] = []

    for country, team_rows in (
        player_population.loc[
            player_population[
                "country"
            ].isin(
                eligible_teams
            )
        ]
        .groupby(
            "country",
            sort=True,
        )
    ):
        if len(team_rows) != 11:
            raise AssertionError(
                "Eligible team does not contain exactly "
                f"11 selected players: {country!r}."
            )

        for specification in specifications:
            specification_record = (
                specification.to_record()
            )

            for (
                dimension,
                value_column,
            ) in DIMENSION_COLUMNS.items():
                values = tuple(
                    float(value)
                    for value in team_rows[
                        value_column
                    ]
                )

                result = evaluate_dimension(
                    values,
                    specification=specification,
                )

                records.append(
                    {
                        "country":
                            country,
                        "formation":
                            team_rows[
                                "formation"
                            ].iloc[0],
                        "dimension":
                            dimension,
                        "specification_id":
                            specification.specification_id,
                        "aggregation_family":
                            specification.aggregation_family,
                        "display_name":
                            specification.display_name,
                        "output_type":
                            specification.output_type,
                        "parameterization":
                            specification_record[
                                "parameterization"
                            ],
                        "historical_control":
                            specification.historical_control,
                        **result,
                    }
                )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "country",
                "specification_id",
                "dimension",
            ]
        )
        .reset_index(drop=True)
    )


def build_specification_summary(
    compatibility: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        compatibility
        .groupby(
            [
                "specification_id",
                "aggregation_family",
                "display_name",
                "output_type",
                "parameterization",
                "historical_control",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            compatibility_rows=(
                "country",
                "size",
            ),
            team_count=(
                "country",
                "nunique",
            ),
            dimension_count=(
                "dimension",
                "nunique",
            ),
            evaluated_rows=(
                "status",
                lambda values: int(
                    values.eq(
                        "evaluated"
                    ).sum()
                ),
            ),
            failed_rows=(
                "status",
                lambda values: int(
                    values.eq(
                        "failed"
                    ).sum()
                ),
            ),
            teams_with_any_failure=(
                "country",
                lambda countries: 0,
            ),
            rows_with_negative_inputs=(
                "negative_value_count",
                lambda values: int(
                    values.gt(0).sum()
                ),
            ),
            minimum_observed_input=(
                "minimum_input_value",
                "min",
            ),
            maximum_observed_input=(
                "maximum_input_value",
                "max",
            ),
            minimum_aggregated_value=(
                "aggregated_value",
                "min",
            ),
            maximum_aggregated_value=(
                "aggregated_value",
                "max",
            ),
        )
    )

    failed_teams = (
        compatibility.loc[
            compatibility[
                "status"
            ].eq("failed")
        ]
        .groupby(
            "specification_id"
        )["country"]
        .nunique()
        .to_dict()
    )

    summary[
        "teams_with_any_failure"
    ] = summary[
        "specification_id"
    ].map(
        failed_teams
    ).fillna(0).astype(int)

    summary[
        "real_scale_compatible"
    ] = summary[
        "failed_rows"
    ].eq(0)

    return (
        summary
        .sort_values(
            "specification_id"
        )
        .reset_index(drop=True)
    )


def validate_outputs(
    *,
    player_population: pd.DataFrame,
    team_coverage: pd.DataFrame,
    compatibility: pd.DataFrame,
    specification_summary: pd.DataFrame,
    specifications: tuple[
        AggregationSpecification,
        ...,
    ],
) -> None:
    if player_population.empty:
        raise AssertionError(
            "Expected-lineup player population is empty."
        )

    if team_coverage.empty:
        raise AssertionError(
            "Expected-lineup team coverage is empty."
        )

    complete_team_count = int(
        team_coverage[
            "eligible_for_real_scale_audit"
        ].sum()
    )

    expected_compatibility_rows = (
        complete_team_count
        * len(specifications)
        * len(DIMENSION_COLUMNS)
    )

    if len(
        compatibility
    ) != expected_compatibility_rows:
        raise AssertionError(
            "Unexpected compatibility row count. "
            f"Expected {expected_compatibility_rows}, "
            f"received {len(compatibility)}."
        )

    if compatibility[
        [
            "country",
            "dimension",
            "specification_id",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Compatibility output contains duplicate "
            "team-dimension-specification rows."
        )

    if len(
        specification_summary
    ) != len(
        specifications
    ):
        raise AssertionError(
            "Specification summary does not cover the complete "
            "shortlisted registry."
        )

    valid_statuses = {
        "evaluated",
        "failed",
    }

    invalid_statuses = (
        set(
            compatibility[
                "status"
            ]
        )
        - valid_statuses
    )

    if invalid_statuses:
        raise AssertionError(
            "Compatibility output contains invalid statuses: "
            f"{sorted(invalid_statuses)}"
        )

    evaluated_values = (
        compatibility.loc[
            compatibility[
                "status"
            ].eq("evaluated"),
            "aggregated_value",
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        evaluated_values
    ).all():
        raise AssertionError(
            "Evaluated compatibility rows contain non-finite values."
        )


def build_metadata(
    *,
    player_population: pd.DataFrame,
    team_coverage: pd.DataFrame,
    compatibility: pd.DataFrame,
    specification_summary: pd.DataFrame,
) -> dict[str, Any]:
    complete_team_count = int(
        team_coverage[
            "complete_expected_xi"
        ].sum()
    )

    incomplete_team_count = int(
        (
            ~team_coverage[
                "complete_expected_xi"
            ]
        ).sum()
    )

    failed_rows = int(
        compatibility[
            "status"
        ].eq("failed").sum()
    )

    compatible_specifications = int(
        specification_summary[
            "real_scale_compatible"
        ].sum()
    )

    return {
        "study_id": "091A",
        "study_name": (
            "Expected-Lineup Aggregation Input Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "expected_lineup_row_count":
            int(
                team_coverage[
                    "expected_slot_rows"
                ].sum()
            ),
        "resolved_player_population_row_count":
            len(player_population),
        "team_count":
            len(team_coverage),
        "complete_expected_xi_team_count":
            complete_team_count,
        "incomplete_expected_xi_team_count":
            incomplete_team_count,
        "shortlisted_specification_count":
            len(specification_summary),
        "compatibility_result_count":
            len(compatibility),
        "failed_compatibility_row_count":
            failed_rows,
        "fully_real_scale_compatible_specification_count":
            compatible_specifications,
        "negative_dimension_projection_player_count":
            int(
                player_population[
                    "any_negative_dimension_projection"
                ].sum()
            ),
        "ranking_generated":
            False,
        "team_repository_generated":
            False,
        "goal_model_fitted":
            False,
        "production_repository_changed":
            False,
        "production_runtime_changed":
            False,
        "population_contract": (
            "Complete expected 4-3-3 starting XIs only. "
            "Player dimension values are reconstructed from the "
            "full role-rating matrix using the frozen role-projection "
            "weights."
        ),
        "interpretation_boundary": (
            "This audit tests identity coverage, expected-lineup "
            "coverage, negative-value exposure, and numerical "
            "compatibility of shortlisted aggregation methods. "
            "It does not compare team quality or predictive performance."
        ),
        "outputs": [
            PLAYER_POPULATION_PATH.name,
            TEAM_COVERAGE_PATH.name,
            COMPATIBILITY_PATH.name,
            SPECIFICATION_SUMMARY_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    player_population: pd.DataFrame,
    team_coverage: pd.DataFrame,
    compatibility: pd.DataFrame,
    specification_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    status_counts = (
        compatibility[
            "status"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    incompatible = (
        specification_summary.loc[
            ~specification_summary[
                "real_scale_compatible"
            ]
        ][
            [
                "specification_id",
                "failed_rows",
                "teams_with_any_failure",
                "minimum_observed_input",
                "maximum_observed_input",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    report = f"""# Study 091A — Expected-Lineup Aggregation Input Audit

## Purpose

Audit whether the shortlisted Study 090 aggregation specifications can
be applied safely to real expected-lineup player projections.

## Population contract

- Expected formation: 4-3-3
- Population unit: selected expected starting XI
- Required selected players per team: 11
- Complete teams included in aggregation audit:
  {metadata["complete_expected_xi_team_count"]}
- Incomplete teams retained in coverage output but excluded from
  aggregation evaluation:
  {metadata["incomplete_expected_xi_team_count"]}

## Data integration

Expected-lineup selections are joined to the complete player-rating
matrix by `player_id`.

Attack, midfield, defense, and goalkeeper projections are reconstructed
using the existing frozen role-projection weights.

- Resolved selected-player rows: {len(player_population)}
- Players with at least one negative projected dimension:
  {metadata["negative_dimension_projection_player_count"]}

## Aggregation compatibility

- Shortlisted specifications: {len(specification_summary)}
- Compatibility rows: {len(compatibility)}

Status counts:

{json.dumps(status_counts, indent=2)}

Specifications with one or more real-scale failures:

{json.dumps(incompatible, indent=2)}

## Methodological boundary

This study:

- creates no team ranking;
- creates no team repository;
- fits no goal model;
- changes no production configuration;
- evaluates no match outcomes;
- makes no claim about predictive superiority.

An aggregation failure is preserved as evidence of incompatibility with
the current real-valued rating scale. No fallback aggregation is used.

## Result

**OVERALL RESULT: {metadata["status"]}**

The expected-lineup data path and shortlisted aggregation methods were
audited without changing production behavior.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 091A — EXPECTED-LINEUP AGGREGATION INPUT AUDIT"
    )
    print("=" * 88)

    (
        expected_lineups,
        player_ratings,
        lineup_coverage,
    ) = load_inputs()

    specifications = (
        selected_specifications()
    )

    player_population = (
        build_player_population(
            expected_lineups=expected_lineups,
            player_ratings=player_ratings,
        )
    )

    team_coverage = build_team_coverage(
        expected_lineups=expected_lineups,
        player_population=player_population,
        lineup_coverage=lineup_coverage,
    )

    compatibility = (
        build_compatibility_results(
            player_population=player_population,
            team_coverage=team_coverage,
            specifications=specifications,
        )
    )

    specification_summary = (
        build_specification_summary(
            compatibility
        )
    )

    validate_outputs(
        player_population=player_population,
        team_coverage=team_coverage,
        compatibility=compatibility,
        specification_summary=(
            specification_summary
        ),
        specifications=specifications,
    )

    metadata = build_metadata(
        player_population=player_population,
        team_coverage=team_coverage,
        compatibility=compatibility,
        specification_summary=(
            specification_summary
        ),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    player_population.to_csv(
        PLAYER_POPULATION_PATH,
        index=False,
    )

    team_coverage.to_csv(
        TEAM_COVERAGE_PATH,
        index=False,
    )

    compatibility.to_csv(
        COMPATIBILITY_PATH,
        index=False,
    )

    specification_summary.to_csv(
        SPECIFICATION_SUMMARY_PATH,
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
        player_population=player_population,
        team_coverage=team_coverage,
        compatibility=compatibility,
        specification_summary=(
            specification_summary
        ),
        metadata=metadata,
    )

    print()
    print("Expected-lineup coverage")
    print("-" * 88)
    print(
        "  Complete expected XIs: "
        f"{metadata['complete_expected_xi_team_count']}"
    )
    print(
        "  Incomplete expected XIs: "
        f"{metadata['incomplete_expected_xi_team_count']}"
    )
    print(
        "  Resolved player rows: "
        f"{len(player_population)}"
    )

    print()
    print("Compatibility statuses")
    print("-" * 88)
    print(
        compatibility[
            "status"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Specification compatibility")
    print("-" * 88)
    print(
        specification_summary[
            [
                "specification_id",
                "evaluated_rows",
                "failed_rows",
                "teams_with_any_failure",
                "real_scale_compatible",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Expected-lineup schema: PASS")
    print("  Player-rating schema: PASS")
    print("  Player identity join: PASS")
    print("  Complete-XI contract: PASS")
    print("  Dimension projection finiteness: PASS")
    print("  Silent aggregation fallback: NONE")
    print("  Team ranking: NOT GENERATED")
    print("  Goal-model fitting: NONE")
    print("  Production behavior: UNCHANGED")

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
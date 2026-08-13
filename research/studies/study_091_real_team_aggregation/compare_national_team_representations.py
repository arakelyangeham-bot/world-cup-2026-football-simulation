# compare_national_team_representations.py

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_091_real_team_aggregation"
)

OUTPUT_DIRECTORY = INPUT_DIRECTORY

COMPATIBILITY_PATH = (
    INPUT_DIRECTORY
    / "aggregation_real_scale_compatibility.csv"
)

SPECIFICATION_SUMMARY_PATH = (
    INPUT_DIRECTORY
    / "aggregation_real_scale_specification_summary.csv"
)

TEAM_COVERAGE_PATH = (
    INPUT_DIRECTORY
    / "expected_lineup_team_coverage.csv"
)

TEAM_DIMENSION_VALUES_PATH = (
    OUTPUT_DIRECTORY
    / "national_team_dimension_aggregation_values.csv"
)

TEAM_PROFILE_VALUES_PATH = (
    OUTPUT_DIRECTORY
    / "national_team_aggregation_profiles.csv"
)

BASELINE_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "national_team_aggregation_baseline_comparisons.csv"
)

RANK_CORRELATION_PATH = (
    OUTPUT_DIRECTORY
    / "national_team_aggregation_rank_correlations.csv"
)

DISAGREEMENT_PATH = (
    OUTPUT_DIRECTORY
    / "national_team_aggregation_disagreements.csv"
)

EXCLUSION_PATH = (
    OUTPUT_DIRECTORY
    / "national_team_aggregation_exclusions.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_091b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_091b_report.md"
)


BASELINE_SPECIFICATION_ID = "top5_arithmetic"

EXPECTED_DIMENSIONS = {
    "attack",
    "midfield",
    "defense",
}

ABSOLUTE_TOLERANCE = 1e-12

TOP_DISAGREEMENTS_PER_GROUP = 10


REQUIRED_COMPATIBILITY_COLUMNS = {
    "country",
    "formation",
    "dimension",
    "specification_id",
    "aggregation_family",
    "display_name",
    "output_type",
    "parameterization",
    "historical_control",
    "status",
    "aggregated_value",
    "error_type",
    "error_message",
    "observed_population_size",
}

REQUIRED_SPECIFICATION_COLUMNS = {
    "specification_id",
    "aggregation_family",
    "display_name",
    "output_type",
    "parameterization",
    "historical_control",
    "evaluated_rows",
    "failed_rows",
    "teams_with_any_failure",
    "real_scale_compatible",
}

REQUIRED_COVERAGE_COLUMNS = {
    "country",
    "complete_expected_xi",
    "eligible_for_real_scale_audit",
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
            f"{frame_name} is missing required columns: "
            f"{missing}"
        )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    compatibility = pd.read_csv(
        COMPATIBILITY_PATH
    )

    specification_summary = pd.read_csv(
        SPECIFICATION_SUMMARY_PATH
    )

    team_coverage = pd.read_csv(
        TEAM_COVERAGE_PATH
    )

    require_columns(
        compatibility,
        REQUIRED_COMPATIBILITY_COLUMNS,
        frame_name="Real-scale compatibility results",
    )

    require_columns(
        specification_summary,
        REQUIRED_SPECIFICATION_COLUMNS,
        frame_name="Specification compatibility summary",
    )

    require_columns(
        team_coverage,
        REQUIRED_COVERAGE_COLUMNS,
        frame_name="Expected-lineup team coverage",
    )

    return (
        compatibility,
        specification_summary,
        team_coverage,
    )


def select_compatible_specifications(
    specification_summary: pd.DataFrame,
) -> tuple[
    tuple[str, ...],
    pd.DataFrame,
]:
    compatible = (
        specification_summary.loc[
            specification_summary[
                "real_scale_compatible"
            ].astype(bool)
            & specification_summary[
                "failed_rows"
            ].eq(0)
        ]
        .copy()
        .sort_values("specification_id")
        .reset_index(drop=True)
    )

    excluded = (
        specification_summary.loc[
            ~(
                specification_summary[
                    "real_scale_compatible"
                ].astype(bool)
                & specification_summary[
                    "failed_rows"
                ].eq(0)
            )
        ]
        .copy()
        .sort_values("specification_id")
        .reset_index(drop=True)
    )

    if compatible.empty:
        raise AssertionError(
            "No real-scale-compatible aggregation "
            "specifications are available."
        )

    compatible_ids = tuple(
        compatible[
            "specification_id"
        ].astype(str)
    )

    if (
        BASELINE_SPECIFICATION_ID
        not in compatible_ids
    ):
        raise AssertionError(
            "The required top-five arithmetic baseline is not "
            "real-scale compatible."
        )

    exclusion_records: list[
        dict[str, object]
    ] = []

    for row in excluded.itertuples(
        index=False
    ):
        exclusion_records.append(
            {
                "specification_id":
                    row.specification_id,
                "aggregation_family":
                    row.aggregation_family,
                "display_name":
                    row.display_name,
                "failed_rows":
                    int(row.failed_rows),
                "teams_with_any_failure":
                    int(
                        row.teams_with_any_failure
                    ),
                "real_scale_compatible":
                    bool(
                        row.real_scale_compatible
                    ),
                "exclusion_reason": (
                    "Excluded from Study 091B because Study 091A "
                    "identified one or more failures on the "
                    "current real-valued player-projection scale."
                ),
            }
        )

    return (
        compatible_ids,
        pd.DataFrame(
            exclusion_records,
            columns=[
                "specification_id",
                "aggregation_family",
                "display_name",
                "failed_rows",
                "teams_with_any_failure",
                "real_scale_compatible",
                "exclusion_reason",
            ],
        ),
    )


def build_team_dimension_values(
    *,
    compatibility: pd.DataFrame,
    compatible_ids: tuple[str, ...],
    team_coverage: pd.DataFrame,
) -> pd.DataFrame:
    eligible_teams = set(
        team_coverage.loc[
            team_coverage[
                "eligible_for_real_scale_audit"
            ].astype(bool),
            "country",
        ].astype(str)
    )

    selected = compatibility.loc[
        compatibility[
            "specification_id"
        ].isin(compatible_ids)
        & compatibility[
            "country"
        ].isin(eligible_teams)
        & compatibility[
            "status"
        ].eq("evaluated")
        & compatibility[
            "dimension"
        ].isin(EXPECTED_DIMENSIONS)
    ].copy()

    selected[
        "aggregated_value"
    ] = pd.to_numeric(
        selected[
            "aggregated_value"
        ],
        errors="coerce",
    )

    if selected[
        "aggregated_value"
    ].isna().any():
        raise ValueError(
            "Evaluated representation rows contain missing "
            "aggregated values."
        )

    values = selected[
        "aggregated_value"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Evaluated representation rows contain non-finite "
            "aggregated values."
        )

    selected[
        "dimension_rank"
    ] = (
        selected
        .groupby(
            [
                "specification_id",
                "dimension",
            ]
        )[
            "aggregated_value"
        ]
        .rank(
            method="average",
            ascending=False,
        )
    )

    selected[
        "dimension_percentile"
    ] = (
        selected
        .groupby(
            [
                "specification_id",
                "dimension",
            ]
        )[
            "aggregated_value"
        ]
        .rank(
            method="average",
            ascending=True,
            pct=True,
        )
    )

    return (
        selected
        .sort_values(
            [
                "specification_id",
                "dimension",
                "dimension_rank",
                "country",
            ]
        )
        .reset_index(drop=True)
    )


def build_team_profiles(
    dimension_values: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "country",
        "formation",
        "specification_id",
        "aggregation_family",
        "display_name",
        "output_type",
        "parameterization",
        "historical_control",
        "observed_population_size",
    ]

    metadata = (
        dimension_values[
            metadata_columns
        ]
        .drop_duplicates()
    )

    duplicated_metadata = metadata[
        [
            "country",
            "specification_id",
        ]
    ].duplicated().any()

    if duplicated_metadata:
        raise AssertionError(
            "Team-specification metadata is not unique."
        )

    value_wide = (
        dimension_values
        .pivot(
            index=[
                "country",
                "specification_id",
            ],
            columns="dimension",
            values="aggregated_value",
        )
        .reset_index()
    )

    missing_dimensions = (
        EXPECTED_DIMENSIONS
        - set(value_wide.columns)
    )

    if missing_dimensions:
        raise AssertionError(
            "Team profiles are missing dimensions: "
            f"{sorted(missing_dimensions)}"
        )

    profiles = metadata.merge(
        value_wide,
        on=[
            "country",
            "specification_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    profiles[
        "equal_dimension_mean"
    ] = profiles[
        [
            "attack",
            "midfield",
            "defense",
        ]
    ].mean(axis=1)

    profiles[
        "dimension_range"
    ] = (
        profiles[
            [
                "attack",
                "midfield",
                "defense",
            ]
        ].max(axis=1)
        - profiles[
            [
                "attack",
                "midfield",
                "defense",
            ]
        ].min(axis=1)
    )

    profiles[
        "profile_rank"
    ] = (
        profiles
        .groupby(
            "specification_id"
        )[
            "equal_dimension_mean"
        ]
        .rank(
            method="average",
            ascending=False,
        )
    )

    profiles[
        "profile_percentile"
    ] = (
        profiles
        .groupby(
            "specification_id"
        )[
            "equal_dimension_mean"
        ]
        .rank(
            method="average",
            ascending=True,
            pct=True,
        )
    )

    return (
        profiles
        .sort_values(
            [
                "specification_id",
                "profile_rank",
                "country",
            ]
        )
        .reset_index(drop=True)
    )


def safe_relative_delta(
    value: float,
    baseline: float,
) -> float | None:
    if math.isclose(
        baseline,
        0.0,
        rel_tol=1e-12,
        abs_tol=ABSOLUTE_TOLERANCE,
    ):
        return None

    return float(
        (value - baseline)
        / abs(baseline)
    )


def build_baseline_comparisons(
    dimension_values: pd.DataFrame,
    team_profiles: pd.DataFrame,
) -> pd.DataFrame:
    dimension_rows = dimension_values[
        [
            "country",
            "dimension",
            "specification_id",
            "aggregation_family",
            "display_name",
            "aggregated_value",
            "dimension_rank",
            "dimension_percentile",
        ]
    ].copy()

    dimension_rows[
        "representation_level"
    ] = dimension_rows[
        "dimension"
    ]

    dimension_rows = dimension_rows.rename(
        columns={
            "aggregated_value":
                "representation_value",
            "dimension_rank":
                "representation_rank",
            "dimension_percentile":
                "representation_percentile",
        }
    )

    profile_rows = team_profiles[
        [
            "country",
            "specification_id",
            "aggregation_family",
            "display_name",
            "equal_dimension_mean",
            "profile_rank",
            "profile_percentile",
        ]
    ].copy()

    profile_rows[
        "representation_level"
    ] = "overall_equal_dimension_mean"

    profile_rows = profile_rows.rename(
        columns={
            "equal_dimension_mean":
                "representation_value",
            "profile_rank":
                "representation_rank",
            "profile_percentile":
                "representation_percentile",
        }
    )

    combined = pd.concat(
        [
            dimension_rows[
                [
                    "country",
                    "representation_level",
                    "specification_id",
                    "aggregation_family",
                    "display_name",
                    "representation_value",
                    "representation_rank",
                    "representation_percentile",
                ]
            ],
            profile_rows[
                [
                    "country",
                    "representation_level",
                    "specification_id",
                    "aggregation_family",
                    "display_name",
                    "representation_value",
                    "representation_rank",
                    "representation_percentile",
                ]
            ],
        ],
        ignore_index=True,
    )

    baseline = combined.loc[
        combined[
            "specification_id"
        ].eq(
            BASELINE_SPECIFICATION_ID
        )
    ][
        [
            "country",
            "representation_level",
            "representation_value",
            "representation_rank",
            "representation_percentile",
        ]
    ].rename(
        columns={
            "representation_value":
                "baseline_value",
            "representation_rank":
                "baseline_rank",
            "representation_percentile":
                "baseline_percentile",
        }
    )

    comparisons = combined.merge(
        baseline,
        on=[
            "country",
            "representation_level",
        ],
        how="left",
        validate="many_to_one",
    )

    if comparisons[
        "baseline_value"
    ].isna().any():
        raise AssertionError(
            "One or more comparison rows lack a top-five "
            "arithmetic baseline."
        )

    comparisons[
        "value_delta_from_baseline"
    ] = (
        comparisons[
            "representation_value"
        ]
        - comparisons[
            "baseline_value"
        ]
    )

    comparisons[
        "absolute_value_delta_from_baseline"
    ] = comparisons[
        "value_delta_from_baseline"
    ].abs()

    comparisons[
        "relative_value_delta_from_baseline"
    ] = [
        safe_relative_delta(
            float(value),
            float(baseline_value),
        )
        for value, baseline_value in zip(
            comparisons[
                "representation_value"
            ],
            comparisons[
                "baseline_value"
            ],
            strict=True,
        )
    ]

    comparisons[
        "rank_shift_from_baseline"
    ] = (
        comparisons[
            "baseline_rank"
        ]
        - comparisons[
            "representation_rank"
        ]
    )

    comparisons[
        "absolute_rank_shift_from_baseline"
    ] = comparisons[
        "rank_shift_from_baseline"
    ].abs()

    comparisons[
        "percentile_shift_from_baseline"
    ] = (
        comparisons[
            "representation_percentile"
        ]
        - comparisons[
            "baseline_percentile"
        ]
    )

    return (
        comparisons
        .sort_values(
            [
                "representation_level",
                "specification_id",
                "representation_rank",
                "country",
            ]
        )
        .reset_index(drop=True)
    )


def build_rank_correlations(
    baseline_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for representation_level, level_rows in (
        baseline_comparisons.groupby(
            "representation_level",
            sort=True,
        )
    ):
        rank_wide = (
            level_rows
            .pivot(
                index="country",
                columns="specification_id",
                values="representation_rank",
            )
            .sort_index(axis=1)
        )

        value_wide = (
            level_rows
            .pivot(
                index="country",
                columns="specification_id",
                values="representation_value",
            )
            .sort_index(axis=1)
        )

        specification_ids = list(
            rank_wide.columns
        )

        for left_index, left_id in enumerate(
            specification_ids
        ):
            for right_id in specification_ids[
                left_index + 1:
            ]:
                valid = (
                    rank_wide[
                        [
                            left_id,
                            right_id,
                        ]
                    ]
                    .dropna()
                )

                if len(valid) < 2:
                    rank_correlation = None
                else:
                    rank_correlation = float(
                        valid[left_id].corr(
                            valid[right_id],
                            method="pearson",
                        )
                    )

                value_valid = (
                    value_wide[
                        [
                            left_id,
                            right_id,
                        ]
                    ]
                    .dropna()
                )

                if len(value_valid) < 2:
                    value_correlation = None
                else:
                    value_correlation = float(
                        value_valid[left_id].corr(
                            value_valid[right_id],
                            method="pearson",
                        )
                    )

                records.append(
                    {
                        "representation_level":
                            representation_level,
                        "specification_a":
                            left_id,
                        "specification_b":
                            right_id,
                        "shared_team_count":
                            len(valid),
                        "rank_correlation":
                            rank_correlation,
                        "value_correlation":
                            value_correlation,
                    }
                )

    return pd.DataFrame(records)


def build_disagreements(
    baseline_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    candidates = baseline_comparisons.loc[
        ~baseline_comparisons[
            "specification_id"
        ].eq(
            BASELINE_SPECIFICATION_ID
        )
    ].copy()

    disagreements = (
        candidates
        .sort_values(
            [
                "representation_level",
                "specification_id",
                "absolute_rank_shift_from_baseline",
                "absolute_value_delta_from_baseline",
                "country",
            ],
            ascending=[
                True,
                True,
                False,
                False,
                True,
            ],
        )
        .groupby(
            [
                "representation_level",
                "specification_id",
            ],
            as_index=False,
            group_keys=False,
        )
        .head(
            TOP_DISAGREEMENTS_PER_GROUP
        )
        .reset_index(drop=True)
    )

    disagreements[
        "movement_direction"
    ] = np.select(
        [
            disagreements[
                "rank_shift_from_baseline"
            ].gt(0),
            disagreements[
                "rank_shift_from_baseline"
            ].lt(0),
        ],
        [
            "higher_under_candidate",
            "lower_under_candidate",
        ],
        default="unchanged_rank",
    )

    return disagreements


def validate_outputs(
    *,
    compatible_ids: tuple[str, ...],
    dimension_values: pd.DataFrame,
    team_profiles: pd.DataFrame,
    baseline_comparisons: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    disagreements: pd.DataFrame,
    team_coverage: pd.DataFrame,
) -> None:
    team_count = int(
        team_coverage[
            "eligible_for_real_scale_audit"
        ].astype(bool).sum()
    )

    expected_dimension_rows = (
        team_count
        * len(compatible_ids)
        * len(EXPECTED_DIMENSIONS)
    )

    if len(
        dimension_values
    ) != expected_dimension_rows:
        raise AssertionError(
            "Unexpected team-dimension row count. "
            f"Expected {expected_dimension_rows}, "
            f"received {len(dimension_values)}."
        )

    expected_profile_rows = (
        team_count
        * len(compatible_ids)
    )

    if len(
        team_profiles
    ) != expected_profile_rows:
        raise AssertionError(
            "Unexpected team-profile row count. "
            f"Expected {expected_profile_rows}, "
            f"received {len(team_profiles)}."
        )

    expected_comparison_rows = (
        expected_profile_rows
        * (
            len(EXPECTED_DIMENSIONS)
            + 1
        )
    )

    if len(
        baseline_comparisons
    ) != expected_comparison_rows:
        raise AssertionError(
            "Unexpected baseline-comparison row count. "
            f"Expected {expected_comparison_rows}, "
            f"received {len(baseline_comparisons)}."
        )

    if dimension_values[
        [
            "country",
            "dimension",
            "specification_id",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Team-dimension output contains duplicate keys."
        )

    if team_profiles[
        [
            "country",
            "specification_id",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Team-profile output contains duplicate keys."
        )

    if rank_correlations.empty:
        raise AssertionError(
            "Rank-correlation output is empty."
        )

    if disagreements.empty:
        raise AssertionError(
            "Disagreement output is empty."
        )

    numeric_frames = (
        dimension_values,
        team_profiles,
        baseline_comparisons,
    )

    for frame in numeric_frames:
        numeric = frame.select_dtypes(
            include="number"
        )

        non_finite = numeric.map(
            lambda value: (
                False
                if pd.isna(value)
                else not math.isfinite(
                    float(value)
                )
            )
        )

        if non_finite.any().any():
            raise AssertionError(
                "Representation output contains non-finite "
                "numeric values."
            )


def build_metadata(
    *,
    compatible_ids: tuple[str, ...],
    exclusions: pd.DataFrame,
    dimension_values: pd.DataFrame,
    team_profiles: pd.DataFrame,
    baseline_comparisons: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    disagreements: pd.DataFrame,
    team_coverage: pd.DataFrame,
) -> dict[str, Any]:
    team_count = int(
        team_coverage[
            "eligible_for_real_scale_audit"
        ].astype(bool).sum()
    )

    return {
        "study_id": "091B",
        "study_name": (
            "National-Team Representation Comparison"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "team_count": team_count,
        "compatible_specification_count":
            len(compatible_ids),
        "excluded_specification_count":
            len(exclusions),
        "dimension_count":
            len(EXPECTED_DIMENSIONS),
        "team_dimension_value_row_count":
            len(dimension_values),
        "team_profile_row_count":
            len(team_profiles),
        "baseline_comparison_row_count":
            len(baseline_comparisons),
        "rank_correlation_row_count":
            len(rank_correlations),
        "disagreement_row_count":
            len(disagreements),
        "baseline_specification_id":
            BASELINE_SPECIFICATION_ID,
        "compatible_specification_ids":
            list(compatible_ids),
        "excluded_specification_ids": (
            exclusions[
                "specification_id"
            ].tolist()
            if not exclusions.empty
            else []
        ),
        "equal_dimension_mean_is_production_score":
            False,
        "ranking_interpretation": (
            "Ranks are descriptive within the 160-team "
            "complete-expected-XI population. The equal-dimension "
            "mean is used only to compare representation movement "
            "and is not a production team-strength formula."
        ),
        "team_repository_generated":
            False,
        "goal_model_fitted":
            False,
        "predictive_superiority_claimed":
            False,
        "production_repository_changed":
            False,
        "production_runtime_changed":
            False,
        "outputs": [
            TEAM_DIMENSION_VALUES_PATH.name,
            TEAM_PROFILE_VALUES_PATH.name,
            BASELINE_COMPARISON_PATH.name,
            RANK_CORRELATION_PATH.name,
            DISAGREEMENT_PATH.name,
            EXCLUSION_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    metadata: dict[str, Any],
    baseline_comparisons: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    disagreements: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> None:
    non_baseline = baseline_comparisons.loc[
        ~baseline_comparisons[
            "specification_id"
        ].eq(
            BASELINE_SPECIFICATION_ID
        )
    ]

    movement_summary = (
        non_baseline
        .groupby(
            [
                "representation_level",
                "specification_id",
            ],
            as_index=False,
        )
        .agg(
            mean_absolute_value_delta=(
                "absolute_value_delta_from_baseline",
                "mean",
            ),
            maximum_absolute_value_delta=(
                "absolute_value_delta_from_baseline",
                "max",
            ),
            mean_absolute_rank_shift=(
                "absolute_rank_shift_from_baseline",
                "mean",
            ),
            maximum_absolute_rank_shift=(
                "absolute_rank_shift_from_baseline",
                "max",
            ),
        )
    )

    overall_movement = (
        movement_summary.loc[
            movement_summary[
                "representation_level"
            ].eq(
                "overall_equal_dimension_mean"
            )
        ]
        .sort_values(
            "mean_absolute_rank_shift",
            ascending=False,
        )
        .to_dict(
            orient="records"
        )
    )

    baseline_correlations = (
        rank_correlations.loc[
            rank_correlations[
                "specification_a"
            ].eq(
                BASELINE_SPECIFICATION_ID
            )
            | rank_correlations[
                "specification_b"
            ].eq(
                BASELINE_SPECIFICATION_ID
            )
        ]
        .sort_values(
            [
                "representation_level",
                "rank_correlation",
            ]
        )
        .to_dict(
            orient="records"
        )
    )

    largest_overall_disagreements = (
        disagreements.loc[
            disagreements[
                "representation_level"
            ].eq(
                "overall_equal_dimension_mean"
            )
        ]
        .sort_values(
            "absolute_rank_shift_from_baseline",
            ascending=False,
        )
        .head(20)[
            [
                "country",
                "specification_id",
                "representation_value",
                "baseline_value",
                "rank_shift_from_baseline",
                "movement_direction",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    exclusion_records = (
        exclusions.to_dict(
            orient="records"
        )
        if not exclusions.empty
        else []
    )

    report = f"""# Study 091B — National-Team Representation Comparison

## Purpose

Compare how mathematically and behaviorally distinct aggregation
specifications reshape real national-team expected-XI representations.

## Population contract

- Teams: {metadata["team_count"]}
- Population per team: complete expected 4-3-3 starting XI
- Player dimensions: attack, midfield, defense
- Compatible aggregation specifications:
  {metadata["compatible_specification_count"]}
- Baseline: `{BASELINE_SPECIFICATION_ID}`

## Compatibility exclusions

{json.dumps(exclusion_records, indent=2)}

Excluded methods remain documented rather than silently omitted.

## Representation outputs

- Team-dimension rows:
  {metadata["team_dimension_value_row_count"]}
- Team-profile rows:
  {metadata["team_profile_row_count"]}
- Baseline-comparison rows:
  {metadata["baseline_comparison_row_count"]}
- Pairwise correlation rows:
  {metadata["rank_correlation_row_count"]}
- Flagged disagreement rows:
  {metadata["disagreement_row_count"]}

## Overall movement from the top-five arithmetic baseline

{json.dumps(overall_movement, indent=2)}

## Rank correlations involving the baseline

{json.dumps(baseline_correlations, indent=2)}

## Largest descriptive overall disagreements

{json.dumps(largest_overall_disagreements, indent=2)}

## Interpretation boundary

The equal-dimension mean is a descriptive comparison device only. It
is not a new production score and does not replace dimension-specific
attack, midfield, or defense values.

A team moving upward under an elite-sensitive method means that its
expected XI contains a player-value distribution favored by that
aggregation philosophy. It does not establish that the team is
objectively stronger or that the method predicts matches better.

## Methodological boundary

This study:

- generates no production repository;
- changes no simulation configuration;
- fits no goal model;
- evaluates no match outcomes;
- creates no universal aggregation ranking;
- makes no claim about predictive superiority.

## Result

**OVERALL RESULT: {metadata["status"]}**

The compatible aggregation philosophies were compared on real
national-team expected XIs while preserving the distinction between
descriptive representation movement and predictive performance.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 091B — NATIONAL-TEAM REPRESENTATION COMPARISON"
    )
    print("=" * 88)

    (
        compatibility,
        specification_summary,
        team_coverage,
    ) = load_inputs()

    (
        compatible_ids,
        exclusions,
    ) = select_compatible_specifications(
        specification_summary
    )

    dimension_values = (
        build_team_dimension_values(
            compatibility=compatibility,
            compatible_ids=compatible_ids,
            team_coverage=team_coverage,
        )
    )

    team_profiles = build_team_profiles(
        dimension_values
    )

    baseline_comparisons = (
        build_baseline_comparisons(
            dimension_values,
            team_profiles,
        )
    )

    rank_correlations = (
        build_rank_correlations(
            baseline_comparisons
        )
    )

    disagreements = build_disagreements(
        baseline_comparisons
    )

    validate_outputs(
        compatible_ids=compatible_ids,
        dimension_values=dimension_values,
        team_profiles=team_profiles,
        baseline_comparisons=baseline_comparisons,
        rank_correlations=rank_correlations,
        disagreements=disagreements,
        team_coverage=team_coverage,
    )

    metadata = build_metadata(
        compatible_ids=compatible_ids,
        exclusions=exclusions,
        dimension_values=dimension_values,
        team_profiles=team_profiles,
        baseline_comparisons=baseline_comparisons,
        rank_correlations=rank_correlations,
        disagreements=disagreements,
        team_coverage=team_coverage,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dimension_values.to_csv(
        TEAM_DIMENSION_VALUES_PATH,
        index=False,
    )

    team_profiles.to_csv(
        TEAM_PROFILE_VALUES_PATH,
        index=False,
    )

    baseline_comparisons.to_csv(
        BASELINE_COMPARISON_PATH,
        index=False,
    )

    rank_correlations.to_csv(
        RANK_CORRELATION_PATH,
        index=False,
    )

    disagreements.to_csv(
        DISAGREEMENT_PATH,
        index=False,
    )

    exclusions.to_csv(
        EXCLUSION_PATH,
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
        metadata=metadata,
        baseline_comparisons=(
            baseline_comparisons
        ),
        rank_correlations=rank_correlations,
        disagreements=disagreements,
        exclusions=exclusions,
    )

    print()
    print("Study population")
    print("-" * 88)
    print(
        f"  Complete expected-XI teams: "
        f"{metadata['team_count']}"
    )
    print(
        f"  Compatible specifications: "
        f"{metadata['compatible_specification_count']}"
    )
    print(
        f"  Excluded specifications: "
        f"{metadata['excluded_specification_count']}"
    )

    print()
    print("Output coverage")
    print("-" * 88)
    print(
        f"  Team-dimension rows: "
        f"{len(dimension_values)}"
    )
    print(
        f"  Team-profile rows: "
        f"{len(team_profiles)}"
    )
    print(
        f"  Baseline comparisons: "
        f"{len(baseline_comparisons)}"
    )
    print(
        f"  Pairwise correlations: "
        f"{len(rank_correlations)}"
    )
    print(
        f"  Flagged disagreements: "
        f"{len(disagreements)}"
    )

    print()
    print("Mean absolute overall rank movement")
    print("-" * 88)

    overall_movement = (
        baseline_comparisons.loc[
            baseline_comparisons[
                "representation_level"
            ].eq(
                "overall_equal_dimension_mean"
            )
            & ~baseline_comparisons[
                "specification_id"
            ].eq(
                BASELINE_SPECIFICATION_ID
            )
        ]
        .groupby(
            "specification_id"
        )[
            "absolute_rank_shift_from_baseline"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        overall_movement.to_string()
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Study 091A compatibility filter: PASS")
    print("  Complete-XI team coverage: PASS")
    print("  Dimension completeness: PASS")
    print("  Baseline availability: PASS")
    print("  Duplicate-key audit: PASS")
    print("  Finite-value audit: PASS")
    print("  Production repository generated: NO")
    print("  Goal model fitted: NO")
    print("  Predictive claims made: NO")
    print("  Production behavior changed: NO")

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
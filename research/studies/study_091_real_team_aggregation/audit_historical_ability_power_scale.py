# audit_historical_ability_power_scale.py

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


PLAYER_POPULATION_PATH = (
    INPUT_DIRECTORY
    / "expected_lineup_player_population.csv"
)

BASELINE_COMPARISON_PATH = (
    INPUT_DIRECTORY
    / "national_team_aggregation_baseline_comparisons.csv"
)

DIMENSION_VALUES_PATH = (
    INPUT_DIRECTORY
    / "national_team_dimension_aggregation_values.csv"
)


TEAM_DIMENSION_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "historical_ability_power_team_dimension_audit.csv"
)

PLAYER_CONTRIBUTION_PATH = (
    OUTPUT_DIRECTORY
    / "historical_ability_power_player_contributions.csv"
)

TEAM_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "historical_ability_power_team_summary.csv"
)

EXTREME_CASE_PATH = (
    OUTPUT_DIRECTORY
    / "historical_ability_power_extreme_cases.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_091c_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_091c_report.md"
)


BASELINE_SPECIFICATION_ID = "top5_arithmetic"
HISTORICAL_SPECIFICATION_ID = "ability_power_gamma_2"

GAMMA = 2.0

DIMENSION_COLUMNS = {
    "attack": "projected_attack",
    "midfield": "projected_midfield",
    "defense": "projected_defense",
}

REQUIRED_PLAYER_COLUMNS = {
    "country",
    "player_id",
    "player_lineup",
    "slot",
    "role",
    "projected_attack",
    "projected_midfield",
    "projected_defense",
}

REQUIRED_COMPARISON_COLUMNS = {
    "country",
    "representation_level",
    "specification_id",
    "representation_value",
    "baseline_value",
    "rank_shift_from_baseline",
    "absolute_rank_shift_from_baseline",
}

REQUIRED_DIMENSION_COLUMNS = {
    "country",
    "dimension",
    "specification_id",
    "aggregated_value",
}

ABSOLUTE_TOLERANCE = 1e-12

TOP_EXTREME_TEAMS = 25


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


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    player_population = pd.read_csv(
        PLAYER_POPULATION_PATH
    )

    baseline_comparisons = pd.read_csv(
        BASELINE_COMPARISON_PATH
    )

    dimension_values = pd.read_csv(
        DIMENSION_VALUES_PATH
    )

    require_columns(
        player_population,
        REQUIRED_PLAYER_COLUMNS,
        frame_name="Expected-lineup player population",
    )

    require_columns(
        baseline_comparisons,
        REQUIRED_COMPARISON_COLUMNS,
        frame_name="Baseline comparisons",
    )

    require_columns(
        dimension_values,
        REQUIRED_DIMENSION_COLUMNS,
        frame_name="Dimension aggregation values",
    )

    return (
        player_population,
        baseline_comparisons,
        dimension_values,
    )


def ability_power_components(
    values: tuple[float, ...],
    *,
    gamma: float = GAMMA,
) -> dict[str, object]:
    """
    Reproduce the frozen historical ability-power calculation.

    Negative inputs are clipped to zero before weight construction:

        clipped_i = max(value_i, 0)
        weight_i = clipped_i ** gamma

    The result is:

        sum(clipped_i * weight_i) / sum(weight_i)
    """

    if not values:
        raise ValueError(
            "Cannot audit an empty player population."
        )

    clipped = tuple(
        max(float(value), 0.0)
        for value in values
    )

    weights = tuple(
        value ** gamma
        for value in clipped
    )

    total_weight = math.fsum(
        weights
    )

    if math.isclose(
        total_weight,
        0.0,
        rel_tol=1e-12,
        abs_tol=ABSOLUTE_TOLERANCE,
    ):
        historical_value = 0.0
        normalized_weights = tuple(
            0.0
            for _ in weights
        )
    else:
        historical_value = (
            math.fsum(
                value * weight
                for value, weight in zip(
                    clipped,
                    weights,
                    strict=True,
                )
            )
            / total_weight
        )

        normalized_weights = tuple(
            weight / total_weight
            for weight in weights
        )

    positive_count = sum(
        value > 0.0
        for value in values
    )

    zero_count = sum(
        math.isclose(
            value,
            0.0,
            rel_tol=1e-12,
            abs_tol=ABSOLUTE_TOLERANCE,
        )
        for value in values
    )

    negative_count = sum(
        value < 0.0
        for value in values
    )

    top_five = sorted(
        values,
        reverse=True,
    )[:5]

    top_five_mean = (
        math.fsum(top_five)
        / len(top_five)
    )

    full_mean = (
        math.fsum(values)
        / len(values)
    )

    positive_values = [
        value
        for value in values
        if value > 0.0
    ]

    positive_mean = (
        math.fsum(
            positive_values
        )
        / len(positive_values)
        if positive_values
        else 0.0
    )

    maximum_weight_share = (
        max(normalized_weights)
        if normalized_weights
        else 0.0
    )

    effective_player_count = (
        1.0
        / math.fsum(
            weight ** 2
            for weight in normalized_weights
        )
        if total_weight > 0.0
        else 0.0
    )

    top_weight_shares = sorted(
        normalized_weights,
        reverse=True,
    )

    top_one_weight_share = (
        top_weight_shares[0]
        if top_weight_shares
        else 0.0
    )

    top_three_weight_share = math.fsum(
        top_weight_shares[:3]
    )

    return {
        "historical_value":
            float(historical_value),
        "top_five_mean":
            float(top_five_mean),
        "full_population_mean":
            float(full_mean),
        "positive_only_mean":
            float(positive_mean),
        "historical_minus_top_five":
            float(
                historical_value
                - top_five_mean
            ),
        "historical_minus_full_mean":
            float(
                historical_value
                - full_mean
            ),
        "positive_player_count":
            positive_count,
        "zero_player_count":
            zero_count,
        "negative_player_count":
            negative_count,
        "positive_player_share":
            positive_count / len(values),
        "negative_player_share":
            negative_count / len(values),
        "minimum_input":
            float(min(values)),
        "maximum_input":
            float(max(values)),
        "input_range":
            float(
                max(values)
                - min(values)
            ),
        "maximum_weight_share":
            float(maximum_weight_share),
        "top_one_weight_share":
            float(top_one_weight_share),
        "top_three_weight_share":
            float(top_three_weight_share),
        "effective_player_count":
            float(effective_player_count),
        "total_raw_weight":
            float(total_weight),
        "clipped_value_sum":
            float(math.fsum(clipped)),
    }


def build_team_dimension_audit(
    player_population: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for country, team_rows in (
        player_population
        .groupby(
            "country",
            sort=True,
        )
    ):
        if len(team_rows) != 11:
            continue

        for dimension, column in (
            DIMENSION_COLUMNS.items()
        ):
            values = tuple(
                float(value)
                for value in team_rows[column]
            )

            components = (
                ability_power_components(
                    values,
                    gamma=GAMMA,
                )
            )

            records.append(
                {
                    "country":
                        country,
                    "dimension":
                        dimension,
                    "player_count":
                        len(values),
                    **components,
                }
            )

    output = pd.DataFrame(
        records
    )

    output[
        "absolute_historical_minus_top_five"
    ] = output[
        "historical_minus_top_five"
    ].abs()

    output[
        "clipping_exposure"
    ] = output[
        "negative_player_count"
    ].gt(0)

    return (
        output
        .sort_values(
            [
                "absolute_historical_minus_top_five",
                "country",
                "dimension",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_player_contributions(
    player_population: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for country, team_rows in (
        player_population
        .groupby(
            "country",
            sort=True,
        )
    ):
        if len(team_rows) != 11:
            continue

        for dimension, column in (
            DIMENSION_COLUMNS.items()
        ):
            values = tuple(
                float(value)
                for value in team_rows[column]
            )

            clipped_values = tuple(
                max(value, 0.0)
                for value in values
            )

            raw_weights = tuple(
                value ** GAMMA
                for value in clipped_values
            )

            total_weight = math.fsum(
                raw_weights
            )

            if total_weight > 0.0:
                normalized_weights = tuple(
                    weight / total_weight
                    for weight in raw_weights
                )
            else:
                normalized_weights = tuple(
                    0.0
                    for _ in raw_weights
                )

            contributions = tuple(
                value * weight
                for value, weight in zip(
                    clipped_values,
                    normalized_weights,
                    strict=True,
                )
            )

            order = sorted(
                range(len(values)),
                key=lambda index: (
                    values[index],
                    str(
                        team_rows.iloc[
                            index
                        ]["player_id"]
                    ),
                ),
                reverse=True,
            )

            dimension_ranks = {
                index: rank
                for rank, index in enumerate(
                    order,
                    start=1,
                )
            }

            for row_index, (
                original_value,
                clipped_value,
                raw_weight,
                normalized_weight,
                contribution,
            ) in enumerate(
                zip(
                    values,
                    clipped_values,
                    raw_weights,
                    normalized_weights,
                    contributions,
                    strict=True,
                )
            ):
                player = team_rows.iloc[
                    row_index
                ]

                records.append(
                    {
                        "country":
                            country,
                        "dimension":
                            dimension,
                        "player_id":
                            player["player_id"],
                        "player_name":
                            player[
                                "player_lineup"
                            ],
                        "slot":
                            player["slot"],
                        "role":
                            player["role"],
                        "dimension_rank":
                            dimension_ranks[
                                row_index
                            ],
                        "original_value":
                            original_value,
                        "clipped_value":
                            clipped_value,
                        "value_was_clipped":
                            original_value < 0.0,
                        "raw_weight":
                            raw_weight,
                        "normalized_weight_share":
                            normalized_weight,
                        "weighted_contribution":
                            contribution,
                    }
                )

    output = pd.DataFrame(
        records
    )

    return (
        output
        .sort_values(
            [
                "country",
                "dimension",
                "dimension_rank",
            ]
        )
        .reset_index(drop=True)
    )


def merge_observed_outputs(
    *,
    audit: pd.DataFrame,
    dimension_values: pd.DataFrame,
) -> pd.DataFrame:
    selected = dimension_values.loc[
        dimension_values[
            "specification_id"
        ].isin(
            {
                BASELINE_SPECIFICATION_ID,
                HISTORICAL_SPECIFICATION_ID,
            }
        )
    ][
        [
            "country",
            "dimension",
            "specification_id",
            "aggregated_value",
        ]
    ].copy()

    wide = (
        selected
        .pivot(
            index=[
                "country",
                "dimension",
            ],
            columns="specification_id",
            values="aggregated_value",
        )
        .reset_index()
    )

    wide = wide.rename(
        columns={
            BASELINE_SPECIFICATION_ID:
                "observed_top5_arithmetic",
            HISTORICAL_SPECIFICATION_ID:
                "observed_ability_power",
        }
    )

    output = audit.merge(
        wide,
        on=[
            "country",
            "dimension",
        ],
        how="left",
        validate="one_to_one",
    )

    required_observed = {
        "observed_top5_arithmetic",
        "observed_ability_power",
    }

    if not required_observed.issubset(
        output.columns
    ):
        raise AssertionError(
            "Observed aggregation output is incomplete."
        )

    output[
        "reproduction_error_top5"
    ] = (
        output[
            "top_five_mean"
        ]
        - output[
            "observed_top5_arithmetic"
        ]
    ).abs()

    output[
        "reproduction_error_ability_power"
    ] = (
        output[
            "historical_value"
        ]
        - output[
            "observed_ability_power"
        ]
    ).abs()

    return output


def build_team_summary(
    *,
    team_dimension_audit: pd.DataFrame,
    baseline_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    dimension_summary = (
        team_dimension_audit
        .groupby(
            "country",
            as_index=False,
        )
        .agg(
            mean_historical_value=(
                "historical_value",
                "mean",
            ),
            mean_top_five_value=(
                "top_five_mean",
                "mean",
            ),
            mean_historical_minus_top_five=(
                "historical_minus_top_five",
                "mean",
            ),
            mean_absolute_historical_minus_top_five=(
                "absolute_historical_minus_top_five",
                "mean",
            ),
            maximum_absolute_historical_minus_top_five=(
                "absolute_historical_minus_top_five",
                "max",
            ),
            total_negative_dimension_values=(
                "negative_player_count",
                "sum",
            ),
            mean_negative_player_share=(
                "negative_player_share",
                "mean",
            ),
            mean_positive_player_share=(
                "positive_player_share",
                "mean",
            ),
            mean_effective_player_count=(
                "effective_player_count",
                "mean",
            ),
            minimum_effective_player_count=(
                "effective_player_count",
                "min",
            ),
            mean_top_one_weight_share=(
                "top_one_weight_share",
                "mean",
            ),
            maximum_top_one_weight_share=(
                "top_one_weight_share",
                "max",
            ),
            mean_top_three_weight_share=(
                "top_three_weight_share",
                "mean",
            ),
        )
    )

    overall_movement = (
        baseline_comparisons.loc[
            baseline_comparisons[
                "representation_level"
            ].eq(
                "overall_equal_dimension_mean"
            )
            & baseline_comparisons[
                "specification_id"
            ].eq(
                HISTORICAL_SPECIFICATION_ID
            )
        ][
            [
                "country",
                "representation_value",
                "baseline_value",
                "rank_shift_from_baseline",
                "absolute_rank_shift_from_baseline",
                "value_delta_from_baseline",
                "absolute_value_delta_from_baseline",
            ]
        ]
        .rename(
            columns={
                "representation_value":
                    "observed_historical_overall",
                "baseline_value":
                    "observed_baseline_overall",
                "rank_shift_from_baseline":
                    "overall_rank_shift",
                "absolute_rank_shift_from_baseline":
                    "absolute_overall_rank_shift",
                "value_delta_from_baseline":
                    "overall_value_delta",
                "absolute_value_delta_from_baseline":
                    "absolute_overall_value_delta",
            }
        )
    )

    output = dimension_summary.merge(
        overall_movement,
        on="country",
        how="inner",
        validate="one_to_one",
    )

    output[
        "has_negative_dimension_exposure"
    ] = output[
        "total_negative_dimension_values"
    ].gt(0)

    return (
        output
        .sort_values(
            [
                "absolute_overall_rank_shift",
                "absolute_overall_value_delta",
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


def build_extreme_cases(
    team_summary: pd.DataFrame,
) -> pd.DataFrame:
    largest_rank_movements = (
        team_summary
        .sort_values(
            [
                "absolute_overall_rank_shift",
                "absolute_overall_value_delta",
            ],
            ascending=False,
        )
        .head(
            TOP_EXTREME_TEAMS
        )
        .copy()
    )

    largest_rank_movements[
        "extreme_case_type"
    ] = "largest_absolute_rank_shift"

    largest_positive_value_changes = (
        team_summary
        .sort_values(
            "overall_value_delta",
            ascending=False,
        )
        .head(
            TOP_EXTREME_TEAMS
        )
        .copy()
    )

    largest_positive_value_changes[
        "extreme_case_type"
    ] = "largest_positive_value_delta"

    largest_negative_value_changes = (
        team_summary
        .sort_values(
            "overall_value_delta",
            ascending=True,
        )
        .head(
            TOP_EXTREME_TEAMS
        )
        .copy()
    )

    largest_negative_value_changes[
        "extreme_case_type"
    ] = "largest_negative_value_delta"

    lowest_effective_player_counts = (
        team_summary
        .sort_values(
            "mean_effective_player_count",
            ascending=True,
        )
        .head(
            TOP_EXTREME_TEAMS
        )
        .copy()
    )

    lowest_effective_player_counts[
        "extreme_case_type"
    ] = "lowest_effective_player_count"

    return (
        pd.concat(
            [
                largest_rank_movements,
                largest_positive_value_changes,
                largest_negative_value_changes,
                lowest_effective_player_counts,
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "extreme_case_type",
                "absolute_overall_rank_shift",
                "country",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def validate_outputs(
    *,
    team_dimension_audit: pd.DataFrame,
    player_contributions: pd.DataFrame,
    team_summary: pd.DataFrame,
    extreme_cases: pd.DataFrame,
) -> None:
    if team_dimension_audit.empty:
        raise AssertionError(
            "Team-dimension audit is empty."
        )

    if player_contributions.empty:
        raise AssertionError(
            "Player-contribution output is empty."
        )

    if team_summary.empty:
        raise AssertionError(
            "Team summary is empty."
        )

    if extreme_cases.empty:
        raise AssertionError(
            "Extreme-case output is empty."
        )

    if team_dimension_audit[
        [
            "country",
            "dimension",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Team-dimension audit contains duplicate keys."
        )

    if player_contributions[
        [
            "country",
            "dimension",
            "player_id",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Player contributions contain duplicate keys."
        )

    if team_summary[
        "country"
    ].duplicated().any():
        raise AssertionError(
            "Team summary contains duplicate countries."
        )

    maximum_top5_error = float(
        team_dimension_audit[
            "reproduction_error_top5"
        ].max()
    )

    maximum_historical_error = float(
        team_dimension_audit[
            "reproduction_error_ability_power"
        ].max()
    )

    if maximum_top5_error > ABSOLUTE_TOLERANCE:
        raise AssertionError(
            "Top-five arithmetic reproduction exceeded tolerance: "
            f"{maximum_top5_error!r}."
        )

    if maximum_historical_error > ABSOLUTE_TOLERANCE:
        raise AssertionError(
            "Historical ability-power reproduction exceeded "
            f"tolerance: {maximum_historical_error!r}."
        )

    weight_sums = (
        player_contributions
        .groupby(
            [
                "country",
                "dimension",
            ]
        )[
            "normalized_weight_share"
        ]
        .sum()
    )

    invalid_weight_sums = weight_sums.loc[
        ~np.isclose(
            weight_sums.to_numpy(
                dtype=float
            ),
            1.0,
            rtol=1e-12,
            atol=1e-12,
        )
        & ~np.isclose(
            weight_sums.to_numpy(
                dtype=float
            ),
            0.0,
            rtol=1e-12,
            atol=1e-12,
        )
    ]

    if not invalid_weight_sums.empty:
        raise AssertionError(
            "Historical normalized weights do not sum to zero "
            "or one."
        )

    numeric_frames = (
        team_dimension_audit,
        player_contributions,
        team_summary,
    )

    for frame in numeric_frames:
        numeric = frame.select_dtypes(
            include="number"
        )

        invalid = numeric.map(
            lambda value: (
                False
                if pd.isna(value)
                else not math.isfinite(
                    float(value)
                )
            )
        )

        if invalid.any().any():
            raise AssertionError(
                "Study 091C output contains non-finite values."
            )


def build_metadata(
    *,
    team_dimension_audit: pd.DataFrame,
    player_contributions: pd.DataFrame,
    team_summary: pd.DataFrame,
    extreme_cases: pd.DataFrame,
) -> dict[str, Any]:
    clipping_rows = int(
        team_dimension_audit[
            "clipping_exposure"
        ].sum()
    )

    clipping_teams = int(
        team_summary[
            "has_negative_dimension_exposure"
        ].sum()
    )

    rank_shift_correlation = float(
        team_summary[
            "absolute_overall_rank_shift"
        ].corr(
            team_summary[
                "mean_negative_player_share"
            ],
            method="spearman",
        )
    )

    effective_count_correlation = float(
        team_summary[
            "absolute_overall_rank_shift"
        ].corr(
            team_summary[
                "mean_effective_player_count"
            ],
            method="spearman",
        )
    )

    top_one_share_correlation = float(
        team_summary[
            "absolute_overall_rank_shift"
        ].corr(
            team_summary[
                "mean_top_one_weight_share"
            ],
            method="spearman",
        )
    )

    return {
        "study_id": "091C",
        "study_name": (
            "Historical Ability-Power Scale Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "historical_specification_id":
            HISTORICAL_SPECIFICATION_ID,
        "baseline_specification_id":
            BASELINE_SPECIFICATION_ID,
        "gamma":
            GAMMA,
        "team_count":
            len(team_summary),
        "team_dimension_audit_row_count":
            len(team_dimension_audit),
        "player_contribution_row_count":
            len(player_contributions),
        "extreme_case_row_count":
            len(extreme_cases),
        "team_dimension_rows_with_negative_input_clipping":
            clipping_rows,
        "teams_with_negative_dimension_exposure":
            clipping_teams,
        "spearman_absolute_rank_shift_vs_negative_share":
            rank_shift_correlation,
        "spearman_absolute_rank_shift_vs_effective_player_count":
            effective_count_correlation,
        "spearman_absolute_rank_shift_vs_top_one_weight_share":
            top_one_share_correlation,
        "negative_value_policy": (
            "Negative projected values are clipped to zero before "
            "ability-power weights and contributions are computed."
        ),
        "ranking_generated":
            False,
        "new_aggregation_created":
            False,
        "goal_model_fitted":
            False,
        "production_repository_changed":
            False,
        "production_runtime_changed":
            False,
        "interpretation_boundary": (
            "This audit explains the historical control's scale "
            "behavior. It does not determine whether that behavior "
            "improves prediction."
        ),
        "outputs": [
            TEAM_DIMENSION_AUDIT_PATH.name,
            PLAYER_CONTRIBUTION_PATH.name,
            TEAM_SUMMARY_PATH.name,
            EXTREME_CASE_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    team_dimension_audit: pd.DataFrame,
    team_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    largest_movements = (
        team_summary
        .head(20)[
            [
                "country",
                "overall_rank_shift",
                "overall_value_delta",
                "total_negative_dimension_values",
                "mean_negative_player_share",
                "mean_effective_player_count",
                "mean_top_one_weight_share",
                "mean_top_three_weight_share",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    highest_clipping_exposure = (
        team_summary
        .sort_values(
            [
                "mean_negative_player_share",
                "absolute_overall_rank_shift",
            ],
            ascending=False,
        )
        .head(20)[
            [
                "country",
                "overall_rank_shift",
                "overall_value_delta",
                "total_negative_dimension_values",
                "mean_negative_player_share",
                "mean_effective_player_count",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    most_concentrated = (
        team_summary
        .sort_values(
            [
                "mean_effective_player_count",
                "mean_top_one_weight_share",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(20)[
            [
                "country",
                "overall_rank_shift",
                "overall_value_delta",
                "mean_effective_player_count",
                "mean_top_one_weight_share",
                "mean_top_three_weight_share",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    dimension_summary = (
        team_dimension_audit
        .groupby(
            "dimension",
            as_index=False,
        )
        .agg(
            mean_absolute_difference=(
                "absolute_historical_minus_top_five",
                "mean",
            ),
            maximum_absolute_difference=(
                "absolute_historical_minus_top_five",
                "max",
            ),
            rows_with_clipping=(
                "clipping_exposure",
                "sum",
            ),
            mean_effective_player_count=(
                "effective_player_count",
                "mean",
            ),
            mean_top_one_weight_share=(
                "top_one_weight_share",
                "mean",
            ),
        )
        .to_dict(
            orient="records"
        )
    )

    report = f"""# Study 091C — Historical Ability-Power Scale Audit

## Purpose

Explain why the historical `ability_power_gamma_2` aggregation produces
substantially larger national-team representation movements than the
top-five arithmetic baseline.

## Frozen historical formula

For each player value:

1. Negative values are clipped to zero.
2. The clipped value is raised to `gamma = {GAMMA}` to construct a weight.
3. The final value is a weighted average of the clipped abilities.

This means the historical formula combines two effects:

- suppression of negative projected values;
- nonlinear concentration on stronger positive values.

## Coverage

- Teams: {metadata["team_count"]}
- Team-dimension audit rows:
  {metadata["team_dimension_audit_row_count"]}
- Player-contribution rows:
  {metadata["player_contribution_row_count"]}
- Team-dimension rows exposed to negative-value clipping:
  {metadata["team_dimension_rows_with_negative_input_clipping"]}
- Teams with negative-value exposure:
  {metadata["teams_with_negative_dimension_exposure"]}

## Association with overall rank movement

Spearman correlations:

- Absolute rank shift versus mean negative-player share:
  {metadata["spearman_absolute_rank_shift_vs_negative_share"]}
- Absolute rank shift versus mean effective player count:
  {metadata["spearman_absolute_rank_shift_vs_effective_player_count"]}
- Absolute rank shift versus mean top-player weight share:
  {metadata["spearman_absolute_rank_shift_vs_top_one_weight_share"]}

These associations are descriptive and do not establish causation.

## Dimension-level behavior

{json.dumps(dimension_summary, indent=2)}

## Largest overall movements

{json.dumps(largest_movements, indent=2)}

## Teams with the greatest negative-value exposure

{json.dumps(highest_clipping_exposure, indent=2)}

## Most concentrated historical weighting

{json.dumps(most_concentrated, indent=2)}

## Interpretation

Large historical-control movements may arise through several mechanisms:

- negative players cease to lower the aggregate after clipping;
- only positive contributors receive nonzero weight;
- the quadratic weighting concentrates influence in a small number of
  strong players;
- teams with similar top-five means may have very different positive
  tails and effective contributor counts.

The effective player count summarizes weight concentration. A value near
one means that one player dominates the historical aggregate. A larger
value means that influence is distributed across more players.

## Methodological boundary

This study:

- introduces no new aggregation;
- changes no frozen formula;
- creates no production repository;
- fits no goal model;
- evaluates no match outcomes;
- makes no claim about predictive superiority.

## Result

**OVERALL RESULT: {metadata["status"]}**

The historical ability-power representation was decomposed into clipping,
weight concentration, and player-level contribution effects.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 091C — HISTORICAL ABILITY-POWER SCALE AUDIT"
    )
    print("=" * 88)

    (
        player_population,
        baseline_comparisons,
        dimension_values,
    ) = load_inputs()

    team_dimension_audit = (
        build_team_dimension_audit(
            player_population
        )
    )

    team_dimension_audit = (
        merge_observed_outputs(
            audit=team_dimension_audit,
            dimension_values=dimension_values,
        )
    )

    player_contributions = (
        build_player_contributions(
            player_population
        )
    )

    team_summary = build_team_summary(
        team_dimension_audit=team_dimension_audit,
        baseline_comparisons=baseline_comparisons,
    )

    extreme_cases = build_extreme_cases(
        team_summary
    )

    validate_outputs(
        team_dimension_audit=team_dimension_audit,
        player_contributions=player_contributions,
        team_summary=team_summary,
        extreme_cases=extreme_cases,
    )

    metadata = build_metadata(
        team_dimension_audit=team_dimension_audit,
        player_contributions=player_contributions,
        team_summary=team_summary,
        extreme_cases=extreme_cases,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    team_dimension_audit.to_csv(
        TEAM_DIMENSION_AUDIT_PATH,
        index=False,
    )

    player_contributions.to_csv(
        PLAYER_CONTRIBUTION_PATH,
        index=False,
    )

    team_summary.to_csv(
        TEAM_SUMMARY_PATH,
        index=False,
    )

    extreme_cases.to_csv(
        EXTREME_CASE_PATH,
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
        team_dimension_audit=(
            team_dimension_audit
        ),
        team_summary=team_summary,
        metadata=metadata,
    )

    print()
    print("Audit coverage")
    print("-" * 88)
    print(
        f"  Teams: "
        f"{metadata['team_count']}"
    )
    print(
        "  Team-dimension rows: "
        f"{len(team_dimension_audit)}"
    )
    print(
        "  Player-contribution rows: "
        f"{len(player_contributions)}"
    )
    print(
        "  Rows with clipping exposure: "
        f"{metadata['team_dimension_rows_with_negative_input_clipping']}"
    )

    print()
    print("Largest overall rank movements")
    print("-" * 88)
    print(
        team_summary[
            [
                "country",
                "overall_rank_shift",
                "overall_value_delta",
                "mean_negative_player_share",
                "mean_effective_player_count",
                "mean_top_one_weight_share",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Historical formula reproduction: PASS")
    print("  Top-five baseline reproduction: PASS")
    print("  Normalized-weight audit: PASS")
    print("  Finite-value audit: PASS")
    print("  New aggregation introduced: NO")
    print("  Team repository generated: NO")
    print("  Goal model fitted: NO")
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
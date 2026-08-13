#fingerprint_aggregation

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FINGERPRINT_METRICS = (
    "goals_per_match",
    "home_goals_per_match",
    "away_goals_per_match",

    "home_win_rate",
    "draw_rate",
    "away_win_rate",

    "both_teams_to_score_rate",

    "zero_goal_match_rate",
    "one_goal_match_rate",
    "two_goal_match_rate",
    "three_goal_match_rate",
    "four_plus_goal_match_rate",

    "one_goal_margin_rate",
    "three_plus_goal_margin_rate",

    "champion_points",
    "bottom_points",

    "points_spread",
    "goal_difference_spread",
)


RATE_METRICS = {
    "home_win_rate",
    "draw_rate",
    "away_win_rate",
    "both_teams_to_score_rate",
    "zero_goal_match_rate",
    "one_goal_match_rate",
    "two_goal_match_rate",
    "three_goal_match_rate",
    "four_plus_goal_match_rate",
    "one_goal_margin_rate",
    "three_plus_goal_margin_rate",
}


@dataclass(frozen=True)
class FingerprintAggregationResult:
    """
    Model-level league fingerprints aggregated across
    repeated simulated seasons.

    One row corresponds to one football model.
    """

    league_fingerprints: pd.DataFrame

    def validate(self) -> None:
        if self.league_fingerprints.empty:
            raise ValueError(
                "League fingerprint table is empty."
            )

def aggregate_league_fingerprints(
    season_statistics: pd.DataFrame,
) -> FingerprintAggregationResult:
    """
    Aggregate repeated season fingerprints into one row
    per football model.

    For every registered fingerprint metric, calculate:

    - arithmetic mean;
    - sample standard deviation.

    No model comparison or historical evaluation is
    performed here.
    """

    required_columns = {
        "model_id",
        "model_name",
        "season_number",
        *FINGERPRINT_METRICS,
    }

    missing_columns = (
        required_columns
        - set(season_statistics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Season statistics are missing fingerprint "
            f"columns: {sorted(missing_columns)}"
        )

    if season_statistics.empty:
        raise ValueError(
            "Season statistics are empty."
        )

    working = season_statistics.copy()

    numeric_columns = list(
        FINGERPRINT_METRICS
    )

    for column in numeric_columns:
        working[column] = pd.to_numeric(
            working[column],
            errors="raise",
        )

    numeric_values = working[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "Season statistics contain non-finite "
            "fingerprint values."
        )

    named_aggregations: dict[
        str,
        tuple[str, str],
    ] = {
        "season_count": (
            "season_number",
            "count",
        ),
    }

    for metric in FINGERPRINT_METRICS:
        named_aggregations[
            f"{metric}_mean"
        ] = (
            metric,
            "mean",
        )

        named_aggregations[
            f"{metric}_std"
        ] = (
            metric,
            "std",
        )

    summary = (
        working
        .groupby(
            [
                "model_id",
                "model_name",
            ],
            as_index=False,
        )
        .agg(
            **named_aggregations
        )
        .sort_values(
            "model_id"
        )
        .reset_index(
            drop=True
        )
    )

    standard_deviation_columns = [
        f"{metric}_std"
        for metric in FINGERPRINT_METRICS
    ]

    # pandas returns NaN sample standard deviation when only
    # one season is present. A one-season benchmark has zero
    # observed between-season dispersion.
    summary[
        standard_deviation_columns
    ] = (
        summary[
            standard_deviation_columns
        ]
        .fillna(0.0)
    )

    result = FingerprintAggregationResult(
        league_fingerprints=summary
    )

    result.validate()

    return result

def validate_league_fingerprints(
    *,
    fingerprint_result: FingerprintAggregationResult,
    season_statistics: pd.DataFrame,
    expected_model_ids: set[str] | None = None,
    expected_season_count: int | None = None,
) -> None:
    fingerprints = (
        fingerprint_result
        .league_fingerprints
    )

    fingerprint_result.validate()

    if fingerprints[
        "model_id"
    ].duplicated().any():
        duplicates = (
            fingerprints.loc[
                fingerprints[
                    "model_id"
                ].duplicated(
                    keep=False
                ),
                [
                    "model_id",
                    "model_name",
                ],
            ]
        )

        raise AssertionError(
            "League fingerprints contain duplicate "
            "model rows:\n"
            f"{duplicates.to_string(index=False)}"
        )

    observed_model_ids = set(
        fingerprints[
            "model_id"
        ].astype(str)
    )

    if (
        expected_model_ids is not None
        and observed_model_ids
        != expected_model_ids
    ):
        raise AssertionError(
            "League fingerprint model population does "
            "not match expectations. "
            f"Expected={sorted(expected_model_ids)}, "
            f"observed={sorted(observed_model_ids)}."
        )

    if (
        expected_season_count is not None
        and not fingerprints[
            "season_count"
        ].eq(
            expected_season_count
        ).all()
    ):
        raise AssertionError(
            "At least one model fingerprint contains an "
            "unexpected season count."
        )

    mean_columns = [
        f"{metric}_mean"
        for metric in FINGERPRINT_METRICS
    ]

    standard_deviation_columns = [
        f"{metric}_std"
        for metric in FINGERPRINT_METRICS
    ]

    numeric_columns = [
        "season_count",
        *mean_columns,
        *standard_deviation_columns,
    ]

    numeric_values = fingerprints[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise AssertionError(
            "League fingerprints contain non-finite "
            "numeric values."
        )

    if (
        fingerprints[
            standard_deviation_columns
        ].lt(0.0)
        .any()
        .any()
    ):
        raise AssertionError(
            "League fingerprint standard deviations "
            "must not be negative."
        )

    for metric in RATE_METRICS:
        mean_column = (
            f"{metric}_mean"
        )

        if not fingerprints[
            mean_column
        ].between(
            0.0,
            1.0,
        ).all():
            raise AssertionError(
                f"{mean_column} contains a value "
                "outside [0, 1]."
            )

    outcome_rate_sum = (
        fingerprints[
            [
                "home_win_rate_mean",
                "draw_rate_mean",
                "away_win_rate_mean",
            ]
        ]
        .sum(
            axis=1
        )
    )

    if not outcome_rate_sum.between(
        1.0 - 1e-12,
        1.0 + 1e-12,
    ).all():
        raise AssertionError(
            "Mean home/draw/away rates do not sum to one."
        )

    # Reconcile every stored mean against the raw season
    # population.
    raw_means = (
        season_statistics
        .groupby(
            "model_id"
        )[
            list(
                FINGERPRINT_METRICS
            )
        ]
        .mean()
        .sort_index()
    )

    fingerprint_means = (
        fingerprints
        .set_index(
            "model_id"
        )[
            mean_columns
        ]
        .rename(
            columns={
                f"{metric}_mean":
                    metric
                for metric
                in FINGERPRINT_METRICS
            }
        )
        .sort_index()
    )

    mean_differences = (
        fingerprint_means
        - raw_means
    )

    if not mean_differences.abs().le(
        1e-12
    ).all().all():
        raise AssertionError(
            "Aggregated fingerprint means do not "
            "reconcile with season statistics."
        )
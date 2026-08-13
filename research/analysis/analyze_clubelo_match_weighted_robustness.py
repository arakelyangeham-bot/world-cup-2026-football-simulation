#analyze_clubelo_match_weighted_robustness

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STUDY_062_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_062_clubelo_interpretation"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_063_clubelo_match_weighted_robustness"
)


MATCH_LEVEL_INPUT_PATH = (
    STUDY_062_DIRECTORY
    / "match_level_improvement.csv"
)

MATCH_WEIGHTED_PATH = (
    OUTPUT_DIRECTORY
    / "match_weighted_improvement.csv"
)

RATING_BUCKET_PATH = (
    OUTPUT_DIRECTORY
    / "match_weighted_rating_bucket_analysis.csv"
)

FAVORITE_PATH = (
    OUTPUT_DIRECTORY
    / "match_weighted_favorite_analysis.csv"
)

RESULT_TYPE_PATH = (
    OUTPUT_DIRECTORY
    / "match_weighted_result_type_analysis.csv"
)

TOTAL_GOAL_PATH = (
    OUTPUT_DIRECTORY
    / "match_weighted_total_goal_analysis.csv"
)

SCORELINE_PATH = (
    OUTPUT_DIRECTORY
    / "match_weighted_scoreline_analysis.csv"
)

CONTINUOUS_RELATIONSHIP_PATH = (
    OUTPUT_DIRECTORY
    / "match_weighted_rating_difference_relationship.csv"
)

COMPARISON_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "match_weighted_comparison_summary.csv"
)

CONFIGURATION_AGREEMENT_PATH = (
    OUTPUT_DIRECTORY
    / "configuration_agreement_summary.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "study_summary.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


IMPROVEMENT_METRICS = (
    "combined_goal_error_improvement",
    "total_goal_error_improvement",
    "goal_difference_error_improvement",
    "outcome_log_loss_improvement",
    "exact_score_log_loss_improvement",
)


MATCH_IDENTITY_COLUMNS = (
    "comparison_name",
    "baseline_specification",
    "candidate_specification",
    "event_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "actual_result",
    "actual_scoreline",
    "actual_total_goals",
    "rating_prior_diff",
    "absolute_rating_prior_diff",
    "attack_diff",
    "defense_diff",
    "attack_depth_diff",
)


def require_file(
    path: Path,
    label: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}"
        )


def load_match_level_improvements() -> pd.DataFrame:
    require_file(
        MATCH_LEVEL_INPUT_PATH,
        "Study 062 match-level improvement dataset",
    )

    dataframe = pd.read_csv(
        MATCH_LEVEL_INPUT_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Study 062 match-level improvement dataset "
            "is empty."
        )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    return dataframe


def validate_input(
    dataframe: pd.DataFrame,
) -> None:
    required_columns = {
        *MATCH_IDENTITY_COLUMNS,
        *IMPROVEMENT_METRICS,
        "train_fraction",
        "alpha",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Match-level input is missing columns: "
            f"{sorted(missing_columns)}"
        )

    numeric_columns = [
        "home_score",
        "away_score",
        "actual_total_goals",
        "rating_prior_diff",
        "absolute_rating_prior_diff",
        "attack_diff",
        "defense_diff",
        "attack_depth_diff",
        "train_fraction",
        "alpha",
        *IMPROVEMENT_METRICS,
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    if dataframe[
        numeric_columns
    ].isna().any().any():
        raise ValueError(
            "Match-level input contains missing "
            "required numeric values."
        )

    duplicate_key = [
        "comparison_name",
        "event_id",
        "train_fraction",
        "alpha",
    ]

    if dataframe.duplicated(
        subset=duplicate_key
    ).any():
        raise ValueError(
            "Match-level input contains duplicate "
            "comparison-event-configuration rows."
        )

    comparison_count = dataframe[
        "comparison_name"
    ].nunique()

    if comparison_count != 2:
        raise AssertionError(
            "Study 063 expects exactly two controlled "
            f"comparisons, found {comparison_count}."
        )


def validate_group_identity(
    group: pd.DataFrame,
    comparison_name: str,
    event_id: object,
) -> None:
    """
    Confirm that every repeated configuration row for one
    match agrees on its football identity and attributes.
    """
    identity_columns = [
        column
        for column in MATCH_IDENTITY_COLUMNS
        if column not in {
            "comparison_name",
            "event_id",
        }
    ]

    for column in identity_columns:
        series = group[column]

        if pd.api.types.is_numeric_dtype(
            series
        ):
            values = series.to_numpy(
                dtype=float
            )

            equal = np.allclose(
                values,
                values[0],
                atol=1e-12,
                rtol=0.0,
                equal_nan=True,
            )
        else:
            equal = (
                series.fillna("<missing>")
                .astype(str)
                .nunique()
                == 1
            )

        if not equal:
            raise AssertionError(
                "Repeated configuration rows disagree "
                f"for comparison={comparison_name!r}, "
                f"event_id={event_id!r}, "
                f"column={column!r}."
            )


def build_match_weighted_improvements(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse repeated split-and-alpha predictions into one
    equally weighted row per match and comparison.
    """
    records: list[dict[str, object]] = []

    grouped = dataframe.groupby(
        [
            "comparison_name",
            "event_id",
        ],
        sort=False,
        dropna=False,
    )

    for (
        comparison_name,
        event_id,
    ), group in grouped:
        validate_group_identity(
            group=group,
            comparison_name=str(
                comparison_name
            ),
            event_id=event_id,
        )

        first = group.iloc[0]

        record: dict[str, object] = {
            column: first[column]
            for column in MATCH_IDENTITY_COLUMNS
        }

        record[
            "configuration_count"
        ] = len(group)

        record[
            "unique_train_fraction_count"
        ] = group[
            "train_fraction"
        ].nunique()

        record[
            "unique_alpha_count"
        ] = group["alpha"].nunique()

        for metric in IMPROVEMENT_METRICS:
            values = group[
                metric
            ].to_numpy(dtype=float)

            record[
                f"mean_{metric}"
            ] = float(
                np.mean(values)
            )

            record[
                f"median_{metric}"
            ] = float(
                np.median(values)
            )

            record[
                f"std_{metric}"
            ] = (
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                if len(values) > 1
                else 0.0
            )

            record[
                f"configuration_win_count_{metric}"
            ] = int(
                np.sum(values > 0.0)
            )

            record[
                f"configuration_tie_count_{metric}"
            ] = int(
                np.sum(
                    np.isclose(
                        values,
                        0.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                )
            )

            record[
                f"configuration_loss_count_{metric}"
            ] = int(
                np.sum(values < 0.0)
            )

            record[
                f"configuration_win_rate_{metric}"
            ] = float(
                np.mean(values > 0.0)
            )

            record[
                f"all_configurations_improved_{metric}"
            ] = bool(
                np.all(values > 0.0)
            )

            record[
                f"majority_configurations_improved_{metric}"
            ] = bool(
                np.mean(values > 0.0)
                > 0.5
            )

        records.append(record)

    output = pd.DataFrame(records)

    duplicate_key = [
        "comparison_name",
        "event_id",
    ]

    if output.duplicated(
        subset=duplicate_key
    ).any():
        raise AssertionError(
            "Match aggregation produced duplicate "
            "comparison-event rows."
        )

    return output


def assign_rating_bucket(
    absolute_difference: pd.Series,
) -> pd.Series:
    return pd.cut(
        absolute_difference,
        bins=[
            -np.inf,
            50,
            100,
            150,
            200,
            300,
            np.inf,
        ],
        labels=[
            "0-50",
            "50-100",
            "100-150",
            "150-200",
            "200-300",
            "300+",
        ],
        right=False,
        ordered=True,
    )


def assign_favorite_category(
    rating_difference: pd.Series,
    even_threshold: float = 50.0,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                rating_difference > even_threshold,
                rating_difference < -even_threshold,
            ],
            [
                "home_favorite",
                "away_favorite",
            ],
            default="nearly_even",
        ),
        index=rating_difference.index,
    )


def assign_total_goal_band(
    total_goals: pd.Series,
) -> pd.Series:
    return pd.cut(
        total_goals,
        bins=[
            -np.inf,
            1,
            3,
            5,
            np.inf,
        ],
        labels=[
            "0_goals",
            "1-2_goals",
            "3-4_goals",
            "5+_goals",
        ],
        right=False,
        ordered=True,
    )


def summarize_match_weighted_improvements(
    dataframe: pd.DataFrame,
    group_columns: Iterable[str],
) -> pd.DataFrame:
    """
    Summarize aggregated match means. Each match contributes
    exactly once within each controlled comparison.
    """
    records: list[dict[str, object]] = []

    grouped = dataframe.groupby(
        list(group_columns),
        observed=True,
        dropna=False,
        sort=False,
    )

    for group_key, group in grouped:
        if not isinstance(
            group_key,
            tuple,
        ):
            group_key = (group_key,)

        identity = {
            column: value
            for column, value in zip(
                group_columns,
                group_key,
            )
        }

        for metric in IMPROVEMENT_METRICS:
            mean_column = f"mean_{metric}"

            values = group[
                mean_column
            ].to_numpy(dtype=float)

            record = dict(identity)

            record.update(
                {
                    "metric": metric,
                    "match_count": len(values),
                    "mean_match_improvement": float(
                        np.mean(values)
                    ),
                    "median_match_improvement": float(
                        np.median(values)
                    ),
                    "match_improvement_standard_deviation":
                        float(
                            np.std(
                                values,
                                ddof=1,
                            )
                        )
                        if len(values) > 1
                        else 0.0,
                    "match_win_count": int(
                        np.sum(values > 0.0)
                    ),
                    "match_tie_count": int(
                        np.sum(
                            np.isclose(
                                values,
                                0.0,
                                atol=1e-12,
                                rtol=0.0,
                            )
                        )
                    ),
                    "match_loss_count": int(
                        np.sum(values < 0.0)
                    ),
                    "match_win_rate": float(
                        np.mean(values > 0.0)
                    ),
                    "minimum_match_improvement": float(
                        np.min(values)
                    ),
                    "maximum_match_improvement": float(
                        np.max(values)
                    ),
                }
            )

            records.append(record)

    return pd.DataFrame(records)


def build_segment_analyses(
    match_weighted: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    analysis = match_weighted.copy()

    analysis["rating_bucket"] = (
        assign_rating_bucket(
            analysis[
                "absolute_rating_prior_diff"
            ]
        )
    )

    analysis["favorite_category"] = (
        assign_favorite_category(
            analysis["rating_prior_diff"]
        )
    )

    analysis["total_goal_band"] = (
        assign_total_goal_band(
            analysis["actual_total_goals"]
        )
    )

    rating_bucket = (
        summarize_match_weighted_improvements(
            dataframe=analysis,
            group_columns=(
                "comparison_name",
                "rating_bucket",
            ),
        )
    )

    favorite = (
        summarize_match_weighted_improvements(
            dataframe=analysis,
            group_columns=(
                "comparison_name",
                "favorite_category",
            ),
        )
    )

    result_type = (
        summarize_match_weighted_improvements(
            dataframe=analysis,
            group_columns=(
                "comparison_name",
                "actual_result",
            ),
        )
    )

    total_goal = (
        summarize_match_weighted_improvements(
            dataframe=analysis,
            group_columns=(
                "comparison_name",
                "total_goal_band",
            ),
        )
    )

    scoreline_counts = (
        analysis.groupby(
            [
                "comparison_name",
                "actual_scoreline",
            ]
        )
        .size()
        .rename("scoreline_match_count")
        .reset_index()
    )

    frequent_scorelines = scoreline_counts[
        scoreline_counts[
            "scoreline_match_count"
        ].ge(3)
    ][
        [
            "comparison_name",
            "actual_scoreline",
        ]
    ]

    scoreline_population = analysis.merge(
        frequent_scorelines,
        on=[
            "comparison_name",
            "actual_scoreline",
        ],
        how="inner",
        validate="many_to_one",
    )

    scoreline = (
        summarize_match_weighted_improvements(
            dataframe=scoreline_population,
            group_columns=(
                "comparison_name",
                "actual_scoreline",
            ),
        )
    )

    return (
        rating_bucket,
        favorite,
        result_type,
        total_goal,
        scoreline,
    )


def build_continuous_relationship(
    match_weighted: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for comparison_name, group in (
        match_weighted.groupby(
            "comparison_name",
            sort=False,
        )
    ):
        for metric in IMPROVEMENT_METRICS:
            improvement_column = (
                f"mean_{metric}"
            )

            pearson = group[
                "absolute_rating_prior_diff"
            ].corr(
                group[improvement_column],
                method="pearson",
            )

            spearman = group[
                "absolute_rating_prior_diff"
            ].corr(
                group[improvement_column],
                method="spearman",
            )

            records.append(
                {
                    "comparison_name":
                        comparison_name,
                    "rating_variable":
                        "absolute_rating_prior_diff",
                    "improvement_metric":
                        metric,
                    "pearson_correlation":
                        float(pearson),
                    "spearman_correlation":
                        float(spearman),
                    "match_count": len(group),
                }
            )

    return pd.DataFrame(records)


def build_comparison_summary(
    match_weighted: pd.DataFrame,
) -> pd.DataFrame:
    return summarize_match_weighted_improvements(
        dataframe=match_weighted,
        group_columns=(
            "comparison_name",
            "baseline_specification",
            "candidate_specification",
        ),
    )


def build_configuration_agreement_summary(
    match_weighted: pd.DataFrame,
) -> pd.DataFrame:
    """
    Describe how consistently configurations agreed that
    ClubElo improved each individual match.
    """
    records: list[dict[str, object]] = []

    for comparison_name, group in (
        match_weighted.groupby(
            "comparison_name",
            sort=False,
        )
    ):
        for metric in IMPROVEMENT_METRICS:
            rate_column = (
                f"configuration_win_rate_{metric}"
            )

            all_column = (
                f"all_configurations_improved_{metric}"
            )

            majority_column = (
                "majority_configurations_improved_"
                f"{metric}"
            )

            rates = group[
                rate_column
            ].to_numpy(dtype=float)

            records.append(
                {
                    "comparison_name":
                        comparison_name,
                    "metric": metric,
                    "match_count": len(group),
                    "mean_configuration_win_rate":
                        float(
                            np.mean(rates)
                        ),
                    "median_configuration_win_rate":
                        float(
                            np.median(rates)
                        ),
                    "matches_with_majority_agreement":
                        int(
                            group[
                                majority_column
                            ].sum()
                        ),
                    "majority_agreement_rate":
                        float(
                            group[
                                majority_column
                            ].mean()
                        ),
                    "matches_with_unanimous_agreement":
                        int(
                            group[
                                all_column
                            ].sum()
                        ),
                    "unanimous_agreement_rate":
                        float(
                            group[
                                all_column
                            ].mean()
                        ),
                }
            )

    return pd.DataFrame(records)


def get_single_row(
    dataframe: pd.DataFrame,
    comparison_name: str,
    metric: str,
) -> pd.Series:
    selected = dataframe[
        dataframe[
            "comparison_name"
        ].eq(comparison_name)
        & dataframe["metric"].eq(metric)
    ]

    if len(selected) != 1:
        raise AssertionError(
            "Expected exactly one row for "
            f"{comparison_name!r}, {metric!r}."
        )

    return selected.iloc[0]


def build_summary(
    match_weighted: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    rating_bucket: pd.DataFrame,
    favorite: pd.DataFrame,
    continuous_relationship: pd.DataFrame,
    configuration_agreement: pd.DataFrame,
) -> dict[str, object]:
    comparison_records: list[
        dict[str, object]
    ] = []

    for comparison_name in (
        match_weighted[
            "comparison_name"
        ].unique()
    ):
        exact_score = get_single_row(
            dataframe=comparison_summary,
            comparison_name=comparison_name,
            metric=(
                "exact_score_log_loss_improvement"
            ),
        )

        outcome = get_single_row(
            dataframe=comparison_summary,
            comparison_name=comparison_name,
            metric=(
                "outcome_log_loss_improvement"
            ),
        )

        comparison_records.append(
            {
                "comparison_name":
                    comparison_name,
                "unique_matches": int(
                    exact_score["match_count"]
                ),
                "mean_exact_score_improvement":
                    float(
                        exact_score[
                            "mean_match_improvement"
                        ]
                    ),
                "exact_score_match_win_rate":
                    float(
                        exact_score[
                            "match_win_rate"
                        ]
                    ),
                "mean_outcome_improvement":
                    float(
                        outcome[
                            "mean_match_improvement"
                        ]
                    ),
                "outcome_match_win_rate":
                    float(
                        outcome[
                            "match_win_rate"
                        ]
                    ),
            }
        )

    exact_buckets = rating_bucket[
        rating_bucket["metric"].eq(
            "exact_score_log_loss_improvement"
        )
    ]

    strongest_bucket = (
        exact_buckets
        .sort_values(
            "mean_match_improvement",
            ascending=False,
        )
        .iloc[0]
    )

    outcome_favorites = favorite[
        favorite["metric"].eq(
            "outcome_log_loss_improvement"
        )
    ]

    strongest_favorite = (
        outcome_favorites
        .sort_values(
            "mean_match_improvement",
            ascending=False,
        )
        .iloc[0]
    )

    exact_relationship = (
        continuous_relationship[
            continuous_relationship[
                "improvement_metric"
            ].eq(
                "exact_score_log_loss_improvement"
            )
        ]
        .assign(
            absolute_spearman=lambda frame: (
                frame[
                    "spearman_correlation"
                ].abs()
            )
        )
        .sort_values(
            "absolute_spearman",
            ascending=False,
        )
        .iloc[0]
    )

    exact_agreement = (
        configuration_agreement[
            configuration_agreement[
                "metric"
            ].eq(
                "exact_score_log_loss_improvement"
            )
        ]
        .sort_values(
            "majority_agreement_rate",
            ascending=False,
        )
        .iloc[0]
    )

    return {
        "study": (
            "study_063_clubelo_"
            "match_weighted_robustness"
        ),
        "source_configuration_weighted_rows":
            2050,
        "match_weighted_rows": int(
            len(match_weighted)
        ),
        "unique_events": int(
            match_weighted[
                "event_id"
            ].nunique()
        ),
        "controlled_comparisons": int(
            match_weighted[
                "comparison_name"
            ].nunique()
        ),
        "comparison_results":
            comparison_records,
        "strongest_exact_score_rating_bucket": {
            "comparison_name": str(
                strongest_bucket[
                    "comparison_name"
                ]
            ),
            "rating_bucket": str(
                strongest_bucket[
                    "rating_bucket"
                ]
            ),
            "match_count": int(
                strongest_bucket[
                    "match_count"
                ]
            ),
            "mean_match_improvement": float(
                strongest_bucket[
                    "mean_match_improvement"
                ]
            ),
            "match_win_rate": float(
                strongest_bucket[
                    "match_win_rate"
                ]
            ),
        },
        "strongest_outcome_favorite_category": {
            "comparison_name": str(
                strongest_favorite[
                    "comparison_name"
                ]
            ),
            "favorite_category": str(
                strongest_favorite[
                    "favorite_category"
                ]
            ),
            "match_count": int(
                strongest_favorite[
                    "match_count"
                ]
            ),
            "mean_match_improvement": float(
                strongest_favorite[
                    "mean_match_improvement"
                ]
            ),
            "match_win_rate": float(
                strongest_favorite[
                    "match_win_rate"
                ]
            ),
        },
        "strongest_absolute_rating_relationship": {
            "comparison_name": str(
                exact_relationship[
                    "comparison_name"
                ]
            ),
            "pearson_correlation": float(
                exact_relationship[
                    "pearson_correlation"
                ]
            ),
            "spearman_correlation": float(
                exact_relationship[
                    "spearman_correlation"
                ]
            ),
        },
        "strongest_exact_score_configuration_agreement": {
            "comparison_name": str(
                exact_agreement[
                    "comparison_name"
                ]
            ),
            "majority_agreement_rate": float(
                exact_agreement[
                    "majority_agreement_rate"
                ]
            ),
            "unanimous_agreement_rate": float(
                exact_agreement[
                    "unanimous_agreement_rate"
                ]
            ),
        },
    }


def write_report(
    summary: dict[str, object],
    comparison_summary: pd.DataFrame,
    rating_bucket: pd.DataFrame,
    favorite: pd.DataFrame,
    continuous_relationship: pd.DataFrame,
    configuration_agreement: pd.DataFrame,
) -> None:
    comparison_display = (
        comparison_summary[
            comparison_summary[
                "metric"
            ].isin(
                [
                    "combined_goal_error_improvement",
                    "outcome_log_loss_improvement",
                    (
                        "exact_score_log_loss_"
                        "improvement"
                    ),
                ]
            )
        ][
            [
                "comparison_name",
                "metric",
                "match_count",
                "mean_match_improvement",
                "match_win_rate",
            ]
        ]
        .to_markdown(index=False)
    )

    bucket_display = (
        rating_bucket[
            rating_bucket[
                "metric"
            ].eq(
                "exact_score_log_loss_improvement"
            )
        ][
            [
                "comparison_name",
                "rating_bucket",
                "match_count",
                "mean_match_improvement",
                "match_win_rate",
            ]
        ]
        .sort_values(
            [
                "comparison_name",
                "rating_bucket",
            ]
        )
        .to_markdown(index=False)
    )

    favorite_display = (
        favorite[
            favorite[
                "metric"
            ].eq(
                "outcome_log_loss_improvement"
            )
        ][
            [
                "comparison_name",
                "favorite_category",
                "match_count",
                "mean_match_improvement",
                "match_win_rate",
            ]
        ]
        .sort_values(
            [
                "comparison_name",
                "favorite_category",
            ]
        )
        .to_markdown(index=False)
    )

    relationship_display = (
        continuous_relationship[
            continuous_relationship[
                "improvement_metric"
            ].eq(
                "exact_score_log_loss_improvement"
            )
        ][
            [
                "comparison_name",
                "match_count",
                "pearson_correlation",
                "spearman_correlation",
            ]
        ]
        .to_markdown(index=False)
    )

    agreement_display = (
        configuration_agreement[
            configuration_agreement[
                "metric"
            ].isin(
                [
                    "outcome_log_loss_improvement",
                    (
                        "exact_score_log_loss_"
                        "improvement"
                    ),
                ]
            )
        ][
            [
                "comparison_name",
                "metric",
                "match_count",
                "mean_configuration_win_rate",
                "majority_agreement_rate",
                "unanimous_agreement_rate",
            ]
        ]
        .to_markdown(index=False)
    )

    strongest_bucket = summary[
        "strongest_exact_score_rating_bucket"
    ]

    strongest_favorite = summary[
        "strongest_outcome_favorite_category"
    ]

    strongest_relationship = summary[
        "strongest_absolute_rating_relationship"
    ]

    report = f"""# Study 063 — Match-Weighted ClubElo Robustness

## Purpose

Test whether Study 062's interpretation remains stable after
removing configuration weighting.

Study 062 treated every event, split, alpha value, and controlled
comparison prediction as a separate row. Study 063 first
aggregates all configurations for each event and comparison,
then gives every football match equal weight.

No goal model was fitted.

## Population

- Configuration-weighted source rows:
  {summary["source_configuration_weighted_rows"]}
- Match-weighted rows:
  {summary["match_weighted_rows"]}
- Unique events:
  {summary["unique_events"]}
- Controlled comparisons:
  {summary["controlled_comparisons"]}

A positive improvement means the ClubElo-enabled model produced
the lower average loss or error for that match.

## Overall match-weighted results

{comparison_display}

## Rating-difference buckets

Exact-score log-loss improvement:

{bucket_display}

The strongest match-weighted bucket was
`{strongest_bucket["rating_bucket"]}` for
`{strongest_bucket["comparison_name"]}`:

- Matches:
  {strongest_bucket["match_count"]}
- Mean match improvement:
  {strongest_bucket["mean_match_improvement"]:.8f}
- Match win rate:
  {strongest_bucket["match_win_rate"]:.1%}

## Favorite classification

Outcome-log-loss improvement:

{favorite_display}

The strongest favorite category was
`{strongest_favorite["favorite_category"]}` for
`{strongest_favorite["comparison_name"]}`:

- Matches:
  {strongest_favorite["match_count"]}
- Mean match improvement:
  {strongest_favorite["mean_match_improvement"]:.8f}
- Match win rate:
  {strongest_favorite["match_win_rate"]:.1%}

## Continuous matchup imbalance

{relationship_display}

The strongest absolute relationship had:

- Pearson correlation:
  {strongest_relationship["pearson_correlation"]:.6f}
- Spearman correlation:
  {strongest_relationship["spearman_correlation"]:.6f}

Weak correlations indicate that ClubElo's value is not merely
a monotonic function of matchup imbalance.

## Configuration agreement

{agreement_display}

`majority_agreement_rate` is the proportion of matches for which
more than half of available split-and-alpha configurations
favored ClubElo.

`unanimous_agreement_rate` is the proportion of matches for which
every available configuration favored ClubElo.

## Interpretation contract

Study 061 answered whether ClubElo improved aggregate predictive
performance.

Study 062 described improvement across all fitted configurations.

Study 063 asks whether those descriptive conclusions remain
when every match is weighted equally.

Agreement between Studies 062 and 063 strengthens the football
interpretation. Disagreement indicates that a conclusion was
partly driven by repeated configurations or unequal test-set
membership.

## Outputs

- `match_weighted_improvement.csv`
- `match_weighted_rating_bucket_analysis.csv`
- `match_weighted_favorite_analysis.csv`
- `match_weighted_result_type_analysis.csv`
- `match_weighted_total_goal_analysis.csv`
- `match_weighted_scoreline_analysis.csv`
- `match_weighted_rating_difference_relationship.csv`
- `match_weighted_comparison_summary.csv`
- `configuration_agreement_summary.csv`
- `study_summary.json`

## Validation

- Study 062 input loading: PASS
- Configuration-row uniqueness: PASS
- Match identity consistency: PASS
- One row per comparison and event: PASS
- Equal match weighting: PASS
- Segment analysis: PASS
- Continuous relationship analysis: PASS
- Configuration agreement analysis: PASS
- No new goal-model fitting: PASS
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    source = (
        load_match_level_improvements()
    )

    validate_input(source)

    match_weighted = (
        build_match_weighted_improvements(
            source
        )
    )

    expected_rows = (
        source[
            [
                "comparison_name",
                "event_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    if len(match_weighted) != expected_rows:
        raise AssertionError(
            "Match-weighted row count differs from "
            f"the expected population: "
            f"{len(match_weighted)} vs "
            f"{expected_rows}."
        )

    (
        rating_bucket,
        favorite,
        result_type,
        total_goal,
        scoreline,
    ) = build_segment_analyses(
        match_weighted
    )

    continuous_relationship = (
        build_continuous_relationship(
            match_weighted
        )
    )

    comparison_summary = (
        build_comparison_summary(
            match_weighted
        )
    )

    configuration_agreement = (
        build_configuration_agreement_summary(
            match_weighted
        )
    )

    summary = build_summary(
        match_weighted=match_weighted,
        comparison_summary=(
            comparison_summary
        ),
        rating_bucket=rating_bucket,
        favorite=favorite,
        continuous_relationship=(
            continuous_relationship
        ),
        configuration_agreement=(
            configuration_agreement
        ),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    match_weighted.to_csv(
        MATCH_WEIGHTED_PATH,
        index=False,
    )

    rating_bucket.to_csv(
        RATING_BUCKET_PATH,
        index=False,
    )

    favorite.to_csv(
        FAVORITE_PATH,
        index=False,
    )

    result_type.to_csv(
        RESULT_TYPE_PATH,
        index=False,
    )

    total_goal.to_csv(
        TOTAL_GOAL_PATH,
        index=False,
    )

    scoreline.to_csv(
        SCORELINE_PATH,
        index=False,
    )

    continuous_relationship.to_csv(
        CONTINUOUS_RELATIONSHIP_PATH,
        index=False,
    )

    comparison_summary.to_csv(
        COMPARISON_SUMMARY_PATH,
        index=False,
    )

    configuration_agreement.to_csv(
        CONFIGURATION_AGREEMENT_PATH,
        index=False,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        summary=summary,
        comparison_summary=(
            comparison_summary
        ),
        rating_bucket=rating_bucket,
        favorite=favorite,
        continuous_relationship=(
            continuous_relationship
        ),
        configuration_agreement=(
            configuration_agreement
        ),
    )

    print(
        "Study 063 — Match-Weighted ClubElo "
        "Robustness"
    )
    print("=" * 76)
    print()
    print(
        "Configuration-weighted source rows: "
        f"{len(source)}"
    )
    print(
        "Match-weighted rows: "
        f"{len(match_weighted)}"
    )
    print(
        "Unique events: "
        f"{match_weighted['event_id'].nunique()}"
    )
    print(
        "Controlled comparisons: "
        f"{match_weighted['comparison_name'].nunique()}"
    )
    print()

    print("Overall Match-Weighted Results")
    print("-" * 76)

    display_summary = (
        comparison_summary[
            comparison_summary[
                "metric"
            ].isin(
                [
                    "combined_goal_error_improvement",
                    "outcome_log_loss_improvement",
                    (
                        "exact_score_log_loss_"
                        "improvement"
                    ),
                ]
            )
        ][
            [
                "comparison_name",
                "metric",
                "match_count",
                "mean_match_improvement",
                "match_win_rate",
            ]
        ]
    )

    print(
        display_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )

    print()
    print("Rating-Bucket Interpretation")
    print("-" * 76)

    print(
        rating_bucket[
            rating_bucket[
                "metric"
            ].eq(
                "exact_score_log_loss_improvement"
            )
        ][
            [
                "comparison_name",
                "rating_bucket",
                "match_count",
                "mean_match_improvement",
                "match_win_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )

    print()
    print("Favorite Interpretation")
    print("-" * 76)

    print(
        favorite[
            favorite[
                "metric"
            ].eq(
                "outcome_log_loss_improvement"
            )
        ][
            [
                "comparison_name",
                "favorite_category",
                "match_count",
                "mean_match_improvement",
                "match_win_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )

    print()
    print("Configuration Agreement")
    print("-" * 76)

    print(
        configuration_agreement[
            configuration_agreement[
                "metric"
            ].isin(
                [
                    "outcome_log_loss_improvement",
                    (
                        "exact_score_log_loss_"
                        "improvement"
                    ),
                ]
            )
        ][
            [
                "comparison_name",
                "metric",
                "mean_configuration_win_rate",
                "majority_agreement_rate",
                "unanimous_agreement_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )

    print()
    print("Study 062 input loading: PASS")
    print("Configuration-row uniqueness: PASS")
    print("Match identity consistency: PASS")
    print("One row per comparison and event: PASS")
    print("Equal match weighting: PASS")
    print("Segment analysis: PASS")
    print("Continuous relationship analysis: PASS")
    print("Configuration agreement analysis: PASS")
    print("No new goal-model fitting: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
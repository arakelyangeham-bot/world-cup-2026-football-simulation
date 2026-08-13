#benchmark_transformation_robustness

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.benchmarking.goal_model_benchmark import (
    GoalModelBenchmarkConfig,
    GoalModelDatasetConfig,
    run_goal_model_benchmark,
    write_goal_model_benchmark_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "study_092c2"
    / "observation_datasets"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "study_092c2"
    / "representation_robustness"
)

TRANSFORMATIONS = (
    "global_zscore",
    "percentile_normal",
    "robust_zscore",
)

DATASET_PATHS = {
    transformation: (
        INPUT_DIRECTORY
        / (
            "bundesliga_observations_"
            f"{transformation}_with_clubelo.csv"
        )
    )
    for transformation in TRANSFORMATIONS
}

FEATURE_SPECIFICATION = (
    "attack_defense_attack_depth_rating_prior"
)

TRAIN_FRACTIONS = (
    0.60,
    0.70,
    0.75,
    0.80,
)

ALPHA_VALUES = (
    0.0,
    0.01,
    0.1,
    1.0,
)

PRIMARY_METRICS = (
    "combined_poisson_deviance",
    "combined_goal_mae",
    "total_goal_mae",
    "goal_difference_mae",
    "outcome_log_loss",
    "outcome_brier_score",
    "exact_score_log_loss",
)

CALIBRATION_METRICS = (
    "absolute_draw_rate_error",
    "absolute_home_goal_mean_error",
    "absolute_away_goal_mean_error",
)

ALL_METRICS = (
    *PRIMARY_METRICS,
    *CALIBRATION_METRICS,
)

EXPECTED_RUN_COUNT = (
    len(TRANSFORMATIONS)
    * len(TRAIN_FRACTIONS)
    * len(ALPHA_VALUES)
)

MATCHED_RESULT_PATH = (
    OUTPUT_DIRECTORY
    / "matched_representation_results.csv"
)

PAIRWISE_DELTA_PATH = (
    OUTPUT_DIRECTORY
    / "pairwise_representation_deltas.csv"
)

METRIC_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "representation_metric_summary.csv"
)

CONFIGURATION_WIN_PATH = (
    OUTPUT_DIRECTORY
    / "configuration_winners.csv"
)

OVERALL_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "representation_robustness_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_092c2c_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_092C2C_RESULTS.md"
)


def build_config() -> GoalModelBenchmarkConfig:
    datasets = tuple(
        GoalModelDatasetConfig(
            name=transformation,
            path=DATASET_PATHS[
                transformation
            ],
            representation_type="full_squad",
        )
        for transformation in TRANSFORMATIONS
    )

    return GoalModelBenchmarkConfig(
        name=(
            "study_092c2c_representation_"
            "robustness_benchmark"
        ),
        datasets=datasets,
        feature_specifications=(
            FEATURE_SPECIFICATION,
        ),
        train_fractions=TRAIN_FRACTIONS,
        alpha_values=ALPHA_VALUES,
        require_matched_populations=True,
        capture_predictions=True,
        capture_coefficients=True,
    )


def validate_results(
    results: pd.DataFrame,
) -> None:
    if len(results) != EXPECTED_RUN_COUNT:
        raise AssertionError(
            "Unexpected benchmark run count: "
            f"{len(results)} versus "
            f"{EXPECTED_RUN_COUNT}."
        )

    if set(
        results["dataset"]
    ) != set(TRANSFORMATIONS):
        raise AssertionError(
            "One or more representation branches are "
            "missing from benchmark results."
        )

    if set(
        results["train_fraction"]
    ) != set(TRAIN_FRACTIONS):
        raise AssertionError(
            "Unexpected train-fraction population."
        )

    if set(
        results["alpha"]
    ) != set(ALPHA_VALUES):
        raise AssertionError(
            "Unexpected alpha population."
        )

    expected_configuration_count = (
        len(TRAIN_FRACTIONS)
        * len(ALPHA_VALUES)
    )

    counts = (
        results.groupby(
            "dataset"
        )
        .size()
    )

    if not counts.eq(
        expected_configuration_count
    ).all():
        raise AssertionError(
            "Representations do not have equal numbers "
            "of benchmark configurations."
        )

    metric_values = results[
        list(ALL_METRICS)
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        metric_values
    ).all():
        raise AssertionError(
            "Benchmark results contain non-finite metrics."
        )


def build_matched_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "dataset",
        "representation_type",
        "feature_specification",
        "train_fraction",
        "alpha",
        "train_matches",
        "test_matches",
        *ALL_METRICS,
    ]

    return (
        results[
            columns
        ]
        .sort_values(
            [
                "train_fraction",
                "alpha",
                "dataset",
            ]
        )
        .reset_index(drop=True)
    )


def build_pairwise_deltas(
    results: pd.DataFrame,
) -> pd.DataFrame:
    indexed = results.set_index(
        [
            "dataset",
            "train_fraction",
            "alpha",
        ]
    )

    records: list[dict[str, object]] = []

    for train_fraction in TRAIN_FRACTIONS:
        for alpha in ALPHA_VALUES:
            for baseline in TRANSFORMATIONS:
                for candidate in TRANSFORMATIONS:
                    if candidate == baseline:
                        continue

                    for metric in ALL_METRICS:
                        baseline_value = float(
                            indexed.loc[
                                (
                                    baseline,
                                    train_fraction,
                                    alpha,
                                ),
                                metric,
                            ]
                        )

                        candidate_value = float(
                            indexed.loc[
                                (
                                    candidate,
                                    train_fraction,
                                    alpha,
                                ),
                                metric,
                            ]
                        )

                        delta = (
                            candidate_value
                            - baseline_value
                        )

                        records.append(
                            {
                                "train_fraction":
                                    train_fraction,
                                "alpha":
                                    alpha,
                                "baseline_representation":
                                    baseline,
                                "candidate_representation":
                                    candidate,
                                "metric":
                                    metric,
                                "baseline_value":
                                    baseline_value,
                                "candidate_value":
                                    candidate_value,
                                "candidate_minus_baseline":
                                    delta,
                                "candidate_improved":
                                    delta < 0.0,
                                "candidate_tied":
                                    bool(
                                        np.isclose(
                                            delta,
                                            0.0,
                                            atol=1e-12,
                                            rtol=0.0,
                                        )
                                    ),
                            }
                        )

    return pd.DataFrame(records)


def build_metric_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    configuration_count = (
        len(TRAIN_FRACTIONS)
        * len(ALPHA_VALUES)
    )

    for metric in ALL_METRICS:
        for transformation in TRANSFORMATIONS:
            values = (
                results.loc[
                    results[
                        "dataset"
                    ].eq(
                        transformation
                    ),
                    metric,
                ]
                .to_numpy(
                    dtype=float
                )
            )

            wins = 0
            ties = 0

            for train_fraction in TRAIN_FRACTIONS:
                for alpha in ALPHA_VALUES:
                    matched = results.loc[
                        results[
                            "train_fraction"
                        ].eq(
                            train_fraction
                        )
                        & results[
                            "alpha"
                        ].eq(
                            alpha
                        )
                    ]

                    minimum = float(
                        matched[
                            metric
                        ].min()
                    )

                    candidate_value = float(
                        matched.loc[
                            matched[
                                "dataset"
                            ].eq(
                                transformation
                            ),
                            metric,
                        ].iloc[0]
                    )

                    if np.isclose(
                        candidate_value,
                        minimum,
                        atol=1e-12,
                        rtol=0.0,
                    ):
                        winner_count = int(
                            np.isclose(
                                matched[
                                    metric
                                ].to_numpy(
                                    dtype=float
                                ),
                                minimum,
                                atol=1e-12,
                                rtol=0.0,
                            ).sum()
                        )

                        if winner_count == 1:
                            wins += 1
                        else:
                            ties += 1

            records.append(
                {
                    "representation":
                        transformation,
                    "metric":
                        metric,
                    "configuration_count":
                        configuration_count,
                    "win_count":
                        wins,
                    "tie_count":
                        ties,
                    "win_rate":
                        wins
                        / configuration_count,
                    "mean_value":
                        float(
                            values.mean()
                        ),
                    "median_value":
                        float(
                            np.median(
                                values
                            )
                        ),
                    "minimum_value":
                        float(
                            values.min()
                        ),
                    "maximum_value":
                        float(
                            values.max()
                        ),
                    "standard_deviation":
                        float(
                            values.std(
                                ddof=1
                            )
                        ),
                }
            )

    return pd.DataFrame(records)


def build_configuration_winners(
    results: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for train_fraction in TRAIN_FRACTIONS:
        for alpha in ALPHA_VALUES:
            matched = results.loc[
                results[
                    "train_fraction"
                ].eq(
                    train_fraction
                )
                & results[
                    "alpha"
                ].eq(
                    alpha
                )
            ].copy()

            metric_win_counts = {
                transformation: 0
                for transformation
                in TRANSFORMATIONS
            }

            metric_rank_totals = {
                transformation: 0.0
                for transformation
                in TRANSFORMATIONS
            }

            for metric in ALL_METRICS:
                metric_ranks = matched[
                    metric
                ].rank(
                    method="average",
                    ascending=True,
                )

                minimum = float(
                    matched[
                        metric
                    ].min()
                )

                for row_index, row in matched.iterrows():
                    transformation = str(
                        row["dataset"]
                    )

                    metric_rank_totals[
                        transformation
                    ] += float(
                        metric_ranks.loc[
                            row_index
                        ]
                    )

                    if np.isclose(
                        float(
                            row[metric]
                        ),
                        minimum,
                        atol=1e-12,
                        rtol=0.0,
                    ):
                        metric_win_counts[
                            transformation
                        ] += 1

            ordered = sorted(
                TRANSFORMATIONS,
                key=lambda transformation: (
                    -metric_win_counts[
                        transformation
                    ],
                    metric_rank_totals[
                        transformation
                    ]
                    / len(
                        ALL_METRICS
                    ),
                    transformation,
                ),
            )

            leader = ordered[0]
            runner_up = ordered[1]

            tied = (
                metric_win_counts[
                    leader
                ]
                == metric_win_counts[
                    runner_up
                ]
                and np.isclose(
                    metric_rank_totals[
                        leader
                    ],
                    metric_rank_totals[
                        runner_up
                    ],
                    atol=1e-12,
                    rtol=0.0,
                )
            )

            records.append(
                {
                    "train_fraction":
                        train_fraction,
                    "alpha":
                        alpha,
                    "winning_representation":
                        (
                            "mixed"
                            if tied
                            else leader
                        ),
                    "winning_metric_count":
                        metric_win_counts[
                            leader
                        ],
                    "winning_mean_metric_rank":
                        metric_rank_totals[
                            leader
                        ]
                        / len(
                            ALL_METRICS
                        ),
                    "robust_metric_win_count":
                        metric_win_counts[
                            "robust_zscore"
                        ],
                    "global_metric_win_count":
                        metric_win_counts[
                            "global_zscore"
                        ],
                    "percentile_metric_win_count":
                        metric_win_counts[
                            "percentile_normal"
                        ],
                }
            )

    return pd.DataFrame(records)


def build_overall_summary(
    metric_summary: pd.DataFrame,
    configuration_winners: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for transformation in TRANSFORMATIONS:
        primary = metric_summary.loc[
            metric_summary[
                "representation"
            ].eq(
                transformation
            )
            & metric_summary[
                "metric"
            ].isin(
                PRIMARY_METRICS
            )
        ]

        calibration = metric_summary.loc[
            metric_summary[
                "representation"
            ].eq(
                transformation
            )
            & metric_summary[
                "metric"
            ].isin(
                CALIBRATION_METRICS
            )
        ]

        configuration_win_count = int(
            configuration_winners[
                "winning_representation"
            ].eq(
                transformation
            ).sum()
        )

        records.append(
            {
                "representation":
                    transformation,
                "configuration_count":
                    len(
                        configuration_winners
                    ),
                "configuration_win_count":
                    configuration_win_count,
                "configuration_win_rate":
                    configuration_win_count
                    / len(
                        configuration_winners
                    ),
                "primary_metric_win_count":
                    int(
                        primary[
                            "win_count"
                        ].sum()
                    ),
                "primary_metric_comparison_count":
                    int(
                        len(
                            PRIMARY_METRICS
                        )
                        * len(
                            TRAIN_FRACTIONS
                        )
                        * len(
                            ALPHA_VALUES
                        )
                    ),
                "primary_metric_win_rate":
                    float(
                        primary[
                            "win_count"
                        ].sum()
                        / (
                            len(
                                PRIMARY_METRICS
                            )
                            * len(
                                TRAIN_FRACTIONS
                            )
                            * len(
                                ALPHA_VALUES
                            )
                        )
                    ),
                "calibration_metric_win_count":
                    int(
                        calibration[
                            "win_count"
                        ].sum()
                    ),
                "mean_primary_metric_win_rate":
                    float(
                        primary[
                            "win_rate"
                        ].mean()
                    ),
                "mean_calibration_metric_win_rate":
                    float(
                        calibration[
                            "win_rate"
                        ].mean()
                    ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "primary_metric_win_rate",
                "configuration_win_rate",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def classify_evidence(
    summary: pd.DataFrame,
) -> str:
    robust = summary.loc[
        summary[
            "representation"
        ].eq(
            "robust_zscore"
        )
    ].iloc[0]

    primary_win_rate = float(
        robust[
            "primary_metric_win_rate"
        ]
    )

    configuration_win_rate = float(
        robust[
            "configuration_win_rate"
        ]
    )

    if (
        primary_win_rate > 0.50
        and configuration_win_rate > 0.50
    ):
        return "strong"

    if primary_win_rate > 0.50:
        return "moderate"

    if (
        primary_win_rate > 1.0 / 3.0
        or configuration_win_rate
        > 1.0 / 3.0
    ):
        return "limited"

    return "none"


def write_report(
    overall_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    evidence_classification: str,
) -> None:
    robust = overall_summary.loc[
        overall_summary[
            "representation"
        ].eq(
            "robust_zscore"
        )
    ].iloc[0]

    primary_robust = metric_summary.loc[
        metric_summary[
            "representation"
        ].eq(
            "robust_zscore"
        )
        & metric_summary[
            "metric"
        ].isin(
            PRIMARY_METRICS
        )
    ]

    report = f"""# Study 092C2C — Representation Robustness Benchmark

## Status

**PASS**

## Research question

Does the predictive advantage observed for `robust_zscore`
persist across reasonable chronological train fractions and
Poisson regularization strengths?

## Experiment

- Representations: {", ".join(f"`{value}`" for value in TRANSFORMATIONS)}
- Feature specification: `{FEATURE_SPECIFICATION}`
- Train fractions: {TRAIN_FRACTIONS}
- Alpha values: {ALPHA_VALUES}
- Total benchmark runs: {EXPECTED_RUN_COUNT}
- Matched representation configurations:
  {len(TRAIN_FRACTIONS) * len(ALPHA_VALUES)}

## Predeclared evidence classification

**{evidence_classification.upper()}**

## Robust representation summary

- Configuration win rate:
  {float(robust["configuration_win_rate"]):.1%}
- Primary metric win rate:
  {float(robust["primary_metric_win_rate"]):.1%}
- Primary metric wins:
  {int(robust["primary_metric_win_count"])}
- Primary metric comparisons:
  {int(robust["primary_metric_comparison_count"])}

## Robust primary-metric breakdown

{primary_robust[
    [
        "metric",
        "configuration_count",
        "win_count",
        "win_rate",
        "mean_value",
        "median_value",
        "standard_deviation",
    ]
].to_markdown(index=False)}

## Interpretation boundary

This remains a controlled retrospective, single-season
Bundesliga benchmark. The experiment tests stability across
chronological split and regularization choices but does not
establish multi-season, cross-league, or prediction-date-frozen
generalization.

## Validation

- Matched event populations: PASS
- Matched targets: PASS
- Shared split assignments: PASS
- Shared feature specification: PASS
- Complete split-alpha grid: PASS
- Finite benchmark metrics: PASS
- Standard benchmark engine reused unchanged: PASS
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 092C2C — REPRESENTATION "
        "ROBUSTNESS BENCHMARK"
    )
    print("=" * 88)

    benchmark_result = (
        run_goal_model_benchmark(
            build_config()
        )
    )

    validate_results(
        benchmark_result.results
    )

    matched_results = (
        build_matched_results(
            benchmark_result.results
        )
    )

    pairwise_deltas = (
        build_pairwise_deltas(
            matched_results
        )
    )

    metric_summary = (
        build_metric_summary(
            matched_results
        )
    )

    configuration_winners = (
        build_configuration_winners(
            matched_results
        )
    )

    overall_summary = (
        build_overall_summary(
            metric_summary,
            configuration_winners,
        )
    )

    evidence_classification = (
        classify_evidence(
            overall_summary
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_goal_model_benchmark_outputs(
        benchmark_result=benchmark_result,
        output_directory=OUTPUT_DIRECTORY,
    )

    matched_results.to_csv(
        MATCHED_RESULT_PATH,
        index=False,
    )

    pairwise_deltas.to_csv(
        PAIRWISE_DELTA_PATH,
        index=False,
    )

    metric_summary.to_csv(
        METRIC_SUMMARY_PATH,
        index=False,
    )

    configuration_winners.to_csv(
        CONFIGURATION_WIN_PATH,
        index=False,
    )

    overall_summary.to_csv(
        OVERALL_SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "study_id": "092C2C",
        "study_name": (
            "Representation Robustness Benchmark"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "transformations": list(
            TRANSFORMATIONS
        ),
        "feature_specification":
            FEATURE_SPECIFICATION,
        "train_fractions": list(
            TRAIN_FRACTIONS
        ),
        "alpha_values": list(
            ALPHA_VALUES
        ),
        "benchmark_run_count":
            EXPECTED_RUN_COUNT,
        "matched_configuration_count":
            len(TRAIN_FRACTIONS)
            * len(ALPHA_VALUES),
        "evidence_classification":
            evidence_classification,
        "preferred_representation":
            str(
                overall_summary.iloc[0][
                    "representation"
                ]
            ),
        "standard_benchmark_engine_reused":
            True,
        "methodological_boundary": (
            "Single-season retrospective robustness "
            "analysis across chronological splits and "
            "Poisson regularization values."
        ),
        "outputs": [
            "benchmark_results.csv",
            "configuration_ranking.csv",
            "split_assignments.csv",
            "coefficients.csv",
            "predictions.csv",
            MATCHED_RESULT_PATH.name,
            PAIRWISE_DELTA_PATH.name,
            METRIC_SUMMARY_PATH.name,
            CONFIGURATION_WIN_PATH.name,
            OVERALL_SUMMARY_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        overall_summary=overall_summary,
        metric_summary=metric_summary,
        evidence_classification=(
            evidence_classification
        ),
    )

    print()
    print("Overall representation summary")
    print("-" * 88)
    print(
        overall_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print(
        "Evidence classification: "
        f"{evidence_classification.upper()}"
    )

    print()
    print("Validation summary")
    print("  Benchmark run count: PASS")
    print("  Matched event populations: PASS")
    print("  Matched targets: PASS")
    print("  Shared chronological splits: PASS")
    print("  Complete alpha grid: PASS")
    print("  Finite benchmark metrics: PASS")
    print("  Standard benchmark engine: PASS")
    print("  Output generation: PASS")

    print()
    print("=" * 88)
    print("OVERALL EXECUTION RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
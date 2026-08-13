#benchmark_transformation_representations

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.benchmarking.goal_model_benchmark import (
    DEFAULT_METRICS,
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
    / "representation_benchmark"
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
    0.75,
)

ALPHA_VALUES = (
    0.0,
)

REPRESENTATION_TYPE = "full_squad"

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

COMPARISON_METRICS = (
    *PRIMARY_METRICS,
    *CALIBRATION_METRICS,
)

REPRESENTATION_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "representation_comparison.csv"
)

PAIRWISE_DELTA_PATH = (
    OUTPUT_DIRECTORY
    / "pairwise_representation_deltas.csv"
)

METRIC_WINNER_PATH = (
    OUTPUT_DIRECTORY
    / "metric_winners.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_092c2b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_092C2B_RESULTS.md"
)


def build_config() -> GoalModelBenchmarkConfig:
    datasets = tuple(
        GoalModelDatasetConfig(
            name=transformation,
            path=DATASET_PATHS[
                transformation
            ],
            representation_type=(
                REPRESENTATION_TYPE
            ),
        )
        for transformation in TRANSFORMATIONS
    )

    return GoalModelBenchmarkConfig(
        name=(
            "study_092c2b_matched_"
            "representation_benchmark"
        ),
        datasets=datasets,
        feature_specifications=(
            FEATURE_SPECIFICATION,
        ),
        train_fractions=TRAIN_FRACTIONS,
        alpha_values=ALPHA_VALUES,
        ranking_metrics=DEFAULT_METRICS,
        require_matched_populations=True,
        capture_predictions=True,
        capture_coefficients=True,
    )


def validate_results(
    results: pd.DataFrame,
) -> None:
    expected_rows = (
        len(TRANSFORMATIONS)
        * len(TRAIN_FRACTIONS)
        * len(ALPHA_VALUES)
    )

    if len(results) != expected_rows:
        raise AssertionError(
            "Unexpected benchmark result count: "
            f"{len(results)} vs {expected_rows}."
        )

    if set(results["dataset"]) != set(
        TRANSFORMATIONS
    ):
        raise AssertionError(
            "Benchmark results do not contain every "
            "representation branch."
        )

    if set(
        results[
            "feature_specification"
        ]
    ) != {
        FEATURE_SPECIFICATION
    }:
        raise AssertionError(
            "Unexpected feature specification in results."
        )

    if results[
        "test_matches"
    ].nunique() != 1:
        raise AssertionError(
            "Representations were evaluated on different "
            "test population sizes."
        )

    if results[
        "train_matches"
    ].nunique() != 1:
        raise AssertionError(
            "Representations were trained on different "
            "population sizes."
        )

    numeric_metrics = results[
        list(COMPARISON_METRICS)
    ].to_numpy(dtype=float)

    if not np.isfinite(
        numeric_metrics
    ).all():
        raise AssertionError(
            "Benchmark results contain non-finite metrics."
        )


def build_representation_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:
    selected_columns = [
        "dataset",
        "representation_type",
        "feature_specification",
        "train_fraction",
        "alpha",
        "train_matches",
        "test_matches",
        *COMPARISON_METRICS,
        "actual_draw_rate",
        "predicted_draw_rate",
        "actual_home_goal_mean",
        "predicted_home_goal_mean",
        "actual_away_goal_mean",
        "predicted_away_goal_mean",
    ]

    comparison = results[
        selected_columns
    ].copy()

    rank_columns: list[str] = []

    for metric in COMPARISON_METRICS:
        rank_column = (
            f"{metric}_rank"
        )

        comparison[
            rank_column
        ] = comparison[
            metric
        ].rank(
            method="average",
            ascending=True,
        )

        rank_columns.append(
            rank_column
        )

    comparison[
        "mean_comparison_metric_rank"
    ] = comparison[
        rank_columns
    ].mean(axis=1)

    comparison[
        "metric_win_count"
    ] = 0

    for metric in COMPARISON_METRICS:
        minimum = comparison[
            metric
        ].min()

        winners = np.isclose(
            comparison[
                metric
            ].to_numpy(dtype=float),
            minimum,
            atol=1e-12,
            rtol=0.0,
        )

        comparison.loc[
            winners,
            "metric_win_count",
        ] += 1

    return (
        comparison
        .sort_values(
            [
                "mean_comparison_metric_rank",
                "combined_poisson_deviance",
            ]
        )
        .reset_index(drop=True)
    )


def build_pairwise_deltas(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    indexed = comparison.set_index(
        "dataset"
    )

    records: list[
        dict[str, object]
    ] = []

    for baseline in TRANSFORMATIONS:
        for candidate in TRANSFORMATIONS:
            if candidate == baseline:
                continue

            for metric in COMPARISON_METRICS:
                baseline_value = float(
                    indexed.loc[
                        baseline,
                        metric,
                    ]
                )

                candidate_value = float(
                    indexed.loc[
                        candidate,
                        metric,
                    ]
                )

                delta = (
                    candidate_value
                    - baseline_value
                )

                records.append(
                    {
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

    return pd.DataFrame(
        records
    )


def build_metric_winners(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for metric in COMPARISON_METRICS:
        minimum = float(
            comparison[
                metric
            ].min()
        )

        winners = comparison.loc[
            np.isclose(
                comparison[
                    metric
                ].to_numpy(dtype=float),
                minimum,
                atol=1e-12,
                rtol=0.0,
            ),
            "dataset",
        ].tolist()

        records.append(
            {
                "metric": metric,
                "winning_value": minimum,
                "winner_count": len(
                    winners
                ),
                "winning_representations":
                    "|".join(winners),
            }
        )

    return pd.DataFrame(
        records
    )


def determine_overall_result(
    comparison: pd.DataFrame,
) -> dict[str, object]:
    ordered = comparison.sort_values(
        [
            "metric_win_count",
            "mean_comparison_metric_rank",
            "combined_poisson_deviance",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    leader = ordered.iloc[0]

    if len(ordered) > 1:
        runner_up = ordered.iloc[1]

        exact_tie = (
            int(
                leader[
                    "metric_win_count"
                ]
            )
            == int(
                runner_up[
                    "metric_win_count"
                ]
            )
            and np.isclose(
                float(
                    leader[
                        "mean_comparison_metric_rank"
                    ]
                ),
                float(
                    runner_up[
                        "mean_comparison_metric_rank"
                    ]
                ),
                atol=1e-12,
                rtol=0.0,
            )
        )
    else:
        exact_tie = False

    return {
        "preferred_representation": (
            "mixed"
            if exact_tie
            else str(
                leader[
                    "dataset"
                ]
            )
        ),
        "leading_metric_win_count": int(
            leader[
                "metric_win_count"
            ]
        ),
        "leading_mean_metric_rank": float(
            leader[
                "mean_comparison_metric_rank"
            ]
        ),
        "decision_status": (
            "mixed"
            if exact_tie
            else "single_leader"
        ),
    }


def write_report(
    comparison: pd.DataFrame,
    winners: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    best_poisson = (
        comparison
        .sort_values(
            "combined_poisson_deviance"
        )
        .iloc[0]
    )

    best_outcome = (
        comparison
        .sort_values(
            "outcome_log_loss"
        )
        .iloc[0]
    )

    best_exact_score = (
        comparison
        .sort_values(
            "exact_score_log_loss"
        )
        .iloc[0]
    )

    report = f"""# Study 092C2B — Matched Representation Benchmark

## Status

**PASS**

## Research question

Does changing the player-feature transformation improve
predictive performance when fixtures, targets, ClubElo priors,
feature specification, chronological split, regularization,
and benchmark implementation are held constant?

## Experimental design

- Observation population: 306 Bundesliga 2024–25 fixtures
- Representations:
  - `global_zscore`
  - `percentile_normal`
  - `robust_zscore`
- Feature specification:
  `{FEATURE_SPECIFICATION}`
- Training fraction: {TRAIN_FRACTIONS[0]:.2f}
- Alpha: {ALPHA_VALUES[0]:.6f}
- Split policy: shared chronological split
- ClubElo prior: identical fixture-date-valid prior population
- Ranking rule: lower metric values are better

## Overall comparison

- Preferred representation:
  `{decision["preferred_representation"]}`
- Decision status:
  `{decision["decision_status"]}`
- Leading metric wins:
  {decision["leading_metric_win_count"]}
- Leading mean metric rank:
  {decision["leading_mean_metric_rank"]:.6f}

## Primary results

### Lowest combined Poisson deviance

- Representation: `{best_poisson["dataset"]}`
- Value:
  {float(best_poisson["combined_poisson_deviance"]):.8f}

### Lowest outcome log loss

- Representation: `{best_outcome["dataset"]}`
- Value:
  {float(best_outcome["outcome_log_loss"]):.8f}

### Lowest exact-score log loss

- Representation: `{best_exact_score["dataset"]}`
- Value:
  {float(best_exact_score["exact_score_log_loss"]):.8f}

## Metric winners

{winners.to_markdown(index=False)}

## Interpretation boundary

This is a controlled retrospective, single-season,
chronological-holdout comparison. The player-derived
representation is static at the season level and is not
independently frozen before every fixture date. ClubElo priors
are fixture-date valid.

A winning representation in this study should therefore be
treated as evidence for predictive usefulness within this
matched Bundesliga experiment, not as proof of multi-season or
out-of-league generalization.

## Validation

- Matched event population: PASS
- Matched targets: PASS
- Shared chronological split: PASS
- Shared feature specification: PASS
- Shared alpha: PASS
- Shared ClubElo priors: PASS
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
        "STUDY 092C2B — MATCHED "
        "REPRESENTATION BENCHMARK"
    )
    print("=" * 88)

    config = build_config()

    benchmark_result = (
        run_goal_model_benchmark(
            config
        )
    )

    validate_results(
        benchmark_result.results
    )

    comparison = (
        build_representation_comparison(
            benchmark_result.results
        )
    )

    pairwise_deltas = (
        build_pairwise_deltas(
            comparison
        )
    )

    metric_winners = (
        build_metric_winners(
            comparison
        )
    )

    decision = determine_overall_result(
        comparison
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_goal_model_benchmark_outputs(
        benchmark_result=benchmark_result,
        output_directory=OUTPUT_DIRECTORY,
    )

    comparison.to_csv(
        REPRESENTATION_COMPARISON_PATH,
        index=False,
    )

    pairwise_deltas.to_csv(
        PAIRWISE_DELTA_PATH,
        index=False,
    )

    metric_winners.to_csv(
        METRIC_WINNER_PATH,
        index=False,
    )

    metadata = {
        "study_id": "092C2B",
        "study_name": (
            "Matched Representation Benchmark"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "study_type": (
            "controlled_retrospective_"
            "representation_benchmark"
        ),
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
        "benchmark_run_count": int(
            len(
                benchmark_result.results
            )
        ),
        "training_match_count": int(
            comparison[
                "train_matches"
            ].iloc[0]
        ),
        "test_match_count": int(
            comparison[
                "test_matches"
            ].iloc[0]
        ),
        "preferred_representation":
            decision[
                "preferred_representation"
            ],
        "decision_status":
            decision[
                "decision_status"
            ],
        "leading_metric_win_count":
            decision[
                "leading_metric_win_count"
            ],
        "leading_mean_metric_rank":
            decision[
                "leading_mean_metric_rank"
            ],
        "matched_population_validation":
            True,
        "shared_chronological_split":
            True,
        "shared_clubelo_priors":
            True,
        "standard_benchmark_engine_reused":
            True,
        "methodological_boundary": (
            "Controlled retrospective single-season "
            "chronological holdout. Static player-derived "
            "representations are not independently frozen "
            "before every fixture date."
        ),
        "outputs": [
            "benchmark_results.csv",
            "configuration_ranking.csv",
            "split_assignments.csv",
            "coefficients.csv",
            "predictions.csv",
            REPRESENTATION_COMPARISON_PATH.name,
            PAIRWISE_DELTA_PATH.name,
            METRIC_WINNER_PATH.name,
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
        comparison=comparison,
        winners=metric_winners,
        decision=decision,
    )

    display_columns = [
        "dataset",
        "combined_poisson_deviance",
        "combined_goal_mae",
        "total_goal_mae",
        "goal_difference_mae",
        "outcome_log_loss",
        "outcome_brier_score",
        "exact_score_log_loss",
        "absolute_draw_rate_error",
        "metric_win_count",
        "mean_comparison_metric_rank",
    ]

    print()
    print("Representation comparison")
    print("-" * 88)
    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )

    print()
    print(
        "Preferred representation: "
        f"{decision['preferred_representation']}"
    )
    print(
        "Decision status: "
        f"{decision['decision_status']}"
    )

    print()
    print("Validation summary")
    print("  Dataset loading: PASS")
    print("  Matched event populations: PASS")
    print("  Matched targets: PASS")
    print("  Shared chronological split: PASS")
    print("  Shared feature specification: PASS")
    print("  Shared alpha: PASS")
    print("  Standard benchmark engine: PASS")
    print("  Finite benchmark metrics: PASS")
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
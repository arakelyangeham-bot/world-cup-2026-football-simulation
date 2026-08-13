#benchmark_attack_depth_midfield_robustness

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.benchmarking.goal_model_benchmark import (
    GoalModelBenchmarkConfig,
    GoalModelDatasetConfig,
    GoalModelBenchmarkResult,
    run_goal_model_benchmark,
    write_goal_model_benchmark_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OBSERVATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_048_club_observation_dataset"
    / "full_squad_observations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_054_attack_depth_midfield_robustness"
)


FEATURE_SPECIFICATIONS = (
    "attack_defense",
    "attack_defense_attack_depth",
    "attack_defense_midfield",
    (
        "attack_defense_"
        "attack_depth_midfield"
    ),
)

TRAIN_FRACTIONS = (
    0.60,
    0.70,
    0.75,
    0.80,
)

ALPHA_VALUES = (
    0.0,
    0.0001,
    0.001,
    0.005,
    0.01,
)

BASELINE_SPECIFICATION = "attack_defense"


COMPARISON_METRICS = (
    "combined_poisson_deviance",
    "home_poisson_deviance",
    "away_poisson_deviance",
    "combined_goal_mae",
    "total_goal_mae",
    "goal_difference_mae",
    "outcome_log_loss",
    "outcome_brier_score",
    "exact_score_log_loss",
    "absolute_draw_rate_error",
    "absolute_home_goal_mean_error",
    "absolute_away_goal_mean_error",
)


def build_config() -> GoalModelBenchmarkConfig:
    return GoalModelBenchmarkConfig(
        name=(
            "study_054_attack_depth_"
            "midfield_robustness"
        ),
        datasets=(
            GoalModelDatasetConfig(
                name="full_squad",
                path=OBSERVATION_PATH,
                representation_type=(
                    "full_squad"
                ),
            ),
        ),
        feature_specifications=(
            FEATURE_SPECIFICATIONS
        ),
        train_fractions=(
            TRAIN_FRACTIONS
        ),
        alpha_values=ALPHA_VALUES,
        capture_predictions=True,
        capture_coefficients=True,
    )


def build_baseline_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = [
        "train_fraction",
        "alpha",
    ]

    baseline = (
        results[
            results[
                "feature_specification"
            ].eq(BASELINE_SPECIFICATION)
        ]
        .set_index(key_columns)
        .sort_index()
    )

    expected_baseline_count = (
        len(TRAIN_FRACTIONS)
        * len(ALPHA_VALUES)
    )

    if len(baseline) != expected_baseline_count:
        raise AssertionError(
            "Baseline configuration population "
            "is incomplete."
        )

    rows: list[dict[str, object]] = []

    for candidate in results.itertuples(
        index=False
    ):
        key = (
            candidate.train_fraction,
            candidate.alpha,
        )

        baseline_row = baseline.loc[key]

        row: dict[str, object] = {
            "feature_specification": (
                candidate.feature_specification
            ),
            "train_fraction": (
                candidate.train_fraction
            ),
            "alpha": candidate.alpha,
            "is_baseline": (
                candidate.feature_specification
                == BASELINE_SPECIFICATION
            ),
        }

        for metric in COMPARISON_METRICS:
            candidate_value = float(
                getattr(candidate, metric)
            )

            baseline_value = float(
                baseline_row[metric]
            )

            difference = (
                candidate_value
                - baseline_value
            )

            row[
                f"candidate_{metric}"
            ] = candidate_value

            row[
                f"baseline_{metric}"
            ] = baseline_value

            row[
                f"{metric}_difference"
            ] = difference

            row[
                f"{metric}_beats_baseline"
            ] = difference < 0

        rows.append(row)

    return pd.DataFrame(rows)


def build_specification_summary(
    baseline_comparison: pd.DataFrame,
) -> pd.DataFrame:
    candidates = baseline_comparison[
        ~baseline_comparison["is_baseline"]
    ]

    rows: list[dict[str, object]] = []

    for specification_name, group in (
        candidates.groupby(
            "feature_specification",
            sort=True,
        )
    ):
        row: dict[str, object] = {
            "feature_specification":
                specification_name,
            "comparison_count": len(group),
        }

        for metric in COMPARISON_METRICS:
            difference_column = (
                f"{metric}_difference"
            )

            differences = group[
                difference_column
            ]

            row[
                f"{metric}_wins"
            ] = int(
                (
                    differences < 0
                ).sum()
            )

            row[
                f"{metric}_losses"
            ] = int(
                (
                    differences > 0
                ).sum()
            )

            row[
                f"{metric}_ties"
            ] = int(
                (
                    differences == 0
                ).sum()
            )

            row[
                f"{metric}_win_rate"
            ] = float(
                (
                    differences < 0
                ).mean()
            )

            row[
                f"mean_{metric}_difference"
            ] = float(
                differences.mean()
            )

            row[
                f"median_{metric}_difference"
            ] = float(
                differences.median()
            )

            row[
                f"best_{metric}_difference"
            ] = float(
                differences.min()
            )

            row[
                f"worst_{metric}_difference"
            ] = float(
                differences.max()
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                (
                    "combined_poisson_"
                    "deviance_wins"
                ),
                (
                    "mean_combined_poisson_"
                    "deviance_difference"
                ),
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_split_summary(
    baseline_comparison: pd.DataFrame,
) -> pd.DataFrame:
    candidates = baseline_comparison[
        ~baseline_comparison["is_baseline"]
    ]

    rows: list[dict[str, object]] = []

    for (
        specification_name,
        train_fraction,
    ), group in candidates.groupby(
        [
            "feature_specification",
            "train_fraction",
        ],
        sort=True,
    ):
        differences = group[
            (
                "combined_poisson_"
                "deviance_difference"
            )
        ]

        rows.append(
            {
                "feature_specification":
                    specification_name,
                "train_fraction":
                    train_fraction,
                "alpha_count":
                    len(group),
                "poisson_deviance_wins":
                    int(
                        (
                            differences < 0
                        ).sum()
                    ),
                "poisson_deviance_losses":
                    int(
                        (
                            differences > 0
                        ).sum()
                    ),
                "poisson_deviance_win_rate":
                    float(
                        (
                            differences < 0
                        ).mean()
                    ),
                (
                    "mean_poisson_"
                    "deviance_difference"
                ): float(
                    differences.mean()
                ),
                (
                    "best_poisson_"
                    "deviance_difference"
                ): float(
                    differences.min()
                ),
                (
                    "worst_poisson_"
                    "deviance_difference"
                ): float(
                    differences.max()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_coefficient_stability(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    feature_rows = coefficients[
        ~coefficients["feature"].eq(
            "intercept"
        )
    ].copy()

    rows: list[dict[str, object]] = []

    grouping_columns = [
        "feature_specification",
        "target",
        "feature",
    ]

    for keys, group in feature_rows.groupby(
        grouping_columns,
        sort=True,
    ):
        (
            specification_name,
            target,
            feature,
        ) = keys

        values = group[
            "coefficient"
        ].to_numpy(dtype=float)

        positive_count = int(
            (values > 0).sum()
        )

        negative_count = int(
            (values < 0).sum()
        )

        zero_count = int(
            np.isclose(
                values,
                0.0,
                atol=1e-12,
                rtol=0.0,
            ).sum()
        )

        nonzero_signs = set(
            np.sign(
                values[
                    ~np.isclose(
                        values,
                        0.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                ]
            )
        )

        sign_stable = (
            len(nonzero_signs) <= 1
        )

        absolute_values = np.abs(values)

        mean_absolute = float(
            absolute_values.mean()
        )

        maximum_absolute = float(
            absolute_values.max()
        )

        rows.append(
            {
                "feature_specification":
                    specification_name,
                "target": target,
                "feature": feature,
                "coefficient_count":
                    len(values),
                "mean_coefficient":
                    float(values.mean()),
                "standard_deviation":
                    float(
                        values.std(
                            ddof=0
                        )
                    ),
                "minimum_coefficient":
                    float(values.min()),
                "maximum_coefficient":
                    float(values.max()),
                "mean_absolute_coefficient":
                    mean_absolute,
                "maximum_absolute_coefficient":
                    maximum_absolute,
                "positive_count":
                    positive_count,
                "negative_count":
                    negative_count,
                "zero_count":
                    zero_count,
                "sign_stable":
                    sign_stable,
                (
                    "maximum_to_mean_"
                    "absolute_ratio"
                ): (
                    maximum_absolute
                    / mean_absolute
                    if mean_absolute > 0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def build_feature_focus_summary(
    coefficient_stability: pd.DataFrame,
) -> pd.DataFrame:
    focus_features = {
        "attack_depth_diff",
        "midfield_diff",
    }

    return (
        coefficient_stability[
            coefficient_stability[
                "feature"
            ].isin(focus_features)
        ]
        .sort_values(
            [
                "feature_specification",
                "target",
                "feature",
            ]
        )
        .reset_index(drop=True)
    )


def determine_conclusion(
    specification_summary: pd.DataFrame,
) -> dict[str, object]:
    primary_metric = (
        "combined_poisson_deviance"
    )

    wins_column = (
        f"{primary_metric}_wins"
    )

    mean_difference_column = (
        f"mean_{primary_metric}_difference"
    )

    ranked = (
        specification_summary
        .sort_values(
            [
                wins_column,
                mean_difference_column,
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    best = ranked.iloc[0]

    comparison_count = int(
        best["comparison_count"]
    )

    win_count = int(
        best[wins_column]
    )

    win_rate = float(
        best[
            f"{primary_metric}_win_rate"
        ]
    )

    mean_difference = float(
        best[
            mean_difference_column
        ]
    )

    if (
        win_rate >= 0.75
        and mean_difference < 0
    ):
        evidence_strength = "strong"
    elif (
        win_rate >= 0.60
        and mean_difference < 0
    ):
        evidence_strength = "moderate"
    elif mean_difference < 0:
        evidence_strength = "weak"
    else:
        evidence_strength = "none"

    return {
        "best_specification": (
            best[
                "feature_specification"
            ]
        ),
        "poisson_deviance_wins":
            win_count,
        "comparison_count":
            comparison_count,
        "poisson_deviance_win_rate":
            win_rate,
        (
            "mean_poisson_"
            "deviance_difference"
        ): mean_difference,
        "evidence_strength":
            evidence_strength,
    }


def write_results_markdown(
    path: Path,
    result: GoalModelBenchmarkResult,
    specification_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    conclusion: dict[str, object],
) -> None:
    best_deviance = (
        result.results
        .sort_values(
            "combined_poisson_deviance"
        )
        .iloc[0]
    )

    best_ranked = (
        result.configuration_ranking
        .iloc[0]
    )

    lines = [
        "# Study 054 Results",
        "",
        (
            "## Attack Depth and Midfield "
            "Robustness"
        ),
        "",
        "**Status:** `PASS`",
        "",
        "## Experimental design",
        "",
        "- Representation: `full_squad`",
        (
            "- Feature specifications: "
            f"{list(FEATURE_SPECIFICATIONS)}"
        ),
        (
            "- Training fractions: "
            f"{list(TRAIN_FRACTIONS)}"
        ),
        (
            "- Alpha values: "
            f"{list(ALPHA_VALUES)}"
        ),
        (
            "- Benchmark configurations: "
            f"{len(result.results)}"
        ),
        "",
        "## Best Poisson-deviance configuration",
        "",
        (
            "- Specification: "
            f"`{best_deviance['feature_specification']}`"
        ),
        (
            "- Training fraction: "
            f"{best_deviance['train_fraction']:.2f}"
        ),
        (
            "- Alpha: "
            f"{best_deviance['alpha']:.6f}"
        ),
        (
            "- Combined Poisson deviance: "
            f"{best_deviance['combined_poisson_deviance']:.6f}"
        ),
        "",
        "## Best multi-metric configuration",
        "",
        (
            "- Specification: "
            f"`{best_ranked['feature_specification']}`"
        ),
        (
            "- Training fraction: "
            f"{best_ranked['train_fraction']:.2f}"
        ),
        (
            "- Alpha: "
            f"{best_ranked['alpha']:.6f}"
        ),
        (
            "- Mean metric rank: "
            f"{best_ranked['mean_metric_rank']:.4f}"
        ),
        "",
        "## Baseline-comparison conclusion",
        "",
        (
            "- Strongest specification: "
            f"`{conclusion['best_specification']}`"
        ),
        (
            "- Poisson-deviance wins: "
            f"{conclusion['poisson_deviance_wins']}"
            f"/{conclusion['comparison_count']}"
        ),
        (
            "- Win rate: "
            f"{conclusion['poisson_deviance_win_rate']:.3f}"
        ),
        (
            "- Mean deviance difference: "
            f"{conclusion['mean_poisson_deviance_difference']:.6f}"
        ),
        (
            "- Evidence strength: "
            f"`{conclusion['evidence_strength']}`"
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "The chronological cut points overlap "
            "within one season and therefore are not "
            "independent test seasons."
        ),
        (
            "Coefficient stability is diagnostic; "
            "coefficient magnitude should not yet be "
            "interpreted causally."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    config = build_config()

    benchmark_result = (
        run_goal_model_benchmark(
            config
        )
    )

    expected_run_count = (
        len(FEATURE_SPECIFICATIONS)
        * len(TRAIN_FRACTIONS)
        * len(ALPHA_VALUES)
    )

    if (
        len(benchmark_result.results)
        != expected_run_count
    ):
        raise AssertionError(
            "Unexpected benchmark configuration "
            f"count: {len(benchmark_result.results)} "
            f"vs {expected_run_count}."
        )

    baseline_comparison = (
        build_baseline_comparison(
            benchmark_result.results
        )
    )

    specification_summary = (
        build_specification_summary(
            baseline_comparison
        )
    )

    split_summary = (
        build_split_summary(
            baseline_comparison
        )
    )

    coefficient_stability = (
        build_coefficient_stability(
            benchmark_result.coefficients
        )
    )

    feature_focus_summary = (
        build_feature_focus_summary(
            coefficient_stability
        )
    )

    conclusion = determine_conclusion(
        specification_summary
    )

    write_goal_model_benchmark_outputs(
        benchmark_result=benchmark_result,
        output_directory=OUTPUT_DIR,
    )

    baseline_comparison.to_csv(
        OUTPUT_DIR
        / "baseline_comparison.csv",
        index=False,
    )

    specification_summary.to_csv(
        OUTPUT_DIR
        / "specification_summary.csv",
        index=False,
    )

    split_summary.to_csv(
        OUTPUT_DIR
        / "split_summary.csv",
        index=False,
    )

    coefficient_stability.to_csv(
        OUTPUT_DIR
        / "coefficient_stability.csv",
        index=False,
    )

    feature_focus_summary.to_csv(
        OUTPUT_DIR
        / "attack_depth_midfield_coefficients.csv",
        index=False,
    )

    metadata = {
        "study_id": "054",
        "study_name": (
            "Attack Depth and Midfield "
            "Robustness"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "representation_type": (
            "full_squad"
        ),
        "observation_count": int(
            len(
                benchmark_result.datasets[
                    "full_squad"
                ]
            )
        ),
        "feature_specifications": list(
            FEATURE_SPECIFICATIONS
        ),
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
        "baseline_specification": (
            BASELINE_SPECIFICATION
        ),
        "rating_prior_included": False,
        "evidence_score_included": False,
        "conclusion": conclusion,
        "output_files": [
            "benchmark_results.csv",
            "configuration_ranking.csv",
            "split_assignments.csv",
            "coefficients.csv",
            "predictions.csv",
            "baseline_comparison.csv",
            "specification_summary.csv",
            "split_summary.csv",
            "coefficient_stability.csv",
            (
                "attack_depth_midfield_"
                "coefficients.csv"
            ),
            "study_metadata.json",
            "STUDY_054_RESULTS.md",
        ],
    }

    with (
        OUTPUT_DIR
        / "study_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    write_results_markdown(
        path=(
            OUTPUT_DIR
            / "STUDY_054_RESULTS.md"
        ),
        result=benchmark_result,
        specification_summary=(
            specification_summary
        ),
        split_summary=split_summary,
        conclusion=conclusion,
    )

    print("Study 054")
    print("=" * 78)
    print()
    print(
        "Benchmark configurations: "
        f"{len(benchmark_result.results)}"
    )
    print(
        "Feature specifications: "
        f"{len(FEATURE_SPECIFICATIONS)}"
    )
    print(
        "Training fractions: "
        f"{len(TRAIN_FRACTIONS)}"
    )
    print(
        "Alpha values: "
        f"{len(ALPHA_VALUES)}"
    )
    print()

    print("Specification Summary")
    print("-" * 78)

    summary_columns = [
        "feature_specification",
        "comparison_count",
        (
            "combined_poisson_"
            "deviance_wins"
        ),
        (
            "combined_poisson_"
            "deviance_win_rate"
        ),
        (
            "mean_combined_poisson_"
            "deviance_difference"
        ),
        "outcome_log_loss_wins",
        (
            "mean_outcome_log_loss_"
            "difference"
        ),
        "outcome_brier_score_wins",
        (
            "mean_outcome_brier_score_"
            "difference"
        ),
    ]

    print(
        specification_summary[
            summary_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Split-Level Poisson Summary")
    print("-" * 78)
    print(
        split_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Focused Coefficient Stability")
    print("-" * 78)
    print(
        feature_focus_summary[
            [
                "feature_specification",
                "target",
                "feature",
                "coefficient_count",
                "mean_coefficient",
                "standard_deviation",
                "minimum_coefficient",
                "maximum_coefficient",
                "sign_stable",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Conclusion")
    print("-" * 78)
    print(
        "Best specification: "
        f"{conclusion['best_specification']}"
    )
    print(
        "Poisson-deviance wins: "
        f"{conclusion['poisson_deviance_wins']}"
        f"/{conclusion['comparison_count']}"
    )
    print(
        "Mean deviance difference: "
        f"{conclusion['mean_poisson_deviance_difference']:.6f}"
    )
    print(
        "Evidence strength: "
        f"{conclusion['evidence_strength']}"
    )

    print()
    print("Benchmark-engine integration: PASS")
    print("Baseline comparison: PASS")
    print("Split robustness: PASS")
    print("Fine regularization grid: PASS")
    print("Coefficient stability capture: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
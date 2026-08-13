#benchmark_dynamic_form_incremental_information

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Importing the module registers named experiments.
import research.experiments.goal_model_experiments  # noqa: F401

from research.benchmarking.goal_model_benchmark import (
    run_goal_model_benchmark,
    write_goal_model_benchmark_outputs,
)
from research.experiments.experiment_registry import (
    GoalModelExperiment,
    get_goal_model_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_NAME = (
    "dynamic_form_incremental_information"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_068_dynamic_form_incremental_information"
)

PAIRED_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "paired_comparison_results.csv"
)

INCREMENTAL_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "dynamic_form_incremental_value_summary.csv"
)

COEFFICIENT_PATH = (
    OUTPUT_DIRECTORY
    / "dynamic_form_coefficients.csv"
)

OVERLAP_PATH = (
    OUTPUT_DIRECTORY
    / "dynamic_form_information_overlap.csv"
)

SPLIT_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "dynamic_form_split_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


PAIRING_COLUMNS = (
    "dataset",
    "representation_type",
    "train_fraction",
    "alpha",
)

COMPARISON_METRICS = (
    "combined_poisson_deviance",
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

DYNAMIC_FORM_FEATURES = (
    "home_attack_form",
    "away_attack_form",
    "home_defense_form",
    "away_defense_form",
)


def validate_experiment_dataset(
    experiment: GoalModelExperiment,
) -> None:
    for dataset in experiment.datasets:
        if not dataset.path.exists():
            raise FileNotFoundError(
                "Registered Dynamic Form dataset does "
                f"not exist: {dataset.path}"
            )


def build_paired_results(
    benchmark_results: pd.DataFrame,
    experiment: GoalModelExperiment,
) -> pd.DataFrame:
    comparison = experiment.paired_comparisons[0]

    baseline = benchmark_results[
        benchmark_results[
            "feature_specification"
        ].eq(
            comparison.baseline_specification
        )
    ][
        [
            *PAIRING_COLUMNS,
            *COMPARISON_METRICS,
        ]
    ].copy()

    candidate = benchmark_results[
        benchmark_results[
            "feature_specification"
        ].eq(
            comparison.candidate_specification
        )
    ][
        [
            *PAIRING_COLUMNS,
            *COMPARISON_METRICS,
        ]
    ].copy()

    baseline = baseline.rename(
        columns={
            metric: f"baseline_{metric}"
            for metric in COMPARISON_METRICS
        }
    )

    candidate = candidate.rename(
        columns={
            metric: f"candidate_{metric}"
            for metric in COMPARISON_METRICS
        }
    )

    paired = baseline.merge(
        candidate,
        on=list(PAIRING_COLUMNS),
        how="inner",
        validate="one_to_one",
    )

    expected_rows = (
        len(experiment.datasets)
        * len(experiment.train_fractions)
        * len(experiment.alpha_values)
    )

    if len(paired) != expected_rows:
        raise AssertionError(
            "Unexpected paired-result count: "
            f"{len(paired)} vs {expected_rows}."
        )

    paired.insert(
        0,
        "comparison_name",
        comparison.name,
    )

    paired.insert(
        1,
        "baseline_specification",
        comparison.baseline_specification,
    )

    paired.insert(
        2,
        "candidate_specification",
        comparison.candidate_specification,
    )

    for metric in COMPARISON_METRICS:
        delta_column = f"delta_{metric}"

        paired[delta_column] = (
            paired[f"candidate_{metric}"]
            - paired[f"baseline_{metric}"]
        )

        paired[
            f"improved_{metric}"
        ] = paired[delta_column] < 0.0

    return paired


def build_incremental_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for metric in COMPARISON_METRICS:
        deltas = paired[
            f"delta_{metric}"
        ].to_numpy(dtype=float)

        tie_mask = np.isclose(
            deltas,
            0.0,
            atol=1e-12,
            rtol=0.0,
        )

        win_count = int(
            np.sum(deltas < 0.0)
        )

        tie_count = int(
            np.sum(tie_mask)
        )

        loss_count = int(
            len(deltas)
            - win_count
            - tie_count
        )

        records.append(
            {
                "metric": metric,
                "comparison_count":
                    len(deltas),
                "win_count": win_count,
                "tie_count": tie_count,
                "loss_count": loss_count,
                "win_rate": float(
                    win_count / len(deltas)
                ),
                "mean_delta": float(
                    np.mean(deltas)
                ),
                "median_delta": float(
                    np.median(deltas)
                ),
                "minimum_delta": float(
                    np.min(deltas)
                ),
                "maximum_delta": float(
                    np.max(deltas)
                ),
                "mean_improvement": float(
                    -np.mean(deltas)
                ),
            }
        )

    return pd.DataFrame(records)


def build_split_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for train_fraction, group in paired.groupby(
        "train_fraction",
        sort=True,
    ):
        for metric in (
            "combined_poisson_deviance",
            "combined_goal_mae",
            "outcome_log_loss",
            "exact_score_log_loss",
        ):
            deltas = group[
                f"delta_{metric}"
            ].to_numpy(dtype=float)

            records.append(
                {
                    "train_fraction":
                        train_fraction,
                    "metric": metric,
                    "comparison_count":
                        len(deltas),
                    "win_count": int(
                        np.sum(deltas < 0.0)
                    ),
                    "win_rate": float(
                        np.mean(deltas < 0.0)
                    ),
                    "mean_delta": float(
                        np.mean(deltas)
                    ),
                    "minimum_delta": float(
                        np.min(deltas)
                    ),
                    "maximum_delta": float(
                        np.max(deltas)
                    ),
                }
            )

    return pd.DataFrame(records)


def extract_dynamic_form_coefficients(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    output = coefficients[
        coefficients[
            "feature"
        ].isin(
            DYNAMIC_FORM_FEATURES
        )
    ].copy()

    if output.empty:
        raise AssertionError(
            "No Dynamic Form coefficients were captured."
        )

    output["coefficient_sign"] = np.select(
        [
            output["coefficient"] > 0.0,
            output["coefficient"] < 0.0,
        ],
        [
            "positive",
            "negative",
        ],
        default="zero",
    )

    return (
        output
        .sort_values(
            [
                "feature",
                "target",
                "train_fraction",
                "alpha",
            ]
        )
        .reset_index(drop=True)
    )


def build_information_overlap(
    dataframe: pd.DataFrame,
    overlap_features: tuple[str, ...],
) -> pd.DataFrame:
    missing = (
        set(overlap_features)
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Dataset is missing overlap features: "
            f"{sorted(missing)}"
        )

    values = dataframe[
        list(overlap_features)
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    records: list[dict[str, object]] = []

    for feature in (
        "attack_form_diff",
        "defense_form_diff",
    ):
        for comparison_feature in (
            "attack_depth_diff",
            "rating_prior_diff",
        ):
            records.append(
                {
                    "dynamic_form_feature":
                        feature,
                    "comparison_feature":
                        comparison_feature,
                    "pearson_correlation": float(
                        values[feature].corr(
                            values[
                                comparison_feature
                            ],
                            method="pearson",
                        )
                    ),
                    "spearman_correlation": float(
                        values[feature].corr(
                            values[
                                comparison_feature
                            ],
                            method="spearman",
                        )
                    ),
                    "observation_count":
                        len(values),
                }
            )

    return pd.DataFrame(records)


def get_summary_row(
    summary: pd.DataFrame,
    metric: str,
) -> pd.Series:
    selected = summary[
        summary["metric"].eq(metric)
    ]

    if len(selected) != 1:
        raise AssertionError(
            f"Expected one summary row for {metric!r}."
        )

    return selected.iloc[0]


def classify_evidence(
    summary: pd.DataFrame,
) -> str:
    poisson = get_summary_row(
        summary,
        "combined_poisson_deviance",
    )

    outcome = get_summary_row(
        summary,
        "outcome_log_loss",
    )

    exact = get_summary_row(
        summary,
        "exact_score_log_loss",
    )

    poisson_better = (
        float(poisson["mean_delta"]) < 0.0
        and float(poisson["win_rate"]) > 0.5
    )

    outcome_better = (
        float(outcome["mean_delta"]) < 0.0
        and float(outcome["win_rate"]) > 0.5
    )

    exact_better = (
        float(exact["mean_delta"]) < 0.0
        and float(exact["win_rate"]) > 0.5
    )

    if (
        poisson_better
        and outcome_better
        and exact_better
    ):
        return "strong"

    if poisson_better and exact_better:
        return "moderate"

    if poisson_better:
        return "limited"

    return "none"


def build_metadata(
    experiment: GoalModelExperiment,
    benchmark_results: pd.DataFrame,
    paired: pd.DataFrame,
    evidence: str,
) -> dict[str, object]:
    return {
        "study_id": "068",
        "study_name": (
            "Dynamic Form Incremental Information "
            "Benchmark"
        ),
        "experiment_name":
            experiment.name,
        "baseline_specification": (
            experiment.paired_comparisons[
                0
            ].baseline_specification
        ),
        "candidate_specification": (
            experiment.paired_comparisons[
                0
            ].candidate_specification
        ),
        "dataset_paths": [
            str(dataset.path)
            for dataset in experiment.datasets
        ],
        "train_fractions": list(
            experiment.train_fractions
        ),
        "alpha_values": list(
            experiment.alpha_values
        ),
        "expected_benchmark_runs":
            experiment.expected_benchmark_runs,
        "actual_benchmark_runs": int(
            len(benchmark_results)
        ),
        "paired_comparison_rows": int(
            len(paired)
        ),
        "evidence_classification":
            evidence,
    }


def write_report(
    experiment: GoalModelExperiment,
    benchmark_results: pd.DataFrame,
    summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    overlap: pd.DataFrame,
    evidence: str,
) -> None:
    poisson = get_summary_row(
        summary,
        "combined_poisson_deviance",
    )

    goal_mae = get_summary_row(
        summary,
        "combined_goal_mae",
    )

    outcome = get_summary_row(
        summary,
        "outcome_log_loss",
    )

    exact = get_summary_row(
        summary,
        "exact_score_log_loss",
    )

    best = (
        benchmark_results
        .sort_values(
            "combined_poisson_deviance"
        )
        .iloc[0]
    )

    report = f"""# Study 068 — Dynamic Form Incremental Information Benchmark

## Research question

Does leakage-safe recent attacking and defensive form improve
prediction beyond the current Integrated Club Goal Model v1?

## Controlled comparison

### Baseline

`{experiment.paired_comparisons[0].baseline_specification}`

### Candidate

`{experiment.paired_comparisons[0].candidate_specification}`

The only additional football concept is Dynamic Form.

## Experiment

- Benchmark configurations: {len(benchmark_results)}
- Paired comparisons: {int(poisson["comparison_count"])}
- Training fractions: {len(experiment.train_fractions)}
- Alpha values: {len(experiment.alpha_values)}

A negative delta indicates that the Dynamic Form candidate
outperformed Version 1.

## Incremental results

- Combined Poisson deviance:
  mean delta {float(poisson["mean_delta"]):.8f};
  wins {int(poisson["win_count"])}/{int(poisson["comparison_count"])}
  ({float(poisson["win_rate"]):.1%})
- Combined goal MAE:
  mean delta {float(goal_mae["mean_delta"]):.8f};
  wins {int(goal_mae["win_count"])}/{int(goal_mae["comparison_count"])}
  ({float(goal_mae["win_rate"]):.1%})
- Outcome log loss:
  mean delta {float(outcome["mean_delta"]):.8f};
  wins {int(outcome["win_count"])}/{int(outcome["comparison_count"])}
  ({float(outcome["win_rate"]):.1%})
- Exact-score log loss:
  mean delta {float(exact["mean_delta"]):.8f};
  wins {int(exact["win_count"])}/{int(exact["comparison_count"])}
  ({float(exact["win_rate"]):.1%})

## Best configuration

- Feature specification:
  `{best["feature_specification"]}`
- Training fraction:
  {float(best["train_fraction"]):.2f}
- Alpha:
  {float(best["alpha"]):.6f}
- Combined Poisson deviance:
  {float(best["combined_poisson_deviance"]):.8f}

## Dynamic Form coefficients

- Coefficient rows: {len(coefficients)}
- Positive: {int((coefficients["coefficient"] > 0).sum())}
- Negative: {int((coefficients["coefficient"] < 0).sum())}
- Mean absolute coefficient:
  {float(coefficients["coefficient"].abs().mean()):.8f}

## Evidence classification

**{evidence.upper()}**

## Interpretation policy

Dynamic Form should be promoted only if improvement is
consistent across chronological splits and is not confined to
one regularization setting.

A mixed or negative result does not invalidate the residual
repository or provider. It means only that this particular
eight-match, 0.80-decay formulation has not earned inclusion
in the predictive baseline.

## Validation

- Experiment Registry integration: PASS
- Matched observation population: PASS
- Expected benchmark run count: PASS
- Controlled baseline/candidate pairing: PASS
- Dynamic Form coefficient extraction: PASS
- Information-overlap analysis: PASS
- Split-stability analysis: PASS
- Output generation: PASS

## Result

**OVERALL EXECUTION RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    experiment = get_goal_model_experiment(
        EXPERIMENT_NAME
    )

    experiment.validate()
    validate_experiment_dataset(
        experiment
    )

    benchmark_result = (
        run_goal_model_benchmark(
            experiment.to_benchmark_config()
        )
    )

    if len(benchmark_result.results) != (
        experiment.expected_benchmark_runs
    ):
        raise AssertionError(
            "Unexpected benchmark run count."
        )

    paired = build_paired_results(
        benchmark_results=(
            benchmark_result.results
        ),
        experiment=experiment,
    )

    summary = build_incremental_summary(
        paired
    )

    split_summary = build_split_summary(
        paired
    )

    coefficients = (
        extract_dynamic_form_coefficients(
            benchmark_result.coefficients
        )
    )

    dataset = next(
        iter(
            benchmark_result.datasets.values()
        )
    )

    overlap = build_information_overlap(
        dataframe=dataset,
        overlap_features=(
            experiment.overlap_features
        ),
    )

    evidence = classify_evidence(
        summary
    )

    metadata = build_metadata(
        experiment=experiment,
        benchmark_results=(
            benchmark_result.results
        ),
        paired=paired,
        evidence=evidence,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_goal_model_benchmark_outputs(
        benchmark_result=benchmark_result,
        output_directory=OUTPUT_DIRECTORY,
    )

    paired.to_csv(
        PAIRED_RESULTS_PATH,
        index=False,
    )

    summary.to_csv(
        INCREMENTAL_SUMMARY_PATH,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENT_PATH,
        index=False,
    )

    overlap.to_csv(
        OVERLAP_PATH,
        index=False,
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_PATH,
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
        experiment=experiment,
        benchmark_results=(
            benchmark_result.results
        ),
        summary=summary,
        split_summary=split_summary,
        coefficients=coefficients,
        overlap=overlap,
        evidence=evidence,
    )

    display = summary[
        summary["metric"].isin(
            [
                "combined_poisson_deviance",
                "combined_goal_mae",
                "outcome_log_loss",
                "exact_score_log_loss",
            ]
        )
    ][
        [
            "metric",
            "win_count",
            "comparison_count",
            "win_rate",
            "mean_delta",
        ]
    ]

    print(
        "Study 068 — Dynamic Form Incremental "
        "Information Benchmark"
    )
    print("=" * 76)
    print()
    print(
        f"Experiment: {experiment.name}"
    )
    print(
        "Benchmark configurations: "
        f"{len(benchmark_result.results)}"
    )
    print(
        f"Paired comparisons: {len(paired)}"
    )
    print()
    print("Incremental Value")
    print("-" * 76)
    print(
        display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )
    print()
    print("Split Stability")
    print("-" * 76)
    print(
        split_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )
    print()
    print(
        f"Evidence classification: "
        f"{evidence.upper()}"
    )
    print()
    print("Experiment Registry integration: PASS")
    print("Benchmark run count: PASS")
    print("Controlled paired comparison: PASS")
    print("Dynamic Form coefficient extraction: PASS")
    print("Information-overlap analysis: PASS")
    print("Split-stability analysis: PASS")
    print("Output generation: PASS")
    print()
    print("OVERALL EXECUTION RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
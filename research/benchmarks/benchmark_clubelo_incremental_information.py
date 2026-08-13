#benchmark_clubelo_incremental_information

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Importing this module registers the named experiment.
import research.experiments.goal_model_experiments  # noqa: F401

from research.benchmarking.goal_model_benchmark import (
    DEFAULT_METRICS,
    run_goal_model_benchmark,
    write_goal_model_benchmark_outputs,
)
from research.experiments.experiment_registry import (
    GoalModelExperiment,
    get_goal_model_experiment,
)
from research.modeling.football_feature_registry import (
    get_club_goal_model_feature_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_NAME = (
    "clubelo_incremental_information"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_061_clubelo_incremental_information"
)

PAIRED_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "paired_comparison_results.csv"
)

INCREMENTAL_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "clubelo_incremental_value_summary.csv"
)

OVERLAP_MATRIX_PATH = (
    OUTPUT_DIRECTORY
    / "feature_overlap_matrix.csv"
)

OVERLAP_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "clubelo_information_overlap.csv"
)

OVERLAP_REGRESSION_PATH = (
    OUTPUT_DIRECTORY
    / "clubelo_overlap_regression.csv"
)

RATING_COEFFICIENT_PATH = (
    OUTPUT_DIRECTORY
    / "rating_prior_coefficients.csv"
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


def validate_experiment_dataset(
    experiment: GoalModelExperiment,
) -> None:
    """
    Confirm that every dataset required by the registered
    experiment exists before invoking the benchmark engine.
    """
    missing_paths = [
        dataset.path
        for dataset in experiment.datasets
        if not dataset.path.exists()
    ]

    if missing_paths:
        formatted = "\n".join(
            f"- {path}"
            for path in missing_paths
        )

        raise FileNotFoundError(
            "One or more registered experiment datasets "
            f"do not exist:\n{formatted}"
        )


def build_paired_comparison_results(
    benchmark_results: pd.DataFrame,
    experiment: GoalModelExperiment,
) -> pd.DataFrame:
    """
    Compare each candidate specification directly against
    its registered baseline at identical dataset, split,
    and alpha settings.

    For every benchmark metric, a negative delta means the
    candidate produced a lower and therefore better value.
    """
    frames: list[pd.DataFrame] = []

    for comparison in experiment.paired_comparisons:
        baseline = benchmark_results[
            benchmark_results[
                "feature_specification"
            ].eq(
                comparison.baseline_specification
            )
        ].copy()

        candidate = benchmark_results[
            benchmark_results[
                "feature_specification"
            ].eq(
                comparison.candidate_specification
            )
        ].copy()

        if baseline.empty:
            raise AssertionError(
                f"{comparison.name}: baseline results "
                "are empty."
            )

        if candidate.empty:
            raise AssertionError(
                f"{comparison.name}: candidate results "
                "are empty."
            )

        baseline_columns = [
            *PAIRING_COLUMNS,
            *COMPARISON_METRICS,
        ]

        candidate_columns = [
            *PAIRING_COLUMNS,
            *COMPARISON_METRICS,
        ]

        baseline = baseline[
            baseline_columns
        ].rename(
            columns={
                metric: f"baseline_{metric}"
                for metric in COMPARISON_METRICS
            }
        )

        candidate = candidate[
            candidate_columns
        ].rename(
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
                f"{comparison.name}: unexpected paired "
                f"row count {len(paired)} vs "
                f"{expected_rows}."
            )

        paired.insert(
            0,
            "comparison_name",
            comparison.name,
        )

        paired.insert(
            1,
            "comparison_description",
            comparison.description,
        )

        paired.insert(
            2,
            "baseline_specification",
            comparison.baseline_specification,
        )

        paired.insert(
            3,
            "candidate_specification",
            comparison.candidate_specification,
        )

        for metric in COMPARISON_METRICS:
            baseline_column = (
                f"baseline_{metric}"
            )

            candidate_column = (
                f"candidate_{metric}"
            )

            delta_column = f"delta_{metric}"
            improved_column = (
                f"improved_{metric}"
            )

            paired[delta_column] = (
                paired[candidate_column]
                - paired[baseline_column]
            )

            paired[improved_column] = (
                paired[delta_column] < 0.0
            )

        frames.append(paired)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_incremental_value_summary(
    paired_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize ClubElo's incremental value across all
    split and alpha combinations for each paired comparison.
    """
    records: list[dict[str, object]] = []

    for (
        comparison_name,
        baseline_specification,
        candidate_specification,
    ), group in paired_results.groupby(
        [
            "comparison_name",
            "baseline_specification",
            "candidate_specification",
        ],
        sort=False,
    ):
        for metric in COMPARISON_METRICS:
            delta_column = f"delta_{metric}"
            deltas = group[
                delta_column
            ].to_numpy(dtype=float)

            win_count = int(
                np.sum(deltas < 0.0)
            )

            tie_count = int(
                np.sum(
                    np.isclose(
                        deltas,
                        0.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                )
            )

            loss_count = int(
                len(deltas)
                - win_count
                - tie_count
            )

            records.append(
                {
                    "comparison_name":
                        comparison_name,
                    "baseline_specification":
                        baseline_specification,
                    "candidate_specification":
                        candidate_specification,
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


def build_feature_overlap_matrix(
    dataframe: pd.DataFrame,
    overlap_features: tuple[str, ...],
) -> pd.DataFrame:
    """
    Produce Pearson and Spearman correlation matrices for
    all registered overlap-analysis features.
    """
    missing = (
        set(overlap_features)
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Observation dataset is missing registered "
            "overlap features: "
            f"{sorted(missing)}"
        )

    values = dataframe[
        list(overlap_features)
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if values.isna().any().any():
        raise ValueError(
            "Overlap-analysis features contain missing "
            "values."
        )

    pearson = values.corr(
        method="pearson"
    )

    spearman = values.corr(
        method="spearman"
    )

    frames: list[pd.DataFrame] = []

    for method, matrix in (
        ("pearson", pearson),
        ("spearman", spearman),
    ):
        long = (
            matrix
            .rename_axis("feature_x")
            .reset_index()
            .melt(
                id_vars="feature_x",
                var_name="feature_y",
                value_name="correlation",
            )
        )

        long.insert(
            0,
            "method",
            method,
        )

        frames.append(long)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_clubelo_overlap_summary(
    dataframe: pd.DataFrame,
    overlap_features: tuple[str, ...],
) -> pd.DataFrame:
    """
    Measure each player-derived feature's direct
    relationship with rating_prior_diff.
    """
    target = "rating_prior_diff"

    if target not in overlap_features:
        raise ValueError(
            "rating_prior_diff is not registered for "
            "overlap analysis."
        )

    records: list[dict[str, object]] = []

    for feature in overlap_features:
        if feature == target:
            continue

        pair = dataframe[
            [
                feature,
                target,
            ]
        ].apply(
            pd.to_numeric,
            errors="raise",
        )

        pearson = pair[feature].corr(
            pair[target],
            method="pearson",
        )

        spearman = pair[feature].corr(
            pair[target],
            method="spearman",
        )

        records.append(
            {
                "predictor": feature,
                "target": target,
                "pearson_correlation": float(
                    pearson
                ),
                "absolute_pearson_correlation": float(
                    abs(pearson)
                ),
                "spearman_correlation": float(
                    spearman
                ),
                "absolute_spearman_correlation": float(
                    abs(spearman)
                ),
                "observation_count": len(pair),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "absolute_pearson_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_overlap_regression(
    dataframe: pd.DataFrame,
    overlap_features: tuple[str, ...],
) -> pd.DataFrame:
    """
    Estimate how much of rating_prior_diff can be recovered
    from the registered player-derived features.

    Both raw and standardized coefficients are retained.
    The R-squared value measures total information overlap.
    """
    target_column = "rating_prior_diff"

    predictor_columns = [
        feature
        for feature in overlap_features
        if feature != target_column
    ]

    if not predictor_columns:
        raise ValueError(
            "No overlap-regression predictors were "
            "configured."
        )

    regression_data = dataframe[
        [
            *predictor_columns,
            target_column,
        ]
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if regression_data.isna().any().any():
        raise ValueError(
            "Overlap-regression data contain missing "
            "values."
        )

    x_raw = regression_data[
        predictor_columns
    ].to_numpy(dtype=float)

    y_raw = regression_data[
        target_column
    ].to_numpy(dtype=float)

    raw_model = LinearRegression()
    raw_model.fit(
        x_raw,
        y_raw,
    )

    predictor_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_standardized = (
        predictor_scaler.fit_transform(
            x_raw
        )
    )

    y_standardized = (
        target_scaler.fit_transform(
            y_raw.reshape(-1, 1)
        )
        .ravel()
    )

    standardized_model = LinearRegression()
    standardized_model.fit(
        x_standardized,
        y_standardized,
    )

    predictions = raw_model.predict(
        x_raw
    )

    residuals = y_raw - predictions

    r_squared = float(
        raw_model.score(
            x_raw,
            y_raw,
        )
    )

    records: list[dict[str, object]] = []

    for (
        predictor,
        raw_coefficient,
        standardized_coefficient,
    ) in zip(
        predictor_columns,
        raw_model.coef_,
        standardized_model.coef_,
    ):
        records.append(
            {
                "target": target_column,
                "predictor": predictor,
                "raw_coefficient": float(
                    raw_coefficient
                ),
                "standardized_coefficient": float(
                    standardized_coefficient
                ),
                "intercept": float(
                    raw_model.intercept_
                ),
                "r_squared": r_squared,
                "adjusted_r_squared":
                    calculate_adjusted_r_squared(
                        r_squared=r_squared,
                        observation_count=len(y_raw),
                        predictor_count=(
                            len(predictor_columns)
                        ),
                    ),
                "residual_standard_deviation": float(
                    np.std(
                        residuals,
                        ddof=1,
                    )
                ),
                "target_standard_deviation": float(
                    np.std(
                        y_raw,
                        ddof=1,
                    )
                ),
                "observation_count": len(y_raw),
                "predictor_count":
                    len(predictor_columns),
            }
        )

    return pd.DataFrame(records)


def calculate_adjusted_r_squared(
    r_squared: float,
    observation_count: int,
    predictor_count: int,
) -> float:
    denominator = (
        observation_count
        - predictor_count
        - 1
    )

    if denominator <= 0:
        raise ValueError(
            "Insufficient observations for adjusted "
            "R-squared."
        )

    return float(
        1.0
        - (
            (1.0 - r_squared)
            * (observation_count - 1)
            / denominator
        )
    )


def extract_rating_prior_coefficients(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """
    Isolate the fitted rating_prior_diff coefficients for
    interpretation and stability analysis.
    """
    if coefficients.empty:
        raise AssertionError(
            "Benchmark did not capture coefficients."
        )

    output = coefficients[
        coefficients["feature"].eq(
            "rating_prior_diff"
        )
    ].copy()

    if output.empty:
        raise AssertionError(
            "No rating_prior_diff coefficients were "
            "captured."
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
                "feature_specification",
                "train_fraction",
                "alpha",
                "target",
            ]
        )
        .reset_index(drop=True)
    )


def get_summary_row(
    summary: pd.DataFrame,
    comparison_name: str,
    metric: str,
) -> pd.Series:
    selected = summary[
        summary[
            "comparison_name"
        ].eq(comparison_name)
        & summary["metric"].eq(metric)
    ]

    if len(selected) != 1:
        raise AssertionError(
            "Expected exactly one summary row for "
            f"{comparison_name!r}, {metric!r}."
        )

    return selected.iloc[0]


def classify_incremental_evidence(
    summary: pd.DataFrame,
    experiment: GoalModelExperiment,
) -> str:
    """
    Apply a predeclared conservative interpretation rule.

    Strong:
      ClubElo improves both combined Poisson deviance and
      outcome log loss in both paired comparisons, with
      win rates above 50%.

    Moderate:
      Both paired comparisons improve combined Poisson
      deviance, but outcome-level evidence is mixed.

    Limited:
      Only one paired comparison improves combined Poisson
      deviance.

    None:
      Neither paired comparison improves combined Poisson
      deviance on average.
    """
    poisson_improvements: list[bool] = []
    outcome_improvements: list[bool] = []
    poisson_majorities: list[bool] = []
    outcome_majorities: list[bool] = []

    for comparison in experiment.paired_comparisons:
        poisson = get_summary_row(
            summary=summary,
            comparison_name=comparison.name,
            metric="combined_poisson_deviance",
        )

        outcome = get_summary_row(
            summary=summary,
            comparison_name=comparison.name,
            metric="outcome_log_loss",
        )

        poisson_improvements.append(
            float(poisson["mean_delta"]) < 0.0
        )

        outcome_improvements.append(
            float(outcome["mean_delta"]) < 0.0
        )

        poisson_majorities.append(
            float(poisson["win_rate"]) > 0.5
        )

        outcome_majorities.append(
            float(outcome["win_rate"]) > 0.5
        )

    if (
        all(poisson_improvements)
        and all(outcome_improvements)
        and all(poisson_majorities)
        and all(outcome_majorities)
    ):
        return "strong"

    if all(poisson_improvements):
        return "moderate"

    if any(poisson_improvements):
        return "limited"

    return "none"


def build_metadata(
    experiment: GoalModelExperiment,
    benchmark_results: pd.DataFrame,
    paired_results: pd.DataFrame,
    incremental_summary: pd.DataFrame,
    overlap_regression: pd.DataFrame,
    evidence_classification: str,
) -> dict[str, object]:
    return {
        "study": (
            "study_061_clubelo_"
            "incremental_information"
        ),
        "experiment_name": experiment.name,
        "experiment_description":
            experiment.description,
        "datasets": [
            {
                "name": dataset.name,
                "path": str(dataset.path),
                "representation_type":
                    dataset.representation_type,
            }
            for dataset in experiment.datasets
        ],
        "feature_specifications": list(
            experiment.feature_specifications
        ),
        "paired_comparisons": [
            {
                "name": comparison.name,
                "baseline_specification":
                    comparison.baseline_specification,
                "candidate_specification":
                    comparison.candidate_specification,
                "description":
                    comparison.description,
            }
            for comparison in (
                experiment.paired_comparisons
            )
        ],
        "overlap_features": list(
            experiment.overlap_features
        ),
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
        "paired_result_rows": int(
            len(paired_results)
        ),
        "incremental_summary_rows": int(
            len(incremental_summary)
        ),
        "clubelo_overlap_r_squared": float(
            overlap_regression[
                "r_squared"
            ].iloc[0]
        ),
        "clubelo_overlap_adjusted_r_squared":
            float(
                overlap_regression[
                    "adjusted_r_squared"
                ].iloc[0]
            ),
        "incremental_evidence_classification":
            evidence_classification,
    }


def format_metric_summary(
    summary: pd.DataFrame,
    comparison_name: str,
    metric: str,
) -> str:
    row = get_summary_row(
        summary=summary,
        comparison_name=comparison_name,
        metric=metric,
    )

    return (
        f"- `{metric}`: mean delta "
        f"{float(row['mean_delta']):.8f}; "
        f"wins {int(row['win_count'])}/"
        f"{int(row['comparison_count'])} "
        f"({float(row['win_rate']):.1%})"
    )


def write_report(
    experiment: GoalModelExperiment,
    benchmark_results: pd.DataFrame,
    incremental_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    overlap_regression: pd.DataFrame,
    rating_coefficients: pd.DataFrame,
    evidence_classification: str,
) -> None:
    best_poisson = (
        benchmark_results
        .sort_values(
            "combined_poisson_deviance"
        )
        .iloc[0]
    )

    best_outcome = (
        benchmark_results
        .sort_values(
            "outcome_log_loss"
        )
        .iloc[0]
    )

    overlap_r_squared = float(
        overlap_regression[
            "r_squared"
        ].iloc[0]
    )

    adjusted_r_squared = float(
        overlap_regression[
            "adjusted_r_squared"
        ].iloc[0]
    )

    strongest_overlap = (
        overlap_summary.iloc[0]
    )

    comparison_sections: list[str] = []

    for comparison in experiment.paired_comparisons:
        section_lines = [
            f"### {comparison.name}",
            "",
            comparison.description,
            "",
            format_metric_summary(
                summary=incremental_summary,
                comparison_name=comparison.name,
                metric=(
                    "combined_poisson_deviance"
                ),
            ),
            format_metric_summary(
                summary=incremental_summary,
                comparison_name=comparison.name,
                metric="combined_goal_mae",
            ),
            format_metric_summary(
                summary=incremental_summary,
                comparison_name=comparison.name,
                metric="outcome_log_loss",
            ),
            format_metric_summary(
                summary=incremental_summary,
                comparison_name=comparison.name,
                metric="exact_score_log_loss",
            ),
        ]

        comparison_sections.append(
            "\n".join(section_lines)
        )

    report = f"""# Study 061 — ClubElo Incremental Information Benchmark

## Research question

Does a temporally valid historical ClubElo rating prior
provide predictive information beyond player-derived club
representations?

## Experiment

- Registered experiment: `{experiment.name}`
- Benchmark configurations: {len(benchmark_results)}
- Feature specifications: {len(experiment.feature_specifications)}
- Training fractions: {len(experiment.train_fractions)}
- Alpha values: {len(experiment.alpha_values)}
- Paired comparisons: {len(experiment.paired_comparisons)}

A negative metric delta means that the ClubElo candidate
outperformed its corresponding player-only baseline.

## Paired comparison results

{chr(10).join(comparison_sections)}

## Information overlap

The linear regression predicting `rating_prior_diff` from
the registered player-derived difference features produced:

- R-squared: {overlap_r_squared:.6f}
- Adjusted R-squared: {adjusted_r_squared:.6f}
- Strongest individual Pearson relationship:
  `{strongest_overlap["predictor"]}`
  ({float(strongest_overlap["pearson_correlation"]):.6f})

This measures how much historical ClubElo information can be
reconstructed from the existing player-derived representation.

## Best configurations

### Lowest combined Poisson deviance

- Feature specification:
  `{best_poisson["feature_specification"]}`
- Training fraction:
  {float(best_poisson["train_fraction"]):.2f}
- Alpha:
  {float(best_poisson["alpha"]):.6f}
- Combined Poisson deviance:
  {float(best_poisson["combined_poisson_deviance"]):.8f}

### Lowest outcome log loss

- Feature specification:
  `{best_outcome["feature_specification"]}`
- Training fraction:
  {float(best_outcome["train_fraction"]):.2f}
- Alpha:
  {float(best_outcome["alpha"]):.6f}
- Outcome log loss:
  {float(best_outcome["outcome_log_loss"]):.8f}

## Rating-prior coefficients

- Captured ClubElo coefficient rows:
  {len(rating_coefficients)}
- Positive coefficients:
  {int((rating_coefficients["coefficient"] > 0).sum())}
- Negative coefficients:
  {int((rating_coefficients["coefficient"] < 0).sum())}
- Mean coefficient:
  {float(rating_coefficients["coefficient"].mean()):.10f}
- Minimum coefficient:
  {float(rating_coefficients["coefficient"].min()):.10f}
- Maximum coefficient:
  {float(rating_coefficients["coefficient"].max()):.10f}

## Predeclared evidence classification

**{evidence_classification.upper()}**

The classification is based on mean paired improvements and
win rates across the registered split and regularization grid.

## Outputs

- `benchmark_results.csv`
- `configuration_ranking.csv`
- `coefficients.csv`
- `predictions.csv`
- `split_assignments.csv`
- `paired_comparison_results.csv`
- `clubelo_incremental_value_summary.csv`
- `feature_overlap_matrix.csv`
- `clubelo_information_overlap.csv`
- `clubelo_overlap_regression.csv`
- `rating_prior_coefficients.csv`
- `study_metadata.json`

## Validation

- Experiment registry loading: PASS
- Benchmark run count: PASS
- Controlled paired comparisons: PASS
- Rating-prior isolation: PASS
- Information-overlap feature coverage: PASS
- ClubElo coefficient extraction: PASS
- Output generation: PASS
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

    benchmark_config = (
        experiment.to_benchmark_config()
    )

    benchmark_result = (
        run_goal_model_benchmark(
            benchmark_config
        )
    )

    if len(benchmark_result.results) != (
        experiment.expected_benchmark_runs
    ):
        raise AssertionError(
            "Benchmark execution produced an unexpected "
            f"run count: {len(benchmark_result.results)} "
            f"vs {experiment.expected_benchmark_runs}."
        )

    if len(benchmark_result.datasets) != 1:
        raise AssertionError(
            "Study 061 currently expects exactly one "
            "observation dataset."
        )

    observation_dataset = next(
        iter(
            benchmark_result.datasets.values()
        )
    )

    paired_results = (
        build_paired_comparison_results(
            benchmark_results=(
                benchmark_result.results
            ),
            experiment=experiment,
        )
    )

    incremental_summary = (
        build_incremental_value_summary(
            paired_results
        )
    )

    feature_overlap_matrix = (
        build_feature_overlap_matrix(
            dataframe=observation_dataset,
            overlap_features=(
                experiment.overlap_features
            ),
        )
    )

    overlap_summary = (
        build_clubelo_overlap_summary(
            dataframe=observation_dataset,
            overlap_features=(
                experiment.overlap_features
            ),
        )
    )

    overlap_regression = (
        build_overlap_regression(
            dataframe=observation_dataset,
            overlap_features=(
                experiment.overlap_features
            ),
        )
    )

    rating_coefficients = (
        extract_rating_prior_coefficients(
            benchmark_result.coefficients
        )
    )

    evidence_classification = (
        classify_incremental_evidence(
            summary=incremental_summary,
            experiment=experiment,
        )
    )

    metadata = build_metadata(
        experiment=experiment,
        benchmark_results=(
            benchmark_result.results
        ),
        paired_results=paired_results,
        incremental_summary=(
            incremental_summary
        ),
        overlap_regression=(
            overlap_regression
        ),
        evidence_classification=(
            evidence_classification
        ),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_goal_model_benchmark_outputs(
        benchmark_result=benchmark_result,
        output_directory=OUTPUT_DIRECTORY,
    )

    paired_results.to_csv(
        PAIRED_RESULTS_PATH,
        index=False,
    )

    incremental_summary.to_csv(
        INCREMENTAL_SUMMARY_PATH,
        index=False,
    )

    feature_overlap_matrix.to_csv(
        OVERLAP_MATRIX_PATH,
        index=False,
    )

    overlap_summary.to_csv(
        OVERLAP_SUMMARY_PATH,
        index=False,
    )

    overlap_regression.to_csv(
        OVERLAP_REGRESSION_PATH,
        index=False,
    )

    rating_coefficients.to_csv(
        RATING_COEFFICIENT_PATH,
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
        incremental_summary=(
            incremental_summary
        ),
        overlap_summary=overlap_summary,
        overlap_regression=(
            overlap_regression
        ),
        rating_coefficients=(
            rating_coefficients
        ),
        evidence_classification=(
            evidence_classification
        ),
    )

    print(
        "Study 061 — ClubElo Incremental "
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
        "Paired comparison rows: "
        f"{len(paired_results)}"
    )
    print()

    print("Information Overlap")
    print("-" * 76)
    print(
        "ClubElo overlap R-squared: "
        f"{float(overlap_regression['r_squared'].iloc[0]):.6f}"
    )
    print(
        "Adjusted R-squared: "
        f"{float(overlap_regression['adjusted_r_squared'].iloc[0]):.6f}"
    )
    print()
    print(
        overlap_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Incremental Value")
    print("-" * 76)

    display_summary = (
        incremental_summary[
            incremental_summary[
                "metric"
            ].isin(
                [
                    "combined_poisson_deviance",
                    "combined_goal_mae",
                    "outcome_log_loss",
                    "exact_score_log_loss",
                ]
            )
        ][
            [
                "comparison_name",
                "metric",
                "win_count",
                "comparison_count",
                "win_rate",
                "mean_delta",
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
    print(
        "Evidence classification: "
        f"{evidence_classification.upper()}"
    )
    print()
    print("Experiment registry loading: PASS")
    print("Benchmark run count: PASS")
    print("Controlled paired comparisons: PASS")
    print("Rating-prior isolation: PASS")
    print("Information-overlap analysis: PASS")
    print("Coefficient extraction: PASS")
    print("Output generation: PASS")
    print()
    print("OVERALL EXECUTION RESULT: PASS")
    print()
    print(
        f"Outputs written to: {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
#analyze_bundesliga_replay_performance

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_083_bundesliga_production_replay"
    / "fixture_replay_predictions.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_084_bundesliga_replay_performance_analysis"
)

PERFORMANCE_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "performance_summary.csv"
)

FIXTURE_ERRORS_PATH = (
    OUTPUT_DIRECTORY
    / "fixture_error_analysis.csv"
)

TEAM_BIAS_PATH = (
    OUTPUT_DIRECTORY
    / "team_bias_summary.csv"
)

CALIBRATION_PATH = (
    OUTPUT_DIRECTORY
    / "outcome_probability_calibration.csv"
)

EXTREME_ERRORS_PATH = (
    OUTPUT_DIRECTORY
    / "largest_prediction_errors.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


EXPECTED_MATCH_COUNT = 306
PROBABILITY_FLOOR = 1e-15
CALIBRATION_BIN_COUNT = 10
EXTREME_ERROR_COUNT = 20


def load_replay_predictions() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Study 083 replay output does not exist: "
            f"{INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Study 083 replay output is empty."
        )

    required_columns = {
        "event_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "goal_difference",
        "total_goals",
        "outcome",
        "lambda_home",
        "lambda_away",
        "pred_total_goals",
        "pred_goal_diff",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "prediction_status",
    }

    missing = required_columns - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Replay dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.copy()

    if len(dataframe) != EXPECTED_MATCH_COUNT:
        raise ValueError(
            "Unexpected replay population. "
            f"Expected {EXPECTED_MATCH_COUNT}, "
            f"received {len(dataframe)}."
        )

    if dataframe["event_id"].duplicated().any():
        raise ValueError(
            "Replay dataset contains duplicate event IDs."
        )

    if not dataframe[
        "prediction_status"
    ].eq(
        "PASS"
    ).all():
        raise ValueError(
            "Replay dataset contains unsuccessful "
            "prediction rows."
        )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    numeric_columns = [
        "home_score",
        "away_score",
        "goal_difference",
        "total_goals",
        "lambda_home",
        "lambda_away",
        "pred_total_goals",
        "pred_goal_diff",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    numeric_values = dataframe[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "Replay dataset contains non-finite required "
            "numeric values."
        )

    if (
        dataframe["lambda_home"].le(0).any()
        or dataframe["lambda_away"].le(0).any()
    ):
        raise ValueError(
            "Replay dataset contains non-positive lambdas."
        )

    probabilities = dataframe[
        [
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ]
    ].to_numpy(
        dtype=float
    )

    if (
        (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise ValueError(
            "Replay dataset contains probabilities outside "
            "[0, 1]."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        np.ones(len(dataframe)),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError(
            "Replay probabilities do not sum to one."
        )

    expected_goal_difference = (
        dataframe["home_score"]
        - dataframe["away_score"]
    )

    if not np.allclose(
        dataframe["goal_difference"],
        expected_goal_difference,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError(
            "Observed goal differences are inconsistent."
        )

    expected_total_goals = (
        dataframe["home_score"]
        + dataframe["away_score"]
    )

    if not np.allclose(
        dataframe["total_goals"],
        expected_total_goals,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError(
            "Observed total goals are inconsistent."
        )

    return (
        dataframe
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )


def poisson_probability(
    goals: int,
    expected_goals: float,
) -> float:
    if goals < 0:
        raise ValueError(
            "goals must not be negative."
        )

    if expected_goals <= 0.0:
        raise ValueError(
            "expected_goals must be positive."
        )

    log_probability = (
        -expected_goals
        + goals * math.log(expected_goals)
        - math.lgamma(goals + 1)
    )

    return math.exp(
        log_probability
    )


def observed_result_index(
    home_score: int,
    away_score: int,
) -> int:
    if home_score > away_score:
        return 0

    if home_score == away_score:
        return 1

    return 2


def multiclass_brier_score(
    observed_indices: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        np.arange(len(observed_indices)),
        observed_indices,
    ] = 1.0

    return float(
        (
            (
                probabilities
                - one_hot
            )
            ** 2
        )
        .sum(axis=1)
        .mean()
    )


def correlation(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    if len(actual) < 2:
        return float("nan")

    if (
        np.unique(actual).size <= 1
        or np.unique(predicted).size <= 1
    ):
        return float("nan")

    return float(
        np.corrcoef(
            actual,
            predicted,
        )[0, 1]
    )


def add_fixture_errors(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["home_goal_error"] = (
        output["lambda_home"]
        - output["home_score"]
    )

    output["away_goal_error"] = (
        output["lambda_away"]
        - output["away_score"]
    )

    output["absolute_home_goal_error"] = (
        output["home_goal_error"].abs()
    )

    output["absolute_away_goal_error"] = (
        output["away_goal_error"].abs()
    )

    output["combined_absolute_goal_error"] = (
        output["absolute_home_goal_error"]
        + output["absolute_away_goal_error"]
    ) / 2.0

    output["total_goal_error"] = (
        output["pred_total_goals"]
        - output["total_goals"]
    )

    output["absolute_total_goal_error"] = (
        output["total_goal_error"].abs()
    )

    output["goal_difference_error"] = (
        output["pred_goal_diff"]
        - output["goal_difference"]
    )

    output["absolute_goal_difference_error"] = (
        output["goal_difference_error"].abs()
    )

    observed_indices = np.array(
        [
            observed_result_index(
                int(row.home_score),
                int(row.away_score),
            )
            for row in output.itertuples(
                index=False
            )
        ],
        dtype=int,
    )

    output["observed_result_index"] = (
        observed_indices
    )

    probability_matrix = output[
        [
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ]
    ].to_numpy(
        dtype=float
    )

    output["predicted_result_index"] = (
        probability_matrix.argmax(axis=1)
    )

    output["outcome_prediction_correct"] = (
        output["observed_result_index"]
        .eq(
            output["predicted_result_index"]
        )
    )

    exact_score_probabilities: list[float] = []

    for row in output.itertuples(
        index=False
    ):
        probability = (
            poisson_probability(
                int(row.home_score),
                float(row.lambda_home),
            )
            * poisson_probability(
                int(row.away_score),
                float(row.lambda_away),
            )
        )

        exact_score_probabilities.append(
            probability
        )

    output[
        "pred_exact_score_probability"
    ] = exact_score_probabilities

    output["exact_score_log_loss"] = (
        -np.log(
            np.clip(
                output[
                    "pred_exact_score_probability"
                ].to_numpy(dtype=float),
                PROBABILITY_FLOOR,
                1.0,
            )
        )
    )

    return output


def build_performance_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    actual_home = predictions[
        "home_score"
    ].to_numpy(dtype=float)

    actual_away = predictions[
        "away_score"
    ].to_numpy(dtype=float)

    predicted_home = predictions[
        "lambda_home"
    ].to_numpy(dtype=float)

    predicted_away = predictions[
        "lambda_away"
    ].to_numpy(dtype=float)

    actual_total = (
        actual_home
        + actual_away
    )

    predicted_total = (
        predicted_home
        + predicted_away
    )

    actual_goal_difference = (
        actual_home
        - actual_away
    )

    predicted_goal_difference = (
        predicted_home
        - predicted_away
    )

    home_poisson_deviance = (
        mean_poisson_deviance(
            actual_home,
            predicted_home,
        )
    )

    away_poisson_deviance = (
        mean_poisson_deviance(
            actual_away,
            predicted_away,
        )
    )

    observed_indices = predictions[
        "observed_result_index"
    ].to_numpy(
        dtype=int
    )

    probability_matrix = predictions[
        [
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ]
    ].to_numpy(
        dtype=float
    )

    probability_matrix = np.clip(
        probability_matrix,
        PROBABILITY_FLOOR,
        1.0,
    )

    probability_matrix = (
        probability_matrix
        / probability_matrix.sum(
            axis=1,
            keepdims=True,
        )
    )

    actual_draw_rate = float(
        (
            actual_home
            == actual_away
        ).mean()
    )

    predicted_draw_rate = float(
        predictions[
            "draw_probability"
        ].mean()
    )

    actual_home_win_rate = float(
        (
            actual_home
            > actual_away
        ).mean()
    )

    actual_away_win_rate = float(
        (
            actual_home
            < actual_away
        ).mean()
    )

    metrics = {
        "match_count": len(
            predictions
        ),

        "home_poisson_deviance": float(
            home_poisson_deviance
        ),
        "away_poisson_deviance": float(
            away_poisson_deviance
        ),
        "combined_poisson_deviance": float(
            (
                home_poisson_deviance
                + away_poisson_deviance
            )
            / 2.0
        ),

        "home_goal_mae": float(
            mean_absolute_error(
                actual_home,
                predicted_home,
            )
        ),
        "away_goal_mae": float(
            mean_absolute_error(
                actual_away,
                predicted_away,
            )
        ),
        "combined_goal_mae": float(
            (
                mean_absolute_error(
                    actual_home,
                    predicted_home,
                )
                + mean_absolute_error(
                    actual_away,
                    predicted_away,
                )
            )
            / 2.0
        ),

        "home_goal_rmse": float(
            mean_squared_error(
                actual_home,
                predicted_home,
            )
            ** 0.5
        ),
        "away_goal_rmse": float(
            mean_squared_error(
                actual_away,
                predicted_away,
            )
            ** 0.5
        ),

        "total_goal_mae": float(
            mean_absolute_error(
                actual_total,
                predicted_total,
            )
        ),
        "goal_difference_mae": float(
            mean_absolute_error(
                actual_goal_difference,
                predicted_goal_difference,
            )
        ),

        "home_goal_correlation": correlation(
            actual_home,
            predicted_home,
        ),
        "away_goal_correlation": correlation(
            actual_away,
            predicted_away,
        ),
        "total_goal_correlation": correlation(
            actual_total,
            predicted_total,
        ),
        "goal_difference_correlation": correlation(
            actual_goal_difference,
            predicted_goal_difference,
        ),

        "outcome_accuracy": float(
            predictions[
                "outcome_prediction_correct"
            ].mean()
        ),
        "outcome_log_loss": float(
            log_loss(
                observed_indices,
                probability_matrix,
                labels=[
                    0,
                    1,
                    2,
                ],
            )
        ),
        "outcome_brier_score": (
            multiclass_brier_score(
                observed_indices,
                probability_matrix,
            )
        ),
        "exact_score_log_loss": float(
            predictions[
                "exact_score_log_loss"
            ].mean()
        ),

        "actual_home_win_rate": (
            actual_home_win_rate
        ),
        "predicted_home_win_rate": float(
            predictions[
                "home_win_probability"
            ].mean()
        ),
        "actual_draw_rate": (
            actual_draw_rate
        ),
        "predicted_draw_rate": (
            predicted_draw_rate
        ),
        "draw_rate_error": float(
            predicted_draw_rate
            - actual_draw_rate
        ),
        "absolute_draw_rate_error": float(
            abs(
                predicted_draw_rate
                - actual_draw_rate
            )
        ),
        "actual_away_win_rate": (
            actual_away_win_rate
        ),
        "predicted_away_win_rate": float(
            predictions[
                "away_win_probability"
            ].mean()
        ),

        "mean_actual_home_goals": float(
            actual_home.mean()
        ),
        "mean_predicted_home_goals": float(
            predicted_home.mean()
        ),
        "home_goal_mean_bias": float(
            predicted_home.mean()
            - actual_home.mean()
        ),

        "mean_actual_away_goals": float(
            actual_away.mean()
        ),
        "mean_predicted_away_goals": float(
            predicted_away.mean()
        ),
        "away_goal_mean_bias": float(
            predicted_away.mean()
            - actual_away.mean()
        ),

        "mean_actual_total_goals": float(
            actual_total.mean()
        ),
        "mean_predicted_total_goals": float(
            predicted_total.mean()
        ),
        "total_goal_mean_bias": float(
            predicted_total.mean()
            - actual_total.mean()
        ),
    }

    return pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
            }
            for metric, value in metrics.items()
        ]
    )


def build_team_bias_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    home = predictions[
        [
            "home_team",
            "home_score",
            "away_score",
            "lambda_home",
            "lambda_away",
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_team": "club",
            "home_score":
                "actual_goals_for",
            "away_score":
                "actual_goals_against",
            "lambda_home":
                "predicted_goals_for",
            "lambda_away":
                "predicted_goals_against",
        }
    )

    away = predictions[
        [
            "away_team",
            "away_score",
            "home_score",
            "lambda_away",
            "lambda_home",
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_team": "club",
            "away_score":
                "actual_goals_for",
            "home_score":
                "actual_goals_against",
            "lambda_away":
                "predicted_goals_for",
            "lambda_home":
                "predicted_goals_against",
        }
    )

    combined = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    combined["goals_for_error"] = (
        combined["predicted_goals_for"]
        - combined["actual_goals_for"]
    )

    combined["goals_against_error"] = (
        combined["predicted_goals_against"]
        - combined["actual_goals_against"]
    )

    combined["actual_goal_difference"] = (
        combined["actual_goals_for"]
        - combined["actual_goals_against"]
    )

    combined["predicted_goal_difference"] = (
        combined["predicted_goals_for"]
        - combined["predicted_goals_against"]
    )

    combined["goal_difference_error"] = (
        combined["predicted_goal_difference"]
        - combined["actual_goal_difference"]
    )

    summary = (
        combined
        .groupby(
            "club",
            as_index=False,
        )
        .agg(
            matches=(
                "club",
                "size",
            ),
            actual_goals_for=(
                "actual_goals_for",
                "sum",
            ),
            predicted_goals_for=(
                "predicted_goals_for",
                "sum",
            ),
            goals_for_bias=(
                "goals_for_error",
                "sum",
            ),
            mean_goals_for_bias=(
                "goals_for_error",
                "mean",
            ),
            actual_goals_against=(
                "actual_goals_against",
                "sum",
            ),
            predicted_goals_against=(
                "predicted_goals_against",
                "sum",
            ),
            goals_against_bias=(
                "goals_against_error",
                "sum",
            ),
            mean_goals_against_bias=(
                "goals_against_error",
                "mean",
            ),
            actual_goal_difference=(
                "actual_goal_difference",
                "sum",
            ),
            predicted_goal_difference=(
                "predicted_goal_difference",
                "sum",
            ),
            goal_difference_bias=(
                "goal_difference_error",
                "sum",
            ),
            mean_goal_difference_bias=(
                "goal_difference_error",
                "mean",
            ),
        )
        .sort_values(
            "mean_goal_difference_bias",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return summary


def build_calibration_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        pd.DataFrame
    ] = []

    outcomes = {
        "home_win": (
            "home_win_probability",
            0,
        ),
        "draw": (
            "draw_probability",
            1,
        ),
        "away_win": (
            "away_win_probability",
            2,
        ),
    }

    bin_edges = np.linspace(
        0.0,
        1.0,
        CALIBRATION_BIN_COUNT + 1,
    )

    for outcome_name, (
        probability_column,
        observed_index,
    ) in outcomes.items():
        working = predictions[
            [
                probability_column,
                "observed_result_index",
            ]
        ].copy()

        working["bin"] = pd.cut(
            working[
                probability_column
            ],
            bins=bin_edges,
            include_lowest=True,
            right=True,
            duplicates="drop",
        )

        working["observed"] = (
            working[
                "observed_result_index"
            ].eq(
                observed_index
            )
        )

        grouped = (
            working
            .groupby(
                "bin",
                observed=False,
            )
            .agg(
                prediction_count=(
                    probability_column,
                    "size",
                ),
                mean_predicted_probability=(
                    probability_column,
                    "mean",
                ),
                observed_frequency=(
                    "observed",
                    "mean",
                ),
            )
            .reset_index()
        )

        grouped = grouped[
            grouped[
                "prediction_count"
            ].gt(0)
        ].copy()

        grouped.insert(
            0,
            "outcome",
            outcome_name,
        )

        grouped["bin"] = (
            grouped["bin"]
            .astype(str)
        )

        grouped[
            "calibration_error"
        ] = (
            grouped[
                "mean_predicted_probability"
            ]
            - grouped[
                "observed_frequency"
            ]
        )

        grouped[
            "absolute_calibration_error"
        ] = grouped[
            "calibration_error"
        ].abs()

        records.append(
            grouped
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


def build_extreme_errors(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    categories = {
        "largest_combined_goal_error": (
            predictions.nlargest(
                EXTREME_ERROR_COUNT,
                "combined_absolute_goal_error",
            )
        ),
        "largest_total_goal_error": (
            predictions.nlargest(
                EXTREME_ERROR_COUNT,
                "absolute_total_goal_error",
            )
        ),
        "largest_goal_difference_error": (
            predictions.nlargest(
                EXTREME_ERROR_COUNT,
                "absolute_goal_difference_error",
            )
        ),
        "lowest_exact_score_probability": (
            predictions.nsmallest(
                EXTREME_ERROR_COUNT,
                "pred_exact_score_probability",
            )
        ),
    }

    selected_columns = [
        "event_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "lambda_home",
        "lambda_away",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "combined_absolute_goal_error",
        "absolute_total_goal_error",
        "absolute_goal_difference_error",
        "pred_exact_score_probability",
    ]

    outputs: list[pd.DataFrame] = []

    for category, dataframe in (
        categories.items()
    ):
        selected = dataframe[
            selected_columns
        ].copy()

        selected.insert(
            0,
            "category",
            category,
        )

        selected.insert(
            1,
            "category_rank",
            range(
                1,
                len(selected) + 1,
            ),
        )

        outputs.append(
            selected
        )

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def build_metadata(
    *,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, object]:
    summary_lookup = {
        row.metric: row.value
        for row in summary.itertuples(
            index=False
        )
    }

    return {
        "study_id": "084",
        "study_name": (
            "Bundesliga Replay Performance Analysis"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "study_type": (
            "overlapping_period_diagnostic_evaluation"
        ),
        "input_dataset": str(
            INPUT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "match_count": len(
            predictions
        ),
        "goal_model_artifact_name": (
            str(
                predictions[
                    "goal_model_artifact_name"
                ].iloc[0]
            )
        ),
        "goal_model_artifact_version": (
            str(
                predictions[
                    "goal_model_artifact_version"
                ].iloc[0]
            )
        ),
        "goal_model_training_end_date": (
            str(
                predictions[
                    "goal_model_training_end_date"
                ].iloc[0]
            )
        ),
        "evaluation_period_start": (
            predictions[
                "date"
            ].min().date().isoformat()
        ),
        "evaluation_period_end": (
            predictions[
                "date"
            ].max().date().isoformat()
        ),
        "interpretation_boundary": (
            "The goal-model training cutoff overlaps the "
            "evaluation season. Results are diagnostic and "
            "must not be interpreted as an unbiased estimate "
            "of prospective Bundesliga performance."
        ),
        "combined_poisson_deviance": (
            summary_lookup[
                "combined_poisson_deviance"
            ]
        ),
        "combined_goal_mae": (
            summary_lookup[
                "combined_goal_mae"
            ]
        ),
        "outcome_log_loss": (
            summary_lookup[
                "outcome_log_loss"
            ]
        ),
        "outcome_brier_score": (
            summary_lookup[
                "outcome_brier_score"
            ]
        ),
        "outputs": [
            PERFORMANCE_SUMMARY_PATH.name,
            FIXTURE_ERRORS_PATH.name,
            TEAM_BIAS_PATH.name,
            CALIBRATION_PATH.name,
            EXTREME_ERRORS_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    summary: pd.DataFrame,
    team_bias: pd.DataFrame,
) -> None:
    values = {
        row.metric: row.value
        for row in summary.itertuples(
            index=False
        )
    }

    largest_positive_bias = (
        team_bias.iloc[0]
    )

    largest_negative_bias = (
        team_bias.iloc[-1]
    )

    report = f"""# Study 084 — Bundesliga Replay Performance Analysis

## Purpose

Evaluate the goal and outcome predictions produced during
Study 083 across the complete Bundesliga 2024–25 replay
population.

## Interpretation boundary

The production goal-model artifact was trained through
2025-05-25, while the replay period covers the 2024–25
Bundesliga season.

Therefore, this analysis is an overlapping-period diagnostic.
It is not an unbiased estimate of prospective Bundesliga
generalization.

## Evaluation population

- Matches: {int(values["match_count"])}
- Competition: Bundesliga
- Season: 2024–25

## Goal prediction performance

- Combined Poisson deviance:
  {values["combined_poisson_deviance"]:.6f}
- Combined goal MAE:
  {values["combined_goal_mae"]:.6f}
- Home goal RMSE:
  {values["home_goal_rmse"]:.6f}
- Away goal RMSE:
  {values["away_goal_rmse"]:.6f}
- Total-goal MAE:
  {values["total_goal_mae"]:.6f}
- Goal-difference MAE:
  {values["goal_difference_mae"]:.6f}

## Outcome performance

- Outcome accuracy:
  {values["outcome_accuracy"]:.6f}
- Outcome log loss:
  {values["outcome_log_loss"]:.6f}
- Multiclass Brier score:
  {values["outcome_brier_score"]:.6f}
- Exact-score log loss:
  {values["exact_score_log_loss"]:.6f}

## Draw-rate behavior

- Actual draw rate:
  {values["actual_draw_rate"]:.6f}
- Predicted draw rate:
  {values["predicted_draw_rate"]:.6f}
- Draw-rate error:
  {values["draw_rate_error"]:.6f}

## Goal-volume behavior

- Mean actual home goals:
  {values["mean_actual_home_goals"]:.6f}
- Mean predicted home goals:
  {values["mean_predicted_home_goals"]:.6f}
- Mean actual away goals:
  {values["mean_actual_away_goals"]:.6f}
- Mean predicted away goals:
  {values["mean_predicted_away_goals"]:.6f}
- Mean actual total goals:
  {values["mean_actual_total_goals"]:.6f}
- Mean predicted total goals:
  {values["mean_predicted_total_goals"]:.6f}

## Largest team-level goal-difference biases

- Most positive:
  `{largest_positive_bias["club"]}`
  ({largest_positive_bias["mean_goal_difference_bias"]:.6f}
  per match)
- Most negative:
  `{largest_negative_bias["club"]}`
  ({largest_negative_bias["mean_goal_difference_bias"]:.6f}
  per match)

## Result

**OVERALL RESULT: PASS**

The performance artifacts were generated successfully.
Interpretation must remain within the stated overlapping-period
diagnostic boundary.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 084 — BUNDESLIGA REPLAY "
        "PERFORMANCE ANALYSIS"
    )
    print("=" * 88)

    replay = load_replay_predictions()

    fixture_errors = add_fixture_errors(
        replay
    )

    performance_summary = (
        build_performance_summary(
            fixture_errors
        )
    )

    team_bias = build_team_bias_summary(
        fixture_errors
    )

    calibration = build_calibration_table(
        fixture_errors
    )

    extreme_errors = build_extreme_errors(
        fixture_errors
    )

    metadata = build_metadata(
        predictions=fixture_errors,
        summary=performance_summary,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance_summary.to_csv(
        PERFORMANCE_SUMMARY_PATH,
        index=False,
    )

    fixture_errors.to_csv(
        FIXTURE_ERRORS_PATH,
        index=False,
    )

    team_bias.to_csv(
        TEAM_BIAS_PATH,
        index=False,
    )

    calibration.to_csv(
        CALIBRATION_PATH,
        index=False,
    )

    extreme_errors.to_csv(
        EXTREME_ERRORS_PATH,
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
        summary=performance_summary,
        team_bias=team_bias,
    )

    values = {
        row.metric: row.value
        for row in (
            performance_summary
            .itertuples(
                index=False
            )
        )
    }

    print()
    print("Evaluation population")
    print(
        f"  Matches: {int(values['match_count'])}"
    )
    print(
        "  Evaluation mode: overlapping-period diagnostic"
    )

    print()
    print("Goal prediction performance")
    print("-" * 88)
    print(
        "  Combined Poisson deviance: "
        f"{values['combined_poisson_deviance']:.6f}"
    )
    print(
        "  Combined goal MAE: "
        f"{values['combined_goal_mae']:.6f}"
    )
    print(
        "  Home goal RMSE: "
        f"{values['home_goal_rmse']:.6f}"
    )
    print(
        "  Away goal RMSE: "
        f"{values['away_goal_rmse']:.6f}"
    )
    print(
        "  Total-goal MAE: "
        f"{values['total_goal_mae']:.6f}"
    )
    print(
        "  Goal-difference MAE: "
        f"{values['goal_difference_mae']:.6f}"
    )

    print()
    print("Outcome performance")
    print("-" * 88)
    print(
        "  Outcome accuracy: "
        f"{values['outcome_accuracy']:.6f}"
    )
    print(
        "  Outcome log loss: "
        f"{values['outcome_log_loss']:.6f}"
    )
    print(
        "  Outcome Brier score: "
        f"{values['outcome_brier_score']:.6f}"
    )
    print(
        "  Exact-score log loss: "
        f"{values['exact_score_log_loss']:.6f}"
    )

    print()
    print("Draw-rate behavior")
    print("-" * 88)
    print(
        "  Actual draw rate: "
        f"{values['actual_draw_rate']:.6f}"
    )
    print(
        "  Predicted draw rate: "
        f"{values['predicted_draw_rate']:.6f}"
    )
    print(
        "  Draw-rate error: "
        f"{values['draw_rate_error']:.6f}"
    )

    print()
    print("Largest team biases")
    print("-" * 88)
    print(
        team_bias[
            [
                "club",
                "mean_goals_for_bias",
                "mean_goals_against_bias",
                "mean_goal_difference_bias",
            ]
        ]
        .head(5)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Validation summary")
    print("  Replay artifact loading: PASS")
    print("  Population preservation: PASS")
    print("  Goal metrics: PASS")
    print("  Outcome metrics: PASS")
    print("  Exact-score metric: PASS")
    print("  Calibration output: PASS")
    print("  Team-bias output: PASS")
    print("  Extreme-error output: PASS")
    print(
        "  Interpretation boundary recorded: PASS"
    )

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
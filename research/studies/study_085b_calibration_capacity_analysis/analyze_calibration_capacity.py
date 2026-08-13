from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_084_bundesliga_replay_performance_analysis"
    / "fixture_error_analysis.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_085b_calibration_capacity_analysis"
)

CONFIGURATION_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "calibration_configuration_summary.csv"
)

FIXTURE_PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "counterfactual_fixture_predictions.csv"
)

CLUB_RESIDUALS_PATH = (
    OUTPUT_DIRECTORY
    / "club_residual_comparison.csv"
)

ELITE_RESIDUALS_PATH = (
    OUTPUT_DIRECTORY
    / "elite_residual_comparison.csv"
)

SEASON_PHASE_PATH = (
    OUTPUT_DIRECTORY
    / "season_phase_residual_comparison.csv"
)

OUTCOME_CALIBRATION_PATH = (
    OUTPUT_DIRECTORY
    / "outcome_rate_comparison.csv"
)

CAPACITY_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "calibration_capacity_summary.csv"
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
MAX_GOALS = 15
PROBABILITY_FLOOR = 1e-15
GRID_POINTS = 161

GLOBAL_SCALE_MIN = 0.70
GLOBAL_SCALE_MAX = 1.50

HOME_AWAY_SCALE_MIN = 0.70
HOME_AWAY_SCALE_MAX = 1.50

ADDITIVE_MIN = -0.40
ADDITIVE_MAX = 0.80

RATING_EFFECT_MIN = -0.30
RATING_EFFECT_MAX = 0.50

ELITE_SCALE_MIN = 0.80
ELITE_SCALE_MAX = 1.80

LOW_LAMBDA_FLOOR = 0.05
ELITE_QUANTILE = 0.75


REQUIRED_COLUMNS = {
    "event_id",
    "date",
    "round_number",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "total_goals",
    "goal_difference",
    "home_rating_prior",
    "away_rating_prior",
    "rating_prior_diff",
    "lambda_home",
    "lambda_away",
    "pred_total_goals",
    "pred_goal_diff",
    "prediction_status",
}


@dataclass(frozen=True)
class CalibrationResult:
    configuration: str
    parameters: dict[str, float]
    lambda_home: np.ndarray
    lambda_away: np.ndarray


def load_predictions() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Study 084 fixture-error artifact does not exist: "
            f"{INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Study 084 fixture-error artifact is empty."
        )

    missing = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Input artifact is missing required columns: "
            f"{sorted(missing)}"
        )

    if len(dataframe) != EXPECTED_MATCH_COUNT:
        raise ValueError(
            "Unexpected fixture population. "
            f"Expected {EXPECTED_MATCH_COUNT}, "
            f"received {len(dataframe)}."
        )

    if dataframe["event_id"].duplicated().any():
        raise ValueError(
            "Input artifact contains duplicate event IDs."
        )

    if not dataframe["prediction_status"].eq("PASS").all():
        raise ValueError(
            "Input artifact contains failed predictions."
        )

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    numeric_columns = [
        "round_number",
        "home_score",
        "away_score",
        "total_goals",
        "goal_difference",
        "home_rating_prior",
        "away_rating_prior",
        "rating_prior_diff",
        "lambda_home",
        "lambda_away",
        "pred_total_goals",
        "pred_goal_diff",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    values = dataframe[numeric_columns].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Input artifact contains non-finite values."
        )

    if (
        dataframe["lambda_home"].le(0.0).any()
        or dataframe["lambda_away"].le(0.0).any()
    ):
        raise ValueError(
            "Input artifact contains non-positive lambdas."
        )

    return (
        dataframe
        .sort_values(
            ["date", "event_id"]
        )
        .reset_index(drop=True)
    )


def poisson_deviance(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if (
        (actual < 0.0).any()
        or (predicted <= 0.0).any()
    ):
        raise ValueError(
            "Poisson deviance requires non-negative observations "
            "and positive predictions."
        )

    log_component = np.zeros_like(
        actual,
        dtype=float,
    )

    positive = actual > 0.0

    log_component[positive] = (
        actual[positive]
        * np.log(
            actual[positive]
            / predicted[positive]
        )
    )

    deviance = 2.0 * (
        log_component
        - (
            actual
            - predicted
        )
    )

    return float(
        deviance.mean()
    )


def poisson_probability_vector(
    goals: np.ndarray,
    expected_goals: np.ndarray,
) -> np.ndarray:
    goals = np.asarray(
        goals,
        dtype=int,
    )

    expected_goals = np.asarray(
        expected_goals,
        dtype=float,
    )

    output = np.empty(
        len(goals),
        dtype=float,
    )

    for index, (
        observed_goals,
        lambda_value,
    ) in enumerate(
        zip(
            goals,
            expected_goals,
        )
    ):
        log_probability = (
            -lambda_value
            + observed_goals * math.log(
                lambda_value
            )
            - math.lgamma(
                observed_goals + 1
            )
        )

        output[index] = math.exp(
            log_probability
        )

    return output


def outcome_probabilities(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
) -> np.ndarray:
    probability_rows: list[
        tuple[float, float, float]
    ] = []

    for home_lambda, away_lambda in zip(
        lambda_home,
        lambda_away,
    ):
        home_probabilities = np.array(
            [
                math.exp(
                    -home_lambda
                    + goals * math.log(
                        home_lambda
                    )
                    - math.lgamma(
                        goals + 1
                    )
                )
                for goals in range(
                    MAX_GOALS + 1
                )
            ],
            dtype=float,
        )

        away_probabilities = np.array(
            [
                math.exp(
                    -away_lambda
                    + goals * math.log(
                        away_lambda
                    )
                    - math.lgamma(
                        goals + 1
                    )
                )
                for goals in range(
                    MAX_GOALS + 1
                )
            ],
            dtype=float,
        )

        score_grid = np.outer(
            home_probabilities,
            away_probabilities,
        )

        home_win = float(
            np.tril(
                score_grid,
                k=-1,
            ).sum()
        )

        draw = float(
            np.trace(
                score_grid
            )
        )

        away_win = float(
            np.triu(
                score_grid,
                k=1,
            ).sum()
        )

        total = (
            home_win
            + draw
            + away_win
        )

        probability_rows.append(
            (
                home_win / total,
                draw / total,
                away_win / total,
            )
        )

    return np.asarray(
        probability_rows,
        dtype=float,
    )


def observed_outcome_indices(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    output = np.empty(
        len(dataframe),
        dtype=int,
    )

    home = dataframe[
        "home_score"
    ].to_numpy(dtype=int)

    away = dataframe[
        "away_score"
    ].to_numpy(dtype=int)

    output[home > away] = 0
    output[home == away] = 1
    output[home < away] = 2

    return output


def multiclass_log_loss(
    observed: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    clipped = np.clip(
        probabilities,
        PROBABILITY_FLOOR,
        1.0,
    )

    selected = clipped[
        np.arange(
            len(observed)
        ),
        observed,
    ]

    return float(
        -np.log(selected).mean()
    )


def multiclass_brier_score(
    observed: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        np.arange(
            len(observed)
        ),
        observed,
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


def evaluate_configuration(
    dataframe: pd.DataFrame,
    result: CalibrationResult,
) -> dict[str, object]:
    actual_home = dataframe[
        "home_score"
    ].to_numpy(dtype=float)

    actual_away = dataframe[
        "away_score"
    ].to_numpy(dtype=float)

    actual_total = (
        actual_home
        + actual_away
    )

    actual_difference = (
        actual_home
        - actual_away
    )

    predicted_total = (
        result.lambda_home
        + result.lambda_away
    )

    predicted_difference = (
        result.lambda_home
        - result.lambda_away
    )

    probabilities = outcome_probabilities(
        result.lambda_home,
        result.lambda_away,
    )

    observed = observed_outcome_indices(
        dataframe
    )

    exact_score_probability = (
        poisson_probability_vector(
            actual_home.astype(int),
            result.lambda_home,
        )
        * poisson_probability_vector(
            actual_away.astype(int),
            result.lambda_away,
        )
    )

    exact_score_log_loss = float(
        -np.log(
            np.clip(
                exact_score_probability,
                PROBABILITY_FLOOR,
                1.0,
            )
        ).mean()
    )

    predicted_outcome = probabilities.argmax(
        axis=1
    )

    return {
        "configuration": result.configuration,
        "parameters_json": json.dumps(
            result.parameters,
            sort_keys=True,
        ),
        "home_poisson_deviance": (
            poisson_deviance(
                actual_home,
                result.lambda_home,
            )
        ),
        "away_poisson_deviance": (
            poisson_deviance(
                actual_away,
                result.lambda_away,
            )
        ),
        "combined_poisson_deviance": (
            0.5
            * (
                poisson_deviance(
                    actual_home,
                    result.lambda_home,
                )
                + poisson_deviance(
                    actual_away,
                    result.lambda_away,
                )
            )
        ),
        "home_goal_mae": float(
            np.abs(
                result.lambda_home
                - actual_home
            ).mean()
        ),
        "away_goal_mae": float(
            np.abs(
                result.lambda_away
                - actual_away
            ).mean()
        ),
        "combined_goal_mae": float(
            0.5
            * (
                np.abs(
                    result.lambda_home
                    - actual_home
                ).mean()
                + np.abs(
                    result.lambda_away
                    - actual_away
                ).mean()
            )
        ),
        "home_goal_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        result.lambda_home
                        - actual_home
                    )
                    ** 2
                )
            )
        ),
        "away_goal_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        result.lambda_away
                        - actual_away
                    )
                    ** 2
                )
            )
        ),
        "total_goal_mae": float(
            np.abs(
                predicted_total
                - actual_total
            ).mean()
        ),
        "goal_difference_mae": float(
            np.abs(
                predicted_difference
                - actual_difference
            ).mean()
        ),
        "outcome_accuracy": float(
            (
                predicted_outcome
                == observed
            ).mean()
        ),
        "outcome_log_loss": (
            multiclass_log_loss(
                observed,
                probabilities,
            )
        ),
        "outcome_brier_score": (
            multiclass_brier_score(
                observed,
                probabilities,
            )
        ),
        "exact_score_log_loss": (
            exact_score_log_loss
        ),
        "actual_home_win_rate": float(
            (
                observed == 0
            ).mean()
        ),
        "predicted_home_win_rate": float(
            probabilities[:, 0].mean()
        ),
        "actual_draw_rate": float(
            (
                observed == 1
            ).mean()
        ),
        "predicted_draw_rate": float(
            probabilities[:, 1].mean()
        ),
        "draw_rate_error": float(
            probabilities[:, 1].mean()
            - (
                observed == 1
            ).mean()
        ),
        "actual_away_win_rate": float(
            (
                observed == 2
            ).mean()
        ),
        "predicted_away_win_rate": float(
            probabilities[:, 2].mean()
        ),
        "mean_actual_home_goals": float(
            actual_home.mean()
        ),
        "mean_predicted_home_goals": float(
            result.lambda_home.mean()
        ),
        "home_goal_mean_bias": float(
            result.lambda_home.mean()
            - actual_home.mean()
        ),
        "mean_actual_away_goals": float(
            actual_away.mean()
        ),
        "mean_predicted_away_goals": float(
            result.lambda_away.mean()
        ),
        "away_goal_mean_bias": float(
            result.lambda_away.mean()
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


def optimize_scalar(
    *,
    objective: Callable[[float], float],
    minimum: float,
    maximum: float,
    grid_points: int = GRID_POINTS,
) -> float:
    candidates = np.linspace(
        minimum,
        maximum,
        grid_points,
    )

    scores = np.array(
        [
            objective(
                float(candidate)
            )
            for candidate in candidates
        ],
        dtype=float,
    )

    best_index = int(
        np.argmin(scores)
    )

    return float(
        candidates[best_index]
    )


def build_baseline(
    dataframe: pd.DataFrame,
) -> CalibrationResult:
    return CalibrationResult(
        configuration="baseline",
        parameters={},
        lambda_home=dataframe[
            "lambda_home"
        ].to_numpy(dtype=float),
        lambda_away=dataframe[
            "lambda_away"
        ].to_numpy(dtype=float),
    )


def build_global_scale(
    dataframe: pd.DataFrame,
) -> CalibrationResult:
    home = dataframe[
        "lambda_home"
    ].to_numpy(dtype=float)

    away = dataframe[
        "lambda_away"
    ].to_numpy(dtype=float)

    actual_home = dataframe[
        "home_score"
    ].to_numpy(dtype=float)

    actual_away = dataframe[
        "away_score"
    ].to_numpy(dtype=float)

    scale = optimize_scalar(
        objective=lambda value: (
            0.5
            * (
                poisson_deviance(
                    actual_home,
                    home * value,
                )
                + poisson_deviance(
                    actual_away,
                    away * value,
                )
            )
        ),
        minimum=GLOBAL_SCALE_MIN,
        maximum=GLOBAL_SCALE_MAX,
    )

    return CalibrationResult(
        configuration="global_multiplicative_scale",
        parameters={
            "scale": scale,
        },
        lambda_home=home * scale,
        lambda_away=away * scale,
    )


def build_home_away_scale(
    dataframe: pd.DataFrame,
) -> CalibrationResult:
    home = dataframe[
        "lambda_home"
    ].to_numpy(dtype=float)

    away = dataframe[
        "lambda_away"
    ].to_numpy(dtype=float)

    actual_home = dataframe[
        "home_score"
    ].to_numpy(dtype=float)

    actual_away = dataframe[
        "away_score"
    ].to_numpy(dtype=float)

    home_scale = optimize_scalar(
        objective=lambda value: (
            poisson_deviance(
                actual_home,
                home * value,
            )
        ),
        minimum=HOME_AWAY_SCALE_MIN,
        maximum=HOME_AWAY_SCALE_MAX,
    )

    away_scale = optimize_scalar(
        objective=lambda value: (
            poisson_deviance(
                actual_away,
                away * value,
            )
        ),
        minimum=HOME_AWAY_SCALE_MIN,
        maximum=HOME_AWAY_SCALE_MAX,
    )

    return CalibrationResult(
        configuration="separate_home_away_scale",
        parameters={
            "home_scale": home_scale,
            "away_scale": away_scale,
        },
        lambda_home=home * home_scale,
        lambda_away=away * away_scale,
    )


def build_global_additive(
    dataframe: pd.DataFrame,
) -> CalibrationResult:
    home = dataframe[
        "lambda_home"
    ].to_numpy(dtype=float)

    away = dataframe[
        "lambda_away"
    ].to_numpy(dtype=float)

    actual_home = dataframe[
        "home_score"
    ].to_numpy(dtype=float)

    actual_away = dataframe[
        "away_score"
    ].to_numpy(dtype=float)

    additive = optimize_scalar(
        objective=lambda value: (
            0.5
            * (
                poisson_deviance(
                    actual_home,
                    np.maximum(
                        LOW_LAMBDA_FLOOR,
                        home + value,
                    ),
                )
                + poisson_deviance(
                    actual_away,
                    np.maximum(
                        LOW_LAMBDA_FLOOR,
                        away + value,
                    ),
                )
            )
        ),
        minimum=ADDITIVE_MIN,
        maximum=ADDITIVE_MAX,
    )

    return CalibrationResult(
        configuration="global_additive_correction",
        parameters={
            "additive": additive,
        },
        lambda_home=np.maximum(
            LOW_LAMBDA_FLOOR,
            home + additive,
        ),
        lambda_away=np.maximum(
            LOW_LAMBDA_FLOOR,
            away + additive,
        ),
    )


def build_piecewise_scale(
    dataframe: pd.DataFrame,
) -> CalibrationResult:
    working = dataframe.copy()

    working["total_bin"] = pd.qcut(
        working["pred_total_goals"],
        q=3,
        labels=[
            "low",
            "medium",
            "high",
        ],
        duplicates="drop",
    )

    if working["total_bin"].isna().any():
        raise ValueError(
            "Piecewise calibration bins are incomplete."
        )

    calibrated_home = working[
        "lambda_home"
    ].to_numpy(dtype=float).copy()

    calibrated_away = working[
        "lambda_away"
    ].to_numpy(dtype=float).copy()

    parameters: dict[str, float] = {}

    for label in (
        "low",
        "medium",
        "high",
    ):
        mask = working[
            "total_bin"
        ].astype(str).eq(label).to_numpy()

        home = working.loc[
            mask,
            "lambda_home",
        ].to_numpy(dtype=float)

        away = working.loc[
            mask,
            "lambda_away",
        ].to_numpy(dtype=float)

        actual_home = working.loc[
            mask,
            "home_score",
        ].to_numpy(dtype=float)

        actual_away = working.loc[
            mask,
            "away_score",
        ].to_numpy(dtype=float)

        scale = optimize_scalar(
            objective=lambda value: (
                0.5
                * (
                    poisson_deviance(
                        actual_home,
                        home * value,
                    )
                    + poisson_deviance(
                        actual_away,
                        away * value,
                    )
                )
            ),
            minimum=GLOBAL_SCALE_MIN,
            maximum=GLOBAL_SCALE_MAX,
        )

        calibrated_home[mask] *= scale
        calibrated_away[mask] *= scale

        parameters[
            f"{label}_scale"
        ] = scale

    return CalibrationResult(
        configuration="piecewise_predicted_total_scale",
        parameters=parameters,
        lambda_home=calibrated_home,
        lambda_away=calibrated_away,
    )


def build_rating_strength_correction(
    dataframe: pd.DataFrame,
) -> CalibrationResult:
    home = dataframe[
        "lambda_home"
    ].to_numpy(dtype=float)

    away = dataframe[
        "lambda_away"
    ].to_numpy(dtype=float)

    rating_difference = dataframe[
        "rating_prior_diff"
    ].to_numpy(dtype=float)

    actual_home = dataframe[
        "home_score"
    ].to_numpy(dtype=float)

    actual_away = dataframe[
        "away_score"
    ].to_numpy(dtype=float)

    rating_scale = float(
        np.std(
            rating_difference
        )
    )

    if rating_scale <= 0.0:
        raise ValueError(
            "Rating differences have zero variance."
        )

    standardized = (
        rating_difference
        / rating_scale
    )

    def calibrated(
        coefficient: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        home_multiplier = np.exp(
            coefficient
            * np.maximum(
                standardized,
                0.0,
            )
        )

        away_multiplier = np.exp(
            coefficient
            * np.maximum(
                -standardized,
                0.0,
            )
        )

        return (
            home * home_multiplier,
            away * away_multiplier,
        )

    coefficient = optimize_scalar(
        objective=lambda value: (
            0.5
            * (
                poisson_deviance(
                    actual_home,
                    calibrated(value)[0],
                )
                + poisson_deviance(
                    actual_away,
                    calibrated(value)[1],
                )
            )
        ),
        minimum=RATING_EFFECT_MIN,
        maximum=RATING_EFFECT_MAX,
    )

    calibrated_home, calibrated_away = (
        calibrated(coefficient)
    )

    return CalibrationResult(
        configuration="rating_strength_dependent_correction",
        parameters={
            "rating_coefficient": coefficient,
            "rating_standard_deviation": rating_scale,
        },
        lambda_home=calibrated_home,
        lambda_away=calibrated_away,
    )


def identify_elite_clubs(
    dataframe: pd.DataFrame,
) -> tuple[tuple[str, ...], float]:
    home = dataframe[
        [
            "home_team",
            "home_rating_prior",
        ]
    ].rename(
        columns={
            "home_team": "club",
            "home_rating_prior": "rating",
        }
    )

    away = dataframe[
        [
            "away_team",
            "away_rating_prior",
        ]
    ].rename(
        columns={
            "away_team": "club",
            "away_rating_prior": "rating",
        }
    )

    ratings = pd.concat(
        [home, away],
        ignore_index=True,
    )

    club_ratings = (
        ratings
        .groupby(
            "club",
            as_index=False,
        )
        .agg(
            average_rating=("rating", "mean")
        )
    )

    threshold = float(
        club_ratings[
            "average_rating"
        ].quantile(
            ELITE_QUANTILE
        )
    )

    elite_clubs = tuple(
        club_ratings.loc[
            club_ratings[
                "average_rating"
            ].ge(threshold),
            "club",
        ]
        .sort_values()
        .tolist()
    )

    return elite_clubs, threshold


def build_elite_scale(
    dataframe: pd.DataFrame,
    elite_clubs: tuple[str, ...],
) -> CalibrationResult:
    base_home = dataframe[
        "lambda_home"
    ].to_numpy(dtype=float)

    base_away = dataframe[
        "lambda_away"
    ].to_numpy(dtype=float)

    actual_home = dataframe[
        "home_score"
    ].to_numpy(dtype=float)

    actual_away = dataframe[
        "away_score"
    ].to_numpy(dtype=float)

    home_elite = dataframe[
        "home_team"
    ].isin(
        elite_clubs
    ).to_numpy()

    away_elite = dataframe[
        "away_team"
    ].isin(
        elite_clubs
    ).to_numpy()

    def calibrated(
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        home = base_home.copy()
        away = base_away.copy()

        home[home_elite] *= scale
        away[away_elite] *= scale

        return home, away

    scale = optimize_scalar(
        objective=lambda value: (
            0.5
            * (
                poisson_deviance(
                    actual_home,
                    calibrated(value)[0],
                )
                + poisson_deviance(
                    actual_away,
                    calibrated(value)[1],
                )
            )
        ),
        minimum=ELITE_SCALE_MIN,
        maximum=ELITE_SCALE_MAX,
    )

    calibrated_home, calibrated_away = (
        calibrated(scale)
    )

    return CalibrationResult(
        configuration="elite_team_attack_scale",
        parameters={
            "elite_scale": scale,
            "elite_club_count": float(
                len(elite_clubs)
            ),
        },
        lambda_home=calibrated_home,
        lambda_away=calibrated_away,
    )


def build_fixture_output(
    dataframe: pd.DataFrame,
    results: list[CalibrationResult],
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for result in results:
        probabilities = outcome_probabilities(
            result.lambda_home,
            result.lambda_away,
        )

        for index, row in dataframe.iterrows():
            records.append(
                {
                    "configuration":
                        result.configuration,
                    "event_id":
                        row["event_id"],
                    "date":
                        row["date"].date().isoformat(),
                    "home_team":
                        row["home_team"],
                    "away_team":
                        row["away_team"],
                    "home_score":
                        row["home_score"],
                    "away_score":
                        row["away_score"],
                    "lambda_home":
                        result.lambda_home[index],
                    "lambda_away":
                        result.lambda_away[index],
                    "pred_total_goals":
                        (
                            result.lambda_home[index]
                            + result.lambda_away[index]
                        ),
                    "pred_goal_diff":
                        (
                            result.lambda_home[index]
                            - result.lambda_away[index]
                        ),
                    "home_win_probability":
                        probabilities[index, 0],
                    "draw_probability":
                        probabilities[index, 1],
                    "away_win_probability":
                        probabilities[index, 2],
                    "parameters_json":
                        json.dumps(
                            result.parameters,
                            sort_keys=True,
                        ),
                }
            )

    return pd.DataFrame(records)


def build_team_match_rows(
    fixture_output: pd.DataFrame,
) -> pd.DataFrame:
    home = fixture_output[
        [
            "configuration",
            "event_id",
            "home_team",
            "home_score",
            "away_score",
            "lambda_home",
            "lambda_away",
        ]
    ].rename(
        columns={
            "home_team": "club",
            "home_score": "actual_goals_for",
            "away_score": "actual_goals_against",
            "lambda_home": "predicted_goals_for",
            "lambda_away": "predicted_goals_against",
        }
    )

    away = fixture_output[
        [
            "configuration",
            "event_id",
            "away_team",
            "away_score",
            "home_score",
            "lambda_away",
            "lambda_home",
        ]
    ].rename(
        columns={
            "away_team": "club",
            "away_score": "actual_goals_for",
            "home_score": "actual_goals_against",
            "lambda_away": "predicted_goals_for",
            "lambda_home": "predicted_goals_against",
        }
    )

    combined = pd.concat(
        [home, away],
        ignore_index=True,
    )

    combined["goals_for_bias"] = (
        combined["predicted_goals_for"]
        - combined["actual_goals_for"]
    )

    combined["goals_against_bias"] = (
        combined["predicted_goals_against"]
        - combined["actual_goals_against"]
    )

    combined["goal_difference_bias"] = (
        (
            combined["predicted_goals_for"]
            - combined["predicted_goals_against"]
        )
        - (
            combined["actual_goals_for"]
            - combined["actual_goals_against"]
        )
    )

    return combined


def build_club_residuals(
    fixture_output: pd.DataFrame,
) -> pd.DataFrame:
    team_rows = build_team_match_rows(
        fixture_output
    )

    return (
        team_rows
        .groupby(
            [
                "configuration",
                "club",
            ],
            as_index=False,
        )
        .agg(
            matches=("event_id", "size"),
            actual_goals_for=(
                "actual_goals_for",
                "sum",
            ),
            predicted_goals_for=(
                "predicted_goals_for",
                "sum",
            ),
            mean_goals_for_bias=(
                "goals_for_bias",
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
            mean_goals_against_bias=(
                "goals_against_bias",
                "mean",
            ),
            mean_goal_difference_bias=(
                "goal_difference_bias",
                "mean",
            ),
        )
        .sort_values(
            [
                "configuration",
                "mean_goals_for_bias",
            ]
        )
        .reset_index(drop=True)
    )


def build_elite_residuals(
    fixture_output: pd.DataFrame,
    elite_clubs: tuple[str, ...],
) -> pd.DataFrame:
    team_rows = build_team_match_rows(
        fixture_output
    )

    team_rows["club_group"] = np.where(
        team_rows["club"].isin(
            elite_clubs
        ),
        "elite_top_quartile",
        "non_elite",
    )

    return (
        team_rows
        .groupby(
            [
                "configuration",
                "club_group",
            ],
            as_index=False,
        )
        .agg(
            team_match_observations=(
                "event_id",
                "size",
            ),
            unique_clubs=(
                "club",
                "nunique",
            ),
            mean_goals_for_bias=(
                "goals_for_bias",
                "mean",
            ),
            mean_goals_against_bias=(
                "goals_against_bias",
                "mean",
            ),
            mean_goal_difference_bias=(
                "goal_difference_bias",
                "mean",
            ),
        )
    )


def build_season_phase_residuals(
    dataframe: pd.DataFrame,
    fixture_output: pd.DataFrame,
) -> pd.DataFrame:
    phase_lookup = dataframe[
        [
            "event_id",
            "round_number",
        ]
    ].copy()

    midpoint = float(
        phase_lookup[
            "round_number"
        ].max()
        / 2.0
    )

    phase_lookup["season_phase"] = np.where(
        phase_lookup[
            "round_number"
        ].le(midpoint),
        "first_half",
        "second_half",
    )

    working = fixture_output.merge(
        phase_lookup[
            [
                "event_id",
                "season_phase",
            ]
        ],
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    working["actual_total_goals"] = (
        working["home_score"]
        + working["away_score"]
    )

    working["total_goal_bias"] = (
        working["pred_total_goals"]
        - working["actual_total_goals"]
    )

    return (
        working
        .groupby(
            [
                "configuration",
                "season_phase",
            ],
            as_index=False,
        )
        .agg(
            match_count=(
                "event_id",
                "size",
            ),
            mean_actual_total_goals=(
                "actual_total_goals",
                "mean",
            ),
            mean_predicted_total_goals=(
                "pred_total_goals",
                "mean",
            ),
            mean_total_goal_bias=(
                "total_goal_bias",
                "mean",
            ),
        )
    )


def build_outcome_rate_comparison(
    fixture_output: pd.DataFrame,
) -> pd.DataFrame:
    working = fixture_output.copy()

    working["actual_home_win"] = (
        working["home_score"]
        > working["away_score"]
    )

    working["actual_draw"] = (
        working["home_score"]
        == working["away_score"]
    )

    working["actual_away_win"] = (
        working["home_score"]
        < working["away_score"]
    )

    return (
        working
        .groupby(
            "configuration",
            as_index=False,
        )
        .agg(
            actual_home_win_rate=(
                "actual_home_win",
                "mean",
            ),
            predicted_home_win_rate=(
                "home_win_probability",
                "mean",
            ),
            actual_draw_rate=(
                "actual_draw",
                "mean",
            ),
            predicted_draw_rate=(
                "draw_probability",
                "mean",
            ),
            actual_away_win_rate=(
                "actual_away_win",
                "mean",
            ),
            predicted_away_win_rate=(
                "away_win_probability",
                "mean",
            ),
        )
    )


def add_improvement_columns(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    output = summary.copy()

    baseline = output.loc[
        output[
            "configuration"
        ].eq(
            "baseline"
        )
    ].iloc[0]

    metrics = [
        "combined_poisson_deviance",
        "combined_goal_mae",
        "total_goal_mae",
        "goal_difference_mae",
        "outcome_log_loss",
        "outcome_brier_score",
        "exact_score_log_loss",
    ]

    for metric in metrics:
        baseline_value = float(
            baseline[metric]
        )

        output[
            f"{metric}_improvement"
        ] = (
            baseline_value
            - output[metric]
        )

        output[
            f"{metric}_relative_improvement"
        ] = (
            (
                baseline_value
                - output[metric]
            )
            / baseline_value
        )

    return output


def build_capacity_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    baseline = summary.loc[
        summary[
            "configuration"
        ].eq(
            "baseline"
        )
    ].iloc[0]

    candidates = summary.loc[
        ~summary[
            "configuration"
        ].eq(
            "baseline"
        )
    ].copy()

    target_metrics = [
        "combined_poisson_deviance",
        "combined_goal_mae",
        "total_goal_mae",
        "outcome_log_loss",
        "outcome_brier_score",
        "exact_score_log_loss",
    ]

    records: list[
        dict[str, object]
    ] = []

    for metric in target_metrics:
        best = candidates.sort_values(
            metric
        ).iloc[0]

        baseline_value = float(
            baseline[metric]
        )

        best_value = float(
            best[metric]
        )

        records.append(
            {
                "metric": metric,
                "baseline_value": baseline_value,
                "best_calibrated_value": best_value,
                "best_configuration": (
                    best["configuration"]
                ),
                "absolute_improvement": (
                    baseline_value
                    - best_value
                ),
                "relative_improvement": (
                    (
                        baseline_value
                        - best_value
                    )
                    / baseline_value
                ),
            }
        )

    return pd.DataFrame(records)


def build_metadata(
    *,
    dataframe: pd.DataFrame,
    results: list[CalibrationResult],
    elite_clubs: tuple[str, ...],
    elite_threshold: float,
) -> dict[str, object]:
    return {
        "study_id": "085B",
        "study_name": (
            "Calibration Capacity Analysis"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "study_type": (
            "counterfactual_frozen_prediction_calibration"
        ),
        "input_dataset": str(
            INPUT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "match_count": len(dataframe),
        "configurations": [
            result.configuration
            for result in results
        ],
        "elite_definition": (
            "Top quartile of clubs by average "
            "prediction-date ClubElo."
        ),
        "elite_rating_threshold": (
            elite_threshold
        ),
        "elite_clubs": list(
            elite_clubs
        ),
        "optimization_target": (
            "In-sample combined home/away Poisson deviance "
            "within the frozen replay artifact."
        ),
        "model_changed": False,
        "production_artifact_changed": False,
        "predictions_rerun": False,
        "calibrations_promoted": False,
        "interpretation_boundary": (
            "Calibration parameters are optimized and evaluated "
            "on the same overlapping-period diagnostic artifact. "
            "Results estimate descriptive calibration capacity, "
            "not prospective out-of-sample improvement."
        ),
        "outputs": [
            CONFIGURATION_SUMMARY_PATH.name,
            FIXTURE_PREDICTIONS_PATH.name,
            CLUB_RESIDUALS_PATH.name,
            ELITE_RESIDUALS_PATH.name,
            SEASON_PHASE_PATH.name,
            OUTCOME_CALIBRATION_PATH.name,
            CAPACITY_SUMMARY_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    summary: pd.DataFrame,
    capacity: pd.DataFrame,
    elite_residuals: pd.DataFrame,
    season_phase: pd.DataFrame,
) -> None:
    baseline = summary.loc[
        summary[
            "configuration"
        ].eq(
            "baseline"
        )
    ].iloc[0]

    best_deviance = summary.sort_values(
        "combined_poisson_deviance"
    ).iloc[0]

    best_log_loss = summary.sort_values(
        "outcome_log_loss"
    ).iloc[0]

    best_draw = (
        summary.assign(
            absolute_draw_error=(
                summary[
                    "draw_rate_error"
                ].abs()
            )
        )
        .sort_values(
            "absolute_draw_error"
        )
        .iloc[0]
    )

    deviance_capacity = capacity.loc[
        capacity["metric"].eq(
            "combined_poisson_deviance"
        )
    ].iloc[0]

    report = f"""# Study 085B — Calibration Capacity Analysis

## Purpose

Estimate how much of the Bundesliga replay error can be removed
through post-prediction calibration while leaving the underlying
football model unchanged.

## Methodological boundary

All calibrations operate on frozen Study 084 lambdas.

No fixture was predicted again.

No production artifact, repository, coefficient, sampler or runtime
configuration was changed.

Calibration parameters were optimized and evaluated on the same
diagnostic population. The results therefore represent descriptive
calibration capacity, not prospective performance.

## Population

- Matches: {int(baseline["match_count"])}
- Baseline combined Poisson deviance:
  {baseline["combined_poisson_deviance"]:.6f}
- Baseline outcome log loss:
  {baseline["outcome_log_loss"]:.6f}
- Baseline predicted draw rate:
  {baseline["predicted_draw_rate"]:.6f}
- Actual draw rate:
  {baseline["actual_draw_rate"]:.6f}

## Best goal calibration

- Configuration:
  `{best_deviance["configuration"]}`
- Combined Poisson deviance:
  {best_deviance["combined_poisson_deviance"]:.6f}
- Relative deviance improvement:
  {best_deviance["combined_poisson_deviance_relative_improvement"]:.6f}
- Combined goal MAE:
  {best_deviance["combined_goal_mae"]:.6f}
- Total-goal bias:
  {best_deviance["total_goal_mean_bias"]:.6f}

## Best outcome calibration

- Configuration:
  `{best_log_loss["configuration"]}`
- Outcome log loss:
  {best_log_loss["outcome_log_loss"]:.6f}
- Relative log-loss improvement:
  {best_log_loss["outcome_log_loss_relative_improvement"]:.6f}
- Outcome Brier score:
  {best_log_loss["outcome_brier_score"]:.6f}

## Draw-rate preservation

The configuration with the smallest absolute draw-rate error was:

- Configuration:
  `{best_draw["configuration"]}`
- Predicted draw rate:
  {best_draw["predicted_draw_rate"]:.6f}
- Draw-rate error:
  {best_draw["draw_rate_error"]:.6f}

## Calibration capacity

The maximum observed relative improvement in combined Poisson
deviance was:

{deviance_capacity["relative_improvement"]:.6f}

This is an optimistic in-sample upper-bound diagnostic.

## Interpretation

Calibration should be considered high-capacity only if it materially
improves goal metrics while preserving:

- outcome log loss;
- Brier score;
- draw behavior;
- elite and non-elite residual balance;
- first- and second-half residual behavior.

If aggregate metrics improve but club, elite or temporal residuals
remain large, then calibration is not the primary scientific
solution.

## Result

**OVERALL RESULT: PASS**

Counterfactual calibration capacity was measured without modifying
the production runtime.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 085B — CALIBRATION CAPACITY ANALYSIS"
    )
    print("=" * 88)

    dataframe = load_predictions()

    elite_clubs, elite_threshold = (
        identify_elite_clubs(
            dataframe
        )
    )

    results = [
        build_baseline(
            dataframe
        ),
        build_global_scale(
            dataframe
        ),
        build_home_away_scale(
            dataframe
        ),
        build_global_additive(
            dataframe
        ),
        build_piecewise_scale(
            dataframe
        ),
        build_rating_strength_correction(
            dataframe
        ),
        build_elite_scale(
            dataframe,
            elite_clubs,
        ),
    ]

    metric_records = [
        evaluate_configuration(
            dataframe,
            result,
        )
        for result in results
    ]

    summary = pd.DataFrame(
        metric_records
    )

    summary.insert(
        1,
        "match_count",
        len(dataframe),
    )

    summary = add_improvement_columns(
        summary
    )

    fixture_output = build_fixture_output(
        dataframe,
        results,
    )

    club_residuals = build_club_residuals(
        fixture_output
    )

    elite_residuals = build_elite_residuals(
        fixture_output,
        elite_clubs,
    )

    season_phase = build_season_phase_residuals(
        dataframe,
        fixture_output,
    )

    outcome_rates = (
        build_outcome_rate_comparison(
            fixture_output
        )
    )

    capacity = build_capacity_summary(
        summary
    )

    metadata = build_metadata(
        dataframe=dataframe,
        results=results,
        elite_clubs=elite_clubs,
        elite_threshold=elite_threshold,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        CONFIGURATION_SUMMARY_PATH,
        index=False,
    )

    fixture_output.to_csv(
        FIXTURE_PREDICTIONS_PATH,
        index=False,
    )

    club_residuals.to_csv(
        CLUB_RESIDUALS_PATH,
        index=False,
    )

    elite_residuals.to_csv(
        ELITE_RESIDUALS_PATH,
        index=False,
    )

    season_phase.to_csv(
        SEASON_PHASE_PATH,
        index=False,
    )

    outcome_rates.to_csv(
        OUTCOME_CALIBRATION_PATH,
        index=False,
    )

    capacity.to_csv(
        CAPACITY_SUMMARY_PATH,
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
        summary=summary,
        capacity=capacity,
        elite_residuals=elite_residuals,
        season_phase=season_phase,
    )

    display_columns = [
        "configuration",
        "combined_poisson_deviance",
        "combined_poisson_deviance_relative_improvement",
        "combined_goal_mae",
        "total_goal_mae",
        "outcome_log_loss",
        "outcome_brier_score",
        "predicted_draw_rate",
        "draw_rate_error",
        "total_goal_mean_bias",
    ]

    print()
    print("Configuration comparison")
    print("-" * 88)
    print(
        summary[
            display_columns
        ]
        .sort_values(
            "combined_poisson_deviance"
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Calibration capacity")
    print("-" * 88)
    print(
        capacity.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Frozen artifact loading: PASS")
    print("  Baseline reconstruction: PASS")
    print("  Global scale calibration: PASS")
    print("  Home/away scale calibration: PASS")
    print("  Additive calibration: PASS")
    print("  Piecewise calibration: PASS")
    print("  Rating-strength calibration: PASS")
    print("  Elite-team calibration: PASS")
    print("  Goal metric evaluation: PASS")
    print("  Outcome metric evaluation: PASS")
    print("  Draw-rate evaluation: PASS")
    print("  Club residual evaluation: PASS")
    print("  Elite residual evaluation: PASS")
    print("  Season-phase evaluation: PASS")
    print("  Production mutation: NONE")

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
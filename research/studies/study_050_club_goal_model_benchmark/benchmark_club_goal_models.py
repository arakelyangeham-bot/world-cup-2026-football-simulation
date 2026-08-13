#benchmark_club_goal_models

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

from simulation.goal_models import (
    GoalPrediction,
    PoissonGoalModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_048_club_observation_dataset"
)

FULL_SQUAD_PATH = (
    INPUT_DIR
    / "full_squad_observations.csv"
)

EXPECTED_XI_PATH = (
    INPUT_DIR
    / "expected_starting_xi_observations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_050_club_goal_model_benchmark"
)


TRAIN_FRACTION = 0.75
MODEL_ALPHA = 0.0
OUTCOME_GRID_MAX_GOALS = 15
PROBABILITY_FLOOR = 1e-15


HOME_FEATURES = [
    "home_attack",
    "away_defense",
    "midfield_diff",
    "defense_diff",
]

AWAY_FEATURES = [
    "away_attack",
    "home_defense",
    "midfield_diff",
    "defense_diff",
]


IDENTITY_COLUMNS = [
    "event_id",
    "date",
    "home_team",
    "home_team_id",
    "away_team",
    "away_team_id",
]

TARGET_COLUMNS = [
    "home_score",
    "away_score",
    "total_goals",
    "goal_difference",
    "result",
]


MODEL_CONFIGS = {
    "full_squad": {
        "path": FULL_SQUAD_PATH,
        "expected_representation_type": "full_squad",
    },
    "expected_starting_xi": {
        "path": EXPECTED_XI_PATH,
        "expected_representation_type": (
            "expected_starting_xi"
        ),
    },
}


def load_observation_dataset(
    path: Path,
    expected_representation_type: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Observation dataset does not exist: {path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Observation dataset is empty: {path}"
        )

    required_columns = {
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *HOME_FEATURES,
        *AWAY_FEATURES,
        "home_representation_type",
        "away_representation_type",
    }

    missing = required_columns - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    numeric_columns = [
        *HOME_FEATURES,
        *AWAY_FEATURES,
        "home_score",
        "away_score",
        "total_goals",
        "goal_difference",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    if dataframe[numeric_columns].isna().any().any():
        raise ValueError(
            f"{path.name} contains missing required "
            "numeric values."
        )

    if dataframe["event_id"].duplicated().any():
        raise ValueError(
            f"{path.name} contains duplicate event IDs."
        )

    if not dataframe[
        "home_representation_type"
    ].eq(expected_representation_type).all():
        raise ValueError(
            f"{path.name} contains unexpected home "
            "representation types."
        )

    if not dataframe[
        "away_representation_type"
    ].eq(expected_representation_type).all():
        raise ValueError(
            f"{path.name} contains unexpected away "
            "representation types."
        )

    return (
        dataframe
        .sort_values(
            ["date", "event_id"]
        )
        .reset_index(drop=True)
    )


def validate_matched_populations(
    datasets: dict[str, pd.DataFrame],
) -> None:
    labels = list(datasets)

    baseline = (
        datasets[labels[0]]
        .set_index("event_id")
        .sort_index()
    )

    comparison_columns = [
        column
        for column in [
            *IDENTITY_COLUMNS,
            *TARGET_COLUMNS,
        ]
        if column != "event_id"
    ]

    for label in labels[1:]:
        candidate = (
            datasets[label]
            .set_index("event_id")
            .sort_index()
        )

        if not baseline.index.equals(
            candidate.index
        ):
            raise AssertionError(
                f"{label}: event population differs "
                "from the baseline dataset."
            )

        for column in comparison_columns:
            left = baseline[column]
            right = candidate[column]

            if pd.api.types.is_numeric_dtype(
                left
            ):
                equal = np.isclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    equal_nan=True,
                ).all()
            else:
                equal = (
                    left.fillna("<missing>")
                    .astype(str)
                    .eq(
                        right.fillna("<missing>")
                        .astype(str)
                    )
                    .all()
                )

            if not equal:
                raise AssertionError(
                    f"{label}: matched datasets disagree "
                    f"on {column!r}."
                )


def build_chronological_split(
    reference: pd.DataFrame,
) -> pd.DataFrame:
    row_count = len(reference)

    split_index = int(
        math.floor(
            row_count
            * TRAIN_FRACTION
        )
    )

    if split_index <= 0:
        raise ValueError(
            "Chronological split produced no "
            "training observations."
        )

    if split_index >= row_count:
        raise ValueError(
            "Chronological split produced no "
            "test observations."
        )

    assignments = reference[
        IDENTITY_COLUMNS
    ].copy()

    assignments["chronological_position"] = (
        np.arange(row_count)
    )

    assignments["split"] = np.where(
        assignments["chronological_position"]
        < split_index,
        "train",
        "test",
    )

    assignments["train_fraction"] = (
        TRAIN_FRACTION
    )

    return assignments


def apply_split(
    dataframe: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_lookup = assignments[
        [
            "event_id",
            "split",
        ]
    ]

    merged = dataframe.merge(
        split_lookup,
        on="event_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(dataframe):
        raise AssertionError(
            "Split assignment did not preserve "
            "the complete event population."
        )

    train = (
        merged[
            merged["split"].eq("train")
        ]
        .sort_values(
            ["date", "event_id"]
        )
        .reset_index(drop=True)
    )

    test = (
        merged[
            merged["split"].eq("test")
        ]
        .sort_values(
            ["date", "event_id"]
        )
        .reset_index(drop=True)
    )

    return train, test


def poisson_probability(
    goals: int,
    expected_goals: float,
) -> float:
    if expected_goals <= 0:
        raise ValueError(
            "Expected goals must be positive."
        )

    log_probability = (
        -expected_goals
        + goals * math.log(expected_goals)
        - math.lgamma(goals + 1)
    )

    return math.exp(log_probability)


def outcome_probabilities(
    lambda_home: float,
    lambda_away: float,
) -> tuple[float, float, float]:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for home_goals in range(
        OUTCOME_GRID_MAX_GOALS + 1
    ):
        home_probability = (
            poisson_probability(
                home_goals,
                lambda_home,
            )
        )

        for away_goals in range(
            OUTCOME_GRID_MAX_GOALS + 1
        ):
            probability = (
                home_probability
                * poisson_probability(
                    away_goals,
                    lambda_away,
                )
            )

            if home_goals > away_goals:
                home_win += probability
            elif home_goals < away_goals:
                away_win += probability
            else:
                draw += probability

    total = (
        home_win
        + draw
        + away_win
    )

    if total <= 0:
        raise ValueError(
            "Outcome probabilities have zero mass."
        )

    return (
        home_win / total,
        draw / total,
        away_win / total,
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


def add_probability_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    result = predictions.copy()

    home_win_probabilities: list[float] = []
    draw_probabilities: list[float] = []
    away_win_probabilities: list[float] = []
    exact_score_probabilities: list[float] = []

    for row in result.itertuples(
        index=False
    ):
        lambda_home = float(
            row.pred_home_goals
        )
        lambda_away = float(
            row.pred_away_goals
        )

        (
            home_win_probability,
            draw_probability,
            away_win_probability,
        ) = outcome_probabilities(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
        )

        exact_score_probability = (
            poisson_probability(
                goals=int(row.home_score),
                expected_goals=lambda_home,
            )
            * poisson_probability(
                goals=int(row.away_score),
                expected_goals=lambda_away,
            )
        )

        home_win_probabilities.append(
            home_win_probability
        )
        draw_probabilities.append(
            draw_probability
        )
        away_win_probabilities.append(
            away_win_probability
        )
        exact_score_probabilities.append(
            exact_score_probability
        )

    result["pred_home_win_probability"] = (
        home_win_probabilities
    )

    result["pred_draw_probability"] = (
        draw_probabilities
    )

    result["pred_away_win_probability"] = (
        away_win_probabilities
    )

    result["pred_exact_score_probability"] = (
        exact_score_probabilities
    )

    result["pred_total_goals"] = (
        result["pred_home_goals"]
        + result["pred_away_goals"]
    )

    result["pred_goal_difference"] = (
        result["pred_home_goals"]
        - result["pred_away_goals"]
    )

    return result


def correlation(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    if len(actual) < 2:
        return float("nan")

    if (
        actual.nunique() <= 1
        or predicted.nunique() <= 1
    ):
        return float("nan")

    return float(
        np.corrcoef(
            actual.to_numpy(dtype=float),
            predicted.to_numpy(dtype=float),
        )[0, 1]
    )


def multiclass_brier_score(
    observed_indices: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        np.arange(
            len(observed_indices)
        ),
        observed_indices,
    ] = 1.0

    squared_error = (
        probabilities
        - one_hot
    ) ** 2

    return float(
        squared_error.sum(axis=1).mean()
    )


def evaluate_predictions(
    model_name: str,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    actual_home = predictions[
        "home_score"
    ].to_numpy(dtype=float)

    actual_away = predictions[
        "away_score"
    ].to_numpy(dtype=float)

    predicted_home = predictions[
        "pred_home_goals"
    ].to_numpy(dtype=float)

    predicted_away = predictions[
        "pred_away_goals"
    ].to_numpy(dtype=float)

    actual_total = (
        actual_home + actual_away
    )

    predicted_total = (
        predicted_home + predicted_away
    )

    actual_goal_difference = (
        actual_home - actual_away
    )

    predicted_goal_difference = (
        predicted_home - predicted_away
    )

    home_deviance = mean_poisson_deviance(
        actual_home,
        predicted_home,
    )

    away_deviance = mean_poisson_deviance(
        actual_away,
        predicted_away,
    )

    combined_deviance = (
        home_deviance
        + away_deviance
    ) / 2.0

    observed_indices = np.array(
        [
            observed_result_index(
                int(home_score),
                int(away_score),
            )
            for home_score, away_score in zip(
                actual_home,
                actual_away,
            )
        ],
        dtype=int,
    )

    outcome_probabilities_array = (
        predictions[
            [
                "pred_home_win_probability",
                "pred_draw_probability",
                "pred_away_win_probability",
            ]
        ]
        .to_numpy(dtype=float)
    )

    outcome_probabilities_array = np.clip(
        outcome_probabilities_array,
        PROBABILITY_FLOOR,
        1.0,
    )

    outcome_probabilities_array = (
        outcome_probabilities_array
        / outcome_probabilities_array.sum(
            axis=1,
            keepdims=True,
        )
    )

    exact_score_probabilities = np.clip(
        predictions[
            "pred_exact_score_probability"
        ].to_numpy(dtype=float),
        PROBABILITY_FLOOR,
        1.0,
    )

    actual_draw_rate = float(
        (
            actual_home
            == actual_away
        ).mean()
    )

    predicted_draw_rate = float(
        predictions[
            "pred_draw_probability"
        ].mean()
    )

    return {
        "model": model_name,
        "test_matches": len(predictions),

        "home_poisson_deviance": float(
            home_deviance
        ),
        "away_poisson_deviance": float(
            away_deviance
        ),
        "combined_poisson_deviance": float(
            combined_deviance
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
            pd.Series(actual_home),
            pd.Series(predicted_home),
        ),
        "away_goal_correlation": correlation(
            pd.Series(actual_away),
            pd.Series(predicted_away),
        ),

        "outcome_log_loss": float(
            log_loss(
                observed_indices,
                outcome_probabilities_array,
                labels=[0, 1, 2],
            )
        ),
        "outcome_brier_score": (
            multiclass_brier_score(
                observed_indices,
                outcome_probabilities_array,
            )
        ),
        "exact_score_log_loss": float(
            -np.log(
                exact_score_probabilities
            ).mean()
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

        "mean_actual_home_goals": float(
            actual_home.mean()
        ),
        "mean_predicted_home_goals": float(
            predicted_home.mean()
        ),
        "mean_actual_away_goals": float(
            actual_away.mean()
        ),
        "mean_predicted_away_goals": float(
            predicted_away.mean()
        ),
    }


def fit_and_predict(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    PoissonGoalModel,
    pd.DataFrame,
]:
    model = PoissonGoalModel(
        name=model_name,
        home_features=HOME_FEATURES,
        away_features=AWAY_FEATURES,
        alpha=MODEL_ALPHA,
    )

    model.fit(train)

    prediction: GoalPrediction = (
        model.predict(test)
    )

    output = test[
        [
            *IDENTITY_COLUMNS,
            *TARGET_COLUMNS,
            *HOME_FEATURES,
            *AWAY_FEATURES,
        ]
    ].copy()

    output["model"] = model_name

    output["pred_home_goals"] = (
        prediction.pred_home_goals
    )

    output["pred_away_goals"] = (
        prediction.pred_away_goals
    )

    output = add_probability_predictions(
        output
    )

    return model, output


def serialize_coefficients(
    model: PoissonGoalModel,
) -> dict[str, object]:
    return {
        "model_name": model.name,
        "alpha": model.alpha,
        "home_model": {
            "features": model.home_features,
            "intercept": float(
                model.home_model.intercept_
            ),
            "coefficients": {
                feature: float(coefficient)
                for feature, coefficient in zip(
                    model.home_features,
                    model.home_model.coef_,
                )
            },
        },
        "away_model": {
            "features": model.away_features,
            "intercept": float(
                model.away_model.intercept_
            ),
            "coefficients": {
                feature: float(coefficient)
                for feature, coefficient in zip(
                    model.away_features,
                    model.away_model.coef_,
                )
            },
        },
    }


def build_paired_prediction_comparison(
    predictions_by_model: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    full_squad = (
        predictions_by_model[
            "full_squad"
        ]
        .set_index("event_id")
        .sort_index()
    )

    expected_xi = (
        predictions_by_model[
            "expected_starting_xi"
        ]
        .set_index("event_id")
        .sort_index()
    )

    if not full_squad.index.equals(
        expected_xi.index
    ):
        raise AssertionError(
            "Prediction populations do not match."
        )

    result = full_squad[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    prediction_columns = [
        "pred_home_goals",
        "pred_away_goals",
        "pred_total_goals",
        "pred_goal_difference",
        "pred_home_win_probability",
        "pred_draw_probability",
        "pred_away_win_probability",
        "pred_exact_score_probability",
    ]

    for column in prediction_columns:
        result[
            f"full_squad_{column}"
        ] = full_squad[column]

        result[
            f"expected_xi_{column}"
        ] = expected_xi[column]

        result[
            f"{column}_difference"
        ] = (
            expected_xi[column]
            - full_squad[column]
        )

    result["full_squad_absolute_goal_error"] = (
        (
            full_squad["home_score"]
            - full_squad["pred_home_goals"]
        ).abs()
        + (
            full_squad["away_score"]
            - full_squad["pred_away_goals"]
        ).abs()
    ) / 2.0

    result["expected_xi_absolute_goal_error"] = (
        (
            expected_xi["home_score"]
            - expected_xi["pred_home_goals"]
        ).abs()
        + (
            expected_xi["away_score"]
            - expected_xi["pred_away_goals"]
        ).abs()
    ) / 2.0

    result["absolute_goal_error_difference"] = (
        result[
            "expected_xi_absolute_goal_error"
        ]
        - result[
            "full_squad_absolute_goal_error"
        ]
    )

    return result.reset_index()


def determine_preferred_model(
    performance: pd.DataFrame,
) -> str:
    lower_is_better = [
        "combined_poisson_deviance",
        "combined_goal_mae",
        "total_goal_mae",
        "goal_difference_mae",
        "outcome_log_loss",
        "outcome_brier_score",
        "exact_score_log_loss",
        "absolute_draw_rate_error",
    ]

    win_counts = {
        model: 0
        for model in performance["model"]
    }

    for metric in lower_is_better:
        winner = (
            performance
            .sort_values(metric)
            .iloc[0]["model"]
        )

        win_counts[winner] += 1

    ordered = sorted(
        win_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    if (
        len(ordered) > 1
        and ordered[0][1] == ordered[1][1]
    ):
        return "mixed"

    return ordered[0][0]


def write_results_markdown(
    path: Path,
    performance: pd.DataFrame,
    preferred_model: str,
) -> None:
    ordered = performance.sort_values(
        "combined_poisson_deviance"
    )

    best_deviance = ordered.iloc[0]
    other_deviance = ordered.iloc[1]

    lines = [
        "# Study 050 Results",
        "",
        "## Full Squad vs Expected XI Club Goal Model",
        "",
        "**Status:** `PASS`",
        "",
        "## Experimental design",
        "",
        (
            f"- Training fraction: "
            f"{TRAIN_FRACTION:.2f}"
        ),
        (
            f"- Model alpha: "
            f"{MODEL_ALPHA:.6f}"
        ),
        (
            "- Representation type is the only intended "
            "experimental difference."
        ),
        "- Historical rating prior: unavailable.",
        "",
        "## Combined Poisson deviance",
        "",
        (
            f"- Best model: `{best_deviance['model']}`"
        ),
        (
            "- Best value: "
            f"{best_deviance['combined_poisson_deviance']:.6f}"
        ),
        (
            f"- Other model: `{other_deviance['model']}`"
        ),
        (
            "- Other value: "
            f"{other_deviance['combined_poisson_deviance']:.6f}"
        ),
        "",
        "## Multi-metric decision",
        "",
        f"- Preferred model: `{preferred_model}`",
        "",
        "## Interpretation boundary",
        "",
        (
            "This is a single-season chronological "
            "holdout benchmark. It does not establish "
            "multi-season generalization."
        ),
        (
            "The expected XI is a season-level projected "
            "lineup rather than a match-specific lineup."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    datasets = {
        label: load_observation_dataset(
            path=config["path"],
            expected_representation_type=(
                config[
                    "expected_representation_type"
                ]
            ),
        )
        for label, config in (
            MODEL_CONFIGS.items()
        )
    }

    validate_matched_populations(
        datasets
    )

    reference = datasets[
        "full_squad"
    ]

    assignments = (
        build_chronological_split(
            reference
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments.to_csv(
        OUTPUT_DIR
        / "chronological_split_assignments.csv",
        index=False,
    )

    performance_rows: list[
        dict[str, object]
    ] = []

    predictions_by_model: dict[
        str,
        pd.DataFrame,
    ] = {}

    for model_name, dataframe in (
        datasets.items()
    ):
        train, test = apply_split(
            dataframe=dataframe,
            assignments=assignments,
        )

        model, predictions = (
            fit_and_predict(
                model_name=model_name,
                train=train,
                test=test,
            )
        )

        predictions_by_model[
            model_name
        ] = predictions

        predictions.to_csv(
            OUTPUT_DIR
            / f"{model_name}_predictions.csv",
            index=False,
        )

        coefficients = (
            serialize_coefficients(
                model
            )
        )

        with (
            OUTPUT_DIR
            / f"{model_name}_coefficients.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                coefficients,
                file,
                indent=2,
            )

        performance_rows.append(
            evaluate_predictions(
                model_name=model_name,
                predictions=predictions,
            )
        )

    performance = (
        pd.DataFrame(
            performance_rows
        )
        .sort_values(
            "combined_poisson_deviance"
        )
        .reset_index(drop=True)
    )

    performance.to_csv(
        OUTPUT_DIR
        / "model_performance_summary.csv",
        index=False,
    )

    paired_predictions = (
        build_paired_prediction_comparison(
            predictions_by_model
        )
    )

    paired_predictions.to_csv(
        OUTPUT_DIR
        / "paired_prediction_comparison.csv",
        index=False,
    )

    preferred_model = (
        determine_preferred_model(
            performance
        )
    )

    metadata = {
        "study_id": "050",
        "study_name": (
            "Full Squad vs Expected XI "
            "Club Goal Model Benchmark"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "observation_count": len(reference),
        "training_count": int(
            assignments["split"]
            .eq("train")
            .sum()
        ),
        "test_count": int(
            assignments["split"]
            .eq("test")
            .sum()
        ),
        "train_fraction": TRAIN_FRACTION,
        "model_alpha": MODEL_ALPHA,
        "home_features": HOME_FEATURES,
        "away_features": AWAY_FEATURES,
        "rating_prior_included": False,
        "preferred_model": (
            preferred_model
        ),
        "output_files": [
            "chronological_split_assignments.csv",
            "model_performance_summary.csv",
            "full_squad_predictions.csv",
            (
                "expected_starting_xi_"
                "predictions.csv"
            ),
            "full_squad_coefficients.json",
            (
                "expected_starting_xi_"
                "coefficients.json"
            ),
            "paired_prediction_comparison.csv",
            "study_metadata.json",
            "STUDY_050_RESULTS.md",
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
            / "STUDY_050_RESULTS.md"
        ),
        performance=performance,
        preferred_model=preferred_model,
    )

    print("Study 050")
    print("=" * 68)
    print()
    print(
        f"Matched observations: "
        f"{len(reference)}"
    )
    print(
        "Training observations: "
        f"{metadata['training_count']}"
    )
    print(
        "Test observations: "
        f"{metadata['test_count']}"
    )
    print()
    print("Model Performance")
    print("-" * 68)
    print(
        performance[
            [
                "model",
                "combined_poisson_deviance",
                "combined_goal_mae",
                "total_goal_mae",
                "goal_difference_mae",
                "outcome_log_loss",
                "outcome_brier_score",
                "exact_score_log_loss",
                "absolute_draw_rate_error",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    print()
    print(
        f"Multi-metric preference: "
        f"{preferred_model}"
    )
    print()
    print("Matched population: PASS")
    print("Chronological split: PASS")
    print("Shared model contract: PASS")
    print("Historical prior exclusion: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
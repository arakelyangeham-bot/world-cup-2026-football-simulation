#goal_model_benchmark

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
)

from research.modeling.football_feature_registry import (
    FootballFeatureSpecification,
    get_club_goal_model_feature_spec,
)
from simulation.goal_models import (
    GoalPrediction,
    PoissonGoalModel,
)


DEFAULT_IDENTITY_COLUMNS = (
    "event_id",
    "date",
    "home_team",
    "home_team_id",
    "away_team",
    "away_team_id",
)

DEFAULT_TARGET_COLUMNS = (
    "home_score",
    "away_score",
    "total_goals",
    "goal_difference",
)

DEFAULT_METRICS = (
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


@dataclass(frozen=True)
class GoalModelDatasetConfig:
    """
    Describes one observation dataset used by the benchmark engine.
    """

    name: str
    path: Path
    representation_type: str


@dataclass(frozen=True)
class GoalModelBenchmarkConfig:
    """
    Complete configuration for one benchmark experiment.
    """

    name: str
    datasets: tuple[GoalModelDatasetConfig, ...]
    feature_specifications: tuple[str, ...]
    train_fractions: tuple[float, ...]
    alpha_values: tuple[float, ...]

    identity_columns: tuple[str, ...] = (
        DEFAULT_IDENTITY_COLUMNS
    )

    target_columns: tuple[str, ...] = (
        DEFAULT_TARGET_COLUMNS
    )

    ranking_metrics: tuple[str, ...] = (
        DEFAULT_METRICS
    )

    outcome_grid_max_goals: int = 15
    probability_floor: float = 1e-15

    require_matched_populations: bool = True
    capture_predictions: bool = True
    capture_coefficients: bool = True


@dataclass
class GoalModelBenchmarkResult:
    """
    Outputs produced by one benchmark execution.
    """

    config: GoalModelBenchmarkConfig
    results: pd.DataFrame
    coefficients: pd.DataFrame
    predictions: pd.DataFrame
    split_assignments: pd.DataFrame
    configuration_ranking: pd.DataFrame
    datasets: dict[str, pd.DataFrame] = field(
        repr=False
    )


def _deduplicate_preserving_order(
    values: Iterable[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        output.append(value)

    return tuple(output)


def _validate_config(
    config: GoalModelBenchmarkConfig,
) -> None:
    if not config.name.strip():
        raise ValueError(
            "Benchmark name cannot be empty."
        )

    if not config.datasets:
        raise ValueError(
            "At least one dataset must be configured."
        )

    dataset_names = [
        dataset.name
        for dataset in config.datasets
    ]

    if len(dataset_names) != len(
        set(dataset_names)
    ):
        raise ValueError(
            "Benchmark dataset names must be unique."
        )

    if not config.feature_specifications:
        raise ValueError(
            "At least one feature specification "
            "must be configured."
        )

    if len(
        config.feature_specifications
    ) != len(
        set(config.feature_specifications)
    ):
        raise ValueError(
            "Feature specifications must be unique."
        )

    for specification_name in (
        config.feature_specifications
    ):
        get_club_goal_model_feature_spec(
            specification_name
        )

    if not config.train_fractions:
        raise ValueError(
            "At least one training fraction "
            "must be configured."
        )

    if len(config.train_fractions) != len(
        set(config.train_fractions)
    ):
        raise ValueError(
            "Training fractions must be unique."
        )

    for fraction in config.train_fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                "Training fractions must lie "
                f"between zero and one: {fraction}"
            )

    if not config.alpha_values:
        raise ValueError(
            "At least one alpha value must "
            "be configured."
        )

    if len(config.alpha_values) != len(
        set(config.alpha_values)
    ):
        raise ValueError(
            "Alpha values must be unique."
        )

    for alpha in config.alpha_values:
        if alpha < 0:
            raise ValueError(
                "Alpha values cannot be negative: "
                f"{alpha}"
            )

    if config.outcome_grid_max_goals < 1:
        raise ValueError(
            "Outcome-grid maximum goals must "
            "be positive."
        )

    if not 0 < config.probability_floor < 1:
        raise ValueError(
            "Probability floor must lie between "
            "zero and one."
        )


def _required_feature_columns(
    config: GoalModelBenchmarkConfig,
) -> tuple[str, ...]:
    columns: list[str] = []

    for specification_name in (
        config.feature_specifications
    ):
        specification = (
            get_club_goal_model_feature_spec(
                specification_name
            )
        )

        columns.extend(
            specification.required_columns()
        )

    return _deduplicate_preserving_order(
        columns
    )


def load_benchmark_dataset(
    dataset_config: GoalModelDatasetConfig,
    benchmark_config: GoalModelBenchmarkConfig,
) -> pd.DataFrame:
    """
    Load and validate one observation dataset.
    """
    path = dataset_config.path

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

    feature_columns = (
        _required_feature_columns(
            benchmark_config
        )
    )

    required_columns = {
        *benchmark_config.identity_columns,
        *benchmark_config.target_columns,
        *feature_columns,
        "home_representation_type",
        "away_representation_type",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    numeric_columns = [
        *benchmark_config.target_columns,
        *feature_columns,
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
            f"{path.name} contains missing "
            "required numeric values."
        )

    if dataframe[
        "event_id"
    ].duplicated().any():
        raise ValueError(
            f"{path.name} contains duplicate "
            "event IDs."
        )

    expected_type = (
        dataset_config.representation_type
    )

    if not dataframe[
        "home_representation_type"
    ].eq(expected_type).all():
        raise ValueError(
            f"{path.name} contains unexpected "
            "home representation types."
        )

    if not dataframe[
        "away_representation_type"
    ].eq(expected_type).all():
        raise ValueError(
            f"{path.name} contains unexpected "
            "away representation types."
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


def load_benchmark_datasets(
    config: GoalModelBenchmarkConfig,
) -> dict[str, pd.DataFrame]:
    """
    Load every configured benchmark dataset.
    """
    datasets = {
        dataset_config.name: (
            load_benchmark_dataset(
                dataset_config=dataset_config,
                benchmark_config=config,
            )
        )
        for dataset_config in config.datasets
    }

    if config.require_matched_populations:
        validate_matched_populations(
            datasets=datasets,
            identity_columns=(
                config.identity_columns
            ),
            target_columns=(
                config.target_columns
            ),
        )

    return datasets


def validate_matched_populations(
    datasets: dict[str, pd.DataFrame],
    identity_columns: Sequence[str],
    target_columns: Sequence[str],
) -> None:
    """
    Confirm that multiple representation datasets contain
    identical matches and targets.
    """
    if len(datasets) <= 1:
        return

    labels = list(datasets)

    baseline = (
        datasets[labels[0]]
        .set_index("event_id")
        .sort_index()
    )

    comparison_columns = [
        column
        for column in (
            *identity_columns,
            *target_columns,
        )
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
                f"{label}: event population does "
                "not match the baseline dataset."
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
                    f"{label}: matched datasets "
                    f"disagree on {column!r}."
                )


def build_split_assignments(
    reference: pd.DataFrame,
    train_fractions: Sequence[float],
    identity_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Create reusable chronological split assignments.
    """
    frames: list[pd.DataFrame] = []

    for train_fraction in train_fractions:
        split_index = int(
            math.floor(
                len(reference)
                * train_fraction
            )
        )

        if split_index <= 0:
            raise ValueError(
                "Chronological split produced "
                "no training rows."
            )

        if split_index >= len(reference):
            raise ValueError(
                "Chronological split produced "
                "no test rows."
            )

        assignments = reference[
            list(identity_columns)
        ].copy()

        assignments[
            "chronological_position"
        ] = np.arange(
            len(reference)
        )

        assignments["train_fraction"] = (
            train_fraction
        )

        assignments["split"] = np.where(
            assignments[
                "chronological_position"
            ] < split_index,
            "train",
            "test",
        )

        train_dates = assignments.loc[
            assignments["split"].eq("train"),
            "date",
        ]

        test_dates = assignments.loc[
            assignments["split"].eq("test"),
            "date",
        ]

        if train_dates.max() > test_dates.min():
            raise AssertionError(
                "Chronological split is invalid."
            )

        frames.append(assignments)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def apply_split_assignment(
    dataframe: pd.DataFrame,
    assignments: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a previously constructed split assignment.
    """
    selected_assignments = assignments[
        assignments[
            "train_fraction"
        ].eq(train_fraction)
    ][
        [
            "event_id",
            "split",
        ]
    ]

    merged = dataframe.merge(
        selected_assignments,
        on="event_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(dataframe):
        raise AssertionError(
            "Split assignment failed to preserve "
            "the complete event population."
        )

    train = (
        merged[
            merged["split"].eq("train")
        ]
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )

    test = (
        merged[
            merged["split"].eq("test")
        ]
        .sort_values(
            [
                "date",
                "event_id",
            ]
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
    maximum_goals: int,
) -> tuple[float, float, float]:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for home_goals in range(
        maximum_goals + 1
    ):
        home_probability = (
            poisson_probability(
                home_goals,
                lambda_home,
            )
        )

        for away_goals in range(
            maximum_goals + 1
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

    total = home_win + draw + away_win

    if total <= 0:
        raise ValueError(
            "Outcome probability mass is zero."
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
    config: GoalModelBenchmarkConfig,
) -> pd.DataFrame:
    output = predictions.copy()

    home_win_probabilities: list[float] = []
    draw_probabilities: list[float] = []
    away_win_probabilities: list[float] = []
    exact_score_probabilities: list[float] = []

    for row in output.itertuples(
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
            maximum_goals=(
                config.outcome_grid_max_goals
            ),
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

    output[
        "pred_home_win_probability"
    ] = home_win_probabilities

    output[
        "pred_draw_probability"
    ] = draw_probabilities

    output[
        "pred_away_win_probability"
    ] = away_win_probabilities

    output[
        "pred_exact_score_probability"
    ] = exact_score_probabilities

    output["pred_total_goals"] = (
        output["pred_home_goals"]
        + output["pred_away_goals"]
    )

    output["pred_goal_difference"] = (
        output["pred_home_goals"]
        - output["pred_away_goals"]
    )

    return output


def multiclass_brier_score(
    observed_results: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        np.arange(
            len(observed_results)
        ),
        observed_results,
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


def fit_goal_model(
    dataset_name: str,
    specification: FootballFeatureSpecification,
    train_fraction: float,
    alpha: float,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: GoalModelBenchmarkConfig,
) -> tuple[
    PoissonGoalModel,
    pd.DataFrame,
]:
    model_name = (
        f"{dataset_name}_"
        f"{specification.name}_"
        f"train_{train_fraction:.4f}_"
        f"alpha_{alpha:.6f}"
    )

    model = PoissonGoalModel(
        name=model_name,
        home_features=list(
            specification.home_features
        ),
        away_features=list(
            specification.away_features
        ),
        alpha=alpha,
    )

    model.fit(train)

    prediction: GoalPrediction = (
        model.predict(test)
    )

    output = test[
        [
            *config.identity_columns,
            *config.target_columns,
        ]
    ].copy()

    output["dataset"] = dataset_name
    output[
        "representation_type"
    ] = dataset_name

    output[
        "feature_specification"
    ] = specification.name

    output["train_fraction"] = (
        train_fraction
    )

    output["alpha"] = alpha

    output["pred_home_goals"] = (
        prediction.pred_home_goals
    )

    output["pred_away_goals"] = (
        prediction.pred_away_goals
    )

    return (
        model,
        add_probability_predictions(
            predictions=output,
            config=config,
        ),
    )


def evaluate_predictions(
    dataset_name: str,
    representation_type: str,
    specification: FootballFeatureSpecification,
    train_fraction: float,
    alpha: float,
    train_count: int,
    predictions: pd.DataFrame,
    config: GoalModelBenchmarkConfig,
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

    home_deviance = mean_poisson_deviance(
        actual_home,
        predicted_home,
    )

    away_deviance = mean_poisson_deviance(
        actual_away,
        predicted_away,
    )

    probabilities = (
        predictions[
            [
                "pred_home_win_probability",
                "pred_draw_probability",
                "pred_away_win_probability",
            ]
        ]
        .to_numpy(dtype=float)
    )

    probabilities = np.clip(
        probabilities,
        config.probability_floor,
        1.0,
    )

    probabilities = (
        probabilities
        / probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    observed_results = np.array(
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

    actual_home_mean = float(
        actual_home.mean()
    )

    predicted_home_mean = float(
        predicted_home.mean()
    )

    actual_away_mean = float(
        actual_away.mean()
    )

    predicted_away_mean = float(
        predicted_away.mean()
    )

    return {
        "benchmark_name": config.name,
        "dataset": dataset_name,
        "representation_type":
            representation_type,

        "feature_specification":
            specification.name,
        "feature_description":
            specification.description,
        "feature_groups":
            "|".join(
                specification.group_names
            ),

        "home_feature_count": len(
            specification.home_features
        ),
        "away_feature_count": len(
            specification.away_features
        ),

        "train_fraction":
            train_fraction,
        "alpha": alpha,
        "train_matches":
            train_count,
        "test_matches":
            len(predictions),

        "home_poisson_deviance": float(
            home_deviance
        ),
        "away_poisson_deviance": float(
            away_deviance
        ),
        "combined_poisson_deviance": float(
            (
                home_deviance
                + away_deviance
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

        "total_goal_mae": float(
            mean_absolute_error(
                actual_home + actual_away,
                predicted_home + predicted_away,
            )
        ),

        "goal_difference_mae": float(
            mean_absolute_error(
                actual_home - actual_away,
                predicted_home - predicted_away,
            )
        ),

        "outcome_log_loss": float(
            log_loss(
                observed_results,
                probabilities,
                labels=[0, 1, 2],
            )
        ),

        "outcome_brier_score":
            multiclass_brier_score(
                observed_results,
                probabilities,
            ),

        "exact_score_log_loss": float(
            -np.log(
                np.clip(
                    predictions[
                        "pred_exact_score_probability"
                    ].to_numpy(dtype=float),
                    config.probability_floor,
                    1.0,
                )
            ).mean()
        ),

        "actual_draw_rate":
            actual_draw_rate,
        "predicted_draw_rate":
            predicted_draw_rate,
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

        "actual_home_goal_mean":
            actual_home_mean,
        "predicted_home_goal_mean":
            predicted_home_mean,
        "home_goal_mean_error": float(
            predicted_home_mean
            - actual_home_mean
        ),
        "absolute_home_goal_mean_error": float(
            abs(
                predicted_home_mean
                - actual_home_mean
            )
        ),

        "actual_away_goal_mean":
            actual_away_mean,
        "predicted_away_goal_mean":
            predicted_away_mean,
        "away_goal_mean_error": float(
            predicted_away_mean
            - actual_away_mean
        ),
        "absolute_away_goal_mean_error": float(
            abs(
                predicted_away_mean
                - actual_away_mean
            )
        ),
    }


def extract_coefficient_rows(
    model: PoissonGoalModel,
    dataset_name: str,
    representation_type: str,
    specification_name: str,
    train_fraction: float,
    alpha: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for target, fitted_model, features in (
        (
            "home_score",
            model.home_model,
            model.home_features,
        ),
        (
            "away_score",
            model.away_model,
            model.away_features,
        ),
    ):
        rows.append(
            {
                "dataset": dataset_name,
                "representation_type":
                    representation_type,
                "feature_specification":
                    specification_name,
                "train_fraction":
                    train_fraction,
                "alpha": alpha,
                "target": target,
                "feature": "intercept",
                "coefficient": float(
                    fitted_model.intercept_
                ),
            }
        )

        for feature, coefficient in zip(
            features,
            fitted_model.coef_,
        ):
            rows.append(
                {
                    "dataset": dataset_name,
                    "representation_type":
                        representation_type,
                    "feature_specification":
                        specification_name,
                    "train_fraction":
                        train_fraction,
                    "alpha": alpha,
                    "target": target,
                    "feature": feature,
                    "coefficient": float(
                        coefficient
                    ),
                }
            )

    return rows


def build_configuration_ranking(
    results: pd.DataFrame,
    ranking_metrics: Sequence[str],
) -> pd.DataFrame:
    ranked = results.copy()
    rank_columns: list[str] = []

    for metric in ranking_metrics:
        if metric not in ranked.columns:
            raise KeyError(
                "Ranking metric does not exist in "
                f"benchmark results: {metric!r}"
            )

        rank_column = f"{metric}_rank"

        ranked[rank_column] = (
            ranked[metric]
            .rank(
                method="average",
                ascending=True,
            )
        )

        rank_columns.append(rank_column)

    ranked["mean_metric_rank"] = (
        ranked[rank_columns]
        .mean(axis=1)
    )

    return (
        ranked
        .sort_values(
            [
                "mean_metric_rank",
                "combined_poisson_deviance",
            ]
        )
        .reset_index(drop=True)
    )


def run_goal_model_benchmark(
    config: GoalModelBenchmarkConfig,
) -> GoalModelBenchmarkResult:
    """
    Execute the complete configured goal-model benchmark.
    """
    _validate_config(config)

    datasets = load_benchmark_datasets(
        config
    )

    reference_name = next(iter(datasets))
    reference = datasets[reference_name]

    split_assignments = (
        build_split_assignments(
            reference=reference,
            train_fractions=(
                config.train_fractions
            ),
            identity_columns=(
                config.identity_columns
            ),
        )
    )

    representation_lookup = {
        dataset.name:
            dataset.representation_type
        for dataset in config.datasets
    }

    result_rows: list[
        dict[str, object]
    ] = []

    coefficient_rows: list[
        dict[str, object]
    ] = []

    prediction_frames: list[
        pd.DataFrame
    ] = []

    for dataset_name, dataframe in (
        datasets.items()
    ):
        representation_type = (
            representation_lookup[
                dataset_name
            ]
        )

        for specification_name in (
            config.feature_specifications
        ):
            specification = (
                get_club_goal_model_feature_spec(
                    specification_name
                )
            )

            for train_fraction in (
                config.train_fractions
            ):
                train, test = (
                    apply_split_assignment(
                        dataframe=dataframe,
                        assignments=(
                            split_assignments
                        ),
                        train_fraction=(
                            train_fraction
                        ),
                    )
                )

                for alpha in (
                    config.alpha_values
                ):
                    model, predictions = (
                        fit_goal_model(
                            dataset_name=(
                                dataset_name
                            ),
                            specification=(
                                specification
                            ),
                            train_fraction=(
                                train_fraction
                            ),
                            alpha=alpha,
                            train=train,
                            test=test,
                            config=config,
                        )
                    )

                    result_rows.append(
                        evaluate_predictions(
                            dataset_name=(
                                dataset_name
                            ),
                            representation_type=(
                                representation_type
                            ),
                            specification=(
                                specification
                            ),
                            train_fraction=(
                                train_fraction
                            ),
                            alpha=alpha,
                            train_count=len(train),
                            predictions=predictions,
                            config=config,
                        )
                    )

                    if (
                        config.capture_coefficients
                    ):
                        coefficient_rows.extend(
                            extract_coefficient_rows(
                                model=model,
                                dataset_name=(
                                    dataset_name
                                ),
                                representation_type=(
                                    representation_type
                                ),
                                specification_name=(
                                    specification_name
                                ),
                                train_fraction=(
                                    train_fraction
                                ),
                                alpha=alpha,
                            )
                        )

                    if (
                        config.capture_predictions
                    ):
                        prediction_frames.append(
                            predictions
                        )

    results = pd.DataFrame(
        result_rows
    )

    expected_run_count = (
        len(config.datasets)
        * len(
            config.feature_specifications
        )
        * len(config.train_fractions)
        * len(config.alpha_values)
    )

    if len(results) != expected_run_count:
        raise AssertionError(
            "Unexpected benchmark run count: "
            f"{len(results)} vs "
            f"{expected_run_count}."
        )

    coefficients = pd.DataFrame(
        coefficient_rows
    )

    predictions = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        if prediction_frames
        else pd.DataFrame()
    )

    configuration_ranking = (
        build_configuration_ranking(
            results=results,
            ranking_metrics=(
                config.ranking_metrics
            ),
        )
    )

    return GoalModelBenchmarkResult(
        config=config,
        results=results,
        coefficients=coefficients,
        predictions=predictions,
        split_assignments=(
            split_assignments
        ),
        configuration_ranking=(
            configuration_ranking
        ),
        datasets=datasets,
    )


def write_goal_model_benchmark_outputs(
    benchmark_result: GoalModelBenchmarkResult,
    output_directory: Path,
) -> None:
    """
    Write standard benchmark artifacts.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_result.results.to_csv(
        output_directory
        / "benchmark_results.csv",
        index=False,
    )

    benchmark_result.configuration_ranking.to_csv(
        output_directory
        / "configuration_ranking.csv",
        index=False,
    )

    benchmark_result.split_assignments.to_csv(
        output_directory
        / "split_assignments.csv",
        index=False,
    )

    if not benchmark_result.coefficients.empty:
        benchmark_result.coefficients.to_csv(
            output_directory
            / "coefficients.csv",
            index=False,
        )

    if not benchmark_result.predictions.empty:
        benchmark_result.predictions.to_csv(
            output_directory
            / "predictions.csv",
            index=False,
        )
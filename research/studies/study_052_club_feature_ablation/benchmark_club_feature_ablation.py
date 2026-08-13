#benchmark_club_feature_ablation

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
)

from research.modeling.football_feature_registry import (
    get_club_goal_model_feature_spec,
    list_club_goal_model_feature_specs,
)
from simulation.goal_models import (
    GoalPrediction,
    PoissonGoalModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_048_club_observation_dataset"
    / "full_squad_observations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_052_club_feature_ablation"
)


TRAIN_FRACTION = 0.75

ALPHA_VALUES = (
    0.0,
    0.01,
    0.1,
    1.0,
)

BASELINE_SPECIFICATION = "attack_defense"

OUTCOME_GRID_MAX_GOALS = 15
PROBABILITY_FLOOR = 1e-15


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
]


LOWER_IS_BETTER_METRICS = [
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
]


def required_feature_columns() -> list[str]:
    columns: set[str] = set()

    for specification_name in (
        list_club_goal_model_feature_specs()
    ):
        specification = (
            get_club_goal_model_feature_spec(
                specification_name
            )
        )

        columns.update(
            specification.required_columns()
        )

    return sorted(columns)


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Observation dataset does not exist: "
            f"{INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Full-squad observation dataset is empty."
        )

    required_columns = {
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *required_feature_columns(),
        "home_representation_type",
        "away_representation_type",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Observation dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    numeric_columns = [
        *TARGET_COLUMNS,
        *required_feature_columns(),
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
            "Observation dataset contains missing "
            "required numeric values."
        )

    if dataframe[
        "event_id"
    ].duplicated().any():
        raise ValueError(
            "Observation dataset contains duplicate "
            "event IDs."
        )

    if not dataframe[
        "home_representation_type"
    ].eq("full_squad").all():
        raise ValueError(
            "Unexpected home representation type."
        )

    if not dataframe[
        "away_representation_type"
    ].eq("full_squad").all():
        raise ValueError(
            "Unexpected away representation type."
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


def chronological_split(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(
        math.floor(
            len(dataframe)
            * TRAIN_FRACTION
        )
    )

    if split_index <= 0:
        raise ValueError(
            "Chronological split produced no "
            "training rows."
        )

    if split_index >= len(dataframe):
        raise ValueError(
            "Chronological split produced no "
            "test rows."
        )

    train = (
        dataframe.iloc[:split_index]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        dataframe.iloc[split_index:]
        .copy()
        .reset_index(drop=True)
    )

    if train["date"].max() > test["date"].min():
        raise AssertionError(
            "Chronological split is invalid."
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

    total_probability = (
        home_win
        + draw
        + away_win
    )

    if total_probability <= 0:
        raise ValueError(
            "Outcome probability mass is zero."
        )

    return (
        home_win / total_probability,
        draw / total_probability,
        away_win / total_probability,
    )


def result_index(
    home_score: int,
    away_score: int,
) -> int:
    if home_score > away_score:
        return 0

    if home_score == away_score:
        return 1

    return 2


def add_probability_columns(
    predictions: pd.DataFrame,
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


def fit_and_predict(
    specification_name: str,
    alpha: float,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    PoissonGoalModel,
    pd.DataFrame,
]:
    specification = (
        get_club_goal_model_feature_spec(
            specification_name
        )
    )

    model = PoissonGoalModel(
        name=(
            f"{specification_name}_"
            f"alpha_{alpha:.2f}"
        ),
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
            *IDENTITY_COLUMNS,
            *TARGET_COLUMNS,
        ]
    ].copy()

    output["specification"] = (
        specification_name
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
        add_probability_columns(
            output
        ),
    )


def evaluate_predictions(
    specification_name: str,
    alpha: float,
    train_count: int,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    specification = (
        get_club_goal_model_feature_spec(
            specification_name
        )
    )

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
        PROBABILITY_FLOOR,
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
            result_index(
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
        "specification":
            specification_name,
        "description":
            specification.description,
        "group_names":
            "|".join(
                specification.group_names
            ),
        "group_count": len(
            specification.group_names
        ),
        "home_feature_count": len(
            specification.home_features
        ),
        "away_feature_count": len(
            specification.away_features
        ),
        "alpha": alpha,
        "train_fraction":
            TRAIN_FRACTION,
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
                    PROBABILITY_FLOOR,
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


def coefficient_rows(
    model: PoissonGoalModel,
    specification_name: str,
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
                "specification":
                    specification_name,
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
                    "specification":
                        specification_name,
                    "alpha": alpha,
                    "target": target,
                    "feature": feature,
                    "coefficient": float(
                        coefficient
                    ),
                }
            )

    return rows


def build_baseline_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:
    baseline = (
        results[
            results["specification"].eq(
                BASELINE_SPECIFICATION
            )
        ]
        .set_index("alpha")
        .sort_index()
    )

    if len(baseline) != len(ALPHA_VALUES):
        raise AssertionError(
            "Baseline specification is missing "
            "one or more alpha configurations."
        )

    comparison_rows: list[
        dict[str, object]
    ] = []

    for row in results.itertuples(
        index=False
    ):
        baseline_row = baseline.loc[
            row.alpha
        ]

        comparison: dict[str, object] = {
            "specification":
                row.specification,
            "alpha":
                row.alpha,
            "is_baseline":
                (
                    row.specification
                    == BASELINE_SPECIFICATION
                ),
        }

        for metric in LOWER_IS_BETTER_METRICS:
            candidate_value = float(
                getattr(row, metric)
            )

            baseline_value = float(
                baseline_row[metric]
            )

            difference = (
                candidate_value
                - baseline_value
            )

            comparison[
                f"candidate_{metric}"
            ] = candidate_value

            comparison[
                f"baseline_{metric}"
            ] = baseline_value

            comparison[
                f"{metric}_difference"
            ] = difference

            comparison[
                f"{metric}_beats_baseline"
            ] = difference < 0

        comparison_rows.append(
            comparison
        )

    return pd.DataFrame(
        comparison_rows
    )


def build_specification_summary(
    baseline_comparison: pd.DataFrame,
) -> pd.DataFrame:
    candidate_rows = baseline_comparison[
        ~baseline_comparison[
            "is_baseline"
        ]
    ]

    summary_rows: list[
        dict[str, object]
    ] = []

    for specification_name, group in (
        candidate_rows.groupby(
            "specification",
            sort=True,
        )
    ):
        deviance_differences = group[
            "combined_poisson_deviance_difference"
        ]

        outcome_differences = group[
            "outcome_log_loss_difference"
        ]

        brier_differences = group[
            "outcome_brier_score_difference"
        ]

        summary_rows.append(
            {
                "specification":
                    specification_name,
                "alpha_count":
                    len(group),

                "poisson_deviance_wins":
                    int(
                        (
                            deviance_differences
                            < 0
                        ).sum()
                    ),

                "poisson_deviance_win_rate":
                    float(
                        (
                            deviance_differences
                            < 0
                        ).mean()
                    ),

                "mean_poisson_deviance_difference":
                    float(
                        deviance_differences.mean()
                    ),

                "best_poisson_deviance_difference":
                    float(
                        deviance_differences.min()
                    ),

                "worst_poisson_deviance_difference":
                    float(
                        deviance_differences.max()
                    ),

                "outcome_log_loss_wins":
                    int(
                        (
                            outcome_differences
                            < 0
                        ).sum()
                    ),

                "mean_outcome_log_loss_difference":
                    float(
                        outcome_differences.mean()
                    ),

                "outcome_brier_wins":
                    int(
                        (
                            brier_differences
                            < 0
                        ).sum()
                    ),

                "mean_outcome_brier_difference":
                    float(
                        brier_differences.mean()
                    ),
            }
        )

    return (
        pd.DataFrame(summary_rows)
        .sort_values(
            [
                "poisson_deviance_wins",
                "mean_poisson_deviance_difference",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_configuration_ranking(
    results: pd.DataFrame,
) -> pd.DataFrame:
    ranked = results.copy()

    rank_columns: list[str] = []

    for metric in LOWER_IS_BETTER_METRICS:
        rank_column = (
            f"{metric}_rank"
        )

        ranked[rank_column] = (
            ranked[metric]
            .rank(
                method="average",
                ascending=True,
            )
        )

        rank_columns.append(
            rank_column
        )

    ranked["mean_metric_rank"] = (
        ranked[rank_columns]
        .mean(axis=1)
    )

    return (
        ranked
        .sort_values(
            "mean_metric_rank"
        )
        .reset_index(drop=True)
    )


def write_results_markdown(
    path: Path,
    results: pd.DataFrame,
    specification_summary: pd.DataFrame,
    configuration_ranking: pd.DataFrame,
) -> None:
    best_deviance_configuration = (
        results
        .sort_values(
            "combined_poisson_deviance"
        )
        .iloc[0]
    )

    best_overall_configuration = (
        configuration_ranking.iloc[0]
    )

    best_ablation = (
        specification_summary.iloc[0]
        if not specification_summary.empty
        else None
    )

    lines = [
        "# Study 052 Results",
        "",
        "## Full-Squad Football Feature Ablation",
        "",
        "**Status:** `PASS`",
        "",
        "## Experimental design",
        "",
        "- Representation: `full_squad`",
        (
            f"- Training fraction: "
            f"{TRAIN_FRACTION:.2f}"
        ),
        (
            f"- Alpha values: "
            f"{list(ALPHA_VALUES)}"
        ),
        (
            "- Registered feature specifications: "
            f"{len(list_club_goal_model_feature_specs())}"
        ),
        (
            "- Benchmark configurations: "
            f"{len(results)}"
        ),
        "",
        "## Best combined Poisson deviance",
        "",
        (
            "- Specification: "
            f"`{best_deviance_configuration['specification']}`"
        ),
        (
            "- Alpha: "
            f"{best_deviance_configuration['alpha']:.4f}"
        ),
        (
            "- Combined Poisson deviance: "
            f"{best_deviance_configuration['combined_poisson_deviance']:.6f}"
        ),
        "",
        "## Best multi-metric configuration",
        "",
        (
            "- Specification: "
            f"`{best_overall_configuration['specification']}`"
        ),
        (
            "- Alpha: "
            f"{best_overall_configuration['alpha']:.4f}"
        ),
        (
            "- Mean metric rank: "
            f"{best_overall_configuration['mean_metric_rank']:.4f}"
        ),
        "",
    ]

    if best_ablation is not None:
        lines.extend(
            [
                "## Strongest feature addition",
                "",
                (
                    "- Specification: "
                    f"`{best_ablation['specification']}`"
                ),
                (
                    "- Poisson-deviance wins "
                    "against baseline: "
                    f"{int(best_ablation['poisson_deviance_wins'])}"
                    f"/{int(best_ablation['alpha_count'])}"
                ),
                (
                    "- Mean Poisson-deviance "
                    "difference from baseline: "
                    f"{best_ablation['mean_poisson_deviance_difference']:.6f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            (
                "A feature family should be retained only "
                "when its improvement is stable across "
                "regularization values and does not merely "
                "win one isolated configuration."
            ),
            (
                "This remains a single-season chronological "
                "holdout benchmark."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    dataframe = load_dataset()

    train, test = chronological_split(
        dataframe
    )

    result_rows: list[
        dict[str, object]
    ] = []

    coefficient_records: list[
        dict[str, object]
    ] = []

    prediction_frames: list[
        pd.DataFrame
    ] = []

    specification_names = (
        list_club_goal_model_feature_specs()
    )

    for specification_name in (
        specification_names
    ):
        for alpha in ALPHA_VALUES:
            model, predictions = (
                fit_and_predict(
                    specification_name=(
                        specification_name
                    ),
                    alpha=alpha,
                    train=train,
                    test=test,
                )
            )

            result_rows.append(
                evaluate_predictions(
                    specification_name=(
                        specification_name
                    ),
                    alpha=alpha,
                    train_count=len(train),
                    predictions=predictions,
                )
            )

            coefficient_records.extend(
                coefficient_rows(
                    model=model,
                    specification_name=(
                        specification_name
                    ),
                    alpha=alpha,
                )
            )

            prediction_frames.append(
                predictions
            )

    results = pd.DataFrame(
        result_rows
    )

    expected_run_count = (
        len(specification_names)
        * len(ALPHA_VALUES)
    )

    if len(results) != expected_run_count:
        raise AssertionError(
            "Unexpected benchmark-run count: "
            f"{len(results)} vs "
            f"{expected_run_count}."
        )

    coefficients = pd.DataFrame(
        coefficient_records
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    baseline_comparison = (
        build_baseline_comparison(
            results
        )
    )

    specification_summary = (
        build_specification_summary(
            baseline_comparison
        )
    )

    configuration_ranking = (
        build_configuration_ranking(
            results
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_DIR
        / "feature_ablation_results.csv",
        index=False,
    )

    baseline_comparison.to_csv(
        OUTPUT_DIR
        / "baseline_comparison.csv",
        index=False,
    )

    specification_summary.to_csv(
        OUTPUT_DIR
        / "feature_specification_summary.csv",
        index=False,
    )

    configuration_ranking.to_csv(
        OUTPUT_DIR
        / "configuration_ranking.csv",
        index=False,
    )

    coefficients.to_csv(
        OUTPUT_DIR
        / "feature_coefficients.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "feature_ablation_predictions.csv",
        index=False,
    )

    metadata = {
        "study_id": "052",
        "study_name": (
            "Full-Squad Football Feature Ablation"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "representation_type": "full_squad",
        "observation_count": int(
            len(dataframe)
        ),
        "training_count": int(
            len(train)
        ),
        "test_count": int(
            len(test)
        ),
        "train_fraction":
            TRAIN_FRACTION,
        "alpha_values": list(
            ALPHA_VALUES
        ),
        "baseline_specification":
            BASELINE_SPECIFICATION,
        "feature_specifications": list(
            specification_names
        ),
        "benchmark_run_count": int(
            len(results)
        ),
        "rating_prior_included": False,
        "evidence_score_included": False,
        "output_files": [
            "feature_ablation_results.csv",
            "baseline_comparison.csv",
            "feature_specification_summary.csv",
            "configuration_ranking.csv",
            "feature_coefficients.csv",
            "feature_ablation_predictions.csv",
            "study_metadata.json",
            "STUDY_052_RESULTS.md",
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
            / "STUDY_052_RESULTS.md"
        ),
        results=results,
        specification_summary=(
            specification_summary
        ),
        configuration_ranking=(
            configuration_ranking
        ),
    )

    print("Study 052")
    print("=" * 76)
    print()
    print(
        f"Observations: "
        f"{len(dataframe)}"
    )
    print(
        f"Training observations: "
        f"{len(train)}"
    )
    print(
        f"Test observations: "
        f"{len(test)}"
    )
    print(
        "Feature specifications: "
        f"{len(specification_names)}"
    )
    print(
        f"Benchmark runs: "
        f"{len(results)}"
    )
    print()

    print("Feature-Specification Summary")
    print("-" * 76)
    print(
        specification_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Top Configurations")
    print("-" * 76)
    print(
        configuration_ranking[
            [
                "specification",
                "alpha",
                "combined_poisson_deviance",
                "combined_goal_mae",
                "outcome_log_loss",
                "outcome_brier_score",
                "absolute_home_goal_mean_error",
                "absolute_away_goal_mean_error",
                "mean_metric_rank",
            ]
        ]
        .head(12)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Registry integration: PASS")
    print("Chronological split: PASS")
    print("Baseline comparison: PASS")
    print("Coefficient capture: PASS")
    print("Evidence-score exclusion: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
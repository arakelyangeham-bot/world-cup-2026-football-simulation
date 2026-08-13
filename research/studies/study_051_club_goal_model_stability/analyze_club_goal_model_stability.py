#analyze_club_goal_model_stability

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
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
    / "study_051_club_goal_model_stability"
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

OUTCOME_GRID_MAX_GOALS = 15
PROBABILITY_FLOOR = 1e-15


FEATURE_SETS = {
    "attack_defense": {
        "home": [
            "home_attack",
            "away_defense",
        ],
        "away": [
            "away_attack",
            "home_defense",
        ],
    },
    "core": {
        "home": [
            "home_attack",
            "away_defense",
            "midfield_diff",
            "defense_diff",
        ],
        "away": [
            "away_attack",
            "home_defense",
            "midfield_diff",
            "defense_diff",
        ],
    },
}


DATASET_CONFIGS = {
    "full_squad": {
        "path": FULL_SQUAD_PATH,
        "representation_type": "full_squad",
    },
    "expected_starting_xi": {
        "path": EXPECTED_XI_PATH,
        "representation_type": (
            "expected_starting_xi"
        ),
    },
}


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


def required_feature_columns() -> list[str]:
    columns: set[str] = set()

    for specification in FEATURE_SETS.values():
        columns.update(specification["home"])
        columns.update(specification["away"])

    return sorted(columns)


def load_dataset(
    path: Path,
    expected_representation_type: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset is empty: {path}"
        )

    required_columns = {
        *IDENTITY_COLUMNS,
        *TARGET_COLUMNS,
        *required_feature_columns(),
        "home_representation_type",
        "away_representation_type",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
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
            f"{path.name} contains missing "
            "required numeric values."
        )

    if dataframe[
        "event_id"
    ].duplicated().any():
        raise ValueError(
            f"{path.name} contains duplicate event IDs."
        )

    if not dataframe[
        "home_representation_type"
    ].eq(
        expected_representation_type
    ).all():
        raise ValueError(
            f"{path.name} contains unexpected "
            "home representation types."
        )

    if not dataframe[
        "away_representation_type"
    ].eq(
        expected_representation_type
    ).all():
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


def validate_matched_populations(
    datasets: dict[str, pd.DataFrame],
) -> None:
    labels = list(datasets)

    baseline = (
        datasets[labels[0]]
        .set_index("event_id")
        .sort_index()
    )

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
                f"{label}: event population mismatch."
            )

        comparison_columns = [
            column
            for column in [
                *IDENTITY_COLUMNS,
                *TARGET_COLUMNS,
            ]
            if column != "event_id"
        ]

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
                    f"{label}: mismatch in "
                    f"{column!r}."
                )


def chronological_split(
    dataframe: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(
        np.floor(
            len(dataframe)
            * train_fraction
        )
    )

    if split_index <= 0:
        raise ValueError(
            "Chronological split produced "
            "no training rows."
        )

    if split_index >= len(dataframe):
        raise ValueError(
            "Chronological split produced "
            "no test rows."
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

    if (
        train["date"].max()
        > test["date"].min()
    ):
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
        + goals * np.log(expected_goals)
        - math.lgamma(goals + 1)
    )

    return float(
        np.exp(log_probability)
    )


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

    total = home_win + draw + away_win

    return (
        home_win / total,
        draw / total,
        away_win / total,
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
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    home_probabilities: list[float] = []
    draw_probabilities: list[float] = []
    away_probabilities: list[float] = []
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
            home_probability,
            draw_probability,
            away_probability,
        ) = outcome_probabilities(
            lambda_home,
            lambda_away,
        )

        exact_probability = (
            poisson_probability(
                int(row.home_score),
                lambda_home,
            )
            * poisson_probability(
                int(row.away_score),
                lambda_away,
            )
        )

        home_probabilities.append(
            home_probability
        )
        draw_probabilities.append(
            draw_probability
        )
        away_probabilities.append(
            away_probability
        )
        exact_score_probabilities.append(
            exact_probability
        )

    output[
        "pred_home_win_probability"
    ] = home_probabilities

    output[
        "pred_draw_probability"
    ] = draw_probabilities

    output[
        "pred_away_win_probability"
    ] = away_probabilities

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
    observed: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        np.arange(len(observed)),
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


def fit_predict(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    home_features: list[str],
    away_features: list[str],
    alpha: float,
) -> tuple[
    PoissonGoalModel,
    pd.DataFrame,
]:
    model = PoissonGoalModel(
        name=name,
        home_features=home_features,
        away_features=away_features,
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
    representation_type: str,
    feature_set: str,
    train_fraction: float,
    alpha: float,
    train_count: int,
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

    home_deviance = (
        mean_poisson_deviance(
            actual_home,
            predicted_home,
        )
    )

    away_deviance = (
        mean_poisson_deviance(
            actual_away,
            predicted_away,
        )
    )

    probability_columns = [
        "pred_home_win_probability",
        "pred_draw_probability",
        "pred_away_win_probability",
    ]

    probabilities = (
        predictions[
            probability_columns
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
            actual_home == actual_away
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
        "representation_type":
            representation_type,
        "feature_set": feature_set,
        "train_fraction": train_fraction,
        "alpha": alpha,
        "train_matches": train_count,
        "test_matches": len(predictions),

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
        "outcome_brier_score": (
            multiclass_brier_score(
                observed_results,
                probabilities,
            )
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
    representation_type: str,
    feature_set: str,
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
                "representation_type":
                    representation_type,
                "feature_set": feature_set,
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
                    "representation_type":
                        representation_type,
                    "feature_set":
                        feature_set,
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


def build_pairwise_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = [
        "feature_set",
        "train_fraction",
        "alpha",
    ]

    full_squad = (
        results[
            results[
                "representation_type"
            ].eq("full_squad")
        ]
        .set_index(key_columns)
        .sort_index()
    )

    expected_xi = (
        results[
            results[
                "representation_type"
            ].eq(
                "expected_starting_xi"
            )
        ]
        .set_index(key_columns)
        .sort_index()
    )

    if not full_squad.index.equals(
        expected_xi.index
    ):
        raise AssertionError(
            "Representation benchmark grids "
            "do not match."
        )

    output = pd.DataFrame(
        index=full_squad.index
    ).reset_index()

    lower_is_better = [
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

    for metric in lower_is_better:
        output[
            f"full_squad_{metric}"
        ] = full_squad[
            metric
        ].to_numpy()

        output[
            f"expected_xi_{metric}"
        ] = expected_xi[
            metric
        ].to_numpy()

        difference = (
            expected_xi[metric]
            - full_squad[metric]
        ).to_numpy()

        output[
            f"{metric}_difference"
        ] = difference

        output[
            f"{metric}_winner"
        ] = np.where(
            difference < 0,
            "expected_starting_xi",
            np.where(
                difference > 0,
                "full_squad",
                "tie",
            ),
        )

    return output


def build_win_summary(
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    winner_columns = [
        column
        for column in pairwise.columns
        if column.endswith("_winner")
    ]

    rows: list[dict[str, object]] = []

    for column in winner_columns:
        metric = column.removesuffix(
            "_winner"
        )

        counts = (
            pairwise[column]
            .value_counts()
            .to_dict()
        )

        rows.append(
            {
                "metric": metric,
                "comparison_count":
                    len(pairwise),
                "full_squad_wins": int(
                    counts.get(
                        "full_squad",
                        0,
                    )
                ),
                "expected_xi_wins": int(
                    counts.get(
                        "expected_starting_xi",
                        0,
                    )
                ),
                "ties": int(
                    counts.get(
                        "tie",
                        0,
                    )
                ),
                "full_squad_win_rate": (
                    counts.get(
                        "full_squad",
                        0,
                    )
                    / len(pairwise)
                ),
                "expected_xi_win_rate": (
                    counts.get(
                        "expected_starting_xi",
                        0,
                    )
                    / len(pairwise)
                ),
            }
        )

    return pd.DataFrame(rows)


def build_configuration_ranking(
    results: pd.DataFrame,
) -> pd.DataFrame:
    ranking_metrics = [
        "combined_poisson_deviance",
        "combined_goal_mae",
        "outcome_log_loss",
        "outcome_brier_score",
        "exact_score_log_loss",
        "absolute_draw_rate_error",
        "absolute_home_goal_mean_error",
        "absolute_away_goal_mean_error",
    ]

    ranked = results.copy()

    rank_columns: list[str] = []

    for metric in ranking_metrics:
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
    win_summary: pd.DataFrame,
    configuration_ranking: pd.DataFrame,
) -> None:
    best_configuration = (
        configuration_ranking.iloc[0]
    )

    deviance_row = (
        win_summary[
            win_summary["metric"].eq(
                "combined_poisson_deviance"
            )
        ]
        .iloc[0]
    )

    lines = [
        "# Study 051 Results",
        "",
        (
            "## Club Goal-Model Stability and "
            "Home–Away Calibration"
        ),
        "",
        "**Status:** `PASS`",
        "",
        "## Experimental grid",
        "",
        (
            f"- Training fractions: "
            f"{list(TRAIN_FRACTIONS)}"
        ),
        (
            f"- Alpha values: "
            f"{list(ALPHA_VALUES)}"
        ),
        (
            f"- Feature sets: "
            f"{list(FEATURE_SETS)}"
        ),
        (
            "- Representation types: "
            "`full_squad`, "
            "`expected_starting_xi`"
        ),
        "",
        "## Poisson-deviance stability",
        "",
        (
            "- Full-squad wins: "
            f"{int(deviance_row['full_squad_wins'])}"
        ),
        (
            "- Expected-XI wins: "
            f"{int(deviance_row['expected_xi_wins'])}"
        ),
        (
            f"- Ties: "
            f"{int(deviance_row['ties'])}"
        ),
        "",
        "## Best overall configuration",
        "",
        (
            "- Representation: "
            f"`{best_configuration['representation_type']}`"
        ),
        (
            "- Feature set: "
            f"`{best_configuration['feature_set']}`"
        ),
        (
            "- Training fraction: "
            f"{best_configuration['train_fraction']:.2f}"
        ),
        (
            "- Alpha: "
            f"{best_configuration['alpha']:.4f}"
        ),
        (
            "- Mean metric rank: "
            f"{best_configuration['mean_metric_rank']:.4f}"
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "All evaluations use one Premier League "
            "season with alternative chronological "
            "cut points. These are not independent "
            "multi-season test sets."
        ),
        (
            "A representation should not be declared "
            "universally superior until the result "
            "survives later-season evaluation."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    datasets = {
        label: load_dataset(
            path=config["path"],
            expected_representation_type=(
                config[
                    "representation_type"
                ]
            ),
        )
        for label, config in (
            DATASET_CONFIGS.items()
        )
    }

    validate_matched_populations(
        datasets
    )

    result_rows: list[
        dict[str, object]
    ] = []

    coefficient_records: list[
        dict[str, object]
    ] = []

    for representation_type, dataframe in (
        datasets.items()
    ):
        for feature_set, specification in (
            FEATURE_SETS.items()
        ):
            for train_fraction in (
                TRAIN_FRACTIONS
            ):
                train, test = (
                    chronological_split(
                        dataframe,
                        train_fraction,
                    )
                )

                for alpha in ALPHA_VALUES:
                    model_name = (
                        f"{representation_type}_"
                        f"{feature_set}_"
                        f"train_{train_fraction:.2f}_"
                        f"alpha_{alpha:.2f}"
                    )

                    model, predictions = (
                        fit_predict(
                            name=model_name,
                            train=train,
                            test=test,
                            home_features=(
                                specification[
                                    "home"
                                ]
                            ),
                            away_features=(
                                specification[
                                    "away"
                                ]
                            ),
                            alpha=alpha,
                        )
                    )

                    result_rows.append(
                        evaluate_predictions(
                            representation_type=(
                                representation_type
                            ),
                            feature_set=(
                                feature_set
                            ),
                            train_fraction=(
                                train_fraction
                            ),
                            alpha=alpha,
                            train_count=len(train),
                            predictions=(
                                predictions
                            ),
                        )
                    )

                    coefficient_records.extend(
                        coefficient_rows(
                            model=model,
                            representation_type=(
                                representation_type
                            ),
                            feature_set=(
                                feature_set
                            ),
                            train_fraction=(
                                train_fraction
                            ),
                            alpha=alpha,
                        )
                    )

    results = pd.DataFrame(
        result_rows
    )

    expected_run_count = (
        len(DATASET_CONFIGS)
        * len(FEATURE_SETS)
        * len(TRAIN_FRACTIONS)
        * len(ALPHA_VALUES)
    )

    if len(results) != expected_run_count:
        raise AssertionError(
            "Unexpected benchmark run count: "
            f"{len(results)} vs "
            f"{expected_run_count}."
        )

    pairwise = build_pairwise_results(
        results
    )

    win_summary = build_win_summary(
        pairwise
    )

    coefficient_table = pd.DataFrame(
        coefficient_records
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
        / "stability_grid_results.csv",
        index=False,
    )

    pairwise.to_csv(
        OUTPUT_DIR
        / "paired_representation_results.csv",
        index=False,
    )

    win_summary.to_csv(
        OUTPUT_DIR
        / "representation_win_summary.csv",
        index=False,
    )

    coefficient_table.to_csv(
        OUTPUT_DIR
        / "coefficient_stability.csv",
        index=False,
    )

    configuration_ranking.to_csv(
        OUTPUT_DIR
        / "configuration_ranking.csv",
        index=False,
    )

    metadata = {
        "study_id": "051",
        "study_name": (
            "Club Goal-Model Stability and "
            "Home-Away Calibration"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "observation_count": int(
            len(
                datasets[
                    "full_squad"
                ]
            )
        ),
        "representation_types": list(
            DATASET_CONFIGS
        ),
        "feature_sets": FEATURE_SETS,
        "train_fractions": list(
            TRAIN_FRACTIONS
        ),
        "alpha_values": list(
            ALPHA_VALUES
        ),
        "benchmark_run_count": int(
            len(results)
        ),
        "paired_configuration_count": int(
            len(pairwise)
        ),
        "rating_prior_included": False,
        "output_files": [
            "stability_grid_results.csv",
            "paired_representation_results.csv",
            "representation_win_summary.csv",
            "coefficient_stability.csv",
            "configuration_ranking.csv",
            "study_metadata.json",
            "STUDY_051_RESULTS.md",
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
            / "STUDY_051_RESULTS.md"
        ),
        win_summary=win_summary,
        configuration_ranking=(
            configuration_ranking
        ),
    )

    print("Study 051")
    print("=" * 72)
    print()
    print(
        f"Matched observations: "
        f"{metadata['observation_count']}"
    )
    print(
        f"Benchmark runs: "
        f"{metadata['benchmark_run_count']}"
    )
    print(
        "Paired representation comparisons: "
        f"{metadata['paired_configuration_count']}"
    )
    print()

    print("Representation Win Summary")
    print("-" * 72)
    print(
        win_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Top Configurations")
    print("-" * 72)
    print(
        configuration_ranking[
            [
                "representation_type",
                "feature_set",
                "train_fraction",
                "alpha",
                "combined_poisson_deviance",
                "combined_goal_mae",
                "outcome_log_loss",
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
    print("Matched population: PASS")
    print("Chronological grid: PASS")
    print("Regularization grid: PASS")
    print("Feature-set comparison: PASS")
    print("Coefficient capture: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
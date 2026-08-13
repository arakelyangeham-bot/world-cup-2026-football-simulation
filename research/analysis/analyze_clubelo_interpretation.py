#analyze_clubelo_interpretation

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STUDY_060_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_060_clubelo_enriched_observations"
)

STUDY_061_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_061_clubelo_incremental_information"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_062_clubelo_interpretation"
)


OBSERVATION_PATH = (
    STUDY_060_DIRECTORY
    / "full_squad_observations_with_clubelo.csv"
)

PREDICTION_PATH = (
    STUDY_061_DIRECTORY
    / "predictions.csv"
)

COEFFICIENT_PATH = (
    STUDY_061_DIRECTORY
    / "rating_prior_coefficients.csv"
)

OVERLAP_REGRESSION_PATH = (
    STUDY_061_DIRECTORY
    / "clubelo_overlap_regression.csv"
)


MATCH_LEVEL_IMPROVEMENT_PATH = (
    OUTPUT_DIRECTORY
    / "match_level_improvement.csv"
)

RATING_BUCKET_ANALYSIS_PATH = (
    OUTPUT_DIRECTORY
    / "rating_bucket_analysis.csv"
)

FAVORITE_ANALYSIS_PATH = (
    OUTPUT_DIRECTORY
    / "favorite_analysis.csv"
)

RESULT_ANALYSIS_PATH = (
    OUTPUT_DIRECTORY
    / "result_type_analysis.csv"
)

TOTAL_GOAL_ANALYSIS_PATH = (
    OUTPUT_DIRECTORY
    / "total_goal_analysis.csv"
)

SCORELINE_ANALYSIS_PATH = (
    OUTPUT_DIRECTORY
    / "scoreline_analysis.csv"
)

CONTINUOUS_RELATIONSHIP_PATH = (
    OUTPUT_DIRECTORY
    / "rating_difference_relationship.csv"
)

COEFFICIENT_STABILITY_PATH = (
    OUTPUT_DIRECTORY
    / "coefficient_stability.csv"
)

COEFFICIENT_BY_SPLIT_PATH = (
    OUTPUT_DIRECTORY
    / "coefficient_stability_by_split.csv"
)

COEFFICIENT_BY_ALPHA_PATH = (
    OUTPUT_DIRECTORY
    / "coefficient_stability_by_alpha.csv"
)

OVERLAP_IMPORTANCE_PATH = (
    OUTPUT_DIRECTORY
    / "overlap_feature_importance.csv"
)

MATCH_RESIDUAL_PATH = (
    OUTPUT_DIRECTORY
    / "clubelo_match_residuals.csv"
)

CLUB_RESIDUAL_PATH = (
    OUTPUT_DIRECTORY
    / "clubelo_club_residuals.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "study_summary.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


PAIR_DEFINITIONS = (
    {
        "comparison_name":
            "rating_prior_increment_attack_defense",
        "baseline_specification":
            "attack_defense",
        "candidate_specification":
            "attack_defense_rating_prior",
    },
    {
        "comparison_name": (
            "rating_prior_increment_"
            "attack_defense_attack_depth"
        ),
        "baseline_specification":
            "attack_defense_attack_depth",
        "candidate_specification": (
            "attack_defense_attack_depth_"
            "rating_prior"
        ),
    },
)


IDENTITY_COLUMNS = (
    "event_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "dataset",
    "representation_type",
    "train_fraction",
    "alpha",
)


PLAYER_OVERLAP_FEATURES = (
    "attack_diff",
    "defense_diff",
    "attack_depth_diff",
)


def require_file(
    path: Path,
    label: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}"
        )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    require_file(
        OBSERVATION_PATH,
        "Enriched observation dataset",
    )

    require_file(
        PREDICTION_PATH,
        "Study 061 prediction dataset",
    )

    require_file(
        COEFFICIENT_PATH,
        "Study 061 rating coefficients",
    )

    require_file(
        OVERLAP_REGRESSION_PATH,
        "Study 061 overlap regression",
    )

    observations = pd.read_csv(
        OBSERVATION_PATH,
        low_memory=False,
    )

    predictions = pd.read_csv(
        PREDICTION_PATH,
        low_memory=False,
    )

    coefficients = pd.read_csv(
        COEFFICIENT_PATH,
        low_memory=False,
    )

    overlap_regression = pd.read_csv(
        OVERLAP_REGRESSION_PATH,
        low_memory=False,
    )

    if observations.empty:
        raise ValueError(
            "Enriched observation dataset is empty."
        )

    if predictions.empty:
        raise ValueError(
            "Prediction dataset is empty."
        )

    if coefficients.empty:
        raise ValueError(
            "Rating-prior coefficient dataset is empty."
        )

    if overlap_regression.empty:
        raise ValueError(
            "Overlap-regression dataset is empty."
        )

    observations["date"] = pd.to_datetime(
        observations["date"],
        errors="raise",
        utc=True,
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"],
        errors="raise",
        utc=True,
    )

    return (
        observations,
        predictions,
        coefficients,
        overlap_regression,
    )


def validate_inputs(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    overlap_regression: pd.DataFrame,
) -> None:
    observation_columns = {
        "event_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "rating_prior_diff",
        *PLAYER_OVERLAP_FEATURES,
    }

    prediction_columns = {
        *IDENTITY_COLUMNS,
        "feature_specification",
        "pred_home_goals",
        "pred_away_goals",
        "pred_home_win_probability",
        "pred_draw_probability",
        "pred_away_win_probability",
        "pred_exact_score_probability",
    }

    coefficient_columns = {
        "feature_specification",
        "train_fraction",
        "alpha",
        "target",
        "feature",
        "coefficient",
    }

    overlap_columns = {
        "target",
        "predictor",
        "raw_coefficient",
        "standardized_coefficient",
        "intercept",
        "r_squared",
        "adjusted_r_squared",
    }

    missing_observation_columns = (
        observation_columns
        - set(observations.columns)
    )

    missing_prediction_columns = (
        prediction_columns
        - set(predictions.columns)
    )

    missing_coefficient_columns = (
        coefficient_columns
        - set(coefficients.columns)
    )

    missing_overlap_columns = (
        overlap_columns
        - set(overlap_regression.columns)
    )

    if missing_observation_columns:
        raise ValueError(
            "Observation dataset is missing columns: "
            f"{sorted(missing_observation_columns)}"
        )

    if missing_prediction_columns:
        raise ValueError(
            "Prediction dataset is missing columns: "
            f"{sorted(missing_prediction_columns)}"
        )

    if missing_coefficient_columns:
        raise ValueError(
            "Coefficient dataset is missing columns: "
            f"{sorted(missing_coefficient_columns)}"
        )

    if missing_overlap_columns:
        raise ValueError(
            "Overlap-regression dataset is missing "
            f"columns: {sorted(missing_overlap_columns)}"
        )

    if observations["event_id"].duplicated().any():
        raise ValueError(
            "Observation dataset contains duplicate "
            "event IDs."
        )

    numeric_observation_columns = [
        "home_score",
        "away_score",
        "rating_prior_diff",
        *PLAYER_OVERLAP_FEATURES,
    ]

    numeric_prediction_columns = [
        "home_score",
        "away_score",
        "train_fraction",
        "alpha",
        "pred_home_goals",
        "pred_away_goals",
        "pred_home_win_probability",
        "pred_draw_probability",
        "pred_away_win_probability",
        "pred_exact_score_probability",
    ]

    for column in numeric_observation_columns:
        observations[column] = pd.to_numeric(
            observations[column],
            errors="raise",
        )

    for column in numeric_prediction_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="raise",
        )

    if observations[
        numeric_observation_columns
    ].isna().any().any():
        raise ValueError(
            "Observation analysis columns contain "
            "missing numeric values."
        )

    if predictions[
        numeric_prediction_columns
    ].isna().any().any():
        raise ValueError(
            "Prediction analysis columns contain "
            "missing numeric values."
        )

    expected_specs = {
        definition["baseline_specification"]
        for definition in PAIR_DEFINITIONS
    } | {
        definition["candidate_specification"]
        for definition in PAIR_DEFINITIONS
    }

    missing_specs = (
        expected_specs
        - set(
            predictions[
                "feature_specification"
            ].unique()
        )
    )

    if missing_specs:
        raise AssertionError(
            "Prediction dataset is missing expected "
            f"feature specifications: {sorted(missing_specs)}"
        )

    if not coefficients[
        "feature"
    ].eq("rating_prior_diff").all():
        raise AssertionError(
            "Coefficient dataset contains features other "
            "than rating_prior_diff."
        )

    if overlap_regression[
        "predictor"
    ].duplicated().any():
        raise AssertionError(
            "Overlap regression contains duplicate "
            "predictors."
        )


def observed_result(
    home_score: int,
    away_score: int,
) -> str:
    if home_score > away_score:
        return "home_win"

    if home_score < away_score:
        return "away_win"

    return "draw"


def observed_outcome_probability(
    row: pd.Series,
    prefix: str,
) -> float:
    result = observed_result(
        int(row["home_score"]),
        int(row["away_score"]),
    )

    if result == "home_win":
        return float(
            row[
                f"{prefix}_pred_home_win_probability"
            ]
        )

    if result == "away_win":
        return float(
            row[
                f"{prefix}_pred_away_win_probability"
            ]
        )

    return float(
        row[
            f"{prefix}_pred_draw_probability"
        ]
    )


def safe_negative_log(
    probability: float,
    floor: float = 1e-15,
) -> float:
    return float(
        -math.log(
            max(
                float(probability),
                floor,
            )
        )
    )


def build_match_level_improvement(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pair each player-only prediction with the corresponding
    ClubElo-enabled prediction at the same event, split,
    and alpha.

    Positive improvement values mean the ClubElo candidate
    produced a lower loss than the baseline.
    """
    observation_features = observations[
        [
            "event_id",
            "rating_prior_diff",
            *PLAYER_OVERLAP_FEATURES,
        ]
    ].copy()

    frames: list[pd.DataFrame] = []

    prediction_value_columns = [
        "pred_home_goals",
        "pred_away_goals",
        "pred_home_win_probability",
        "pred_draw_probability",
        "pred_away_win_probability",
        "pred_exact_score_probability",
    ]

    for definition in PAIR_DEFINITIONS:
        baseline_name = str(
            definition[
                "baseline_specification"
            ]
        )

        candidate_name = str(
            definition[
                "candidate_specification"
            ]
        )

        baseline = predictions[
            predictions[
                "feature_specification"
            ].eq(baseline_name)
        ][
            [
                *IDENTITY_COLUMNS,
                *prediction_value_columns,
            ]
        ].copy()

        candidate = predictions[
            predictions[
                "feature_specification"
            ].eq(candidate_name)
        ][
            [
                *IDENTITY_COLUMNS,
                *prediction_value_columns,
            ]
        ].copy()

        baseline = baseline.rename(
            columns={
                column: f"baseline_{column}"
                for column in prediction_value_columns
            }
        )

        candidate = candidate.rename(
            columns={
                column: f"candidate_{column}"
                for column in prediction_value_columns
            }
        )

        paired = baseline.merge(
            candidate,
            on=list(IDENTITY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )

        expected_count = len(baseline)

        if len(candidate) != expected_count:
            raise AssertionError(
                f"{definition['comparison_name']}: "
                "baseline and candidate prediction counts "
                "do not agree."
            )

        if len(paired) != expected_count:
            raise AssertionError(
                f"{definition['comparison_name']}: "
                "pairing did not preserve all predictions."
            )

        paired = paired.merge(
            observation_features,
            on="event_id",
            how="left",
            validate="many_to_one",
        )

        if paired[
            [
                "rating_prior_diff",
                *PLAYER_OVERLAP_FEATURES,
            ]
        ].isna().any().any():
            raise AssertionError(
                "Match-level analysis failed to attach "
                "observation features."
            )

        paired.insert(
            0,
            "comparison_name",
            definition["comparison_name"],
        )

        paired.insert(
            1,
            "baseline_specification",
            baseline_name,
        )

        paired.insert(
            2,
            "candidate_specification",
            candidate_name,
        )

        paired["absolute_rating_prior_diff"] = (
            paired["rating_prior_diff"].abs()
        )

        paired["actual_result"] = [
            observed_result(
                int(home_score),
                int(away_score),
            )
            for home_score, away_score in zip(
                paired["home_score"],
                paired["away_score"],
            )
        ]

        paired["actual_scoreline"] = (
            paired["home_score"]
            .astype(int)
            .astype(str)
            + "-"
            + paired["away_score"]
            .astype(int)
            .astype(str)
        )

        paired["actual_total_goals"] = (
            paired["home_score"]
            + paired["away_score"]
        )

        paired["baseline_combined_goal_error"] = (
            (
                (
                    paired["home_score"]
                    - paired[
                        "baseline_pred_home_goals"
                    ]
                ).abs()
                + (
                    paired["away_score"]
                    - paired[
                        "baseline_pred_away_goals"
                    ]
                ).abs()
            )
            / 2.0
        )

        paired["candidate_combined_goal_error"] = (
            (
                (
                    paired["home_score"]
                    - paired[
                        "candidate_pred_home_goals"
                    ]
                ).abs()
                + (
                    paired["away_score"]
                    - paired[
                        "candidate_pred_away_goals"
                    ]
                ).abs()
            )
            / 2.0
        )

        paired["combined_goal_error_improvement"] = (
            paired[
                "baseline_combined_goal_error"
            ]
            - paired[
                "candidate_combined_goal_error"
            ]
        )

        paired["baseline_total_goal_error"] = (
            (
                paired["actual_total_goals"]
                - (
                    paired[
                        "baseline_pred_home_goals"
                    ]
                    + paired[
                        "baseline_pred_away_goals"
                    ]
                )
            ).abs()
        )

        paired["candidate_total_goal_error"] = (
            (
                paired["actual_total_goals"]
                - (
                    paired[
                        "candidate_pred_home_goals"
                    ]
                    + paired[
                        "candidate_pred_away_goals"
                    ]
                )
            ).abs()
        )

        paired["total_goal_error_improvement"] = (
            paired[
                "baseline_total_goal_error"
            ]
            - paired[
                "candidate_total_goal_error"
            ]
        )

        paired[
            "baseline_goal_difference_error"
        ] = (
            (
                (
                    paired["home_score"]
                    - paired["away_score"]
                )
                - (
                    paired[
                        "baseline_pred_home_goals"
                    ]
                    - paired[
                        "baseline_pred_away_goals"
                    ]
                )
            ).abs()
        )

        paired[
            "candidate_goal_difference_error"
        ] = (
            (
                (
                    paired["home_score"]
                    - paired["away_score"]
                )
                - (
                    paired[
                        "candidate_pred_home_goals"
                    ]
                    - paired[
                        "candidate_pred_away_goals"
                    ]
                )
            ).abs()
        )

        paired[
            "goal_difference_error_improvement"
        ] = (
            paired[
                "baseline_goal_difference_error"
            ]
            - paired[
                "candidate_goal_difference_error"
            ]
        )

        paired[
            "baseline_outcome_probability"
        ] = paired.apply(
            observed_outcome_probability,
            axis=1,
            prefix="baseline",
        )

        paired[
            "candidate_outcome_probability"
        ] = paired.apply(
            observed_outcome_probability,
            axis=1,
            prefix="candidate",
        )

        paired["baseline_outcome_log_loss"] = (
            paired[
                "baseline_outcome_probability"
            ].map(safe_negative_log)
        )

        paired["candidate_outcome_log_loss"] = (
            paired[
                "candidate_outcome_probability"
            ].map(safe_negative_log)
        )

        paired["outcome_log_loss_improvement"] = (
            paired[
                "baseline_outcome_log_loss"
            ]
            - paired[
                "candidate_outcome_log_loss"
            ]
        )

        paired["baseline_exact_score_log_loss"] = (
            paired[
                "baseline_pred_exact_score_probability"
            ].map(safe_negative_log)
        )

        paired["candidate_exact_score_log_loss"] = (
            paired[
                "candidate_pred_exact_score_probability"
            ].map(safe_negative_log)
        )

        paired[
            "exact_score_log_loss_improvement"
        ] = (
            paired[
                "baseline_exact_score_log_loss"
            ]
            - paired[
                "candidate_exact_score_log_loss"
            ]
        )

        frames.append(paired)

    output = pd.concat(
        frames,
        ignore_index=True,
    )

    return output


def assign_rating_bucket(
    absolute_difference: pd.Series,
) -> pd.Series:
    return pd.cut(
        absolute_difference,
        bins=[
            -np.inf,
            50,
            100,
            150,
            200,
            300,
            np.inf,
        ],
        labels=[
            "0-50",
            "50-100",
            "100-150",
            "150-200",
            "200-300",
            "300+",
        ],
        right=False,
        ordered=True,
    )


def assign_favorite_category(
    rating_difference: pd.Series,
    even_threshold: float = 50.0,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                rating_difference > even_threshold,
                rating_difference < -even_threshold,
            ],
            [
                "home_favorite",
                "away_favorite",
            ],
            default="nearly_even",
        ),
        index=rating_difference.index,
    )


def assign_total_goal_band(
    total_goals: pd.Series,
) -> pd.Series:
    return pd.cut(
        total_goals,
        bins=[
            -np.inf,
            1,
            3,
            5,
            np.inf,
        ],
        labels=[
            "0_goals",
            "1-2_goals",
            "3-4_goals",
            "5+_goals",
        ],
        right=False,
        ordered=True,
    )


def summarize_improvements(
    dataframe: pd.DataFrame,
    group_columns: Iterable[str],
) -> pd.DataFrame:
    metric_columns = (
        "combined_goal_error_improvement",
        "total_goal_error_improvement",
        "goal_difference_error_improvement",
        "outcome_log_loss_improvement",
        "exact_score_log_loss_improvement",
    )

    records: list[dict[str, object]] = []

    grouped = dataframe.groupby(
        list(group_columns),
        observed=True,
        dropna=False,
        sort=False,
    )

    for group_key, group in grouped:
        if not isinstance(
            group_key,
            tuple,
        ):
            group_key = (group_key,)

        base_record = {
            column: value
            for column, value in zip(
                group_columns,
                group_key,
            )
        }

        for metric in metric_columns:
            values = group[
                metric
            ].to_numpy(dtype=float)

            record = dict(base_record)

            record.update(
                {
                    "metric": metric,
                    "observation_count": len(values),
                    "mean_improvement": float(
                        np.mean(values)
                    ),
                    "median_improvement": float(
                        np.median(values)
                    ),
                    "improvement_standard_deviation":
                        float(
                            np.std(
                                values,
                                ddof=1,
                            )
                        )
                        if len(values) > 1
                        else 0.0,
                    "win_count": int(
                        np.sum(values > 0.0)
                    ),
                    "tie_count": int(
                        np.sum(
                            np.isclose(
                                values,
                                0.0,
                                atol=1e-12,
                                rtol=0.0,
                            )
                        )
                    ),
                    "loss_count": int(
                        np.sum(values < 0.0)
                    ),
                    "win_rate": float(
                        np.mean(values > 0.0)
                    ),
                    "minimum_improvement": float(
                        np.min(values)
                    ),
                    "maximum_improvement": float(
                        np.max(values)
                    ),
                }
            )

            records.append(record)

    return pd.DataFrame(records)


def build_segment_analyses(
    match_level: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    analysis = match_level.copy()

    analysis["rating_bucket"] = (
        assign_rating_bucket(
            analysis[
                "absolute_rating_prior_diff"
            ]
        )
    )

    analysis["favorite_category"] = (
        assign_favorite_category(
            analysis["rating_prior_diff"]
        )
    )

    analysis["total_goal_band"] = (
        assign_total_goal_band(
            analysis["actual_total_goals"]
        )
    )

    rating_bucket = summarize_improvements(
        dataframe=analysis,
        group_columns=(
            "comparison_name",
            "rating_bucket",
        ),
    )

    favorite = summarize_improvements(
        dataframe=analysis,
        group_columns=(
            "comparison_name",
            "favorite_category",
        ),
    )

    result_type = summarize_improvements(
        dataframe=analysis,
        group_columns=(
            "comparison_name",
            "actual_result",
        ),
    )

    total_goal = summarize_improvements(
        dataframe=analysis,
        group_columns=(
            "comparison_name",
            "total_goal_band",
        ),
    )

    scoreline_counts = (
        analysis.groupby(
            [
                "comparison_name",
                "actual_scoreline",
            ]
        )
        .size()
        .rename("scoreline_count")
        .reset_index()
    )

    frequent_scorelines = scoreline_counts[
        scoreline_counts[
            "scoreline_count"
        ].ge(5)
    ][
        [
            "comparison_name",
            "actual_scoreline",
        ]
    ]

    scoreline_population = analysis.merge(
        frequent_scorelines,
        on=[
            "comparison_name",
            "actual_scoreline",
        ],
        how="inner",
        validate="many_to_one",
    )

    scoreline = summarize_improvements(
        dataframe=scoreline_population,
        group_columns=(
            "comparison_name",
            "actual_scoreline",
        ),
    )

    return (
        rating_bucket,
        favorite,
        result_type,
        total_goal,
        scoreline,
    )


def build_continuous_relationship(
    match_level: pd.DataFrame,
) -> pd.DataFrame:
    improvement_columns = (
        "combined_goal_error_improvement",
        "total_goal_error_improvement",
        "goal_difference_error_improvement",
        "outcome_log_loss_improvement",
        "exact_score_log_loss_improvement",
    )

    records: list[dict[str, object]] = []

    for comparison_name, group in (
        match_level.groupby(
            "comparison_name",
            sort=False,
        )
    ):
        for improvement_column in (
            improvement_columns
        ):
            pearson = group[
                "absolute_rating_prior_diff"
            ].corr(
                group[improvement_column],
                method="pearson",
            )

            spearman = group[
                "absolute_rating_prior_diff"
            ].corr(
                group[improvement_column],
                method="spearman",
            )

            records.append(
                {
                    "comparison_name":
                        comparison_name,
                    "rating_variable":
                        "absolute_rating_prior_diff",
                    "improvement_metric":
                        improvement_column,
                    "pearson_correlation":
                        float(pearson),
                    "spearman_correlation":
                        float(spearman),
                    "observation_count":
                        len(group),
                }
            )

    return pd.DataFrame(records)


def coefficient_summary_record(
    group: pd.DataFrame,
) -> dict[str, object]:
    values = group[
        "coefficient"
    ].to_numpy(dtype=float)

    return {
        "coefficient_count": len(values),
        "mean_coefficient": float(
            np.mean(values)
        ),
        "median_coefficient": float(
            np.median(values)
        ),
        "standard_deviation": float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(values) > 1
        else 0.0,
        "minimum_coefficient": float(
            np.min(values)
        ),
        "maximum_coefficient": float(
            np.max(values)
        ),
        "positive_count": int(
            np.sum(values > 0.0)
        ),
        "negative_count": int(
            np.sum(values < 0.0)
        ),
        "zero_count": int(
            np.sum(
                np.isclose(
                    values,
                    0.0,
                    atol=1e-15,
                    rtol=0.0,
                )
            )
        ),
        "sign_consistency_rate": float(
            max(
                np.mean(values > 0.0),
                np.mean(values < 0.0),
                np.mean(
                    np.isclose(
                        values,
                        0.0,
                        atol=1e-15,
                        rtol=0.0,
                    )
                ),
            )
        ),
        "coefficient_of_variation": (
            float(
                np.std(
                    values,
                    ddof=1,
                )
                / abs(
                    np.mean(values)
                )
            )
            if (
                len(values) > 1
                and not np.isclose(
                    np.mean(values),
                    0.0,
                    atol=1e-15,
                )
            )
            else np.nan
        ),
    }


def build_coefficient_stability(
    coefficients: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    overall_records: list[
        dict[str, object]
    ] = []

    for (
        feature_specification,
        target,
    ), group in coefficients.groupby(
        [
            "feature_specification",
            "target",
        ],
        sort=False,
    ):
        record = {
            "feature_specification":
                feature_specification,
            "target": target,
        }

        record.update(
            coefficient_summary_record(group)
        )

        overall_records.append(record)

    split_records: list[
        dict[str, object]
    ] = []

    for (
        feature_specification,
        target,
        train_fraction,
    ), group in coefficients.groupby(
        [
            "feature_specification",
            "target",
            "train_fraction",
        ],
        sort=False,
    ):
        record = {
            "feature_specification":
                feature_specification,
            "target": target,
            "train_fraction":
                train_fraction,
        }

        record.update(
            coefficient_summary_record(group)
        )

        split_records.append(record)

    alpha_records: list[
        dict[str, object]
    ] = []

    for (
        feature_specification,
        target,
        alpha,
    ), group in coefficients.groupby(
        [
            "feature_specification",
            "target",
            "alpha",
        ],
        sort=False,
    ):
        record = {
            "feature_specification":
                feature_specification,
            "target": target,
            "alpha": alpha,
        }

        record.update(
            coefficient_summary_record(group)
        )

        alpha_records.append(record)

    return (
        pd.DataFrame(overall_records),
        pd.DataFrame(split_records),
        pd.DataFrame(alpha_records),
    )


def build_overlap_feature_importance(
    overlap_regression: pd.DataFrame,
) -> pd.DataFrame:
    output = overlap_regression.copy()

    output[
        "absolute_standardized_coefficient"
    ] = output[
        "standardized_coefficient"
    ].abs()

    output[
        "standardized_importance_share"
    ] = (
        output[
            "absolute_standardized_coefficient"
        ]
        / output[
            "absolute_standardized_coefficient"
        ].sum()
    )

    output["importance_rank"] = (
        output[
            "absolute_standardized_coefficient"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    return (
        output
        .sort_values(
            [
                "importance_rank",
                "predictor",
            ]
        )
        .reset_index(drop=True)
    )


def validate_overlap_regression_contract(
    overlap_regression: pd.DataFrame,
) -> None:
    expected_predictors = set(
        PLAYER_OVERLAP_FEATURES
    )

    actual_predictors = set(
        overlap_regression[
            "predictor"
        ].astype(str)
    )

    if actual_predictors != expected_predictors:
        raise AssertionError(
            "Overlap-regression predictor population "
            "does not match the registered player "
            "features. "
            f"Expected {sorted(expected_predictors)}, "
            f"found {sorted(actual_predictors)}."
        )

    intercept_values = overlap_regression[
        "intercept"
    ].to_numpy(dtype=float)

    if not np.allclose(
        intercept_values,
        intercept_values[0],
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Overlap-regression rows disagree on the "
            "model intercept."
        )


def build_clubelo_residual_analysis(
    observations: pd.DataFrame,
    overlap_regression: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Reconstruct the Study 061 linear overlap model without
    fitting a new model.

    The match residual is:

        actual rating_prior_diff
        minus
        rating_prior_diff reconstructed from player features

    A positive residual means the home club's historical
    advantage was larger than the player-derived features
    implied. A negative residual favors the away club.
    """
    validate_overlap_regression_contract(
        overlap_regression
    )

    coefficient_lookup = {
        str(row.predictor):
            float(row.raw_coefficient)
        for row in overlap_regression.itertuples(
            index=False
        )
    }

    intercept = float(
        overlap_regression[
            "intercept"
        ].iloc[0]
    )

    match_residuals = observations[
        [
            "event_id",
            "date",
            "home_team",
            "away_team",
            "rating_prior_diff",
            *PLAYER_OVERLAP_FEATURES,
        ]
    ].copy()

    predicted = np.full(
        len(match_residuals),
        intercept,
        dtype=float,
    )

    for feature in PLAYER_OVERLAP_FEATURES:
        predicted += (
            match_residuals[
                feature
            ].to_numpy(dtype=float)
            * coefficient_lookup[feature]
        )

    match_residuals[
        "player_implied_rating_prior_diff"
    ] = predicted

    match_residuals[
        "rating_prior_residual"
    ] = (
        match_residuals[
            "rating_prior_diff"
        ]
        - match_residuals[
            "player_implied_rating_prior_diff"
        ]
    )

    match_residuals[
        "absolute_rating_prior_residual"
    ] = match_residuals[
        "rating_prior_residual"
    ].abs()

    home_view = match_residuals[
        [
            "event_id",
            "date",
            "home_team",
            "rating_prior_residual",
        ]
    ].rename(
        columns={
            "home_team": "club",
            "rating_prior_residual":
                "club_perspective_residual",
        }
    )

    home_view["venue_role"] = "home"

    away_view = match_residuals[
        [
            "event_id",
            "date",
            "away_team",
            "rating_prior_residual",
        ]
    ].rename(
        columns={
            "away_team": "club",
            "rating_prior_residual":
                "club_perspective_residual",
        }
    )

    away_view[
        "club_perspective_residual"
    ] = -away_view[
        "club_perspective_residual"
    ]

    away_view["venue_role"] = "away"

    club_rows = pd.concat(
        [
            home_view,
            away_view,
        ],
        ignore_index=True,
    )

    club_records: list[
        dict[str, object]
    ] = []

    for club, group in club_rows.groupby(
        "club",
        sort=True,
    ):
        values = group[
            "club_perspective_residual"
        ].to_numpy(dtype=float)

        club_records.append(
            {
                "club": club,
                "match_count": len(values),
                "mean_signed_residual": float(
                    np.mean(values)
                ),
                "median_signed_residual": float(
                    np.median(values)
                ),
                "mean_absolute_residual": float(
                    np.mean(
                        np.abs(values)
                    )
                ),
                "residual_standard_deviation":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    )
                    if len(values) > 1
                    else 0.0,
                "positive_residual_count": int(
                    np.sum(values > 0.0)
                ),
                "negative_residual_count": int(
                    np.sum(values < 0.0)
                ),
            }
        )

    club_residuals = pd.DataFrame(
        club_records
    )

    club_residuals[
        "absolute_mean_signed_residual"
    ] = club_residuals[
        "mean_signed_residual"
    ].abs()

    club_residuals = (
        club_residuals
        .sort_values(
            [
                "absolute_mean_signed_residual",
                "mean_absolute_residual",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        match_residuals,
        club_residuals,
    )


def select_metric_rows(
    dataframe: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    selected = dataframe[
        dataframe["metric"].eq(metric)
    ].copy()

    if selected.empty:
        raise AssertionError(
            f"No analysis rows exist for {metric!r}."
        )

    return selected


def build_summary(
    match_level: pd.DataFrame,
    rating_bucket_analysis: pd.DataFrame,
    favorite_analysis: pd.DataFrame,
    continuous_relationship: pd.DataFrame,
    coefficient_stability: pd.DataFrame,
    overlap_importance: pd.DataFrame,
    match_residuals: pd.DataFrame,
    club_residuals: pd.DataFrame,
) -> dict[str, object]:
    exact_bucket = select_metric_rows(
        rating_bucket_analysis,
        "exact_score_log_loss_improvement",
    )

    strongest_bucket_row = (
        exact_bucket
        .sort_values(
            "mean_improvement",
            ascending=False,
        )
        .iloc[0]
    )

    outcome_favorite = select_metric_rows(
        favorite_analysis,
        "outcome_log_loss_improvement",
    )

    strongest_favorite_row = (
        outcome_favorite
        .sort_values(
            "mean_improvement",
            ascending=False,
        )
        .iloc[0]
    )

    continuous_exact = (
        continuous_relationship[
            continuous_relationship[
                "improvement_metric"
            ].eq(
                "exact_score_log_loss_improvement"
            )
        ]
    )

    strongest_continuous_row = (
        continuous_exact.assign(
            absolute_spearman=(
                continuous_exact[
                    "spearman_correlation"
                ].abs()
            )
        )
        .sort_values(
            "absolute_spearman",
            ascending=False,
        )
        .iloc[0]
    )

    strongest_overlap_row = (
        overlap_importance.iloc[0]
    )

    largest_positive_club = (
        club_residuals
        .sort_values(
            "mean_signed_residual",
            ascending=False,
        )
        .iloc[0]
    )

    largest_negative_club = (
        club_residuals
        .sort_values(
            "mean_signed_residual",
            ascending=True,
        )
        .iloc[0]
    )

    return {
        "study":
            "study_062_clubelo_interpretation",
        "source_prediction_rows": int(
            len(match_level)
        ),
        "comparisons": int(
            match_level[
                "comparison_name"
            ].nunique()
        ),
        "unique_events": int(
            match_level["event_id"].nunique()
        ),
        "strongest_exact_score_rating_bucket": {
            "comparison_name": str(
                strongest_bucket_row[
                    "comparison_name"
                ]
            ),
            "rating_bucket": str(
                strongest_bucket_row[
                    "rating_bucket"
                ]
            ),
            "mean_improvement": float(
                strongest_bucket_row[
                    "mean_improvement"
                ]
            ),
            "win_rate": float(
                strongest_bucket_row[
                    "win_rate"
                ]
            ),
        },
        "strongest_outcome_favorite_category": {
            "comparison_name": str(
                strongest_favorite_row[
                    "comparison_name"
                ]
            ),
            "favorite_category": str(
                strongest_favorite_row[
                    "favorite_category"
                ]
            ),
            "mean_improvement": float(
                strongest_favorite_row[
                    "mean_improvement"
                ]
            ),
            "win_rate": float(
                strongest_favorite_row[
                    "win_rate"
                ]
            ),
        },
        "strongest_rating_difference_relationship": {
            "comparison_name": str(
                strongest_continuous_row[
                    "comparison_name"
                ]
            ),
            "improvement_metric": str(
                strongest_continuous_row[
                    "improvement_metric"
                ]
            ),
            "pearson_correlation": float(
                strongest_continuous_row[
                    "pearson_correlation"
                ]
            ),
            "spearman_correlation": float(
                strongest_continuous_row[
                    "spearman_correlation"
                ]
            ),
        },
        "strongest_overlap_feature": {
            "predictor": str(
                strongest_overlap_row[
                    "predictor"
                ]
            ),
            "standardized_coefficient": float(
                strongest_overlap_row[
                    "standardized_coefficient"
                ]
            ),
            "importance_share": float(
                strongest_overlap_row[
                    "standardized_importance_share"
                ]
            ),
        },
        "coefficient_groups": int(
            len(coefficient_stability)
        ),
        "coefficient_sign_consistency_pass": bool(
            coefficient_stability[
                "sign_consistency_rate"
            ].eq(1.0).all()
        ),
        "match_residual_standard_deviation": float(
            match_residuals[
                "rating_prior_residual"
            ].std(ddof=1)
        ),
        "largest_positive_club_residual": {
            "club": str(
                largest_positive_club["club"]
            ),
            "mean_signed_residual": float(
                largest_positive_club[
                    "mean_signed_residual"
                ]
            ),
            "match_count": int(
                largest_positive_club[
                    "match_count"
                ]
            ),
        },
        "largest_negative_club_residual": {
            "club": str(
                largest_negative_club["club"]
            ),
            "mean_signed_residual": float(
                largest_negative_club[
                    "mean_signed_residual"
                ]
            ),
            "match_count": int(
                largest_negative_club[
                    "match_count"
                ]
            ),
        },
    }


def write_report(
    summary: dict[str, object],
    rating_bucket_analysis: pd.DataFrame,
    favorite_analysis: pd.DataFrame,
    coefficient_stability: pd.DataFrame,
    overlap_importance: pd.DataFrame,
    club_residuals: pd.DataFrame,
) -> None:
    strongest_bucket = summary[
        "strongest_exact_score_rating_bucket"
    ]

    strongest_favorite = summary[
        "strongest_outcome_favorite_category"
    ]

    strongest_relationship = summary[
        "strongest_rating_difference_relationship"
    ]

    strongest_overlap = summary[
        "strongest_overlap_feature"
    ]

    positive_residual = summary[
        "largest_positive_club_residual"
    ]

    negative_residual = summary[
        "largest_negative_club_residual"
    ]

    bucket_display = (
        select_metric_rows(
            rating_bucket_analysis,
            "exact_score_log_loss_improvement",
        )[
            [
                "comparison_name",
                "rating_bucket",
                "observation_count",
                "mean_improvement",
                "win_rate",
            ]
        ]
        .sort_values(
            [
                "comparison_name",
                "rating_bucket",
            ]
        )
        .to_markdown(index=False)
    )

    favorite_display = (
        select_metric_rows(
            favorite_analysis,
            "outcome_log_loss_improvement",
        )[
            [
                "comparison_name",
                "favorite_category",
                "observation_count",
                "mean_improvement",
                "win_rate",
            ]
        ]
        .sort_values(
            [
                "comparison_name",
                "favorite_category",
            ]
        )
        .to_markdown(index=False)
    )

    overlap_display = overlap_importance[
        [
            "importance_rank",
            "predictor",
            "standardized_coefficient",
            "absolute_standardized_coefficient",
            "standardized_importance_share",
        ]
    ].to_markdown(index=False)

    coefficient_display = (
        coefficient_stability[
            [
                "feature_specification",
                "target",
                "coefficient_count",
                "mean_coefficient",
                "standard_deviation",
                "minimum_coefficient",
                "maximum_coefficient",
                "sign_consistency_rate",
            ]
        ]
        .to_markdown(index=False)
    )

    residual_display = (
        club_residuals[
            [
                "club",
                "match_count",
                "mean_signed_residual",
                "mean_absolute_residual",
            ]
        ]
        .head(10)
        .to_markdown(index=False)
    )

    report = f"""# Study 062 — ClubElo Interpretation and Robustness

## Purpose

Interpret the predictive contribution identified in Study 061
without fitting any new goal models.

Study 062 asks where ClubElo helped, whether its value changed
with matchup imbalance, whether its coefficients remained
stable, and which portions of ClubElo were not reconstructed
by the player-derived representation.

## Inputs

- Study 060 enriched observations
- Study 061 match predictions
- Study 061 rating-prior coefficients
- Study 061 overlap-regression coefficients

## Analysis population

- Match-level paired rows: {summary["source_prediction_rows"]}
- Unique events: {summary["unique_events"]}
- Controlled comparisons: {summary["comparisons"]}

Each row compares a ClubElo-enabled model against its
corresponding player-only model at the same event, chronological
split, and regularization value.

Positive improvement values mean that the ClubElo model
produced the lower loss or error.

## Rating-difference buckets

Exact-score log-loss improvement by absolute ClubElo difference:

{bucket_display}

The largest mean bucket-level improvement occurred in
`{strongest_bucket["rating_bucket"]}` for
`{strongest_bucket["comparison_name"]}`:

- Mean improvement:
  {strongest_bucket["mean_improvement"]:.8f}
- Match-level win rate:
  {strongest_bucket["win_rate"]:.1%}

## Favorite classification

Outcome-log-loss improvement by pre-match favorite category:

{favorite_display}

The strongest favorite-category result occurred for
`{strongest_favorite["favorite_category"]}` in
`{strongest_favorite["comparison_name"]}`:

- Mean improvement:
  {strongest_favorite["mean_improvement"]:.8f}
- Match-level win rate:
  {strongest_favorite["win_rate"]:.1%}

## Continuous matchup imbalance

The strongest relationship between absolute ClubElo difference
and exact-score improvement occurred for
`{strongest_relationship["comparison_name"]}`:

- Pearson correlation:
  {strongest_relationship["pearson_correlation"]:.6f}
- Spearman correlation:
  {strongest_relationship["spearman_correlation"]:.6f}

A positive relationship means ClubElo tended to help more as
the historical rating gap increased. A weak relationship means
its value was distributed more broadly across matchup types.

## Overlap feature importance

The Study 061 overlap model was not refitted. Its stored
standardized coefficients were ranked directly:

{overlap_display}

The strongest standardized overlap feature was
`{strongest_overlap["predictor"]}`:

- Standardized coefficient:
  {strongest_overlap["standardized_coefficient"]:.6f}
- Absolute-coefficient share:
  {strongest_overlap["importance_share"]:.1%}

These shares describe relative coefficient magnitude, not
causal importance.

## ClubElo coefficient stability

{coefficient_display}

Complete sign consistency across all groups:
**{str(summary["coefficient_sign_consistency_pass"]).upper()}**

The home- and away-goal coefficients should be interpreted
separately because the same positive rating difference has
opposite football implications for the two scoring targets.

## Residual ClubElo information

The stored Study 061 overlap regression was used to reconstruct:

`rating_prior_diff`

from:

- `attack_diff`
- `defense_diff`
- `attack_depth_diff`

No new regression was fitted.

The residual equals:

`actual ClubElo difference - player-implied ClubElo difference`

The residual standard deviation was:

{summary["match_residual_standard_deviation"]:.6f} ClubElo points.

The largest positive average club-perspective residual was:

- Club: `{positive_residual["club"]}`
- Mean signed residual:
  {positive_residual["mean_signed_residual"]:.6f}
- Matches:
  {positive_residual["match_count"]}

The largest negative average club-perspective residual was:

- Club: `{negative_residual["club"]}`
- Mean signed residual:
  {negative_residual["mean_signed_residual"]:.6f}
- Matches:
  {negative_residual["match_count"]}

Largest club-level absolute average residuals:

{residual_display}

Club residuals are descriptive and schedule-dependent. They
should not be interpreted as direct measures of coaching,
chemistry, or organizational quality without further evidence.

## Outputs

- `match_level_improvement.csv`
- `rating_bucket_analysis.csv`
- `favorite_analysis.csv`
- `result_type_analysis.csv`
- `total_goal_analysis.csv`
- `scoreline_analysis.csv`
- `rating_difference_relationship.csv`
- `coefficient_stability.csv`
- `coefficient_stability_by_split.csv`
- `coefficient_stability_by_alpha.csv`
- `overlap_feature_importance.csv`
- `clubelo_match_residuals.csv`
- `clubelo_club_residuals.csv`
- `study_summary.json`

## Validation

- Study 060 observation loading: PASS
- Study 061 prediction loading: PASS
- Controlled prediction pairing: PASS
- Match-population enrichment: PASS
- Segment analysis: PASS
- Coefficient-stability analysis: PASS
- Stored overlap-model reconstruction: PASS
- Residual analysis: PASS
- No new goal-model fitting: PASS
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    (
        observations,
        predictions,
        coefficients,
        overlap_regression,
    ) = load_inputs()

    validate_inputs(
        observations=observations,
        predictions=predictions,
        coefficients=coefficients,
        overlap_regression=overlap_regression,
    )

    match_level = (
        build_match_level_improvement(
            observations=observations,
            predictions=predictions,
        )
    )

    (
        rating_bucket_analysis,
        favorite_analysis,
        result_type_analysis,
        total_goal_analysis,
        scoreline_analysis,
    ) = build_segment_analyses(
        match_level
    )

    continuous_relationship = (
        build_continuous_relationship(
            match_level
        )
    )

    (
        coefficient_stability,
        coefficient_by_split,
        coefficient_by_alpha,
    ) = build_coefficient_stability(
        coefficients
    )

    overlap_importance = (
        build_overlap_feature_importance(
            overlap_regression
        )
    )

    (
        match_residuals,
        club_residuals,
    ) = build_clubelo_residual_analysis(
        observations=observations,
        overlap_regression=overlap_regression,
    )

    summary = build_summary(
        match_level=match_level,
        rating_bucket_analysis=(
            rating_bucket_analysis
        ),
        favorite_analysis=favorite_analysis,
        continuous_relationship=(
            continuous_relationship
        ),
        coefficient_stability=(
            coefficient_stability
        ),
        overlap_importance=(
            overlap_importance
        ),
        match_residuals=match_residuals,
        club_residuals=club_residuals,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    match_level.to_csv(
        MATCH_LEVEL_IMPROVEMENT_PATH,
        index=False,
    )

    rating_bucket_analysis.to_csv(
        RATING_BUCKET_ANALYSIS_PATH,
        index=False,
    )

    favorite_analysis.to_csv(
        FAVORITE_ANALYSIS_PATH,
        index=False,
    )

    result_type_analysis.to_csv(
        RESULT_ANALYSIS_PATH,
        index=False,
    )

    total_goal_analysis.to_csv(
        TOTAL_GOAL_ANALYSIS_PATH,
        index=False,
    )

    scoreline_analysis.to_csv(
        SCORELINE_ANALYSIS_PATH,
        index=False,
    )

    continuous_relationship.to_csv(
        CONTINUOUS_RELATIONSHIP_PATH,
        index=False,
    )

    coefficient_stability.to_csv(
        COEFFICIENT_STABILITY_PATH,
        index=False,
    )

    coefficient_by_split.to_csv(
        COEFFICIENT_BY_SPLIT_PATH,
        index=False,
    )

    coefficient_by_alpha.to_csv(
        COEFFICIENT_BY_ALPHA_PATH,
        index=False,
    )

    overlap_importance.to_csv(
        OVERLAP_IMPORTANCE_PATH,
        index=False,
    )

    match_residuals.to_csv(
        MATCH_RESIDUAL_PATH,
        index=False,
    )

    club_residuals.to_csv(
        CLUB_RESIDUAL_PATH,
        index=False,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        summary=summary,
        rating_bucket_analysis=(
            rating_bucket_analysis
        ),
        favorite_analysis=favorite_analysis,
        coefficient_stability=(
            coefficient_stability
        ),
        overlap_importance=(
            overlap_importance
        ),
        club_residuals=club_residuals,
    )

    print(
        "Study 062 — ClubElo Interpretation "
        "and Robustness"
    )
    print("=" * 76)
    print()
    print(
        "Match-level paired rows: "
        f"{len(match_level)}"
    )
    print(
        "Unique events: "
        f"{match_level['event_id'].nunique()}"
    )
    print(
        "Controlled comparisons: "
        f"{match_level['comparison_name'].nunique()}"
    )
    print()

    print("Rating-Bucket Interpretation")
    print("-" * 76)

    bucket_display = (
        rating_bucket_analysis[
            rating_bucket_analysis[
                "metric"
            ].eq(
                "exact_score_log_loss_improvement"
            )
        ][
            [
                "comparison_name",
                "rating_bucket",
                "observation_count",
                "mean_improvement",
                "win_rate",
            ]
        ]
    )

    print(
        bucket_display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Favorite Interpretation")
    print("-" * 76)

    favorite_display = (
        favorite_analysis[
            favorite_analysis[
                "metric"
            ].eq(
                "outcome_log_loss_improvement"
            )
        ][
            [
                "comparison_name",
                "favorite_category",
                "observation_count",
                "mean_improvement",
                "win_rate",
            ]
        ]
    )

    print(
        favorite_display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Overlap Feature Importance")
    print("-" * 76)

    print(
        overlap_importance[
            [
                "importance_rank",
                "predictor",
                "standardized_coefficient",
                "standardized_importance_share",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Coefficient Stability")
    print("-" * 76)

    print(
        coefficient_stability[
            [
                "feature_specification",
                "target",
                "mean_coefficient",
                "standard_deviation",
                "sign_consistency_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8f}"
            ),
        )
    )

    print()
    print("Largest Club Residuals")
    print("-" * 76)

    print(
        club_residuals[
            [
                "club",
                "match_count",
                "mean_signed_residual",
                "mean_absolute_residual",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Controlled prediction pairing: PASS")
    print("Rating-bucket analysis: PASS")
    print("Favorite analysis: PASS")
    print("Result and scoreline analysis: PASS")
    print("Continuous relationship analysis: PASS")
    print("Coefficient stability: PASS")
    print("Overlap feature ranking: PASS")
    print("Stored regression reconstruction: PASS")
    print("Residual analysis: PASS")
    print("No new goal-model fitting: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
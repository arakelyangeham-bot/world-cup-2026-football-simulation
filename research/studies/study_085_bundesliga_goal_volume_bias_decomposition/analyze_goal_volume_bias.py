#analyze_goal_volume_bias

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

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
    / "study_085_bundesliga_goal_volume_bias_decomposition"
)

OVERALL_BIAS_PATH = (
    OUTPUT_DIRECTORY
    / "overall_bias_decomposition.csv"
)

PREDICTED_TOTAL_BINS_PATH = (
    OUTPUT_DIRECTORY
    / "predicted_total_goal_bins.csv"
)

RATING_DIFFERENCE_BINS_PATH = (
    OUTPUT_DIRECTORY
    / "rating_difference_bins.csv"
)

MATCHUP_BALANCE_BINS_PATH = (
    OUTPUT_DIRECTORY
    / "matchup_balance_bins.csv"
)

MONTHLY_BIAS_PATH = (
    OUTPUT_DIRECTORY
    / "monthly_bias_summary.csv"
)

SEASON_PHASE_PATH = (
    OUTPUT_DIRECTORY
    / "season_phase_bias_summary.csv"
)

CLUB_ATTACK_BIAS_PATH = (
    OUTPUT_DIRECTORY
    / "club_attack_bias.csv"
)

CLUB_DEFENSIVE_BIAS_PATH = (
    OUTPUT_DIRECTORY
    / "club_defensive_bias.csv"
)

CLUB_GOAL_DIFFERENCE_BIAS_PATH = (
    OUTPUT_DIRECTORY
    / "club_goal_difference_bias.csv"
)

HOME_AWAY_BIAS_PATH = (
    OUTPUT_DIRECTORY
    / "home_away_bias_summary.csv"
)

ELITE_TEAM_BIAS_PATH = (
    OUTPUT_DIRECTORY
    / "elite_team_bias_summary.csv"
)

BIAS_CONCENTRATION_PATH = (
    OUTPUT_DIRECTORY
    / "bias_concentration_summary.csv"
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
BIN_COUNT = 4
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

    missing = REQUIRED_COLUMNS - set(
        dataframe.columns
    )

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

    if not dataframe[
        "prediction_status"
    ].eq("PASS").all():
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

    numeric_values = dataframe[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Input artifact contains non-finite required values."
        )

    if (
        dataframe["lambda_home"].le(0.0).any()
        or dataframe["lambda_away"].le(0.0).any()
    ):
        raise ValueError(
            "Input artifact contains non-positive lambdas."
        )

    expected_total = (
        dataframe["lambda_home"]
        + dataframe["lambda_away"]
    )

    if not np.allclose(
        dataframe["pred_total_goals"],
        expected_total,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError(
            "Predicted total goals are inconsistent with lambdas."
        )

    expected_goal_difference = (
        dataframe["lambda_home"]
        - dataframe["lambda_away"]
    )

    if not np.allclose(
        dataframe["pred_goal_diff"],
        expected_goal_difference,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError(
            "Predicted goal differences are inconsistent."
        )

    dataframe["home_goal_bias"] = (
        dataframe["lambda_home"]
        - dataframe["home_score"]
    )

    dataframe["away_goal_bias"] = (
        dataframe["lambda_away"]
        - dataframe["away_score"]
    )

    dataframe["total_goal_bias"] = (
        dataframe["pred_total_goals"]
        - dataframe["total_goals"]
    )

    dataframe["goal_difference_bias"] = (
        dataframe["pred_goal_diff"]
        - dataframe["goal_difference"]
    )

    dataframe["absolute_total_goal_error"] = (
        dataframe["total_goal_bias"].abs()
    )

    dataframe["absolute_goal_difference_error"] = (
        dataframe["goal_difference_bias"].abs()
    )

    dataframe["absolute_rating_difference"] = (
        dataframe["rating_prior_diff"].abs()
    )

    dataframe["absolute_predicted_goal_difference"] = (
        dataframe["pred_goal_diff"].abs()
    )

    dataframe["aggregate_goal_shortfall"] = (
        dataframe["total_goals"]
        - dataframe["pred_total_goals"]
    )

    return (
        dataframe
        .sort_values(
            ["date", "event_id"]
        )
        .reset_index(drop=True)
    )


def poisson_deviance(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> float:
    actual_values = np.asarray(
        actual,
        dtype=float,
    )

    predicted_values = np.asarray(
        predicted,
        dtype=float,
    )

    if (
        (actual_values < 0.0).any()
        or (predicted_values <= 0.0).any()
    ):
        raise ValueError(
            "Poisson deviance requires non-negative observations "
            "and positive predictions."
        )

    log_term = np.zeros_like(
        actual_values,
        dtype=float,
    )

    positive = actual_values > 0.0

    log_term[positive] = (
        actual_values[positive]
        * np.log(
            actual_values[positive]
            / predicted_values[positive]
        )
    )

    deviance = 2.0 * (
        log_term
        - (
            actual_values
            - predicted_values
        )
    )

    return float(
        np.mean(deviance)
    )


def safe_share(
    numerator: float,
    denominator: float,
) -> float:
    if math.isclose(
        denominator,
        0.0,
        abs_tol=1e-12,
    ):
        return float("nan")

    return float(
        numerator / denominator
    )


def build_overall_bias(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    actual_home = float(
        dataframe["home_score"].mean()
    )

    predicted_home = float(
        dataframe["lambda_home"].mean()
    )

    actual_away = float(
        dataframe["away_score"].mean()
    )

    predicted_away = float(
        dataframe["lambda_away"].mean()
    )

    actual_total = float(
        dataframe["total_goals"].mean()
    )

    predicted_total = float(
        dataframe["pred_total_goals"].mean()
    )

    actual_goal_difference = float(
        dataframe["goal_difference"].mean()
    )

    predicted_goal_difference = float(
        dataframe["pred_goal_diff"].mean()
    )

    records = [
        {
            "target": "home_goals",
            "match_count": len(dataframe),
            "mean_actual": actual_home,
            "mean_predicted": predicted_home,
            "mean_bias_predicted_minus_actual": (
                predicted_home - actual_home
            ),
            "mean_shortfall_actual_minus_predicted": (
                actual_home - predicted_home
            ),
            "mae": float(
                np.mean(
                    np.abs(
                        dataframe["lambda_home"]
                        - dataframe["home_score"]
                    )
                )
            ),
            "poisson_deviance": poisson_deviance(
                dataframe["home_score"],
                dataframe["lambda_home"],
            ),
        },
        {
            "target": "away_goals",
            "match_count": len(dataframe),
            "mean_actual": actual_away,
            "mean_predicted": predicted_away,
            "mean_bias_predicted_minus_actual": (
                predicted_away - actual_away
            ),
            "mean_shortfall_actual_minus_predicted": (
                actual_away - predicted_away
            ),
            "mae": float(
                np.mean(
                    np.abs(
                        dataframe["lambda_away"]
                        - dataframe["away_score"]
                    )
                )
            ),
            "poisson_deviance": poisson_deviance(
                dataframe["away_score"],
                dataframe["lambda_away"],
            ),
        },
        {
            "target": "total_goals",
            "match_count": len(dataframe),
            "mean_actual": actual_total,
            "mean_predicted": predicted_total,
            "mean_bias_predicted_minus_actual": (
                predicted_total - actual_total
            ),
            "mean_shortfall_actual_minus_predicted": (
                actual_total - predicted_total
            ),
            "mae": float(
                dataframe[
                    "absolute_total_goal_error"
                ].mean()
            ),
            "poisson_deviance": poisson_deviance(
                dataframe["total_goals"],
                dataframe["pred_total_goals"],
            ),
        },
        {
            "target": "goal_difference",
            "match_count": len(dataframe),
            "mean_actual": actual_goal_difference,
            "mean_predicted": predicted_goal_difference,
            "mean_bias_predicted_minus_actual": (
                predicted_goal_difference
                - actual_goal_difference
            ),
            "mean_shortfall_actual_minus_predicted": (
                actual_goal_difference
                - predicted_goal_difference
            ),
            "mae": float(
                dataframe[
                    "absolute_goal_difference_error"
                ].mean()
            ),
            "poisson_deviance": float("nan"),
        },
    ]

    return pd.DataFrame(records)


def add_quantile_bin(
    dataframe: pd.DataFrame,
    *,
    source_column: str,
    output_column: str,
    labels: list[str],
) -> pd.DataFrame:
    output = dataframe.copy()

    output[output_column] = pd.qcut(
        output[source_column],
        q=len(labels),
        labels=labels,
        duplicates="drop",
    )

    if output[output_column].isna().any():
        raise ValueError(
            f"Could not assign complete quantile bins for "
            f"{source_column!r}."
        )

    return output


def summarize_fixture_groups(
    dataframe: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    total_shortfall = float(
        dataframe[
            "aggregate_goal_shortfall"
        ].sum()
    )

    records: list[dict[str, object]] = []

    for group_name, group in dataframe.groupby(
        group_column,
        observed=False,
        sort=False,
    ):
        group_shortfall = float(
            group[
                "aggregate_goal_shortfall"
            ].sum()
        )

        records.append(
            {
                group_column: str(group_name),
                "match_count": len(group),
                "mean_actual_home_goals": float(
                    group["home_score"].mean()
                ),
                "mean_predicted_home_goals": float(
                    group["lambda_home"].mean()
                ),
                "mean_home_goal_bias": float(
                    group["home_goal_bias"].mean()
                ),
                "mean_actual_away_goals": float(
                    group["away_score"].mean()
                ),
                "mean_predicted_away_goals": float(
                    group["lambda_away"].mean()
                ),
                "mean_away_goal_bias": float(
                    group["away_goal_bias"].mean()
                ),
                "mean_actual_total_goals": float(
                    group["total_goals"].mean()
                ),
                "mean_predicted_total_goals": float(
                    group["pred_total_goals"].mean()
                ),
                "mean_total_goal_bias": float(
                    group["total_goal_bias"].mean()
                ),
                "mean_total_goal_shortfall": float(
                    group[
                        "aggregate_goal_shortfall"
                    ].mean()
                ),
                "total_goal_mae": float(
                    group[
                        "absolute_total_goal_error"
                    ].mean()
                ),
                "total_goal_poisson_deviance": (
                    poisson_deviance(
                        group["total_goals"],
                        group["pred_total_goals"],
                    )
                ),
                "aggregate_goal_shortfall": (
                    group_shortfall
                ),
                "share_of_league_goal_shortfall": (
                    safe_share(
                        group_shortfall,
                        total_shortfall,
                    )
                ),
                "mean_rating_difference": float(
                    group["rating_prior_diff"].mean()
                ),
                "mean_absolute_rating_difference": float(
                    group[
                        "absolute_rating_difference"
                    ].mean()
                ),
                "mean_predicted_goal_difference": float(
                    group["pred_goal_diff"].mean()
                ),
            }
        )

    return pd.DataFrame(records)


def build_predicted_total_bins(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    labels = [
        "Q1_lowest",
        "Q2_lower_middle",
        "Q3_upper_middle",
        "Q4_highest",
    ]

    working = add_quantile_bin(
        dataframe,
        source_column="pred_total_goals",
        output_column="predicted_total_goal_bin",
        labels=labels,
    )

    summary = summarize_fixture_groups(
        working,
        group_column="predicted_total_goal_bin",
    )

    bounds = (
        working
        .groupby(
            "predicted_total_goal_bin",
            observed=False,
        )
        ["pred_total_goals"]
        .agg(
            bin_minimum="min",
            bin_maximum="max",
        )
        .reset_index()
    )

    bounds[
        "predicted_total_goal_bin"
    ] = bounds[
        "predicted_total_goal_bin"
    ].astype(str)

    return summary.merge(
        bounds,
        on="predicted_total_goal_bin",
        how="left",
        validate="one_to_one",
    )


def build_rating_difference_bins(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    labels = [
        "Q1_away_favored",
        "Q2_slight_away_or_balanced",
        "Q3_slight_home_or_balanced",
        "Q4_home_favored",
    ]

    working = add_quantile_bin(
        dataframe,
        source_column="rating_prior_diff",
        output_column="rating_difference_bin",
        labels=labels,
    )

    summary = summarize_fixture_groups(
        working,
        group_column="rating_difference_bin",
    )

    bounds = (
        working
        .groupby(
            "rating_difference_bin",
            observed=False,
        )
        ["rating_prior_diff"]
        .agg(
            bin_minimum="min",
            bin_maximum="max",
        )
        .reset_index()
    )

    bounds[
        "rating_difference_bin"
    ] = bounds[
        "rating_difference_bin"
    ].astype(str)

    return summary.merge(
        bounds,
        on="rating_difference_bin",
        how="left",
        validate="one_to_one",
    )


def build_matchup_balance_bins(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    labels = [
        "Q1_most_balanced",
        "Q2_moderately_balanced",
        "Q3_clear_favorite",
        "Q4_largest_mismatch",
    ]

    working = add_quantile_bin(
        dataframe,
        source_column="absolute_rating_difference",
        output_column="matchup_balance_bin",
        labels=labels,
    )

    summary = summarize_fixture_groups(
        working,
        group_column="matchup_balance_bin",
    )

    bounds = (
        working
        .groupby(
            "matchup_balance_bin",
            observed=False,
        )
        ["absolute_rating_difference"]
        .agg(
            bin_minimum="min",
            bin_maximum="max",
        )
        .reset_index()
    )

    bounds[
        "matchup_balance_bin"
    ] = bounds[
        "matchup_balance_bin"
    ].astype(str)

    return summary.merge(
        bounds,
        on="matchup_balance_bin",
        how="left",
        validate="one_to_one",
    )


def build_monthly_bias(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    working = dataframe.copy()

    working["month"] = (
        working["date"]
        .dt.to_period("M")
        .astype(str)
    )

    return summarize_fixture_groups(
        working,
        group_column="month",
    )


def build_season_phase_bias(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    working = dataframe.copy()

    maximum_round = int(
        working["round_number"].max()
    )

    midpoint = maximum_round / 2.0

    working["season_phase"] = np.where(
        working["round_number"] <= midpoint,
        "first_half",
        "second_half",
    )

    summary = summarize_fixture_groups(
        working,
        group_column="season_phase",
    )

    summary["round_midpoint"] = midpoint

    return summary


def build_team_match_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    home = dataframe[
        [
            "event_id",
            "date",
            "home_team",
            "home_score",
            "away_score",
            "lambda_home",
            "lambda_away",
            "home_rating_prior",
            "away_rating_prior",
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_team": "club",
            "home_score": "actual_goals_for",
            "away_score": "actual_goals_against",
            "lambda_home": "predicted_goals_for",
            "lambda_away": "predicted_goals_against",
            "home_rating_prior": "club_rating_prior",
            "away_rating_prior": "opponent_rating_prior",
        }
    )

    home["venue"] = "home"

    away = dataframe[
        [
            "event_id",
            "date",
            "away_team",
            "away_score",
            "home_score",
            "lambda_away",
            "lambda_home",
            "away_rating_prior",
            "home_rating_prior",
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_team": "club",
            "away_score": "actual_goals_for",
            "home_score": "actual_goals_against",
            "lambda_away": "predicted_goals_for",
            "lambda_home": "predicted_goals_against",
            "away_rating_prior": "club_rating_prior",
            "home_rating_prior": "opponent_rating_prior",
        }
    )

    away["venue"] = "away"

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

    combined["actual_goal_difference"] = (
        combined["actual_goals_for"]
        - combined["actual_goals_against"]
    )

    combined["predicted_goal_difference"] = (
        combined["predicted_goals_for"]
        - combined["predicted_goals_against"]
    )

    combined["goal_difference_bias"] = (
        combined["predicted_goal_difference"]
        - combined["actual_goal_difference"]
    )

    combined["attack_shortfall"] = (
        combined["actual_goals_for"]
        - combined["predicted_goals_for"]
    )

    combined["defensive_concession_shortfall"] = (
        combined["actual_goals_against"]
        - combined["predicted_goals_against"]
    )

    return combined


def build_club_attack_bias(
    team_matches: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        team_matches
        .groupby(
            "club",
            as_index=False,
        )
        .agg(
            matches=("event_id", "size"),
            average_rating_prior=(
                "club_rating_prior",
                "mean",
            ),
            actual_goals_scored=(
                "actual_goals_for",
                "sum",
            ),
            predicted_goals_scored=(
                "predicted_goals_for",
                "sum",
            ),
            total_scoring_bias=(
                "goals_for_bias",
                "sum",
            ),
            mean_scoring_bias_per_match=(
                "goals_for_bias",
                "mean",
            ),
            total_attack_shortfall=(
                "attack_shortfall",
                "sum",
            ),
            mean_attack_shortfall_per_match=(
                "attack_shortfall",
                "mean",
            ),
        )
    )

    scoring_mae = (
        team_matches.assign(
            absolute_scoring_error=(
                team_matches[
                    "goals_for_bias"
                ].abs()
            )
        )
        .groupby(
            "club",
            as_index=False,
        )
        .agg(
            scoring_mae=(
                "absolute_scoring_error",
                "mean",
            )
        )
    )

    summary = summary.merge(
        scoring_mae,
        on="club",
        how="left",
        validate="one_to_one",
    )

    summary["actual_goals_per_match"] = (
        summary["actual_goals_scored"]
        / summary["matches"]
    )

    summary["predicted_goals_per_match"] = (
        summary["predicted_goals_scored"]
        / summary["matches"]
    )

    return (
        summary
        .sort_values(
            [
                "mean_scoring_bias_per_match",
                "club",
            ]
        )
        .reset_index(drop=True)
    )


def build_club_defensive_bias(
    team_matches: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        team_matches
        .groupby(
            "club",
            as_index=False,
        )
        .agg(
            matches=("event_id", "size"),
            average_rating_prior=(
                "club_rating_prior",
                "mean",
            ),
            actual_goals_conceded=(
                "actual_goals_against",
                "sum",
            ),
            predicted_goals_conceded=(
                "predicted_goals_against",
                "sum",
            ),
            total_concession_bias=(
                "goals_against_bias",
                "sum",
            ),
            mean_concession_bias_per_match=(
                "goals_against_bias",
                "mean",
            ),
            total_concession_shortfall=(
                "defensive_concession_shortfall",
                "sum",
            ),
            mean_concession_shortfall_per_match=(
                "defensive_concession_shortfall",
                "mean",
            ),
        )
    )

    defensive_mae = (
        team_matches.assign(
            absolute_concession_error=(
                team_matches[
                    "goals_against_bias"
                ].abs()
            )
        )
        .groupby(
            "club",
            as_index=False,
        )
        .agg(
            concession_mae=(
                "absolute_concession_error",
                "mean",
            )
        )
    )

    summary = summary.merge(
        defensive_mae,
        on="club",
        how="left",
        validate="one_to_one",
    )

    summary[
        "actual_goals_conceded_per_match"
    ] = (
        summary["actual_goals_conceded"]
        / summary["matches"]
    )

    summary[
        "predicted_goals_conceded_per_match"
    ] = (
        summary["predicted_goals_conceded"]
        / summary["matches"]
    )

    return (
        summary
        .sort_values(
            [
                "mean_concession_bias_per_match",
                "club",
            ]
        )
        .reset_index(drop=True)
    )


def build_club_goal_difference_bias(
    team_matches: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        team_matches
        .groupby(
            "club",
            as_index=False,
        )
        .agg(
            matches=("event_id", "size"),
            average_rating_prior=(
                "club_rating_prior",
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
            total_goal_difference_bias=(
                "goal_difference_bias",
                "sum",
            ),
            mean_goal_difference_bias_per_match=(
                "goal_difference_bias",
                "mean",
            ),
        )
    )

    return (
        summary
        .sort_values(
            [
                "mean_goal_difference_bias_per_match",
                "club",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_home_away_bias(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    records = [
        {
            "venue": "home",
            "observation_count": len(dataframe),
            "mean_actual_goals": float(
                dataframe["home_score"].mean()
            ),
            "mean_predicted_goals": float(
                dataframe["lambda_home"].mean()
            ),
            "mean_bias_predicted_minus_actual": float(
                dataframe["home_goal_bias"].mean()
            ),
            "mean_shortfall_actual_minus_predicted": float(
                -dataframe["home_goal_bias"].mean()
            ),
            "mae": float(
                dataframe[
                    "absolute_home_goal_error"
                ].mean()
            ),
            "poisson_deviance": poisson_deviance(
                dataframe["home_score"],
                dataframe["lambda_home"],
            ),
        },
        {
            "venue": "away",
            "observation_count": len(dataframe),
            "mean_actual_goals": float(
                dataframe["away_score"].mean()
            ),
            "mean_predicted_goals": float(
                dataframe["lambda_away"].mean()
            ),
            "mean_bias_predicted_minus_actual": float(
                dataframe["away_goal_bias"].mean()
            ),
            "mean_shortfall_actual_minus_predicted": float(
                -dataframe["away_goal_bias"].mean()
            ),
            "mae": float(
                dataframe[
                    "absolute_away_goal_error"
                ].mean()
            ),
            "poisson_deviance": poisson_deviance(
                dataframe["away_score"],
                dataframe["lambda_away"],
            ),
        },
    ]

    return pd.DataFrame(records)


def build_elite_team_bias(
    team_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], float]:
    club_ratings = (
        team_matches
        .groupby(
            "club",
            as_index=False,
        )
        .agg(
            average_rating_prior=(
                "club_rating_prior",
                "mean",
            )
        )
    )

    threshold = float(
        club_ratings[
            "average_rating_prior"
        ].quantile(
            ELITE_QUANTILE
        )
    )

    elite_clubs = tuple(
        club_ratings.loc[
            club_ratings[
                "average_rating_prior"
            ].ge(threshold),
            "club",
        ]
        .sort_values()
        .tolist()
    )

    working = team_matches.copy()

    working["club_group"] = np.where(
        working["club"].isin(
            elite_clubs
        ),
        "elite_top_quartile",
        "non_elite",
    )

    summary = (
        working
        .groupby(
            "club_group",
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
            average_rating_prior=(
                "club_rating_prior",
                "mean",
            ),
            mean_actual_goals_for=(
                "actual_goals_for",
                "mean",
            ),
            mean_predicted_goals_for=(
                "predicted_goals_for",
                "mean",
            ),
            mean_goals_for_bias=(
                "goals_for_bias",
                "mean",
            ),
            mean_actual_goals_against=(
                "actual_goals_against",
                "mean",
            ),
            mean_predicted_goals_against=(
                "predicted_goals_against",
                "mean",
            ),
            mean_goals_against_bias=(
                "goals_against_bias",
                "mean",
            ),
            mean_actual_goal_difference=(
                "actual_goal_difference",
                "mean",
            ),
            mean_predicted_goal_difference=(
                "predicted_goal_difference",
                "mean",
            ),
            mean_goal_difference_bias=(
                "goal_difference_bias",
                "mean",
            ),
            aggregate_attack_shortfall=(
                "attack_shortfall",
                "sum",
            ),
        )
    )

    summary["elite_rating_threshold"] = threshold

    summary["elite_clubs"] = (
        ", ".join(elite_clubs)
    )

    return (
        summary,
        elite_clubs,
        threshold,
    )


def build_bias_concentration(
    dataframe: pd.DataFrame,
    club_attack_bias: pd.DataFrame,
    elite_clubs: tuple[str, ...],
) -> pd.DataFrame:
    total_shortfall = float(
        dataframe[
            "aggregate_goal_shortfall"
        ].sum()
    )

    attack_ranked = (
        club_attack_bias
        .sort_values(
            "total_attack_shortfall",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    top_five_shortfall = float(
        attack_ranked
        .head(5)[
            "total_attack_shortfall"
        ]
        .sum()
    )

    top_ten_shortfall = float(
        attack_ranked
        .head(10)[
            "total_attack_shortfall"
        ]
        .sum()
    )

    elite_fixture_mask = (
        dataframe["home_team"].isin(
            elite_clubs
        )
        | dataframe["away_team"].isin(
            elite_clubs
        )
    )

    elite_fixture_shortfall = float(
        dataframe.loc[
            elite_fixture_mask,
            "aggregate_goal_shortfall",
        ].sum()
    )

    monthly = dataframe.copy()

    monthly["month"] = (
        monthly["date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_shortfall = (
        monthly
        .groupby(
            "month",
            as_index=False,
        )
        .agg(
            aggregate_goal_shortfall=(
                "aggregate_goal_shortfall",
                "sum",
            )
        )
        .sort_values(
            "aggregate_goal_shortfall",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    largest_month = (
        monthly_shortfall.iloc[0]
    )

    predicted_bin_working = add_quantile_bin(
        dataframe,
        source_column="pred_total_goals",
        output_column="predicted_total_goal_bin",
        labels=[
            "Q1_lowest",
            "Q2_lower_middle",
            "Q3_upper_middle",
            "Q4_highest",
        ],
    )

    bin_shortfall = (
        predicted_bin_working
        .groupby(
            "predicted_total_goal_bin",
            observed=False,
            as_index=False,
        )
        .agg(
            aggregate_goal_shortfall=(
                "aggregate_goal_shortfall",
                "sum",
            )
        )
        .sort_values(
            "aggregate_goal_shortfall",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    largest_bin = bin_shortfall.iloc[0]

    records = [
        {
            "concentration_measure":
                "five_most_underpredicted_attacks",
            "population_description": ", ".join(
                attack_ranked.head(5)["club"]
            ),
            "aggregate_goal_shortfall": (
                top_five_shortfall
            ),
            "share_of_league_goal_shortfall": (
                safe_share(
                    top_five_shortfall,
                    total_shortfall,
                )
            ),
        },
        {
            "concentration_measure":
                "ten_most_underpredicted_attacks",
            "population_description": ", ".join(
                attack_ranked.head(10)["club"]
            ),
            "aggregate_goal_shortfall": (
                top_ten_shortfall
            ),
            "share_of_league_goal_shortfall": (
                safe_share(
                    top_ten_shortfall,
                    total_shortfall,
                )
            ),
        },
        {
            "concentration_measure":
                "fixtures_involving_elite_club",
            "population_description": ", ".join(
                elite_clubs
            ),
            "aggregate_goal_shortfall": (
                elite_fixture_shortfall
            ),
            "share_of_league_goal_shortfall": (
                safe_share(
                    elite_fixture_shortfall,
                    total_shortfall,
                )
            ),
        },
        {
            "concentration_measure":
                "largest_monthly_shortfall",
            "population_description": str(
                largest_month["month"]
            ),
            "aggregate_goal_shortfall": float(
                largest_month[
                    "aggregate_goal_shortfall"
                ]
            ),
            "share_of_league_goal_shortfall": (
                safe_share(
                    float(
                        largest_month[
                            "aggregate_goal_shortfall"
                        ]
                    ),
                    total_shortfall,
                )
            ),
        },
        {
            "concentration_measure":
                "largest_predicted_total_bin_shortfall",
            "population_description": str(
                largest_bin[
                    "predicted_total_goal_bin"
                ]
            ),
            "aggregate_goal_shortfall": float(
                largest_bin[
                    "aggregate_goal_shortfall"
                ]
            ),
            "share_of_league_goal_shortfall": (
                safe_share(
                    float(
                        largest_bin[
                            "aggregate_goal_shortfall"
                        ]
                    ),
                    total_shortfall,
                )
            ),
        },
        {
            "concentration_measure":
                "entire_league",
            "population_description":
                "all_306_matches",
            "aggregate_goal_shortfall": (
                total_shortfall
            ),
            "share_of_league_goal_shortfall": 1.0,
        },
    ]

    return pd.DataFrame(records)


def build_metadata(
    *,
    dataframe: pd.DataFrame,
    elite_clubs: tuple[str, ...],
    elite_threshold: float,
) -> dict[str, object]:
    return {
        "study_id": "085A",
        "study_name": (
            "Bundesliga Goal-Volume Bias Decomposition"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "study_type": (
            "descriptive_frozen_prediction_diagnostic"
        ),
        "input_dataset": str(
            INPUT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "match_count": len(dataframe),
        "competition": str(
            dataframe[
                "competition_key"
            ].iloc[0]
            if "competition_key" in dataframe.columns
            else "bundesliga"
        ),
        "evaluation_period_start": (
            dataframe["date"]
            .min()
            .date()
            .isoformat()
        ),
        "evaluation_period_end": (
            dataframe["date"]
            .max()
            .date()
            .isoformat()
        ),
        "mean_actual_total_goals": float(
            dataframe[
                "total_goals"
            ].mean()
        ),
        "mean_predicted_total_goals": float(
            dataframe[
                "pred_total_goals"
            ].mean()
        ),
        "mean_total_goal_bias": float(
            dataframe[
                "total_goal_bias"
            ].mean()
        ),
        "aggregate_goal_shortfall": float(
            dataframe[
                "aggregate_goal_shortfall"
            ].sum()
        ),
        "elite_definition": (
            "Top quartile of clubs by average "
            "prediction-date ClubElo."
        ),
        "elite_quantile": ELITE_QUANTILE,
        "elite_rating_threshold": (
            elite_threshold
        ),
        "elite_clubs": list(
            elite_clubs
        ),
        "model_changed": False,
        "predictions_rerun": False,
        "interpretation_boundary": (
            "This study decomposes errors in a frozen replay "
            "artifact whose evaluation period overlaps the "
            "goal-model training period. It is diagnostic and "
            "does not estimate clean prospective performance."
        ),
        "outputs": [
            OVERALL_BIAS_PATH.name,
            PREDICTED_TOTAL_BINS_PATH.name,
            RATING_DIFFERENCE_BINS_PATH.name,
            MATCHUP_BALANCE_BINS_PATH.name,
            MONTHLY_BIAS_PATH.name,
            SEASON_PHASE_PATH.name,
            CLUB_ATTACK_BIAS_PATH.name,
            CLUB_DEFENSIVE_BIAS_PATH.name,
            CLUB_GOAL_DIFFERENCE_BIAS_PATH.name,
            HOME_AWAY_BIAS_PATH.name,
            ELITE_TEAM_BIAS_PATH.name,
            BIAS_CONCENTRATION_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    dataframe: pd.DataFrame,
    predicted_bins: pd.DataFrame,
    monthly_bias: pd.DataFrame,
    club_attack_bias: pd.DataFrame,
    club_defensive_bias: pd.DataFrame,
    club_goal_difference_bias: pd.DataFrame,
    elite_bias: pd.DataFrame,
    elite_clubs: tuple[str, ...],
    concentration: pd.DataFrame,
) -> None:
    mean_actual_total = float(
        dataframe["total_goals"].mean()
    )

    mean_predicted_total = float(
        dataframe["pred_total_goals"].mean()
    )

    mean_total_bias = float(
        dataframe["total_goal_bias"].mean()
    )

    mean_home_bias = float(
        dataframe["home_goal_bias"].mean()
    )

    mean_away_bias = float(
        dataframe["away_goal_bias"].mean()
    )

    worst_predicted_bin = (
        predicted_bins
        .sort_values(
            "mean_total_goal_bias"
        )
        .iloc[0]
    )

    worst_month = (
        monthly_bias
        .sort_values(
            "mean_total_goal_bias"
        )
        .iloc[0]
    )

    most_underpredicted_attack = (
        club_attack_bias.iloc[0]
    )

    most_overpredicted_attack = (
        club_attack_bias.iloc[-1]
    )

    most_underpredicted_concessions = (
        club_defensive_bias.iloc[0]
    )

    most_negative_goal_difference = (
        club_goal_difference_bias.iloc[0]
    )

    elite_row = elite_bias.loc[
        elite_bias[
            "club_group"
        ].eq(
            "elite_top_quartile"
        )
    ].iloc[0]

    non_elite_row = elite_bias.loc[
        elite_bias[
            "club_group"
        ].eq(
            "non_elite"
        )
    ].iloc[0]

    top_five_row = concentration.loc[
        concentration[
            "concentration_measure"
        ].eq(
            "five_most_underpredicted_attacks"
        )
    ].iloc[0]

    report = f"""# Study 085A — Bundesliga Goal-Volume Bias Decomposition

## Purpose

Determine whether the Bundesliga goal-volume shortfall identified
in Study 084 is primarily global, temporal, matchup-dependent,
elite-team concentrated, or club-specific.

## Methodological boundary

This analysis consumes only the frozen Study 084 fixture-error
artifact.

No predictions were rerun.

No goal model, repository, sampler, coefficient, intercept or
lambda scale was changed.

The replay period overlaps the model-training period, so the
results remain diagnostic rather than a clean prospective test.

## Evaluation population

- Matches: {len(dataframe)}
- Mean actual total goals: {mean_actual_total:.6f}
- Mean predicted total goals: {mean_predicted_total:.6f}
- Mean total-goal bias:
  {mean_total_bias:.6f}
- Aggregate goal shortfall:
  {dataframe["aggregate_goal_shortfall"].sum():.6f}

Negative bias means predicted goals were lower than observed goals.

## Home and away decomposition

- Mean home-goal bias:
  {mean_home_bias:.6f}
- Mean away-goal bias:
  {mean_away_bias:.6f}

The bias affects both home and away scoring.

## Predicted scoring-environment decomposition

The most negatively biased predicted-total-goal bin was:

- Bin: `{worst_predicted_bin["predicted_total_goal_bin"]}`
- Matches: {int(worst_predicted_bin["match_count"])}
- Mean actual total goals:
  {worst_predicted_bin["mean_actual_total_goals"]:.6f}
- Mean predicted total goals:
  {worst_predicted_bin["mean_predicted_total_goals"]:.6f}
- Mean total-goal bias:
  {worst_predicted_bin["mean_total_goal_bias"]:.6f}

## Temporal decomposition

The most negatively biased month was:

- Month: `{worst_month["month"]}`
- Matches: {int(worst_month["match_count"])}
- Mean total-goal bias:
  {worst_month["mean_total_goal_bias"]:.6f}

Monthly differences are descriptive and should not be treated as
stable effects without further validation.

## Club attacking bias

Most underpredicted attack:

- Club:
  `{most_underpredicted_attack["club"]}`
- Mean scoring bias per match:
  {most_underpredicted_attack["mean_scoring_bias_per_match"]:.6f}
- Total attack shortfall:
  {most_underpredicted_attack["total_attack_shortfall"]:.6f}

Most overpredicted attack:

- Club:
  `{most_overpredicted_attack["club"]}`
- Mean scoring bias per match:
  {most_overpredicted_attack["mean_scoring_bias_per_match"]:.6f}

## Club defensive bias

Largest negative concession bias:

- Club:
  `{most_underpredicted_concessions["club"]}`
- Mean concession bias per match:
  {most_underpredicted_concessions["mean_concession_bias_per_match"]:.6f}

A negative concession bias means the model predicted fewer goals
against that club than the club actually conceded.

## Goal-difference bias

Most negative team goal-difference bias:

- Club:
  `{most_negative_goal_difference["club"]}`
- Mean goal-difference bias per match:
  {most_negative_goal_difference["mean_goal_difference_bias_per_match"]:.6f}

## Elite-team comparison

Elite definition:

Top quartile of clubs by average prediction-date ClubElo.

Elite clubs:

`{", ".join(elite_clubs)}`

Elite mean goals-for bias:

{elite_row["mean_goals_for_bias"]:.6f}

Non-elite mean goals-for bias:

{non_elite_row["mean_goals_for_bias"]:.6f}

Elite mean goal-difference bias:

{elite_row["mean_goal_difference_bias"]:.6f}

Non-elite mean goal-difference bias:

{non_elite_row["mean_goal_difference_bias"]:.6f}

## Bias concentration

The five most underpredicted attacks account for:

{top_five_row["share_of_league_goal_shortfall"]:.6f}

of the league-wide aggregate goal shortfall.

A share above one is possible when some other clubs are
overpredicted and offset part of the underprediction.

## Interpretation

This study does not automatically recommend global lambda scaling.

The decomposition should be used to determine whether the next
diagnostic should prioritize:

- global scoring-environment calibration;
- elite-team nonlinearities;
- static repository drift;
- club-specific representation audits;
- richer match-level information.

## Result

**OVERALL RESULT: PASS**

The Bundesliga scoring bias was decomposed without modifying or
rerunning the production model.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 085A — BUNDESLIGA GOAL-VOLUME "
        "BIAS DECOMPOSITION"
    )
    print("=" * 88)

    predictions = load_predictions()

    overall_bias = build_overall_bias(
        predictions
    )

    predicted_total_bins = (
        build_predicted_total_bins(
            predictions
        )
    )

    rating_difference_bins = (
        build_rating_difference_bins(
            predictions
        )
    )

    matchup_balance_bins = (
        build_matchup_balance_bins(
            predictions
        )
    )

    monthly_bias = build_monthly_bias(
        predictions
    )

    season_phase_bias = (
        build_season_phase_bias(
            predictions
        )
    )

    team_matches = build_team_match_table(
        predictions
    )

    club_attack_bias = (
        build_club_attack_bias(
            team_matches
        )
    )

    club_defensive_bias = (
        build_club_defensive_bias(
            team_matches
        )
    )

    club_goal_difference_bias = (
        build_club_goal_difference_bias(
            team_matches
        )
    )

    home_away_bias = build_home_away_bias(
        predictions
    )

    (
        elite_team_bias,
        elite_clubs,
        elite_threshold,
    ) = build_elite_team_bias(
        team_matches
    )

    bias_concentration = (
        build_bias_concentration(
            predictions,
            club_attack_bias,
            elite_clubs,
        )
    )

    metadata = build_metadata(
        dataframe=predictions,
        elite_clubs=elite_clubs,
        elite_threshold=elite_threshold,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_bias.to_csv(
        OVERALL_BIAS_PATH,
        index=False,
    )

    predicted_total_bins.to_csv(
        PREDICTED_TOTAL_BINS_PATH,
        index=False,
    )

    rating_difference_bins.to_csv(
        RATING_DIFFERENCE_BINS_PATH,
        index=False,
    )

    matchup_balance_bins.to_csv(
        MATCHUP_BALANCE_BINS_PATH,
        index=False,
    )

    monthly_bias.to_csv(
        MONTHLY_BIAS_PATH,
        index=False,
    )

    season_phase_bias.to_csv(
        SEASON_PHASE_PATH,
        index=False,
    )

    club_attack_bias.to_csv(
        CLUB_ATTACK_BIAS_PATH,
        index=False,
    )

    club_defensive_bias.to_csv(
        CLUB_DEFENSIVE_BIAS_PATH,
        index=False,
    )

    club_goal_difference_bias.to_csv(
        CLUB_GOAL_DIFFERENCE_BIAS_PATH,
        index=False,
    )

    home_away_bias.to_csv(
        HOME_AWAY_BIAS_PATH,
        index=False,
    )

    elite_team_bias.to_csv(
        ELITE_TEAM_BIAS_PATH,
        index=False,
    )

    bias_concentration.to_csv(
        BIAS_CONCENTRATION_PATH,
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
        dataframe=predictions,
        predicted_bins=predicted_total_bins,
        monthly_bias=monthly_bias,
        club_attack_bias=club_attack_bias,
        club_defensive_bias=club_defensive_bias,
        club_goal_difference_bias=(
            club_goal_difference_bias
        ),
        elite_bias=elite_team_bias,
        elite_clubs=elite_clubs,
        concentration=bias_concentration,
    )

    print()
    print("Evaluation population")
    print("-" * 88)
    print(
        f"  Matches: {len(predictions)}"
    )
    print(
        "  Predictions rerun: NO"
    )
    print(
        "  Production model changed: NO"
    )

    print()
    print("Overall goal-volume bias")
    print("-" * 88)
    print(
        "  Mean actual total goals: "
        f"{predictions['total_goals'].mean():.6f}"
    )
    print(
        "  Mean predicted total goals: "
        f"{predictions['pred_total_goals'].mean():.6f}"
    )
    print(
        "  Mean bias (predicted - actual): "
        f"{predictions['total_goal_bias'].mean():.6f}"
    )
    print(
        "  Aggregate goal shortfall: "
        f"{predictions['aggregate_goal_shortfall'].sum():.6f}"
    )

    print()
    print("Home and away bias")
    print("-" * 88)
    print(
        "  Home-goal bias: "
        f"{predictions['home_goal_bias'].mean():.6f}"
    )
    print(
        "  Away-goal bias: "
        f"{predictions['away_goal_bias'].mean():.6f}"
    )

    print()
    print("Most underpredicted attacks")
    print("-" * 88)
    print(
        club_attack_bias[
            [
                "club",
                "mean_scoring_bias_per_match",
                "total_attack_shortfall",
                "scoring_mae",
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
    print("Elite-team comparison")
    print("-" * 88)
    print(
        elite_team_bias[
            [
                "club_group",
                "unique_clubs",
                "mean_goals_for_bias",
                "mean_goals_against_bias",
                "mean_goal_difference_bias",
            ]
        ]
        .to_string(
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
    print("  Fixture population preservation: PASS")
    print("  Home/away decomposition: PASS")
    print("  Predicted-total binning: PASS")
    print("  Rating-difference binning: PASS")
    print("  Matchup-balance binning: PASS")
    print("  Monthly decomposition: PASS")
    print("  Season-phase decomposition: PASS")
    print("  Club attack decomposition: PASS")
    print("  Club defensive decomposition: PASS")
    print("  Elite-team decomposition: PASS")
    print("  Bias concentration analysis: PASS")
    print("  No production mutation: PASS")

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
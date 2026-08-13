#benchmark_001b_league_fingerprint

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from research import ExperimentCondition
from research.adapters import FootballModelAdapter
from research.benchmarking.football_model_benchmark_engine import (
    FootballModelBenchmarkConfig,
    FootballModelBenchmarkEngine,
    FootballModelBenchmarkSpec,
)

from research.benchmarking.fingerprint_aggregation import (
    FINGERPRINT_METRICS,
    RATE_METRICS,
    aggregate_league_fingerprints,
    validate_league_fingerprints,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "football_model_benchmarks"
    / "benchmark_001b"
)

RAW_OUTPUT_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "raw"
)

SUMMARY_OUTPUT_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "summary"
)

SEASON_STATISTICS_PATH = (
    RAW_OUTPUT_DIRECTORY
    / "season_statistics.csv"
)

CLUB_SEASON_STATISTICS_PATH = (
    RAW_OUTPUT_DIRECTORY
    / "club_season_statistics.csv"
)

MATCH_STATISTICS_PATH = (
    RAW_OUTPUT_DIRECTORY
    / "match_statistics.csv"
)

CLUB_SUMMARY_PATH = (
    SUMMARY_OUTPUT_DIRECTORY
    / "club_summary.csv"
)

LEAGUE_FINGERPRINT_PATH = (
    SUMMARY_OUTPUT_DIRECTORY
    / "league_fingerprint.csv"
)

FOOTBALL_MODEL_COMPARISON_PATH = (
    SUMMARY_OUTPUT_DIRECTORY
    / "football_model_comparison.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "benchmark_001b_metadata.json"
)


BENCHMARK_ID = "001B"

BENCHMARK_NAME = (
    "Football Model League Fingerprint Benchmark"
)

SIMULATION_COUNT = 25
BASE_SEED = 2001

SEASON_START_DATE = date(
    2025,
    8,
    16,
)

DAYS_BETWEEN_MATCHDAYS = 7

CANONICAL_CLUBS = (
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
)


LEGACY_RUNTIME_NAMES = {
    club: club
    for club in CANONICAL_CLUBS
}


PRODUCTION_RUNTIME_NAMES = {
    **{
        club: club
        for club in CANONICAL_CLUBS
    },
    "Liverpool": "Liverpool FC",
    "Wolverhampton Wanderers": "Wolverhampton",
}

FINGERPRINT_METRIC_LABELS = {
    "goals_per_match":
        "Goals per match",
    "home_goals_per_match":
        "Home goals per match",
    "away_goals_per_match":
        "Away goals per match",

    "home_win_rate":
        "Home-win rate",
    "draw_rate":
        "Draw rate",
    "away_win_rate":
        "Away-win rate",

    "both_teams_to_score_rate":
        "Both-teams-to-score rate",

    "zero_goal_match_rate":
        "Zero-goal match rate",
    "one_goal_match_rate":
        "One-goal match rate",
    "two_goal_match_rate":
        "Two-goal match rate",
    "three_goal_match_rate":
        "Three-goal match rate",
    "four_plus_goal_match_rate":
        "Four-plus-goal match rate",

    "one_goal_margin_rate":
        "One-goal-margin rate",
    "three_plus_goal_margin_rate":
        "Three-plus-goal-margin rate",

    "champion_points":
        "Champion points",
    "bottom_points":
        "Bottom-club points",

    "points_spread":
        "League points spread",
    "goal_difference_spread":
        "League goal-difference spread",
}

MODEL_COMPARISON_PAIRS = (
    {
        "comparison_id":
            "FM002_minus_FM001",
        "baseline_model_id":
            "FM001",
        "candidate_model_id":
            "FM002",
    },
    {
        "comparison_id":
            "FM003_minus_FM001",
        "baseline_model_id":
            "FM001",
        "candidate_model_id":
            "FM003",
    },
    {
        "comparison_id":
            "FM003_minus_FM002",
        "baseline_model_id":
            "FM002",
        "candidate_model_id":
            "FM003",
    },
)

def build_condition(
    *,
    name: str,
    repository_source: str,
    match_engine: str,
) -> ExperimentCondition:
    return ExperimentCondition(
        name=name,
        competition_format=(
            "double_round_robin"
        ),
        repository_source=(
            repository_source
        ),
        match_engine=(
            match_engine
        ),
        simulation_count=(
            SIMULATION_COUNT
        ),
        random_seed=BASE_SEED,
        parameters={
            "benchmark_id":
                BENCHMARK_ID,
            "benchmark_name":
                BENCHMARK_NAME,
            "canonical_club_count":
                len(CANONICAL_CLUBS),
            "benchmark_phase":
                "raw_output_validation",
        },
    )

def build_model_specs() -> tuple[
    FootballModelBenchmarkSpec,
    ...,
]:
    adapter = FootballModelAdapter()

    legacy_condition = build_condition(
        name=(
            "Benchmark 001B — FM001 "
            "Legacy Validation"
        ),
        repository_source=(
            "premier_league_validation"
        ),
        match_engine=(
            "production_scoreline_first"
        ),
    )

    integrated_condition = build_condition(
        name=(
            "Benchmark 001B — FM002 "
            "Integrated Club Goal Model v1"
        ),
        repository_source=(
            "premier_league_production_v1"
        ),
        match_engine=(
            "integrated_club_goal_model_v1"
        ),
    )

    robust_condition = build_condition(
        name=(
            "Benchmark 001B — FM003 "
            "Integrated Club Goal Model Robust Candidate"
        ),
        repository_source=(
            "premier_league_robust_candidate_v1"
        ),
        match_engine=(
            "integrated_club_goal_model_v1"
        ),
    )

    legacy_model = adapter.from_condition(
        legacy_condition
    )

    integrated_model = adapter.from_condition(
        integrated_condition
    )

    robust_model = adapter.from_condition(
        robust_condition
    )

    return (
        FootballModelBenchmarkSpec(
            model_id="FM001",
            display_name=(
                "Legacy Premier League "
                "Validation Model"
            ),
            football_model=legacy_model,
            runtime_names=(
                LEGACY_RUNTIME_NAMES
            ),
        ),

        FootballModelBenchmarkSpec(
            model_id="FM002",
            display_name=(
                "Integrated Club Goal Model v1"
            ),
            football_model=integrated_model,
            runtime_names=(
                PRODUCTION_RUNTIME_NAMES
            ),
        ),

        FootballModelBenchmarkSpec(
            model_id="FM003",
            display_name=(
                "Integrated Club Goal Model Robust Candidate"
            ),
            football_model=robust_model,
            runtime_names=PRODUCTION_RUNTIME_NAMES,
        ),
    )

def build_benchmark_config(
) -> FootballModelBenchmarkConfig:
    return FootballModelBenchmarkConfig(
        benchmark_id=BENCHMARK_ID,
        benchmark_name=BENCHMARK_NAME,
        canonical_clubs=(
            CANONICAL_CLUBS
        ),
        simulation_count=(
            SIMULATION_COUNT
        ),
        base_seed=BASE_SEED,
        season_start_date=(
            SEASON_START_DATE
        ),
        days_between_matchdays=(
            DAYS_BETWEEN_MATCHDAYS
        ),
        double_round_robin=True,
    )

def validate_phase_one_outputs(
    *,
    season_statistics: pd.DataFrame,
    club_statistics: pd.DataFrame,
    match_statistics: pd.DataFrame,
) -> None:
    expected_model_ids = {
        "FM001",
        "FM002",
        "FM003",
    }

    if set(
        season_statistics[
            "model_id"
        ].astype(str)
    ) != expected_model_ids:
        raise AssertionError(
            "Season statistics do not contain all "
            "football models."
        )

    expected_season_rows = (
        len(expected_model_ids)
        * SIMULATION_COUNT
    )

    if len(
        season_statistics
    ) != expected_season_rows:
        raise AssertionError(
            "Unexpected number of season-statistics rows."
        )

    expected_club_rows = (
        len(expected_model_ids)
        * SIMULATION_COUNT
        * len(CANONICAL_CLUBS)
    )

    if len(
        club_statistics
    ) != expected_club_rows:
        raise AssertionError(
            "Unexpected number of club-season rows."
        )

    fixture_count_per_season = (
        len(CANONICAL_CLUBS)
        * (
            len(CANONICAL_CLUBS) - 1
        )
    )

    expected_match_rows = (
        len(expected_model_ids)
        * SIMULATION_COUNT
        * fixture_count_per_season
    )

    if len(
        match_statistics
    ) != expected_match_rows:
        raise AssertionError(
            "Unexpected number of match-statistics rows."
        )

    expected_seasons = set(
        range(
            1,
            SIMULATION_COUNT + 1,
        )
    )

    for model_id in expected_model_ids:
        model_seasons = set(
            season_statistics.loc[
                season_statistics[
                    "model_id"
                ].eq(model_id),
                "season_number",
            ].astype(int)
        )

        if model_seasons != expected_seasons:
            raise AssertionError(
                f"{model_id} does not contain the expected "
                "season population."
            )

    expected_seed_sequence = {
        BASE_SEED + index
        for index in range(
            SIMULATION_COUNT
        )
    }

    observed_seed_sequence = set(
        season_statistics[
            "season_seed"
        ].astype(int)
    )

    if observed_seed_sequence != (
        expected_seed_sequence
    ):
        raise AssertionError(
            "Benchmark used an unexpected season-seed "
            "sequence."
        )

    club_population = set(
        CANONICAL_CLUBS
    )

    observed_clubs = set(
        club_statistics[
            "club"
        ].astype(str)
    )

    if observed_clubs != (
        club_population
    ):
        raise AssertionError(
            "Club-season outputs do not preserve the "
            "canonical club population."
        )

    match_clubs = (
        set(
            match_statistics[
                "home_team"
            ].astype(str)
        )
        | set(
            match_statistics[
                "away_team"
            ].astype(str)
        )
    )

    if match_clubs != club_population:
        raise AssertionError(
            "Runtime aliases leaked into match-statistics "
            "outputs."
        )

    grouped_match_counts = (
        match_statistics
        .groupby(
            [
                "model_id",
                "season_number",
            ]
        )
        .size()
    )

    if not grouped_match_counts.eq(
        fixture_count_per_season
    ).all():
        raise AssertionError(
            "At least one model-season contains an "
            "unexpected fixture count."
        )

    if not club_statistics[
        "matches_played"
    ].eq(
        2 * (
            len(CANONICAL_CLUBS) - 1
        )
    ).all():
        raise AssertionError(
            "At least one club played an unexpected "
            "number of league matches."
        )

    if not season_statistics[
        "matches"
    ].eq(
        fixture_count_per_season
    ).all():
        raise AssertionError(
            "At least one league fingerprint contains "
            "an unexpected match count."
        )

    if season_statistics[
        [
            "goals_per_match",
            "home_win_rate",
            "draw_rate",
            "away_win_rate",
        ]
    ].isna().any().any():
        raise AssertionError(
            "Season statistics contain missing core "
            "fingerprint metrics."
        )

    outcome_rate_sum = (
        season_statistics[
            [
                "home_win_rate",
                "draw_rate",
                "away_win_rate",
            ]
        ].sum(
            axis=1
        )
    )

    if not outcome_rate_sum.between(
        0.999999999,
        1.000000001,
    ).all():
        raise AssertionError(
            "Season outcome rates do not sum to one."
        )

def build_club_summary(
    club_statistics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate one row per football model and club.

    This function summarizes repeated-season standings only.
    It does not compare the models or evaluate them against
    historical reality.
    """

    required_columns = {
        "model_id",
        "model_name",
        "season_number",
        "club",
        "position",
        "points",
        "goals_for",
        "goals_against",
        "goal_difference",
        "is_champion",
        "is_top_four",
        "is_bottom_three",
    }

    missing_columns = (
        required_columns
        - set(club_statistics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Club-season statistics are missing required "
            f"columns: {sorted(missing_columns)}"
        )

    summary = (
        club_statistics
        .groupby(
            [
                "model_id",
                "model_name",
                "club",
            ],
            as_index=False,
        )
        .agg(
            seasons=(
                "season_number",
                "count",
            ),
            mean_position=(
                "position",
                "mean",
            ),
            median_position=(
                "position",
                "median",
            ),
            position_std=(
                "position",
                "std",
            ),
            best_position=(
                "position",
                "min",
            ),
            worst_position=(
                "position",
                "max",
            ),
            mean_points=(
                "points",
                "mean",
            ),
            points_std=(
                "points",
                "std",
            ),
            minimum_points=(
                "points",
                "min",
            ),
            maximum_points=(
                "points",
                "max",
            ),
            mean_goals_for=(
                "goals_for",
                "mean",
            ),
            mean_goals_against=(
                "goals_against",
                "mean",
            ),
            mean_goal_difference=(
                "goal_difference",
                "mean",
            ),
            championship_probability=(
                "is_champion",
                "mean",
            ),
            top_four_probability=(
                "is_top_four",
                "mean",
            ),
            bottom_three_probability=(
                "is_bottom_three",
                "mean",
            ),
        )
        .sort_values(
            [
                "model_id",
                "mean_position",
                "club",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # pandas uses sample standard deviation by default.
    # A one-season benchmark would therefore produce NaN.
    # Fill only these dispersion columns with zero.
    summary[
        [
            "position_std",
            "points_std",
        ]
    ] = (
        summary[
            [
                "position_std",
                "points_std",
            ]
        ]
        .fillna(0.0)
    )

    return summary

def validate_club_summary(
    *,
    club_summary: pd.DataFrame,
    club_statistics: pd.DataFrame,
) -> None:
    if club_summary.empty:
        raise AssertionError(
            "Club summary is empty."
        )

    expected_model_ids = {
        "FM001",
        "FM002",
        "FM003",
    }

    expected_row_count = (
        len(expected_model_ids)
        * len(CANONICAL_CLUBS)
    )

    if len(club_summary) != (
        expected_row_count
    ):
        raise AssertionError(
            "Unexpected club-summary row count: "
            f"{len(club_summary)} vs "
            f"{expected_row_count}."
        )

    if club_summary[
        [
            "model_id",
            "club",
        ]
    ].duplicated().any():
        duplicates = (
            club_summary.loc[
                club_summary[
                    [
                        "model_id",
                        "club",
                    ]
                ].duplicated(
                    keep=False
                ),
                [
                    "model_id",
                    "club",
                ],
            ]
        )

        raise AssertionError(
            "Club summary contains duplicate "
            "model-club rows:\n"
            f"{duplicates.to_string(index=False)}"
        )

    if set(
        club_summary[
            "model_id"
        ].astype(str)
    ) != expected_model_ids:
        raise AssertionError(
            "Club summary does not contain all models."
        )

    for model_id in expected_model_ids:
        model_clubs = set(
            club_summary.loc[
                club_summary[
                    "model_id"
                ].eq(model_id),
                "club",
            ].astype(str)
        )

        if model_clubs != set(
            CANONICAL_CLUBS
        ):
            missing = sorted(
                set(CANONICAL_CLUBS)
                - model_clubs
            )

            extra = sorted(
                model_clubs
                - set(CANONICAL_CLUBS)
            )

            raise AssertionError(
                "Club-summary population mismatch for "
                f"{model_id}. Missing={missing}, "
                f"extra={extra}."
            )

    if not club_summary[
        "seasons"
    ].eq(
        SIMULATION_COUNT
    ).all():
        raise AssertionError(
            "At least one club summary row contains an "
            "unexpected season count."
        )

    probability_columns = [
        "championship_probability",
        "top_four_probability",
        "bottom_three_probability",
    ]

    for column in probability_columns:
        if not club_summary[
            column
        ].between(
            0.0,
            1.0,
        ).all():
            raise AssertionError(
                f"{column} contains a value outside [0, 1]."
            )

    championship_totals = (
        club_summary
        .groupby(
            "model_id"
        )[
            "championship_probability"
        ]
        .sum()
    )

    if not championship_totals.between(
        0.999999999,
        1.000000001,
    ).all():
        raise AssertionError(
            "Championship probabilities do not sum to one "
            "within each model."
        )

    expected_top_four_total = 4.0

    top_four_totals = (
        club_summary
        .groupby(
            "model_id"
        )[
            "top_four_probability"
        ]
        .sum()
    )

    if not top_four_totals.between(
        expected_top_four_total - 1e-9,
        expected_top_four_total + 1e-9,
    ).all():
        raise AssertionError(
            "Top-four probabilities do not sum to four "
            "within each model."
        )

    expected_bottom_three_total = 3.0

    bottom_three_totals = (
        club_summary
        .groupby(
            "model_id"
        )[
            "bottom_three_probability"
        ]
        .sum()
    )

    if not bottom_three_totals.between(
        expected_bottom_three_total - 1e-9,
        expected_bottom_three_total + 1e-9,
    ).all():
        raise AssertionError(
            "Bottom-three probabilities do not sum to "
            "three within each model."
        )

    numeric_columns = [
        "mean_position",
        "median_position",
        "position_std",
        "best_position",
        "worst_position",
        "mean_points",
        "points_std",
        "minimum_points",
        "maximum_points",
        "mean_goals_for",
        "mean_goals_against",
        "mean_goal_difference",
        *probability_columns,
    ]

    numeric_values = (
        club_summary[
            numeric_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not pd.notna(
        numeric_values
    ).all():
        raise AssertionError(
            "Club summary contains missing numeric values."
        )

    # Reconcile the summary means against the raw club-season
    # population rather than trusting the groupby blindly.
    raw_points_mean = (
        club_statistics
        .groupby(
            [
                "model_id",
                "club",
            ]
        )[
            "points"
        ]
        .mean()
        .sort_index()
    )

    summary_points_mean = (
        club_summary
        .set_index(
            [
                "model_id",
                "club",
            ]
        )[
            "mean_points"
        ]
        .sort_index()
    )

    if not raw_points_mean.equals(
        summary_points_mean
    ):
        difference = (
            summary_points_mean
            - raw_points_mean
        )

        if not difference.abs().le(
            1e-12
        ).all():
            raise AssertionError(
                "Club-summary mean points do not reconcile "
                "with the raw club-season statistics."
            )

def build_console_summary(
    season_statistics: pd.DataFrame,
) -> pd.DataFrame:
    return (
        season_statistics
        .groupby(
            [
                "model_id",
                "model_name",
            ],
            as_index=False,
        )
        .agg(
            seasons=(
                "season_number",
                "count",
            ),
            mean_goals_per_match=(
                "goals_per_match",
                "mean",
            ),
            mean_home_win_rate=(
                "home_win_rate",
                "mean",
            ),
            mean_draw_rate=(
                "draw_rate",
                "mean",
            ),
            mean_away_win_rate=(
                "away_win_rate",
                "mean",
            ),
            mean_champion_points=(
                "champion_points",
                "mean",
            ),
            mean_bottom_points=(
                "bottom_points",
                "mean",
            ),
        )
        .sort_values(
            "model_id"
        )
        .reset_index(drop=True)
    )

def build_club_summary_display(
    club_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Show the five best mean-position clubs for each model.

    This is only a structural inspection aid while the
    simulation count remains two.
    """

    return (
        club_summary
        .sort_values(
            [
                "model_id",
                "mean_position",
                "club",
            ]
        )
        .groupby(
            "model_id",
            as_index=False,
            group_keys=False,
        )
        .head(5)
        [
            [
                "model_id",
                "club",
                "mean_position",
                "mean_points",
                "mean_goals_for",
                "mean_goals_against",
                "championship_probability",
                "top_four_probability",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

def build_league_fingerprint_display(
    league_fingerprints: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a compact subset of the model fingerprints for
    console inspection.
    """

    return (
        league_fingerprints[
            [
                "model_id",
                "model_name",
                "season_count",
                "goals_per_match_mean",
                "goals_per_match_std",
                "home_win_rate_mean",
                "draw_rate_mean",
                "away_win_rate_mean",
                "champion_points_mean",
                "champion_points_std",
                "bottom_points_mean",
                "points_spread_mean",
                "goal_difference_spread_mean",
            ]
        ]
        .sort_values(
            "model_id"
        )
        .reset_index(
            drop=True
        )
    )

def build_football_model_comparison(
    league_fingerprints: pd.DataFrame,
    *,
    baseline_model_id: str,
    candidate_model_id: str,
) -> pd.DataFrame:
    """
    Compare two model-level league fingerprints.

    Differences are always calculated as:

        candidate - baseline

    This function records descriptive differences only. It does
    not perform significance testing or select a preferred model.
    """

    required_columns = {
        "model_id",
        "model_name",
        "season_count",
    }

    for metric in FINGERPRINT_METRICS:
        required_columns.add(
            f"{metric}_mean"
        )

        required_columns.add(
            f"{metric}_std"
        )

    missing_columns = (
        required_columns
        - set(
            league_fingerprints.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "League fingerprints are missing comparison "
            f"columns: {sorted(missing_columns)}"
        )

    indexed = (
        league_fingerprints
        .set_index(
            "model_id",
            drop=False,
        )
    )

    missing_models = (
        {
            baseline_model_id,
            candidate_model_id,
        }
        - set(indexed.index)
    )

    if missing_models:
        raise ValueError(
            "Comparison models are missing from the league "
            f"fingerprints: {sorted(missing_models)}"
        )

    baseline = indexed.loc[
        baseline_model_id
    ]

    candidate = indexed.loc[
        candidate_model_id
    ]

    if isinstance(
        baseline,
        pd.DataFrame,
    ):
        raise ValueError(
            "Baseline model ID appears more than once."
        )

    if isinstance(
        candidate,
        pd.DataFrame,
    ):
        raise ValueError(
            "Candidate model ID appears more than once."
        )

    rows: list[
        dict[str, object]
    ] = []

    for metric in FINGERPRINT_METRICS:
        baseline_mean = float(
            baseline[
                f"{metric}_mean"
            ]
        )

        candidate_mean = float(
            candidate[
                f"{metric}_mean"
            ]
        )

        baseline_std = float(
            baseline[
                f"{metric}_std"
            ]
        )

        candidate_std = float(
            candidate[
                f"{metric}_std"
            ]
        )

        absolute_difference = (
            candidate_mean
            - baseline_mean
        )

        if baseline_mean == 0.0:
            relative_difference = None
        else:
            relative_difference = (
                absolute_difference
                / abs(
                    baseline_mean
                )
            )

        rows.append(
            {
                "metric_key":
                    metric,
                "metric_name":
                    FINGERPRINT_METRIC_LABELS[
                        metric
                    ],
                "metric_type":
                    (
                        "rate"
                        if metric
                        in RATE_METRICS
                        else "continuous"
                    ),

                "baseline_model_id":
                    baseline_model_id,
                "baseline_model_name":
                    baseline[
                        "model_name"
                    ],
                "baseline_season_count":
                    int(
                        baseline[
                            "season_count"
                        ]
                    ),
                "baseline_mean":
                    baseline_mean,
                "baseline_std":
                    baseline_std,

                "candidate_model_id":
                    candidate_model_id,
                "candidate_model_name":
                    candidate[
                        "model_name"
                    ],
                "candidate_season_count":
                    int(
                        candidate[
                            "season_count"
                        ]
                    ),
                "candidate_mean":
                    candidate_mean,
                "candidate_std":
                    candidate_std,

                "difference_candidate_minus_baseline":
                    absolute_difference,
                "absolute_difference":
                    abs(
                        absolute_difference
                    ),
                "relative_difference":
                    relative_difference,
                "relative_difference_percent":
                    (
                        relative_difference
                        * 100.0
                        if relative_difference
                        is not None
                        else None
                    ),
                "direction":
                    (
                        "candidate_higher"
                        if absolute_difference > 0.0
                        else (
                            "candidate_lower"
                            if absolute_difference < 0.0
                            else "equal"
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )

def build_all_football_model_comparisons(
    league_fingerprints: pd.DataFrame,
) -> pd.DataFrame:
    comparisons: list[pd.DataFrame] = []

    for comparison_spec in (
        MODEL_COMPARISON_PAIRS
    ):
        comparison = (
            build_football_model_comparison(
                league_fingerprints,
                baseline_model_id=(
                    comparison_spec[
                        "baseline_model_id"
                    ]
                ),
                candidate_model_id=(
                    comparison_spec[
                        "candidate_model_id"
                    ]
                ),
            )
        )

        comparison.insert(
            0,
            "comparison_id",
            comparison_spec[
                "comparison_id"
            ],
        )

        comparisons.append(
            comparison
        )

    combined = pd.concat(
        comparisons,
        ignore_index=True,
    )

    return combined

def validate_football_model_comparison(
    *,
    comparison: pd.DataFrame,
    league_fingerprints: pd.DataFrame,
    baseline_model_id: str,
    candidate_model_id: str,
) -> None:
    if comparison.empty:
        raise AssertionError(
            "Football-model comparison is empty."
        )

    if len(comparison) != len(
        FINGERPRINT_METRICS
    ):
        raise AssertionError(
            "Unexpected comparison row count: "
            f"{len(comparison)} vs "
            f"{len(FINGERPRINT_METRICS)}."
        )

    if comparison[
        "metric_key"
    ].duplicated().any():
        raise AssertionError(
            "Football-model comparison contains duplicate "
            "metric rows."
        )

    if set(
        comparison[
            "metric_key"
        ].astype(str)
    ) != set(
        FINGERPRINT_METRICS
    ):
        raise AssertionError(
            "Football-model comparison does not contain "
            "the complete metric registry."
        )

    if not comparison[
        "baseline_model_id"
    ].eq(
        baseline_model_id
    ).all():
        raise AssertionError(
            "Comparison contains an unexpected baseline "
            "model ID."
        )

    if not comparison[
        "candidate_model_id"
    ].eq(
        candidate_model_id
    ).all():
        raise AssertionError(
            "Comparison contains an unexpected candidate "
            "model ID."
        )

    finite_columns = [
        "baseline_mean",
        "baseline_std",
        "candidate_mean",
        "candidate_std",
        "difference_candidate_minus_baseline",
        "absolute_difference",
    ]

    finite_values = comparison[
        finite_columns
    ].to_numpy(
        dtype=float
    )

    if not pd.notna(
        finite_values
    ).all():
        raise AssertionError(
            "Comparison contains missing required numeric "
            "values."
        )

    if not (
        comparison[
            "absolute_difference"
        ]
        - comparison[
            "difference_candidate_minus_baseline"
        ].abs()
    ).abs().le(
        1e-12
    ).all():
        raise AssertionError(
            "Stored absolute differences are inconsistent."
        )

    indexed = (
        league_fingerprints
        .set_index(
            "model_id"
        )
    )

    baseline = indexed.loc[
        baseline_model_id
    ]

    candidate = indexed.loc[
        candidate_model_id
    ]

    for row in comparison.itertuples(
        index=False
    ):
        expected_baseline = float(
            baseline[
                f"{row.metric_key}_mean"
            ]
        )

        expected_candidate = float(
            candidate[
                f"{row.metric_key}_mean"
            ]
        )

        expected_difference = (
            expected_candidate
            - expected_baseline
        )

        if abs(
            row.baseline_mean
            - expected_baseline
        ) > 1e-12:
            raise AssertionError(
                "Comparison baseline mean does not "
                f"reconcile for {row.metric_key!r}."
            )

        if abs(
            row.candidate_mean
            - expected_candidate
        ) > 1e-12:
            raise AssertionError(
                "Comparison candidate mean does not "
                f"reconcile for {row.metric_key!r}."
            )

        if abs(
            row.difference_candidate_minus_baseline
            - expected_difference
        ) > 1e-12:
            raise AssertionError(
                "Comparison difference does not reconcile "
                f"for {row.metric_key!r}."
            )

    if not comparison[
        "direction"
    ].isin(
        {
            "candidate_higher",
            "candidate_lower",
            "equal",
        }
    ).all():
        raise AssertionError(
            "Comparison contains an invalid direction label."
        )

def validate_all_football_model_comparisons(
    *,
    comparisons: pd.DataFrame,
    league_fingerprints: pd.DataFrame,
) -> None:
    if comparisons.empty:
        raise AssertionError(
            "Combined football-model comparison is empty."
        )

    expected_comparison_ids = {
        comparison_spec[
            "comparison_id"
        ]
        for comparison_spec
        in MODEL_COMPARISON_PAIRS
    }

    observed_comparison_ids = set(
        comparisons[
            "comparison_id"
        ].astype(str)
    )

    if observed_comparison_ids != (
        expected_comparison_ids
    ):
        raise AssertionError(
            "Combined comparison contains an unexpected "
            "comparison population. "
            f"Expected={sorted(expected_comparison_ids)}, "
            f"observed={sorted(observed_comparison_ids)}."
        )

    expected_row_count = (
        len(
            MODEL_COMPARISON_PAIRS
        )
        * len(
            FINGERPRINT_METRICS
        )
    )

    if len(comparisons) != (
        expected_row_count
    ):
        raise AssertionError(
            "Unexpected combined comparison row count: "
            f"{len(comparisons)} vs "
            f"{expected_row_count}."
        )

    if comparisons[
        [
            "comparison_id",
            "metric_key",
        ]
    ].duplicated().any():
        duplicates = comparisons.loc[
            comparisons[
                [
                    "comparison_id",
                    "metric_key",
                ]
            ].duplicated(
                keep=False
            ),
            [
                "comparison_id",
                "metric_key",
            ],
        ]

        raise AssertionError(
            "Combined comparison contains duplicate "
            "comparison-metric rows:\n"
            f"{duplicates.to_string(index=False)}"
        )

    for comparison_spec in (
        MODEL_COMPARISON_PAIRS
    ):
        comparison_id = (
            comparison_spec[
                "comparison_id"
            ]
        )

        selected = comparisons.loc[
            comparisons[
                "comparison_id"
            ].eq(
                comparison_id
            )
        ].copy()

        validate_football_model_comparison(
            comparison=selected,
            league_fingerprints=(
                league_fingerprints
            ),
            baseline_model_id=(
                comparison_spec[
                    "baseline_model_id"
                ]
            ),
            candidate_model_id=(
                comparison_spec[
                    "candidate_model_id"
                ]
            ),
        )

def build_model_comparison_display(
    comparisons: pd.DataFrame,
    *,
    comparison_id: str,
) -> pd.DataFrame:
    """
    Display a compact subset of one pairwise model
    comparison.
    """

    selected_metrics = (
        "goals_per_match",
        "home_goals_per_match",
        "away_goals_per_match",
        "home_win_rate",
        "draw_rate",
        "away_win_rate",
        "champion_points",
        "bottom_points",
        "points_spread",
        "goal_difference_spread",
    )

    selected = comparisons.loc[
        comparisons[
            "comparison_id"
        ].eq(
            comparison_id
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            "No comparison rows exist for "
            f"{comparison_id!r}."
        )

    return (
        selected.loc[
            selected[
                "metric_key"
            ].isin(
                selected_metrics
            ),
            [
                "comparison_id",
                "metric_key",
                "metric_name",
                "baseline_model_id",
                "candidate_model_id",
                "baseline_mean",
                "candidate_mean",
                (
                    "difference_candidate_"
                    "minus_baseline"
                ),
                "relative_difference_percent",
                "direction",
            ],
        ]
        .set_index(
            "metric_key"
        )
        .loc[
            list(
                selected_metrics
            )
        ]
        .reset_index()
    )

def main() -> None:
    print("=" * 88)
    print(
        "FOOTBALL MODEL BENCHMARK 001B — "
        "LEAGUE FINGERPRINT PHASE 1"
    )
    print("=" * 88)

    config = build_benchmark_config()

    model_specs = build_model_specs()

    print()
    print("Benchmark configuration")
    print("-" * 88)
    print(
        f"  Football models: "
        f"{len(model_specs)}"
    )
    print(
        f"  Canonical clubs: "
        f"{len(CANONICAL_CLUBS)}"
    )
    print(
        f"  Seasons per model: "
        f"{SIMULATION_COUNT}"
    )
    print(
        f"  Base seed: "
        f"{BASE_SEED}"
    )
    print(
        "  Competition format: "
        "double round robin"
    )

    print()
    print("Running repeated league seasons...")

    result = (
        FootballModelBenchmarkEngine(
            config
        ).run(
            model_specs
        )
    )

    validate_phase_one_outputs(
        season_statistics=(
            result.season_statistics
        ),
        club_statistics=(
            result.club_season_statistics
        ),
        match_statistics=(
            result.match_statistics
        ),
    )

    club_summary = build_club_summary(
        result.club_season_statistics
    )

    validate_club_summary(
        club_summary=club_summary,
        club_statistics=(
            result.club_season_statistics
        ),
    )

    fingerprint_result = (
        aggregate_league_fingerprints(
            result.season_statistics
        )
    )

    league_fingerprints = (
        fingerprint_result
        .league_fingerprints
    )

    validate_league_fingerprints(
        fingerprint_result=(
            fingerprint_result
        ),
        season_statistics=(
            result.season_statistics
        ),
        expected_model_ids={
            "FM001",
            "FM002",
            "FM003",
        },
        expected_season_count=(
            SIMULATION_COUNT
        ),
    )

    football_model_comparisons = (
        build_all_football_model_comparisons(
            league_fingerprints
        )
    )

    validate_all_football_model_comparisons(
        comparisons=(
            football_model_comparisons
        ),
        league_fingerprints=(
            league_fingerprints
        ),
    )
    RAW_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.season_statistics.to_csv(
        SEASON_STATISTICS_PATH,
        index=False,
    )

    result.club_season_statistics.to_csv(
        CLUB_SEASON_STATISTICS_PATH,
        index=False,
    )

    result.match_statistics.to_csv(
        MATCH_STATISTICS_PATH,
        index=False,
    )

    SUMMARY_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    club_summary.to_csv(
        CLUB_SUMMARY_PATH,
        index=False,
    )

    league_fingerprints.to_csv(
        LEAGUE_FINGERPRINT_PATH,
        index=False,
    )

    football_model_comparisons.to_csv(
        FOOTBALL_MODEL_COMPARISON_PATH,
        index=False,
    )

    fixture_count_per_season = (
        len(CANONICAL_CLUBS)
        * (
            len(CANONICAL_CLUBS) - 1
        )
    )

    metadata = {
        "benchmark_id":
            BENCHMARK_ID,
        "benchmark_name":
            BENCHMARK_NAME,
        "benchmark_phase":
            "phase_5_25_season_evaluation",
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "status":
            "PASS",
        "football_model_count":
            len(model_specs),
        "football_models": [
            {
                "model_id":
                    spec.model_id,
                "display_name":
                    spec.display_name,
                "repository_source":
                    spec.football_model
                    .repository_source,
                "match_engine":
                    spec.football_model
                    .match_engine,
            }
            for spec in model_specs
        ],
        "canonical_club_count":
            len(CANONICAL_CLUBS),
        "simulation_count_per_model":
            SIMULATION_COUNT,
        "base_seed":
            BASE_SEED,
        "season_seed_sequence": [
            BASE_SEED + index
            for index in range(
                SIMULATION_COUNT
            )
        ],
        "season_start_date":
            SEASON_START_DATE.isoformat(),
        "days_between_matchdays":
            DAYS_BETWEEN_MATCHDAYS,
        "double_round_robin":
            True,
        "fixture_count_per_season":
            fixture_count_per_season,
        "season_statistics_rows":
            len(
                result.season_statistics
            ),
        "club_season_statistics_rows":
            len(
                result
                .club_season_statistics
            ),
        "match_statistics_rows":
            len(
                result.match_statistics
            ),
        "club_summary_rows":
            len(club_summary),

        "league_fingerprint_rows":
            len(
                league_fingerprints
            ),

        "league_fingerprint_model_count":
            int(
                league_fingerprints[
                    "model_id"
                ].nunique()
            ),

        "league_fingerprint_metric_count":
            len(
                FINGERPRINT_METRICS
            ),

        "league_fingerprint_mean_std_generated":
            True,

        "league_fingerprint_validation_pass":
            True,
        
        "club_summary_model_count":
            int(
                club_summary[
                    "model_id"
                ].nunique()
            ),

        "club_summary_club_count":
            int(
                club_summary[
                    "club"
                ].nunique()
            ),

        "football_model_comparison_rows":
            len(
                football_model_comparisons
            ),

        "football_model_comparison_count":
            len(
                MODEL_COMPARISON_PAIRS
            ),

        "football_model_comparisons": [
            {
                "comparison_id":
                    comparison_spec[
                        "comparison_id"
                    ],
                "baseline_model_id":
                    comparison_spec[
                        "baseline_model_id"
                    ],
                "candidate_model_id":
                    comparison_spec[
                        "candidate_model_id"
                    ],
                "difference_definition":
                    "candidate_minus_baseline",
            }
            for comparison_spec
            in MODEL_COMPARISON_PAIRS
        ],

        "comparison_metric_count_per_pair":
            len(
                FINGERPRINT_METRICS
            ),

        "total_comparison_metric_rows":
            (
                len(
                    MODEL_COMPARISON_PAIRS
                )
                * len(
                    FINGERPRINT_METRICS
                )
            ),

        "comparison_validation_pass":
            True,
        "club_summary_probability_checks_pass":
            True,
        "canonical_names_preserved":
            True,
        "complete_standings_generated":
            True,
        "study_042_fingerprint_reused":
            True,
        "aggregated_model_comparison_performed":
            True,
        "historical_reality_comparison_performed":
            False,
        "model_selection_decision":
            False,
        "interpretation_boundary": (
            "This benchmark compares the descriptive league "
            "fingerprints of FM001, FM002, and FM003 across "
            "25 simulated seasons per model. Pairwise differences "
            "are reported for FM002 minus FM001, FM003 minus FM001, "
            "and FM003 minus FM002. The results support exploratory "
            "football-model interpretation, but they are not yet "
            "a comparison against historical reality and do not "
            "constitute a production model-selection decision."
        ),
        "outputs": [
            str(
                SEASON_STATISTICS_PATH
                .relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                CLUB_SEASON_STATISTICS_PATH
                .relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                MATCH_STATISTICS_PATH
                .relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                CLUB_SUMMARY_PATH
                .relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                LEAGUE_FINGERPRINT_PATH
                .relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                FOOTBALL_MODEL_COMPARISON_PATH
                .relative_to(
                    PROJECT_ROOT
                )
            ),
        ],
    }

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = build_console_summary(
        result.season_statistics
    )

    print()
    print("Phase 1 summary")
    print("-" * 88)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    club_display = (
        build_club_summary_display(
            club_summary
        )
    )

    print()
    print("Club summary inspection")
    print("-" * 88)
    print(
        club_display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    fingerprint_display = (
        build_league_fingerprint_display(
            league_fingerprints
        )
    )

    print()
    print("League fingerprint inspection")
    print("-" * 88)
    print(
        fingerprint_display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    for comparison_spec in (
        MODEL_COMPARISON_PAIRS
    ):
        comparison_id = (
            comparison_spec[
                "comparison_id"
            ]
        )

        comparison_display = (
            build_model_comparison_display(
                football_model_comparisons,
                comparison_id=(
                    comparison_id
                ),
            )
        )

        print()
        print(
            "Football model comparison "
            f"({comparison_spec['candidate_model_id']} "
            "minus "
            f"{comparison_spec['baseline_model_id']})"
        )
        print("-" * 88)
        print(
            comparison_display.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )
    print()
    print("Validation summary")
    print(
        "  Football-model construction: PASS"
    )
    print(
        "  Repeated-season execution: PASS"
    )
    print(
        "  Complete standings generation: PASS"
    )
    print(
        "  Match population preservation: PASS"
    )
    print(
        "  Canonical club names preserved: PASS"
    )
    print(
        "  Study 042 fingerprint integration: PASS"
    )
    print(
        "  Raw CSV outputs written: PASS"
    )
    print(
        "  Model comparison performed: YES"
    )
    print(
        "  Historical comparison performed: NO"
    )

    print(
        "  Club-summary aggregation: PASS"
    )
    print(
        "  Club-summary population: PASS"
    )
    print(
        "  Championship probability totals: PASS"
    )
    print(
        "  Top-four probability totals: PASS"
    )
    print(
        "  Bottom-three probability totals: PASS"
    )
    print(
        "  Club summary CSV written: PASS"
    )
    print(
        "  League-fingerprint aggregation: PASS"
    )
    print(
        "  One fingerprint per football model: PASS"
    )
    print(
        "  Fingerprint mean reconciliation: PASS"
    )
    print(
        "  Fingerprint dispersion validation: PASS"
    )
    print(
        "  Fingerprint rate validation: PASS"
    )
    print(
        "  League fingerprint CSV written: PASS"
    )
    print(
        "  Pairwise football-model comparisons: PASS"
    )
    print(
        "  Comparison metric population: PASS"
    )
    print(
        "  All pairwise mean reconciliations: PASS"
    )
    print(
        "  All candidate-minus-baseline differences: PASS"
    )
    print(
        "  Football model comparison CSV written: PASS"
    )
    print(
        "  Aggregated model comparison performed: YES"
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
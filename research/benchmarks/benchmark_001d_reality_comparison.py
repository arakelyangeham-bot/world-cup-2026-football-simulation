#benchmark_001d_reality_comparison

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.benchmarking.fingerprint_aggregation import (
    RATE_METRICS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SIMULATED_FINGERPRINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "football_model_benchmarks"
    / "benchmark_001b"
    / "summary"
    / "league_fingerprint.csv"
)

REALITY_FINGERPRINT_PATHS = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_042_cross_league_opta_prior_calibration"
    / "outputs"
)

REALITY_FILES = (
    REALITY_FINGERPRINT_PATHS
    / "league_fingerprints_2022.csv",

    REALITY_FINGERPRINT_PATHS
    / "league_fingerprints_2023.csv",

    REALITY_FINGERPRINT_PATHS
    / "league_fingerprints_2024.csv",

    REALITY_FINGERPRINT_PATHS
    / "league_fingerprints_2025.csv",
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "football_model_benchmarks"
    / "benchmark_001d"
)

REALITY_SEASONS_PATH = (
    OUTPUT_DIRECTORY
    / "reality_season_fingerprints.csv"
)

REALITY_MEAN_PATH = (
    OUTPUT_DIRECTORY
    / "reality_mean_fingerprint.csv"
)

MODEL_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "football_model_vs_reality.csv"
)

MODEL_ERROR_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "model_error_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "benchmark_001d_metadata.json"
)

COMPARABLE_METRICS = (
    "goals_per_match",
    "home_goals_per_match",
    "away_goals_per_match",

    "home_win_rate",
    "draw_rate",
    "away_win_rate",

    "both_teams_to_score_rate",

    "zero_goal_match_rate",
    "one_goal_match_rate",
    "two_goal_match_rate",
    "three_goal_match_rate",
    "four_plus_goal_match_rate",

    "one_goal_margin_rate",
    "three_plus_goal_margin_rate",
)


UNAVAILABLE_REALITY_METRICS = (
    "champion_points",
    "bottom_points",
    "points_spread",
    "goal_difference_spread",
)


METRIC_LABELS = {
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
}

def load_reality_seasons() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    required_columns = {
        "competition_key",
        "competition_name",
        "season_start_year",
        "matches",
        "clubs",
        *COMPARABLE_METRICS,
    }

    for path in REALITY_FILES:
        if not path.exists():
            raise FileNotFoundError(
                "Real league-fingerprint file does not exist: "
                f"{path}"
            )

        dataframe = pd.read_csv(
            path
        )

        if dataframe.empty:
            raise ValueError(
                f"Real fingerprint file is empty: {path}"
            )

        missing = (
            required_columns
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: "
                f"{sorted(missing)}"
            )

        premier_league = dataframe.loc[
            dataframe[
                "competition_key"
            ].eq(
                "premier_league"
            )
        ].copy()

        if len(premier_league) != 1:
            raise ValueError(
                f"{path.name} must contain exactly one "
                "Premier League row."
            )

        premier_league.insert(
            0,
            "source_file",
            path.name,
        )

        rows.append(
            premier_league
        )

    combined = pd.concat(
        rows,
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values(
            "season_start_year"
        )
        .reset_index(
            drop=True
        )
    )

    return combined

def validate_reality_seasons(
    reality_seasons: pd.DataFrame,
) -> None:
    expected_years = {
        2022,
        2023,
        2024,
        2025,
    }

    observed_years = set(
        pd.to_numeric(
            reality_seasons[
                "season_start_year"
            ],
            errors="raise",
        ).astype(int)
    )

    if observed_years != expected_years:
        raise AssertionError(
            "Unexpected reality-season population. "
            f"Expected={sorted(expected_years)}, "
            f"observed={sorted(observed_years)}."
        )

    if reality_seasons[
        "season_start_year"
    ].duplicated().any():
        raise AssertionError(
            "Reality fingerprints contain duplicate seasons."
        )

    if not reality_seasons[
        "matches"
    ].eq(
        380
    ).all():
        raise AssertionError(
            "At least one real Premier League fingerprint "
            "does not contain 380 matches."
        )

    if not reality_seasons[
        "clubs"
    ].eq(
        20
    ).all():
        raise AssertionError(
            "At least one real Premier League fingerprint "
            "does not contain 20 clubs."
        )

    values = reality_seasons[
        list(
            COMPARABLE_METRICS
        )
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise AssertionError(
            "Reality fingerprints contain non-finite values."
        )

    for metric in (
        set(COMPARABLE_METRICS)
        & RATE_METRICS
    ):
        if not reality_seasons[
            metric
        ].between(
            0.0,
            1.0,
        ).all():
            raise AssertionError(
                f"Reality metric {metric!r} contains a "
                "value outside [0, 1]."
            )

    outcome_sum = reality_seasons[
        [
            "home_win_rate",
            "draw_rate",
            "away_win_rate",
        ]
    ].sum(
        axis=1
    )

    if not outcome_sum.between(
        1.0 - 1e-12,
        1.0 + 1e-12,
    ).all():
        raise AssertionError(
            "Reality home/draw/away rates do not sum to one."
        )

def build_reality_mean_fingerprint(
    reality_seasons: pd.DataFrame,
) -> pd.DataFrame:
    row: dict[str, object] = {
        "competition_key":
            "premier_league",
        "competition_name":
            "Premier League",
        "season_count":
            len(reality_seasons),
        "season_start_year_min":
            int(
                reality_seasons[
                    "season_start_year"
                ].min()
            ),
        "season_start_year_max":
            int(
                reality_seasons[
                    "season_start_year"
                ].max()
            ),
        "matches_per_season":
            380,
        "clubs_per_season":
            20,
    }

    for metric in COMPARABLE_METRICS:
        values = pd.to_numeric(
            reality_seasons[
                metric
            ],
            errors="raise",
        )

        row[
            f"{metric}_mean"
        ] = float(
            values.mean()
        )

        row[
            f"{metric}_std"
        ] = float(
            values.std()
        )

        row[
            f"{metric}_minimum"
        ] = float(
            values.min()
        )

        row[
            f"{metric}_maximum"
        ] = float(
            values.max()
        )

    return pd.DataFrame(
        [row]
    )

def load_simulated_fingerprints() -> pd.DataFrame:
    if not SIMULATED_FINGERPRINT_PATH.exists():
        raise FileNotFoundError(
            "Simulated league fingerprint does not exist: "
            f"{SIMULATED_FINGERPRINT_PATH}"
        )

    dataframe = pd.read_csv(
        SIMULATED_FINGERPRINT_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "Simulated league fingerprint is empty."
        )

    required = {
        "model_id",
        "model_name",
        "season_count",
    }

    for metric in COMPARABLE_METRICS:
        required.add(
            f"{metric}_mean"
        )

        required.add(
            f"{metric}_std"
        )

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Simulated fingerprint is missing columns: "
            f"{sorted(missing)}"
        )

    expected_models = {
        "FM001",
        "FM002",
        "FM003",
    }

    observed_models = set(
        dataframe[
            "model_id"
        ].astype(str)
    )

    if observed_models != expected_models:
        raise ValueError(
            "Unexpected simulated model population. "
            f"Expected={sorted(expected_models)}, "
            f"observed={sorted(observed_models)}."
        )

    return dataframe

def build_model_vs_reality(
    *,
    simulated: pd.DataFrame,
    reality_mean: pd.DataFrame,
) -> pd.DataFrame:
    reality = reality_mean.iloc[0]

    rows: list[
        dict[str, object]
    ] = []

    for model in simulated.itertuples(
        index=False
    ):
        model_values = model._asdict()

        for metric in COMPARABLE_METRICS:
            reality_value = float(
                reality[
                    f"{metric}_mean"
                ]
            )

            simulated_value = float(
                model_values[
                    f"{metric}_mean"
                ]
            )

            signed_error = (
                simulated_value
                - reality_value
            )

            absolute_error = abs(
                signed_error
            )

            relative_absolute_error = (
                absolute_error
                / abs(
                    reality_value
                )
                if reality_value != 0.0
                else np.nan
            )

            rows.append(
                {
                    "model_id":
                        model_values[
                            "model_id"
                        ],
                    "model_name":
                        model_values[
                            "model_name"
                        ],
                    "simulated_season_count":
                        int(
                            model_values[
                                "season_count"
                            ]
                        ),

                    "metric_key":
                        metric,
                    "metric_name":
                        METRIC_LABELS[
                            metric
                        ],
                    "metric_type":
                        (
                            "rate"
                            if metric in RATE_METRICS
                            else "continuous"
                        ),

                    "reality_season_count":
                        int(
                            reality[
                                "season_count"
                            ]
                        ),
                    "reality_mean":
                        reality_value,
                    "reality_std":
                        float(
                            reality[
                                f"{metric}_std"
                            ]
                        ),

                    "simulated_mean":
                        simulated_value,
                    "simulated_std":
                        float(
                            model_values[
                                f"{metric}_std"
                            ]
                        ),

                    "signed_error_simulation_minus_reality":
                        signed_error,
                    "absolute_error":
                        absolute_error,
                    "relative_absolute_error":
                        relative_absolute_error,
                    "relative_absolute_error_percent":
                        (
                            relative_absolute_error
                            * 100.0
                            if np.isfinite(
                                relative_absolute_error
                            )
                            else np.nan
                        ),

                    "bias_direction":
                        (
                            "overestimates"
                            if signed_error > 0.0
                            else (
                                "underestimates"
                                if signed_error < 0.0
                                else "matches"
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )

def add_metric_winners(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    output = comparison.copy()

    minimum_errors = (
        output
        .groupby(
            "metric_key"
        )[
            "absolute_error"
        ]
        .transform(
            "min"
        )
    )

    output[
        "is_closest_model"
    ] = (
        output[
            "absolute_error"
        ]
        .sub(
            minimum_errors
        )
        .abs()
        .le(
            1e-12
        )
    )

    winner_counts = (
        output
        .groupby(
            "metric_key"
        )[
            "is_closest_model"
        ]
        .sum()
    )

    tied_metrics = set(
        winner_counts.loc[
            winner_counts.gt(1)
        ].index
    )

    output[
        "metric_result"
    ] = np.where(
        output[
            "metric_key"
        ].isin(
            tied_metrics
        ),
        "tie",
        np.where(
            output[
                "is_closest_model"
            ],
            "win",
            "loss",
        ),
    )

    return output

def build_model_error_summary(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    working = comparison.copy()

    working[
        "metric_win"
    ] = working[
        "metric_result"
    ].eq(
        "win"
    )

    working[
        "metric_tie"
    ] = working[
        "metric_result"
    ].eq(
        "tie"
    )

    summary = (
        working
        .groupby(
            [
                "model_id",
                "model_name",
            ],
            as_index=False,
        )
        .agg(
            metric_count=(
                "metric_key",
                "count",
            ),
            metric_wins=(
                "metric_win",
                "sum",
            ),
            metric_ties=(
                "metric_tie",
                "sum",
            ),
            mean_absolute_error=(
                "absolute_error",
                "mean",
            ),
            median_absolute_error=(
                "absolute_error",
                "median",
            ),
            mean_relative_absolute_error=(
                "relative_absolute_error",
                "mean",
            ),
            median_relative_absolute_error=(
                "relative_absolute_error",
                "median",
            ),
        )
        .sort_values(
            [
                "mean_relative_absolute_error",
                "model_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary

def validate_outputs(
    *,
    reality_seasons: pd.DataFrame,
    reality_mean: pd.DataFrame,
    simulated: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    expected_comparison_rows = (
        3
        * len(
            COMPARABLE_METRICS
        )
    )

    if len(comparison) != (
        expected_comparison_rows
    ):
        raise AssertionError(
            "Unexpected model-versus-reality row count: "
            f"{len(comparison)} vs "
            f"{expected_comparison_rows}."
        )

    if comparison[
        [
            "model_id",
            "metric_key",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Model-versus-reality comparison contains "
            "duplicate model-metric rows."
        )

    expected_models = {
        "FM001",
        "FM002",
        "FM003",
    }

    if len(summary) != len(expected_models):
        raise AssertionError(
            "Model error summary contains an unexpected "
            "number of rows."
        )

    if len(reality_mean) != 1:
        raise AssertionError(
            "Reality mean fingerprint must contain one row."
        )

    numeric_columns = [
        "reality_mean",
        "reality_std",
        "simulated_mean",
        "simulated_std",
        "signed_error_simulation_minus_reality",
        "absolute_error",
        "relative_absolute_error",
    ]

    values = comparison[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise AssertionError(
            "Model-versus-reality comparison contains "
            "non-finite numeric values."
        )

    error_reconciliation = (
        comparison[
            "signed_error_simulation_minus_reality"
        ]
        - (
            comparison[
                "simulated_mean"
            ]
            - comparison[
                "reality_mean"
            ]
        )
    )

    if not error_reconciliation.abs().le(
        1e-12
    ).all():
        raise AssertionError(
            "Signed errors do not reconcile."
        )

    absolute_reconciliation = (
        comparison[
            "absolute_error"
        ]
        - comparison[
            "signed_error_simulation_minus_reality"
        ].abs()
    )

    if not absolute_reconciliation.abs().le(
        1e-12
    ).all():
        raise AssertionError(
            "Absolute errors do not reconcile."
        )

    for metric_key, group in comparison.groupby("metric_key"):

        winner_count = (
            group["metric_result"]
            .eq("win")
            .sum()
        )

        tie_count = (
            group["metric_result"]
            .eq("tie")
            .sum()
        )

        #
        # Exactly one winner
        #
        if winner_count == 1:
            continue

        #
        # Or two-or-more tied winners
        #
        if winner_count == 0 and tie_count >= 2:
            continue

        raise AssertionError(
            "Invalid winner assignment for "
            f"{metric_key!r}. "
            f"wins={winner_count}, "
            f"ties={tie_count}."
        )

    if set(
        simulated[
            "model_id"
        ]
    ) != set(
        summary[
            "model_id"
        ]
    ):
        raise AssertionError(
            "Model error summary does not preserve the "
            "simulated model population."
        )

def main() -> None:
    print("=" * 88)
    print(
        "FOOTBALL MODEL BENCHMARK 001D — "
        "REALITY FINGERPRINT COMPARISON"
    )
    print("=" * 88)

    reality_seasons = (
        load_reality_seasons()
    )

    validate_reality_seasons(
        reality_seasons
    )

    reality_mean = (
        build_reality_mean_fingerprint(
            reality_seasons
        )
    )

    simulated = (
        load_simulated_fingerprints()
    )

    comparison = (
        build_model_vs_reality(
            simulated=simulated,
            reality_mean=reality_mean,
        )
    )

    comparison = add_metric_winners(
        comparison
    )

    summary = build_model_error_summary(
        comparison
    )

    validate_outputs(
        reality_seasons=reality_seasons,
        reality_mean=reality_mean,
        simulated=simulated,
        comparison=comparison,
        summary=summary,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    reality_seasons.to_csv(
        REALITY_SEASONS_PATH,
        index=False,
    )

    reality_mean.to_csv(
        REALITY_MEAN_PATH,
        index=False,
    )

    comparison.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    summary.to_csv(
        MODEL_ERROR_SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "benchmark_id":
            "001D",
        "benchmark_name":
            "Football Model vs Reality",
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "status":
            "PASS",

        "competition_key":
            "premier_league",
        "reality_season_start_years":
            [2022, 2023, 2024, 2025],
        "reality_season_count":
            len(reality_seasons),
        "reality_matches_per_season":
            380,
        "reality_clubs_per_season":
            20,

        "simulated_model_ids":
            sorted(
                simulated[
                    "model_id"
                ].astype(str)
            ),
        "simulated_seasons_per_model":
            {
                str(row.model_id):
                    int(
                        row.season_count
                    )
                for row in simulated.itertuples(
                    index=False
                )
            },

        "comparable_metric_count":
            len(
                COMPARABLE_METRICS
            ),
        "unavailable_reality_metrics":
            list(
                UNAVAILABLE_REALITY_METRICS
            ),

        "comparison_rows":
            len(comparison),
        "summary_rows":
            len(summary),

        "simulation_run":
            False,
        "historical_reality_comparison_performed":
            True,
        "model_selection_decision":
            False,

        "interpretation_boundary": (
            "This benchmark compares 25-season simulated "
            "match-profile fingerprints against the mean of "
            "four completed Premier League seasons. The real "
            "reference contains 20 clubs and 380 matches per "
            "season, while the matched simulated population "
            "contains 17 clubs and 272 matches per season. "
            "Only rate and per-match metrics shared by both "
            "schemas are compared. Table-level metrics are "
            "excluded because the uploaded reality "
            "fingerprints do not contain them."
        ),

        "outputs": [
            REALITY_SEASONS_PATH.name,
            REALITY_MEAN_PATH.name,
            MODEL_COMPARISON_PATH.name,
            MODEL_ERROR_SUMMARY_PATH.name,
            METADATA_PATH.name,
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    display = comparison[
        [
            "metric_name",
            "model_id",
            "reality_mean",
            "simulated_mean",
            "signed_error_simulation_minus_reality",
            "absolute_error",
            "relative_absolute_error_percent",
            "bias_direction",
            "metric_result",
        ]
    ].copy()

    print()
    print("Model-versus-reality comparison")
    print("-" * 88)
    print(
        display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Model error summary")
    print("-" * 88)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Validation summary")
    print("  Four real Premier League seasons loaded: PASS")
    print("  Reality fingerprint aggregation: PASS")
    print("  Simulated fingerprint loading: PASS")
    print("  Shared metric population: PASS")
    print("  Signed-error reconciliation: PASS")
    print("  Absolute-error reconciliation: PASS")
    print("  Per-metric winner assignment: PASS")
    print("  Reality comparison outputs written: PASS")
    print("  Simulation run: NO")
    print("  Model-selection decision: NO")

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
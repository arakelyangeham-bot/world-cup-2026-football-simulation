#build_league_stability_matrix

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "research"
    / "datasets"
    / "league_season_repository"
    / "league_season_repository.csv"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_042_cross_league_opta_prior_calibration"
    / "outputs"
)

DEFAULT_BASE_YEAR = 2023
DEFAULT_COMPARISON_YEAR = 2024

METRICS = {
    "goals_per_match": "Goals per match",
    "home_goals_per_match": "Home goals per match",
    "away_goals_per_match": "Away goals per match",
    "mean_home_goal_difference": (
        "Mean home goal difference"
    ),
    "home_win_rate": "Home win rate",
    "draw_rate": "Draw rate",
    "away_win_rate": "Away win rate",
    "both_teams_to_score_rate": (
        "Both teams to score rate"
    ),
    "home_clean_sheet_rate": (
        "Home clean-sheet rate"
    ),
    "away_clean_sheet_rate": (
        "Away clean-sheet rate"
    ),
    "zero_goal_match_rate": (
        "Zero-goal match rate"
    ),
    "one_goal_match_rate": (
        "One-goal match rate"
    ),
    "two_goal_match_rate": (
        "Two-goal match rate"
    ),
    "three_goal_match_rate": (
        "Three-goal match rate"
    ),
    "four_plus_goal_match_rate": (
        "Four-plus-goal match rate"
    ),
    "one_goal_margin_rate": (
        "One-goal-margin rate"
    ),
    "three_plus_goal_margin_rate": (
        "Three-plus-goal-margin rate"
    ),
    "home_points_per_match": (
        "Home points per match"
    ),
    "away_points_per_match": (
        "Away points per match"
    ),
}

RATE_METRICS = {
    metric
    for metric in METRICS
    if metric.endswith("_rate")
}

REQUIRED_IDENTITY_COLUMNS = {
    "competition_key",
    "competition_name",
    "season_start_year",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build league-level and metric-level temporal "
            "stability outputs from the canonical "
            "league-season repository."
        )
    )

    parser.add_argument(
        "--repository",
        type=Path,
        default=DEFAULT_REPOSITORY_PATH,
        help=(
            "Path to the canonical league-season repository."
        ),
    )

    parser.add_argument(
        "--base-year",
        type=int,
        default=DEFAULT_BASE_YEAR,
        help="Earlier season start year.",
    )

    parser.add_argument(
        "--comparison-year",
        type=int,
        default=DEFAULT_COMPARISON_YEAR,
        help="Later season start year.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for Study 042 stability outputs.",
    )

    arguments = parser.parse_args()

    if (
        arguments.base_year
        >= arguments.comparison_year
    ):
        parser.error(
            "--base-year must be earlier than "
            "--comparison-year."
        )

    return arguments


def load_repository(
    repository_path: Path,
) -> pd.DataFrame:
    if not repository_path.exists():
        raise FileNotFoundError(
            "League-season repository was not found:\n"
            f"{repository_path}"
        )

    repository = pd.read_csv(
        repository_path
    )

    if repository.empty:
        raise ValueError(
            "League-season repository is empty."
        )

    required_columns = (
        REQUIRED_IDENTITY_COLUMNS
        | set(METRICS)
    )

    missing_columns = (
        required_columns
        - set(repository.columns)
    )

    if missing_columns:
        raise ValueError(
            "League-season repository is missing "
            f"required columns: {sorted(missing_columns)}"
        )

    repository = repository.copy()

    repository["competition_key"] = (
        repository["competition_key"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    repository["competition_name"] = (
        repository["competition_name"]
        .astype(str)
        .str.strip()
    )

    repository["season_start_year"] = (
        pd.to_numeric(
            repository["season_start_year"],
            errors="raise",
        )
        .astype(int)
    )

    for metric_key in METRICS:
        repository[metric_key] = (
            pd.to_numeric(
                repository[metric_key],
                errors="raise",
            )
        )

    duplicates = repository[
        repository.duplicated(
            subset=[
                "competition_key",
                "season_start_year",
            ],
            keep=False,
        )
    ]

    if not duplicates.empty:
        raise ValueError(
            "Duplicate league-season observations found: "
            f"{duplicates[['competition_key', 'season_start_year']].to_dict('records')}"
        )

    return repository


def select_comparison_seasons(
    repository: pd.DataFrame,
    base_year: int,
    comparison_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = repository[
        repository["season_start_year"]
        == base_year
    ].copy()

    comparison = repository[
        repository["season_start_year"]
        == comparison_year
    ].copy()

    if base.empty:
        raise ValueError(
            f"No repository rows found for {base_year}."
        )

    if comparison.empty:
        raise ValueError(
            f"No repository rows found for "
            f"{comparison_year}."
        )

    base_keys = set(
        base["competition_key"]
    )

    comparison_keys = set(
        comparison["competition_key"]
    )

    if base_keys != comparison_keys:
        raise ValueError(
            "Competition coverage differs between seasons. "
            f"Only in {base_year}: "
            f"{sorted(base_keys - comparison_keys)}. "
            f"Only in {comparison_year}: "
            f"{sorted(comparison_keys - base_keys)}."
        )

    return base, comparison


def percentile_positions(
    values: pd.Series,
) -> pd.Series:
    """
    Convert league metric values into percentile positions.

    The highest observed value receives 1.0 and the lowest
    receives 0.0. Ties receive their average position.
    """

    count = len(values)

    if count == 1:
        return pd.Series(
            [0.5],
            index=values.index,
            dtype=float,
        )

    ascending_rank = values.rank(
        method="average",
        ascending=True,
    )

    return (
        (ascending_rank - 1.0)
        / (count - 1.0)
    )


def build_stability_matrix(
    base: pd.DataFrame,
    comparison: pd.DataFrame,
    base_year: int,
    comparison_year: int,
) -> pd.DataFrame:
    base_lookup = (
        base.set_index("competition_key")
    )

    comparison_lookup = (
        comparison.set_index(
            "competition_key"
        )
    )

    rows: list[dict[str, object]] = []

    for metric_key, metric_name in (
        METRICS.items()
    ):
        base_values = base_lookup[
            metric_key
        ]

        comparison_values = (
            comparison_lookup[metric_key]
        )

        base_ranks = base_values.rank(
            method="min",
            ascending=False,
        ).astype(int)

        comparison_ranks = (
            comparison_values.rank(
                method="min",
                ascending=False,
            ).astype(int)
        )

        base_percentiles = (
            percentile_positions(
                base_values
            )
        )

        comparison_percentiles = (
            percentile_positions(
                comparison_values
            )
        )

        for competition_key in sorted(
            base_lookup.index
        ):
            base_value = float(
                base_values.loc[
                    competition_key
                ]
            )

            comparison_value = float(
                comparison_values.loc[
                    competition_key
                ]
            )

            absolute_change = (
                comparison_value
                - base_value
            )

            if np.isclose(
                base_value,
                0.0,
            ):
                relative_change = np.nan
            else:
                relative_change = (
                    absolute_change
                    / abs(base_value)
                )

            base_rank = int(
                base_ranks.loc[
                    competition_key
                ]
            )

            comparison_rank = int(
                comparison_ranks.loc[
                    competition_key
                ]
            )

            rank_change = (
                comparison_rank
                - base_rank
            )

            base_percentile = float(
                base_percentiles.loc[
                    competition_key
                ]
            )

            comparison_percentile = float(
                comparison_percentiles.loc[
                    competition_key
                ]
            )

            percentile_change = (
                comparison_percentile
                - base_percentile
            )

            rows.append(
                {
                    "competition_key": (
                        competition_key
                    ),
                    "competition_name": (
                        base_lookup.loc[
                            competition_key,
                            "competition_name",
                        ]
                    ),
                    "metric_key": metric_key,
                    "metric_name": metric_name,
                    "base_year": base_year,
                    "comparison_year": (
                        comparison_year
                    ),
                    "base_value": base_value,
                    "comparison_value": (
                        comparison_value
                    ),
                    "absolute_change": (
                        absolute_change
                    ),
                    "absolute_change_magnitude": (
                        abs(absolute_change)
                    ),
                    "relative_change": (
                        relative_change
                    ),
                    "relative_change_magnitude": (
                        abs(relative_change)
                        if not np.isnan(
                            relative_change
                        )
                        else np.nan
                    ),
                    "base_rank": base_rank,
                    "comparison_rank": (
                        comparison_rank
                    ),
                    "rank_change": rank_change,
                    "rank_change_magnitude": (
                        abs(rank_change)
                    ),
                    "rank_preserved": (
                        base_rank
                        == comparison_rank
                    ),
                    "base_percentile": (
                        base_percentile
                    ),
                    "comparison_percentile": (
                        comparison_percentile
                    ),
                    "percentile_change": (
                        percentile_change
                    ),
                    "percentile_change_magnitude": (
                        abs(percentile_change)
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "metric_key",
                "competition_name",
            ]
        )
        .reset_index(drop=True)
    )


def classify_metric_stability(
    mean_relative_change: float,
    mean_rank_movement: float,
    rank_correlation: float,
) -> str:
    """
    Assign a descriptive stability tier.

    These are screening labels, not statistical tests.
    """

    if (
        mean_relative_change <= 0.08
        and mean_rank_movement <= 0.8
        and rank_correlation >= 0.80
    ):
        return "high"

    if (
        mean_relative_change <= 0.15
        and mean_rank_movement <= 1.4
        and rank_correlation >= 0.40
    ):
        return "moderate"

    return "low"


def build_metric_summary(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (
        metric_key,
        metric_group,
    ) in matrix.groupby(
        "metric_key",
        sort=True,
    ):
        metric_name = str(
            metric_group[
                "metric_name"
            ].iloc[0]
        )

        valid_relative_changes = (
            metric_group[
                "relative_change_magnitude"
            ]
            .dropna()
        )

        mean_relative_change = float(
            valid_relative_changes.mean()
        )

        maximum_relative_change = float(
            valid_relative_changes.max()
        )

        mean_rank_movement = float(
            metric_group[
                "rank_change_magnitude"
            ].mean()
        )

        maximum_rank_movement = int(
            metric_group[
                "rank_change_magnitude"
            ].max()
        )

        rank_preservation_rate = float(
            metric_group[
                "rank_preserved"
            ].mean()
        )

        mean_percentile_movement = float(
            metric_group[
                "percentile_change_magnitude"
            ].mean()
        )

        rank_correlation = float(
            metric_group["base_rank"]
            .corr(
                metric_group[
                    "comparison_rank"
                ],
                method="spearman",
            )
        )

        stability_tier = (
            classify_metric_stability(
                mean_relative_change=(
                    mean_relative_change
                ),
                mean_rank_movement=(
                    mean_rank_movement
                ),
                rank_correlation=(
                    rank_correlation
                ),
            )
        )

        rows.append(
            {
                "metric_key": metric_key,
                "metric_name": metric_name,
                "league_count": (
                    len(metric_group)
                ),
                "mean_absolute_change": float(
                    metric_group[
                        "absolute_change_magnitude"
                    ].mean()
                ),
                "mean_relative_change": (
                    mean_relative_change
                ),
                "maximum_relative_change": (
                    maximum_relative_change
                ),
                "mean_rank_movement": (
                    mean_rank_movement
                ),
                "maximum_rank_movement": (
                    maximum_rank_movement
                ),
                "rank_preservation_rate": (
                    rank_preservation_rate
                ),
                "mean_percentile_movement": (
                    mean_percentile_movement
                ),
                "spearman_rank_correlation": (
                    rank_correlation
                ),
                "stability_tier": (
                    stability_tier
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "stability_tier",
                "mean_relative_change",
                "mean_rank_movement",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def format_metric_value(
    metric_key: str,
    value: float,
) -> str:
    if metric_key in RATE_METRICS:
        return f"{value * 100:.2f}%"

    return f"{value:.3f}"


def build_console_matrix(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    display_rows: list[
        dict[str, object]
    ] = []

    selected_metrics = [
        "goals_per_match",
        "draw_rate",
        "both_teams_to_score_rate",
        "mean_home_goal_difference",
        "three_plus_goal_margin_rate",
    ]

    selected = matrix[
        matrix["metric_key"].isin(
            selected_metrics
        )
    ]

    for row in selected.itertuples(
        index=False
    ):
        display_rows.append(
            {
                "League": (
                    row.competition_name
                ),
                "Metric": row.metric_name,
                "Base": format_metric_value(
                    row.metric_key,
                    row.base_value,
                ),
                "Comparison": (
                    format_metric_value(
                        row.metric_key,
                        row.comparison_value,
                    )
                ),
                "Relative Δ": (
                    (
                        f"{row.relative_change:+.1%}"
                    )
                    if not np.isnan(
                        row.relative_change
                    )
                    else "n/a"
                ),
                "Rank": (
                    f"{row.base_rank}"
                    f"→{row.comparison_rank}"
                ),
            }
        )

    return pd.DataFrame(
        display_rows
    )


def build_console_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    display = summary[
        [
            "metric_name",
            "mean_relative_change",
            "mean_rank_movement",
            "rank_preservation_rate",
            "spearman_rank_correlation",
            "stability_tier",
        ]
    ].copy()

    display = display.rename(
        columns={
            "metric_name": "Metric",
            "mean_relative_change": (
                "Mean Relative Δ"
            ),
            "mean_rank_movement": (
                "Mean Rank Move"
            ),
            "rank_preservation_rate": (
                "Rank Preserved"
            ),
            "spearman_rank_correlation": (
                "Spearman"
            ),
            "stability_tier": (
                "Stability"
            ),
        }
    )

    display["Mean Relative Δ"] = (
        display["Mean Relative Δ"]
        .map(lambda value: f"{value:.1%}")
    )

    display["Mean Rank Move"] = (
        display["Mean Rank Move"]
        .round(2)
    )

    display["Rank Preserved"] = (
        display["Rank Preserved"]
        .map(lambda value: f"{value:.1%}")
    )

    display["Spearman"] = (
        display["Spearman"]
        .round(3)
    )

    return display


def main() -> None:
    arguments = parse_arguments()

    repository = load_repository(
        arguments.repository
    )

    base, comparison = (
        select_comparison_seasons(
            repository=repository,
            base_year=arguments.base_year,
            comparison_year=(
                arguments.comparison_year
            ),
        )
    )

    print(
        "Study 042 — League Temporal Stability"
    )
    print(
        "====================================="
    )
    print(
        f"Repository: "
        f"{arguments.repository}"
    )
    print(
        f"Base season start year: "
        f"{arguments.base_year}"
    )
    print(
        f"Comparison season start year: "
        f"{arguments.comparison_year}"
    )
    print(
        f"Competitions: "
        f"{base['competition_key'].nunique()}"
    )
    print(
        f"Metrics: {len(METRICS)}"
    )
    print()

    matrix = build_stability_matrix(
        base=base,
        comparison=comparison,
        base_year=arguments.base_year,
        comparison_year=(
            arguments.comparison_year
        ),
    )

    expected_rows = (
        base["competition_key"].nunique()
        * len(METRICS)
    )

    if len(matrix) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} stability rows, "
            f"but found {len(matrix)}."
        )

    metric_summary = (
        build_metric_summary(
            matrix
        )
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    year_pair = (
        f"{arguments.base_year}_"
        f"{arguments.comparison_year}"
    )

    matrix_path = (
        arguments.output_directory
        / (
            "league_stability_matrix_"
            f"{year_pair}.csv"
        )
    )

    summary_path = (
        arguments.output_directory
        / (
            "metric_stability_summary_"
            f"{year_pair}.csv"
        )
    )

    matrix.to_csv(
        matrix_path,
        index=False,
        encoding="utf-8",
    )

    metric_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    console_matrix = (
        build_console_matrix(
            matrix
        )
    )

    console_summary = (
        build_console_summary(
            metric_summary
        )
    )

    print("Selected Stability Matrix")
    print("-------------------------")
    print(
        console_matrix.to_string(
            index=False
        )
    )
    print()

    print("Metric Stability Summary")
    print("------------------------")
    print(
        console_summary.to_string(
            index=False
        )
    )
    print()

    print("Output Summary")
    print("--------------")
    print(
        f"Matrix rows: {len(matrix)}"
    )
    print(
        f"Metric summary rows: "
        f"{len(metric_summary)}"
    )
    print(f"Matrix: {matrix_path}")
    print(f"Summary: {summary_path}")
    print()

    print("Study Result")
    print("------------")
    print("PASSED")
    print(
        "League temporal-stability outputs "
        "written successfully."
    )


if __name__ == "__main__":
    main()
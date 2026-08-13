#build_league_fingerprint_uncertainty

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared.competition_registry import (
    get_competition,
)
from research.studies.study_042_cross_league_opta_prior_calibration.build_league_fingerprints import (
    DEFAULT_COMPETITIONS,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_DIRECTORY,
    build_input_path,
    load_canonical_dataset,
    parse_competition_keys,
)


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_RANDOM_SEED = 42

METRIC_LABELS = {
    "goals_per_match": "Goals per match",
    "home_win_rate": "Home win rate",
    "draw_rate": "Draw rate",
    "away_win_rate": "Away win rate",
    "both_teams_to_score_rate": (
        "Both teams to score rate"
    ),
    "mean_home_goal_difference": (
        "Mean home goal difference"
    ),
    "three_plus_goal_margin_rate": (
        "Three-plus goal margin rate"
    ),
}

RATE_METRICS = {
    "home_win_rate",
    "draw_rate",
    "away_win_rate",
    "both_teams_to_score_rate",
    "three_plus_goal_margin_rate",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate bootstrap confidence intervals for "
            "Study 042 league-fingerprint metrics."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help=(
            "Season start year. The default is 2023, "
            "representing 2023–24."
        ),
    )

    parser.add_argument(
        "--competitions",
        type=parse_competition_keys,
        default=DEFAULT_COMPETITIONS,
        help=(
            "Comma-separated competition keys. "
            "Defaults to the five supported major leagues."
        ),
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=(
            "Number of bootstrap resamples per league. "
            "Default: 10000."
        ),
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help=(
            "Confidence level for percentile intervals. "
            "Default: 0.95."
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Root directory containing canonical "
            "historical-match datasets."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for Study 042 outputs.",
    )

    arguments = parser.parse_args()

    if arguments.bootstrap_samples <= 0:
        parser.error(
            "--bootstrap-samples must be greater than zero."
        )

    if not (
        0.0
        < arguments.confidence_level
        < 1.0
    ):
        parser.error(
            "--confidence-level must lie between 0 and 1."
        )

    return arguments


def calculate_metrics_from_scores(
    home_scores: np.ndarray,
    away_scores: np.ndarray,
) -> dict[str, float]:
    """
    Calculate the selected league-fingerprint metrics.

    The inputs may be one-dimensional observed-score arrays or
    two-dimensional bootstrap arrays whose rows are resamples.
    """

    total_goals = home_scores + away_scores
    home_goal_difference = (
        home_scores - away_scores
    )

    return {
        "goals_per_match": float(
            np.mean(total_goals)
        ),
        "home_win_rate": float(
            np.mean(home_scores > away_scores)
        ),
        "draw_rate": float(
            np.mean(home_scores == away_scores)
        ),
        "away_win_rate": float(
            np.mean(home_scores < away_scores)
        ),
        "both_teams_to_score_rate": float(
            np.mean(
                (home_scores > 0)
                & (away_scores > 0)
            )
        ),
        "mean_home_goal_difference": float(
            np.mean(home_goal_difference)
        ),
        "three_plus_goal_margin_rate": float(
            np.mean(
                np.abs(home_goal_difference) >= 3
            )
        ),
    }


def calculate_bootstrap_distributions(
    dataframe: pd.DataFrame,
    bootstrap_samples: int,
    random_generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Resample complete matches with replacement and calculate
    every selected metric for each bootstrap sample.
    """

    home_scores = (
        dataframe["home_score"]
        .to_numpy(dtype=float)
    )

    away_scores = (
        dataframe["away_score"]
        .to_numpy(dtype=float)
    )

    match_count = len(dataframe)

    sampled_indices = random_generator.integers(
        low=0,
        high=match_count,
        size=(
            bootstrap_samples,
            match_count,
        ),
    )

    sampled_home_scores = home_scores[
        sampled_indices
    ]

    sampled_away_scores = away_scores[
        sampled_indices
    ]

    total_goals = (
        sampled_home_scores
        + sampled_away_scores
    )

    home_goal_difference = (
        sampled_home_scores
        - sampled_away_scores
    )

    return {
        "goals_per_match": np.mean(
            total_goals,
            axis=1,
        ),
        "home_win_rate": np.mean(
            sampled_home_scores
            > sampled_away_scores,
            axis=1,
        ),
        "draw_rate": np.mean(
            sampled_home_scores
            == sampled_away_scores,
            axis=1,
        ),
        "away_win_rate": np.mean(
            sampled_home_scores
            < sampled_away_scores,
            axis=1,
        ),
        "both_teams_to_score_rate": np.mean(
            (
                (sampled_home_scores > 0)
                & (sampled_away_scores > 0)
            ),
            axis=1,
        ),
        "mean_home_goal_difference": np.mean(
            home_goal_difference,
            axis=1,
        ),
        "three_plus_goal_margin_rate": np.mean(
            np.abs(
                home_goal_difference
            )
            >= 3,
            axis=1,
        ),
    }


def calculate_percentile_interval(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level

    lower_percentile = (
        100.0 * alpha / 2.0
    )

    upper_percentile = (
        100.0
        * (1.0 - alpha / 2.0)
    )

    lower = float(
        np.percentile(
            values,
            lower_percentile,
        )
    )

    upper = float(
        np.percentile(
            values,
            upper_percentile,
        )
    )

    return lower, upper


def build_uncertainty_rows(
    dataframe: pd.DataFrame,
    competition_key: str,
    season_start_year: int,
    bootstrap_samples: int,
    confidence_level: float,
    random_generator: np.random.Generator,
) -> list[dict[str, object]]:
    competition = get_competition(
        competition_key
    )

    home_scores = (
        dataframe["home_score"]
        .to_numpy(dtype=float)
    )

    away_scores = (
        dataframe["away_score"]
        .to_numpy(dtype=float)
    )

    observed_metrics = (
        calculate_metrics_from_scores(
            home_scores=home_scores,
            away_scores=away_scores,
        )
    )

    bootstrap_distributions = (
        calculate_bootstrap_distributions(
            dataframe=dataframe,
            bootstrap_samples=(
                bootstrap_samples
            ),
            random_generator=(
                random_generator
            ),
        )
    )

    rows: list[dict[str, object]] = []

    for metric_key, observed_value in (
        observed_metrics.items()
    ):
        lower_bound, upper_bound = (
            calculate_percentile_interval(
                values=(
                    bootstrap_distributions[
                        metric_key
                    ]
                ),
                confidence_level=(
                    confidence_level
                ),
            )
        )

        rows.append(
            {
                "competition_key": (
                    competition_key
                ),
                "competition_name": (
                    competition.display_name
                ),
                "season_start_year": (
                    season_start_year
                ),
                "matches": len(dataframe),
                "metric_key": metric_key,
                "metric_name": (
                    METRIC_LABELS[metric_key]
                ),
                "observed_value": (
                    observed_value
                ),
                "confidence_level": (
                    confidence_level
                ),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "interval_width": (
                    upper_bound
                    - lower_bound
                ),
                "bootstrap_samples": (
                    bootstrap_samples
                ),
            }
        )

    return rows


def format_value(
    metric_key: str,
    value: float,
) -> str:
    if metric_key in RATE_METRICS:
        return f"{value * 100:.2f}%"

    return f"{value:.3f}"


def build_console_table(
    uncertainty: pd.DataFrame,
) -> pd.DataFrame:
    display_rows: list[
        dict[str, object]
    ] = []

    for row in uncertainty.itertuples(
        index=False
    ):
        interval = (
            f"{format_value(row.metric_key, row.observed_value)} "
            f"[{format_value(row.metric_key, row.lower_bound)}, "
            f"{format_value(row.metric_key, row.upper_bound)}]"
        )

        display_rows.append(
            {
                "League": (
                    row.competition_name
                ),
                "Metric": row.metric_name,
                "Estimate [CI]": interval,
            }
        )

    return pd.DataFrame(display_rows)


def main() -> None:
    arguments = parse_arguments()

    random_generator = (
        np.random.default_rng(
            arguments.random_seed
        )
    )

    uncertainty_rows: list[
        dict[str, object]
    ] = []

    print(
        "Study 042 — League Fingerprint Uncertainty"
    )
    print(
        "=========================================="
    )
    print(
        f"Season start year: "
        f"{arguments.year}"
    )
    print(
        "Competitions: "
        f"{arguments.competitions}"
    )
    print(
        "Bootstrap samples per league: "
        f"{arguments.bootstrap_samples}"
    )
    print(
        "Confidence level: "
        f"{arguments.confidence_level:.1%}"
    )
    print(
        f"Random seed: "
        f"{arguments.random_seed}"
    )
    print()

    for competition_key in (
        arguments.competitions
    ):
        competition = get_competition(
            competition_key
        )

        if (
            competition.category
            != "domestic_league"
        ):
            raise ValueError(
                f"{competition_key!r} is not "
                "registered as a domestic league."
            )

        input_path = build_input_path(
            input_root=arguments.input_root,
            competition_key=competition_key,
            season_start_year=arguments.year,
        )

        dataframe = load_canonical_dataset(
            input_path=input_path,
            competition_key=competition_key,
            season_start_year=arguments.year,
        )

        league_rows = (
            build_uncertainty_rows(
                dataframe=dataframe,
                competition_key=(
                    competition_key
                ),
                season_start_year=(
                    arguments.year
                ),
                bootstrap_samples=(
                    arguments.bootstrap_samples
                ),
                confidence_level=(
                    arguments.confidence_level
                ),
                random_generator=(
                    random_generator
                ),
            )
        )

        uncertainty_rows.extend(
            league_rows
        )

        print(
            f"Bootstrapped "
            f"{competition.display_name}: "
            f"{len(dataframe)} matches"
        )

    uncertainty_dataframe = (
        pd.DataFrame(
            uncertainty_rows
        )
        .sort_values(
            [
                "metric_key",
                "competition_name",
            ]
        )
        .reset_index(drop=True)
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        arguments.output_directory
        / (
            "league_fingerprint_uncertainty_"
            f"{arguments.year}.csv"
        )
    )

    uncertainty_dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    display_table = build_console_table(
        uncertainty_dataframe
    )

    print()
    print("Bootstrap Confidence Intervals")
    print("------------------------------")
    print(
        display_table.to_string(
            index=False
        )
    )

    print()
    print(f"Output: {output_path}")
    print()
    print("Study Result")
    print("------------")
    print("PASSED")
    print(
        "League-fingerprint uncertainty "
        "estimates written successfully."
    )


if __name__ == "__main__":
    main()
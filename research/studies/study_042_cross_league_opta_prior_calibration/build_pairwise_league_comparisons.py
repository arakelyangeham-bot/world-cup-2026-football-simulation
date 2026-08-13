#build_pairwise_league_comparisons

from __future__ import annotations

import argparse
from itertools import combinations
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
from research.studies.study_042_cross_league_opta_prior_calibration.build_league_fingerprint_uncertainty import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
    METRIC_LABELS,
    RATE_METRICS,
    calculate_bootstrap_distributions,
    calculate_metrics_from_scores,
    calculate_percentile_interval,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pairwise bootstrap comparisons of "
            "Study 042 league-fingerprint metrics."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help=(
            "Season start year. Default: 2023, "
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
            "Independent bootstrap comparisons per "
            "league pair. Default: 10000."
        ),
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help=(
            "Confidence level for difference intervals. "
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


def load_league_datasets(
    competition_keys: list[str],
    year: int,
    input_root: Path,
) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}

    for competition_key in competition_keys:
        competition = get_competition(
            competition_key
        )

        if competition.category != "domestic_league":
            raise ValueError(
                f"{competition_key!r} is not "
                "registered as a domestic league."
            )

        input_path = build_input_path(
            input_root=input_root,
            competition_key=competition_key,
            season_start_year=year,
        )

        dataframe = load_canonical_dataset(
            input_path=input_path,
            competition_key=competition_key,
            season_start_year=year,
        )

        datasets[competition_key] = dataframe

        print(
            f"Loaded {competition.display_name}: "
            f"{len(dataframe)} matches"
        )

    return datasets


def calculate_observed_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    home_scores = (
        dataframe["home_score"]
        .to_numpy(dtype=float)
    )

    away_scores = (
        dataframe["away_score"]
        .to_numpy(dtype=float)
    )

    return calculate_metrics_from_scores(
        home_scores=home_scores,
        away_scores=away_scores,
    )


def interpret_direction(
    probability_a_greater: float,
    probability_b_greater: float,
) -> str:
    if probability_a_greater >= 0.975:
        return "strong_evidence_a_greater"

    if probability_b_greater >= 0.975:
        return "strong_evidence_b_greater"

    if probability_a_greater >= 0.90:
        return "moderate_evidence_a_greater"

    if probability_b_greater >= 0.90:
        return "moderate_evidence_b_greater"

    return "inconclusive"


def build_pairwise_rows(
    datasets: dict[str, pd.DataFrame],
    competition_keys: list[str],
    year: int,
    bootstrap_samples: int,
    confidence_level: float,
    random_generator: np.random.Generator,
) -> list[dict[str, object]]:
    observed_by_league: dict[
        str,
        dict[str, float],
    ] = {}

    bootstrap_by_league: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    for competition_key in competition_keys:
        dataframe = datasets[
            competition_key
        ]

        observed_by_league[
            competition_key
        ] = calculate_observed_metrics(
            dataframe
        )

        bootstrap_by_league[
            competition_key
        ] = calculate_bootstrap_distributions(
            dataframe=dataframe,
            bootstrap_samples=bootstrap_samples,
            random_generator=random_generator,
        )

    rows: list[dict[str, object]] = []

    for (
        competition_a_key,
        competition_b_key,
    ) in combinations(
        competition_keys,
        2,
    ):
        competition_a = get_competition(
            competition_a_key
        )

        competition_b = get_competition(
            competition_b_key
        )

        for metric_key in METRIC_LABELS:
            observed_a = (
                observed_by_league[
                    competition_a_key
                ][metric_key]
            )

            observed_b = (
                observed_by_league[
                    competition_b_key
                ][metric_key]
            )

            observed_difference = (
                observed_a - observed_b
            )

            bootstrap_differences = (
                bootstrap_by_league[
                    competition_a_key
                ][metric_key]
                - bootstrap_by_league[
                    competition_b_key
                ][metric_key]
            )

            lower_bound, upper_bound = (
                calculate_percentile_interval(
                    values=bootstrap_differences,
                    confidence_level=confidence_level,
                )
            )

            probability_a_greater = float(
                np.mean(
                    bootstrap_differences > 0
                )
            )

            probability_b_greater = float(
                np.mean(
                    bootstrap_differences < 0
                )
            )

            probability_equal = float(
                np.mean(
                    bootstrap_differences == 0
                )
            )

            interval_excludes_zero = bool(
                lower_bound > 0
                or upper_bound < 0
            )

            if lower_bound > 0:
                interval_direction = (
                    "a_greater"
                )

            elif upper_bound < 0:
                interval_direction = (
                    "b_greater"
                )

            else:
                interval_direction = (
                    "includes_zero"
                )

            rows.append(
                {
                    "season_start_year": year,
                    "competition_a_key": (
                        competition_a_key
                    ),
                    "competition_a_name": (
                        competition_a.display_name
                    ),
                    "competition_b_key": (
                        competition_b_key
                    ),
                    "competition_b_name": (
                        competition_b.display_name
                    ),
                    "matches_a": len(
                        datasets[
                            competition_a_key
                        ]
                    ),
                    "matches_b": len(
                        datasets[
                            competition_b_key
                        ]
                    ),
                    "metric_key": metric_key,
                    "metric_name": (
                        METRIC_LABELS[
                            metric_key
                        ]
                    ),
                    "observed_a": observed_a,
                    "observed_b": observed_b,
                    "observed_difference_a_minus_b": (
                        observed_difference
                    ),
                    "confidence_level": (
                        confidence_level
                    ),
                    "difference_lower_bound": (
                        lower_bound
                    ),
                    "difference_upper_bound": (
                        upper_bound
                    ),
                    "difference_interval_width": (
                        upper_bound
                        - lower_bound
                    ),
                    "interval_excludes_zero": (
                        interval_excludes_zero
                    ),
                    "interval_direction": (
                        interval_direction
                    ),
                    "probability_a_greater": (
                        probability_a_greater
                    ),
                    "probability_b_greater": (
                        probability_b_greater
                    ),
                    "probability_equal": (
                        probability_equal
                    ),
                    "evidence_label": (
                        interpret_direction(
                            probability_a_greater=(
                                probability_a_greater
                            ),
                            probability_b_greater=(
                                probability_b_greater
                            ),
                        )
                    ),
                    "bootstrap_samples": (
                        bootstrap_samples
                    ),
                }
            )

    return rows


def format_metric_value(
    metric_key: str,
    value: float,
) -> str:
    if metric_key in RATE_METRICS:
        return f"{value * 100:.2f}%"

    return f"{value:.3f}"


def build_console_table(
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    display_rows: list[
        dict[str, object]
    ] = []

    for row in comparisons.itertuples(
        index=False
    ):
        difference_interval = (
            f"{format_metric_value(row.metric_key, row.observed_difference_a_minus_b)} "
            f"[{format_metric_value(row.metric_key, row.difference_lower_bound)}, "
            f"{format_metric_value(row.metric_key, row.difference_upper_bound)}]"
        )

        display_rows.append(
            {
                "League A": (
                    row.competition_a_name
                ),
                "League B": (
                    row.competition_b_name
                ),
                "Metric": row.metric_name,
                "A−B [95% CI]": (
                    difference_interval
                ),
                "P(A>B)": (
                    f"{row.probability_a_greater:.1%}"
                ),
                "Evidence": (
                    row.evidence_label
                ),
            }
        )

    return pd.DataFrame(
        display_rows
    )


def main() -> None:
    arguments = parse_arguments()

    print(
        "Study 042 — Pairwise League Comparisons"
    )
    print(
        "======================================="
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
        "Bootstrap samples: "
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

    datasets = load_league_datasets(
        competition_keys=(
            arguments.competitions
        ),
        year=arguments.year,
        input_root=arguments.input_root,
    )

    random_generator = (
        np.random.default_rng(
            arguments.random_seed
        )
    )

    comparison_rows = (
        build_pairwise_rows(
            datasets=datasets,
            competition_keys=(
                arguments.competitions
            ),
            year=arguments.year,
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

    comparison_dataframe = (
        pd.DataFrame(
            comparison_rows
        )
        .sort_values(
            [
                "metric_key",
                "competition_a_name",
                "competition_b_name",
            ]
        )
        .reset_index(drop=True)
    )

    expected_pair_count = (
        len(arguments.competitions)
        * (
            len(arguments.competitions)
            - 1
        )
        // 2
    )

    expected_row_count = (
        expected_pair_count
        * len(METRIC_LABELS)
    )

    if (
        len(comparison_dataframe)
        != expected_row_count
    ):
        raise ValueError(
            f"Expected {expected_row_count} "
            "comparison rows, but found "
            f"{len(comparison_dataframe)}."
        )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        arguments.output_directory
        / (
            "pairwise_league_comparisons_"
            f"{arguments.year}.csv"
        )
    )

    comparison_dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    display_table = build_console_table(
        comparison_dataframe
    )

    print()
    print("Pairwise Comparison Summary")
    print("---------------------------")
    print(
        display_table.to_string(
            index=False
        )
    )

    print()
    print("Comparison Counts")
    print("-----------------")
    print(
        f"League pairs: "
        f"{expected_pair_count}"
    )
    print(
        f"Metrics per pair: "
        f"{len(METRIC_LABELS)}"
    )
    print(
        f"Rows written: "
        f"{len(comparison_dataframe)}"
    )
    print()
    print(f"Output: {output_path}")
    print()
    print("Study Result")
    print("------------")
    print("PASSED")
    print(
        "Pairwise probabilistic league "
        "comparisons written successfully."
    )


if __name__ == "__main__":
    main()
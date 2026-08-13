#estimate_rating_prior_scaling.py

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_COEFFICIENT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "poisson_goal_model_coefficients.csv"
)

DEFAULT_CLUB_DATASET_PATH = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "global_club_prior_dataset.csv"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_042_opta_rating_prior_calibration"
    / "outputs"
    / "phase_2_candidate_scaling"
)

CANDIDATE_SLOPES = [
    1.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
]

OPTA_DIFFERENCES = [
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    15.0,
    20.0,
]

FIFA_REFERENCE_DIFFERENCES = [
    25.0,
    50.0,
    100.0,
    150.0,
    200.0,
    300.0,
]


def load_coefficients(
    path: Path,
) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Coefficient file not found: {path}"
        )

    coefficients: dict[str, dict[str, float]] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            model = row["model"]
            feature = row["feature"]
            coefficient = float(row["coefficient"])

            coefficients.setdefault(
                model,
                {},
            )[feature] = coefficient

    required = {
        "home_goal_model",
        "away_goal_model",
    }

    missing_models = required - set(coefficients)

    if missing_models:
        raise ValueError(
            "Coefficient file is missing models: "
            f"{sorted(missing_models)}"
        )

    for model in required:
        if "fifa_points_diff" not in coefficients[model]:
            raise ValueError(
                f"{model} is missing fifa_points_diff."
            )

    return coefficients


def load_club_dataset(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Club-prior dataset not found: {path}"
        )

    dataframe = pd.read_csv(path)

    required_columns = {
        "club_id",
        "club",
        "opta_rating",
        "global_rank",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Club dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["opta_rating"] = pd.to_numeric(
        dataframe["opta_rating"],
        errors="raise",
    )

    dataframe["global_rank"] = pd.to_numeric(
        dataframe["global_rank"],
        errors="raise",
    ).astype(int)

    return dataframe.sort_values(
        "global_rank"
    ).reset_index(drop=True)


def multiplier(
    coefficient: float,
    prior_difference: float,
) -> float:
    return math.exp(
        coefficient * prior_difference
    )


def percent_change(
    multiplier_value: float,
) -> float:
    return (
        multiplier_value - 1.0
    ) * 100.0


def build_fifa_reference_table(
    home_coefficient: float,
    away_coefficient: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for difference in FIFA_REFERENCE_DIFFERENCES:
        home_multiplier = multiplier(
            home_coefficient,
            difference,
        )

        away_multiplier = multiplier(
            away_coefficient,
            difference,
        )

        rows.append(
            {
                "fifa_point_difference": difference,
                "home_goal_multiplier": home_multiplier,
                "home_goal_percent_change": percent_change(
                    home_multiplier
                ),
                "away_goal_multiplier": away_multiplier,
                "away_goal_percent_change": percent_change(
                    away_multiplier
                ),
            }
        )

    return rows


def build_candidate_effect_table(
    home_coefficient: float,
    away_coefficient: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for slope in CANDIDATE_SLOPES:
        for opta_difference in OPTA_DIFFERENCES:
            prior_difference = (
                slope * opta_difference
            )

            home_multiplier = multiplier(
                home_coefficient,
                prior_difference,
            )

            away_multiplier = multiplier(
                away_coefficient,
                prior_difference,
            )

            rows.append(
                {
                    "candidate_slope": slope,
                    "opta_rating_difference": opta_difference,
                    "implied_prior_difference": prior_difference,
                    "home_goal_multiplier": home_multiplier,
                    "home_goal_percent_change": percent_change(
                        home_multiplier
                    ),
                    "away_goal_multiplier": away_multiplier,
                    "away_goal_percent_change": percent_change(
                        away_multiplier
                    ),
                }
            )

    return rows


def build_elite_difference_summary(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    cutoffs = [
        20,
        50,
        100,
        250,
        500,
    ]

    rows: list[dict[str, Any]] = []

    for cutoff in cutoffs:
        subset = dataframe[
            dataframe["global_rank"] <= cutoff
        ].copy()

        ratings = subset["opta_rating"]

        pairwise_differences = []

        values = ratings.to_numpy()

        for first_index in range(len(values)):
            for second_index in range(
                first_index + 1,
                len(values),
            ):
                pairwise_differences.append(
                    abs(
                        values[first_index]
                        - values[second_index]
                    )
                )

        difference_series = pd.Series(
            pairwise_differences,
            dtype=float,
        )

        rows.append(
            {
                "rank_cutoff": cutoff,
                "club_count": len(subset),
                "comparison_count": len(difference_series),
                "mean_absolute_difference": (
                    difference_series.mean()
                ),
                "median_absolute_difference": (
                    difference_series.median()
                ),
                "percentile_75_difference": (
                    difference_series.quantile(0.75)
                ),
                "percentile_90_difference": (
                    difference_series.quantile(0.90)
                ),
                "maximum_difference": (
                    difference_series.max()
                ),
            }
        )

    return rows


def write_rows_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError(
            f"Cannot write empty output: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    home_coefficient: float,
    away_coefficient: float,
    fifa_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    elite_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Study 042 – Candidate Rating-Prior Scaling Audit",
        "",
        "## Production prior coefficients",
        "",
        (
            "- Home-goal prior coefficient: "
            f"{home_coefficient:.10f}"
        ),
        (
            "- Away-goal prior coefficient: "
            f"{away_coefficient:.10f}"
        ),
        "",
        "The model uses a log link, so prior differences act "
        "multiplicatively on expected goals.",
        "",
        "## FIFA-reference effects",
        "",
        (
            "| FIFA-point difference | Home multiplier "
            "| Home change | Away multiplier | Away change |"
        ),
        "|---:|---:|---:|---:|---:|",
    ]

    for row in fifa_rows:
        lines.append(
            f"| {row['fifa_point_difference']:.0f} "
            f"| {row['home_goal_multiplier']:.4f} "
            f"| {row['home_goal_percent_change']:.2f}% "
            f"| {row['away_goal_multiplier']:.4f} "
            f"| {row['away_goal_percent_change']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Elite Opta pairwise differences",
            "",
            (
                "| Rank cutoff | Clubs | Median difference "
                "| 75th percentile | 90th percentile | Maximum |"
            ),
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in elite_rows:
        lines.append(
            f"| {row['rank_cutoff']} "
            f"| {row['club_count']} "
            f"| {row['median_absolute_difference']:.4f} "
            f"| {row['percentile_75_difference']:.4f} "
            f"| {row['percentile_90_difference']:.4f} "
            f"| {row['maximum_difference']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Candidate slope effects",
            "",
            (
                "| Slope | Opta difference | Implied prior difference "
                "| Home change | Away change |"
            ),
            "|---:|---:|---:|---:|---:|",
        ]
    )

    selected_differences = {
        2.5,
        5.0,
        10.0,
    }

    for row in candidate_rows:
        if (
            row["opta_rating_difference"]
            not in selected_differences
        ):
            continue

        lines.append(
            f"| {row['candidate_slope']:.1f} "
            f"| {row['opta_rating_difference']:.1f} "
            f"| {row['implied_prior_difference']:.1f} "
            f"| {row['home_goal_percent_change']:.2f}% "
            f"| {row['away_goal_percent_change']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This audit does not select a production slope. "
                "It converts each candidate slope into the actual "
                "multiplicative effect imposed by the fitted Poisson "
                "goal coefficients."
            ),
            "",
            (
                "Because the model uses prior differences, an affine "
                "intercept would cancel. Only the slope affects match "
                "predictions under the current model."
            ),
            "",
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the production-model effects of candidate "
            "Opta-to-rating-prior scaling factors."
        )
    )

    parser.add_argument(
        "--coefficients",
        type=Path,
        default=DEFAULT_COEFFICIENT_PATH,
        help="Path to fitted Poisson coefficients.",
    )

    parser.add_argument(
        "--club-dataset",
        type=Path,
        default=DEFAULT_CLUB_DATASET_PATH,
        help="Path to global_club_prior_dataset.csv.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for candidate-scaling outputs.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    coefficients = load_coefficients(
        arguments.coefficients
    )

    dataframe = load_club_dataset(
        arguments.club_dataset
    )

    home_coefficient = coefficients[
        "home_goal_model"
    ]["fifa_points_diff"]

    away_coefficient = coefficients[
        "away_goal_model"
    ]["fifa_points_diff"]

    fifa_rows = build_fifa_reference_table(
        home_coefficient=home_coefficient,
        away_coefficient=away_coefficient,
    )

    candidate_rows = build_candidate_effect_table(
        home_coefficient=home_coefficient,
        away_coefficient=away_coefficient,
    )

    elite_rows = build_elite_difference_summary(
        dataframe
    )

    output_directory = (
        arguments.output_directory
    )

    write_rows_csv(
        fifa_rows,
        output_directory
        / "fifa_reference_effects.csv",
    )

    write_rows_csv(
        candidate_rows,
        output_directory
        / "candidate_slope_effects.csv",
    )

    write_rows_csv(
        elite_rows,
        output_directory
        / "elite_opta_pairwise_differences.csv",
    )

    write_markdown_report(
        home_coefficient=home_coefficient,
        away_coefficient=away_coefficient,
        fifa_rows=fifa_rows,
        candidate_rows=candidate_rows,
        elite_rows=elite_rows,
        output_path=(
            output_directory
            / "candidate_scaling_audit.md"
        ),
    )

    print("Study 042 – Candidate Scaling Audit")
    print("===================================")
    print(
        "Home prior coefficient: "
        f"{home_coefficient:.10f}"
    )
    print(
        "Away prior coefficient: "
        f"{away_coefficient:.10f}"
    )
    print(f"Club records: {len(dataframe)}")
    print()
    print(f"Outputs: {output_directory}")
    print()
    print(
        "Candidate effects calculated. "
        "No production slope was selected."
    )


if __name__ == "__main__":
    main()
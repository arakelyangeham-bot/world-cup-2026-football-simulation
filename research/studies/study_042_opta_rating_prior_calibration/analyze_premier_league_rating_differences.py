#analyze_premier_league_rating_differences

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CLUB_DATASET_PATH = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "global_club_prior_dataset.csv"
)

DEFAULT_COEFFICIENT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "poisson_goal_model_coefficients.csv"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_042_opta_rating_prior_calibration"
    / "outputs"
    / "phase_2b_premier_league_difference_audit"
)

PREMIER_LEAGUE_TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]

CANDIDATE_SLOPES = [
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
]

REFERENCE_QUANTILES = [
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    1.00,
]


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
        "club_full",
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

    return dataframe


def load_prior_coefficients(
    path: Path,
) -> tuple[float, float]:
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

    try:
        home_coefficient = coefficients[
            "home_goal_model"
        ]["fifa_points_diff"]

        away_coefficient = coefficients[
            "away_goal_model"
        ]["fifa_points_diff"]
    except KeyError as exc:
        raise ValueError(
            "Could not locate fitted fifa_points_diff coefficients."
        ) from exc

    return home_coefficient, away_coefficient


def select_premier_league_clubs(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one canonical Opta record for each expected club.

    Where duplicate human-readable names exist globally, use the
    highest-ranked matching record, consistent with Study 041.
    """

    selected_rows: list[pd.Series] = []
    missing_teams: list[str] = []

    for team in PREMIER_LEAGUE_TEAMS:
        matches = dataframe[
            dataframe["club"].eq(team)
        ].copy()

        if matches.empty:
            missing_teams.append(team)
            continue

        matches = matches.sort_values(
            ["global_rank", "opta_rating"],
            ascending=[True, False],
        )

        selected_rows.append(matches.iloc[0])

    if missing_teams:
        raise ValueError(
            "Missing Premier League clubs: "
            f"{missing_teams}"
        )

    selected = pd.DataFrame(
        selected_rows
    ).sort_values(
        "opta_rating",
        ascending=False,
    ).reset_index(drop=True)

    if len(selected) != len(PREMIER_LEAGUE_TEAMS):
        raise ValueError(
            "Premier League selection did not produce exactly "
            f"{len(PREMIER_LEAGUE_TEAMS)} clubs."
        )

    if selected["club_id"].duplicated().any():
        raise ValueError(
            "Premier League selection contains duplicate club IDs."
        )

    return selected


def build_pairwise_difference_table(
    clubs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for first_index, second_index in itertools.combinations(
        range(len(clubs)),
        2,
    ):
        first = clubs.iloc[first_index]
        second = clubs.iloc[second_index]

        if first["opta_rating"] >= second["opta_rating"]:
            stronger = first
            weaker = second
        else:
            stronger = second
            weaker = first

        rating_difference = (
            float(stronger["opta_rating"])
            - float(weaker["opta_rating"])
        )

        rows.append(
            {
                "stronger_club_id": stronger["club_id"],
                "stronger_club": stronger["club"],
                "stronger_rating": float(
                    stronger["opta_rating"]
                ),
                "stronger_global_rank": int(
                    stronger["global_rank"]
                ),
                "weaker_club_id": weaker["club_id"],
                "weaker_club": weaker["club"],
                "weaker_rating": float(
                    weaker["opta_rating"]
                ),
                "weaker_global_rank": int(
                    weaker["global_rank"]
                ),
                "opta_rating_difference": rating_difference,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "opta_rating_difference",
        ascending=False,
    ).reset_index(drop=True)


def build_difference_summary(
    pairwise_table: pd.DataFrame,
) -> dict[str, Any]:
    differences = pairwise_table[
        "opta_rating_difference"
    ]

    return {
        "club_count": 20,
        "pairwise_comparison_count": int(
            len(pairwise_table)
        ),
        "minimum_difference": float(
            differences.min()
        ),
        "mean_difference": float(
            differences.mean()
        ),
        "median_difference": float(
            differences.median()
        ),
        "standard_deviation": float(
            differences.std()
        ),
        "percentile_25_difference": float(
            differences.quantile(0.25)
        ),
        "percentile_75_difference": float(
            differences.quantile(0.75)
        ),
        "percentile_90_difference": float(
            differences.quantile(0.90)
        ),
        "percentile_95_difference": float(
            differences.quantile(0.95)
        ),
        "maximum_difference": float(
            differences.max()
        ),
    }


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


def build_quantile_effect_table(
    pairwise_table: pd.DataFrame,
    home_coefficient: float,
    away_coefficient: float,
) -> list[dict[str, Any]]:
    differences = pairwise_table[
        "opta_rating_difference"
    ]

    rows: list[dict[str, Any]] = []

    for quantile in REFERENCE_QUANTILES:
        opta_difference = float(
            differences.quantile(quantile)
        )

        for slope in CANDIDATE_SLOPES:
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
                    "difference_quantile": quantile,
                    "difference_percentile": (
                        quantile * 100.0
                    ),
                    "opta_rating_difference": (
                        opta_difference
                    ),
                    "candidate_slope": slope,
                    "implied_prior_difference": (
                        prior_difference
                    ),
                    "home_goal_multiplier": (
                        home_multiplier
                    ),
                    "home_goal_percent_change": (
                        percent_change(
                            home_multiplier
                        )
                    ),
                    "away_goal_multiplier": (
                        away_multiplier
                    ),
                    "away_goal_percent_change": (
                        percent_change(
                            away_multiplier
                        )
                    ),
                }
            )

    return rows


def build_matchup_effect_table(
    pairwise_table: pd.DataFrame,
    home_coefficient: float,
    away_coefficient: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for _, matchup in pairwise_table.iterrows():
        opta_difference = float(
            matchup["opta_rating_difference"]
        )

        for slope in CANDIDATE_SLOPES:
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
                    "stronger_club": matchup[
                        "stronger_club"
                    ],
                    "weaker_club": matchup[
                        "weaker_club"
                    ],
                    "opta_rating_difference": (
                        opta_difference
                    ),
                    "candidate_slope": slope,
                    "implied_prior_difference": (
                        prior_difference
                    ),
                    "home_goal_percent_change": (
                        percent_change(
                            home_multiplier
                        )
                    ),
                    "away_goal_percent_change": (
                        percent_change(
                            away_multiplier
                        )
                    ),
                }
            )

    return rows


def write_dict_csv(
    row: dict[str, Any],
    output_path: Path,
) -> None:
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
            fieldnames=list(row.keys()),
        )
        writer.writeheader()
        writer.writerow(row)


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
    clubs: pd.DataFrame,
    pairwise_table: pd.DataFrame,
    summary: dict[str, Any],
    quantile_effect_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Study 042 – Premier League Rating-Difference Audit",
        "",
        "## Competition sample",
        "",
        "| Club | Opta rating | Global rank | Club ID |",
        "|---|---:|---:|---|",
    ]

    for _, row in clubs.iterrows():
        lines.append(
            f"| {row['club']} "
            f"| {float(row['opta_rating']):.4f} "
            f"| {int(row['global_rank'])} "
            f"| {row['club_id']} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise difference distribution",
            "",
            f"- Clubs: {summary['club_count']}",
            (
                "- Pairwise comparisons: "
                f"{summary['pairwise_comparison_count']}"
            ),
            (
                "- Mean difference: "
                f"{summary['mean_difference']:.4f}"
            ),
            (
                "- Median difference: "
                f"{summary['median_difference']:.4f}"
            ),
            (
                "- 75th percentile: "
                f"{summary['percentile_75_difference']:.4f}"
            ),
            (
                "- 90th percentile: "
                f"{summary['percentile_90_difference']:.4f}"
            ),
            (
                "- 95th percentile: "
                f"{summary['percentile_95_difference']:.4f}"
            ),
            (
                "- Maximum difference: "
                f"{summary['maximum_difference']:.4f}"
            ),
            "",
            "## Candidate effects at Premier League quantiles",
            "",
            (
                "| Percentile | Opta difference | Slope "
                "| Implied prior difference | Home change "
                "| Away change |"
            ),
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in quantile_effect_rows:
        lines.append(
            f"| {row['difference_percentile']:.0f}% "
            f"| {row['opta_rating_difference']:.4f} "
            f"| {row['candidate_slope']:.0f} "
            f"| {row['implied_prior_difference']:.2f} "
            f"| {row['home_goal_percent_change']:.2f}% "
            f"| {row['away_goal_percent_change']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Largest Premier League rating gaps",
            "",
            "| Stronger club | Weaker club | Opta difference |",
            "|---|---|---:|",
        ]
    )

    for _, row in pairwise_table.head(15).iterrows():
        lines.append(
            f"| {row['stronger_club']} "
            f"| {row['weaker_club']} "
            f"| {float(row['opta_rating_difference']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This audit evaluates candidate slopes against the "
                "observed rating differences among the 20 selected "
                "Premier League clubs. It does not select a final "
                "production slope."
            ),
            "",
            (
                "The fitted Poisson model uses only rating-prior "
                "differences. Therefore, the implied prior difference "
                "and its multiplicative expected-goal effect are the "
                "quantities relevant to calibration."
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
            "Audit Premier League Opta rating differences and "
            "candidate prior-scaling effects."
        )
    )

    parser.add_argument(
        "--club-dataset",
        type=Path,
        default=DEFAULT_CLUB_DATASET_PATH,
        help="Path to global_club_prior_dataset.csv.",
    )

    parser.add_argument(
        "--coefficients",
        type=Path,
        default=DEFAULT_COEFFICIENT_PATH,
        help="Path to fitted Poisson coefficients.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for audit outputs.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    dataframe = load_club_dataset(
        arguments.club_dataset
    )

    home_coefficient, away_coefficient = (
        load_prior_coefficients(
            arguments.coefficients
        )
    )

    premier_league_clubs = (
        select_premier_league_clubs(
            dataframe
        )
    )

    pairwise_table = (
        build_pairwise_difference_table(
            premier_league_clubs
        )
    )

    summary = build_difference_summary(
        pairwise_table
    )

    quantile_effect_rows = (
        build_quantile_effect_table(
            pairwise_table=pairwise_table,
            home_coefficient=home_coefficient,
            away_coefficient=away_coefficient,
        )
    )

    matchup_effect_rows = (
        build_matchup_effect_table(
            pairwise_table=pairwise_table,
            home_coefficient=home_coefficient,
            away_coefficient=away_coefficient,
        )
    )

    output_directory = (
        arguments.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    premier_league_clubs.to_csv(
        output_directory
        / "premier_league_selected_clubs.csv",
        index=False,
        encoding="utf-8",
    )

    pairwise_table.to_csv(
        output_directory
        / "premier_league_pairwise_differences.csv",
        index=False,
        encoding="utf-8",
    )

    write_dict_csv(
        summary,
        output_directory
        / "premier_league_difference_summary.csv",
    )

    write_rows_csv(
        quantile_effect_rows,
        output_directory
        / "premier_league_quantile_slope_effects.csv",
    )

    write_rows_csv(
        matchup_effect_rows,
        output_directory
        / "premier_league_matchup_slope_effects.csv",
    )

    write_markdown_report(
        clubs=premier_league_clubs,
        pairwise_table=pairwise_table,
        summary=summary,
        quantile_effect_rows=quantile_effect_rows,
        output_path=(
            output_directory
            / "premier_league_difference_audit.md"
        ),
    )

    print(
        "Study 042 – Premier League "
        "Rating-Difference Audit"
    )
    print(
        "=============================================="
    )
    print(
        f"Selected clubs: "
        f"{len(premier_league_clubs)}"
    )
    print(
        "Pairwise comparisons: "
        f"{len(pairwise_table)}"
    )
    print(
        "Median rating difference: "
        f"{summary['median_difference']:.4f}"
    )
    print(
        "90th-percentile difference: "
        f"{summary['percentile_90_difference']:.4f}"
    )
    print(
        "Maximum difference: "
        f"{summary['maximum_difference']:.4f}"
    )
    print()
    print(f"Outputs: {output_directory}")
    print()
    print(
        "Premier League difference audit completed. "
        "No production slope was selected."
    )


if __name__ == "__main__":
    main()
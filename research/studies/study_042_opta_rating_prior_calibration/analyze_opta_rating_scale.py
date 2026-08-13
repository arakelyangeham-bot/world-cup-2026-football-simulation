#analyze_opta_rating_scale

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_PATH = (
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
    / "phase_1_rating_scale_audit"
)

REQUIRED_COLUMNS = {
    "club_id",
    "club",
    "opta_rating",
    "global_rank",
    "snapshot_date",
}

QUANTILES = [
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]

TOP_RANK_CUTOFFS = [
    10,
    20,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
]

RANK_SEPARATIONS = [
    1,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Global Club Prior Dataset not found: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            "Global Club Prior Dataset is empty."
        )

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
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

    dataframe = dataframe.sort_values(
        "global_rank"
    ).reset_index(drop=True)

    return dataframe


def build_overall_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    ratings = dataframe["opta_rating"]

    return {
        "record_count": int(len(dataframe)),
        "snapshot_count": int(
            dataframe["snapshot_date"].nunique()
        ),
        "minimum_rating": float(ratings.min()),
        "maximum_rating": float(ratings.max()),
        "rating_range": float(
            ratings.max() - ratings.min()
        ),
        "mean_rating": float(ratings.mean()),
        "median_rating": float(ratings.median()),
        "standard_deviation": float(ratings.std()),
        "variance": float(ratings.var()),
        "interquartile_range": float(
            ratings.quantile(0.75)
            - ratings.quantile(0.25)
        ),
        "skewness": float(ratings.skew()),
        "kurtosis": float(ratings.kurt()),
    }


def build_quantile_table(
    dataframe: pd.DataFrame,
) -> list[dict[str, float]]:
    ratings = dataframe["opta_rating"]

    rows: list[dict[str, float]] = []

    for quantile in QUANTILES:
        rows.append(
            {
                "quantile": quantile,
                "percentile": quantile * 100.0,
                "opta_rating": float(
                    ratings.quantile(quantile)
                ),
            }
        )

    return rows


def build_top_rank_summary(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for cutoff in TOP_RANK_CUTOFFS:
        subset = dataframe[
            dataframe["global_rank"] <= cutoff
        ]

        if subset.empty:
            continue

        rows.append(
            {
                "rank_cutoff": cutoff,
                "club_count": int(len(subset)),
                "highest_rating": float(
                    subset["opta_rating"].max()
                ),
                "lowest_rating": float(
                    subset["opta_rating"].min()
                ),
                "mean_rating": float(
                    subset["opta_rating"].mean()
                ),
                "median_rating": float(
                    subset["opta_rating"].median()
                ),
                "standard_deviation": float(
                    subset["opta_rating"].std()
                ),
                "rating_span": float(
                    subset["opta_rating"].max()
                    - subset["opta_rating"].min()
                ),
            }
        )

    return rows


def build_adjacent_rank_gap_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    table = dataframe[
        [
            "global_rank",
            "club",
            "club_id",
            "opta_rating",
        ]
    ].copy()

    table["next_rank"] = table[
        "global_rank"
    ].shift(-1)

    table["next_club"] = table[
        "club"
    ].shift(-1)

    table["next_club_id"] = table[
        "club_id"
    ].shift(-1)

    table["next_rating"] = table[
        "opta_rating"
    ].shift(-1)

    table["adjacent_rating_gap"] = (
        table["opta_rating"]
        - table["next_rating"]
    )

    return table.dropna(
        subset=["next_rating"]
    ).reset_index(drop=True)


def build_rank_separation_summary(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    ratings = dataframe["opta_rating"]

    rows: list[dict[str, Any]] = []

    for separation in RANK_SEPARATIONS:
        differences = (
            ratings
            - ratings.shift(-separation)
        ).dropna()

        if differences.empty:
            continue

        rows.append(
            {
                "rank_separation": separation,
                "comparison_count": int(
                    len(differences)
                ),
                "mean_rating_difference": float(
                    differences.mean()
                ),
                "median_rating_difference": float(
                    differences.median()
                ),
                "standard_deviation": float(
                    differences.std()
                ),
                "minimum_difference": float(
                    differences.min()
                ),
                "maximum_difference": float(
                    differences.max()
                ),
                "percentile_90_difference": float(
                    differences.quantile(0.90)
                ),
                "percentile_95_difference": float(
                    differences.quantile(0.95)
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
    overall_summary: dict[str, Any],
    quantile_rows: list[dict[str, float]],
    top_rank_rows: list[dict[str, Any]],
    rank_separation_rows: list[dict[str, Any]],
    adjacent_gap_table: pd.DataFrame,
    output_path: Path,
) -> None:
    largest_adjacent_gaps = (
        adjacent_gap_table
        .sort_values(
            "adjacent_rating_gap",
            ascending=False,
        )
        .head(15)
    )

    lines = [
        "# Study 042 – Phase 1 Opta Rating-Scale Audit",
        "",
        "## Overall distribution",
        "",
        f"- Records: {overall_summary['record_count']}",
        (
            "- Rating range: "
            f"{overall_summary['minimum_rating']:.4f}–"
            f"{overall_summary['maximum_rating']:.4f}"
        ),
        (
            "- Mean rating: "
            f"{overall_summary['mean_rating']:.4f}"
        ),
        (
            "- Median rating: "
            f"{overall_summary['median_rating']:.4f}"
        ),
        (
            "- Standard deviation: "
            f"{overall_summary['standard_deviation']:.4f}"
        ),
        (
            "- Interquartile range: "
            f"{overall_summary['interquartile_range']:.4f}"
        ),
        (
            "- Skewness: "
            f"{overall_summary['skewness']:.4f}"
        ),
        "",
        "## Rating percentiles",
        "",
        "| Percentile | Opta rating |",
        "|---:|---:|",
    ]

    for row in quantile_rows:
        lines.append(
            f"| {row['percentile']:.0f}% "
            f"| {row['opta_rating']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Elite-rank compression",
            "",
            (
                "| Rank cutoff | Clubs | Highest | Lowest "
                "| Mean | Standard deviation | Span |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in top_rank_rows:
        lines.append(
            f"| {row['rank_cutoff']} "
            f"| {row['club_count']} "
            f"| {row['highest_rating']:.4f} "
            f"| {row['lowest_rating']:.4f} "
            f"| {row['mean_rating']:.4f} "
            f"| {row['standard_deviation']:.4f} "
            f"| {row['rating_span']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Differences by rank separation",
            "",
            (
                "| Rank separation | Comparisons | Mean difference "
                "| Median difference | 90th percentile | Maximum |"
            ),
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in rank_separation_rows:
        lines.append(
            f"| {row['rank_separation']} "
            f"| {row['comparison_count']} "
            f"| {row['mean_rating_difference']:.4f} "
            f"| {row['median_rating_difference']:.4f} "
            f"| {row['percentile_90_difference']:.4f} "
            f"| {row['maximum_difference']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Largest adjacent-rank rating gaps",
            "",
            (
                "| Rank | Club | Rating | Next rank | Next club "
                "| Next rating | Gap |"
            ),
            "|---:|---|---:|---:|---|---:|---:|",
        ]
    )

    for _, row in largest_adjacent_gaps.iterrows():
        lines.append(
            f"| {int(row['global_rank'])} "
            f"| {row['club']} "
            f"| {float(row['opta_rating']):.4f} "
            f"| {int(row['next_rank'])} "
            f"| {row['next_club']} "
            f"| {float(row['next_rating']):.4f} "
            f"| {float(row['adjacent_rating_gap']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Methodological note",
            "",
            (
                "This phase describes the raw Opta scale only. "
                "No transformation has been selected, and the "
                "`rating_prior` field remains unassigned."
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
            "Audit the raw Opta rating scale for Study 042."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to global_club_prior_dataset.csv.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for Phase 1 audit outputs.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    dataframe = load_dataset(
        arguments.input
    )

    overall_summary = build_overall_summary(
        dataframe
    )

    quantile_rows = build_quantile_table(
        dataframe
    )

    top_rank_rows = build_top_rank_summary(
        dataframe
    )

    adjacent_gap_table = (
        build_adjacent_rank_gap_table(
            dataframe
        )
    )

    rank_separation_rows = (
        build_rank_separation_summary(
            dataframe
        )
    )

    output_directory = (
        arguments.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_dict_csv(
        overall_summary,
        output_directory
        / "overall_rating_summary.csv",
    )

    write_rows_csv(
        quantile_rows,
        output_directory
        / "rating_quantiles.csv",
    )

    write_rows_csv(
        top_rank_rows,
        output_directory
        / "top_rank_rating_summary.csv",
    )

    write_rows_csv(
        rank_separation_rows,
        output_directory
        / "rank_separation_differences.csv",
    )

    adjacent_gap_table.to_csv(
        output_directory
        / "adjacent_rank_rating_gaps.csv",
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "study": "Study 042",
        "phase": "Phase 1 – Opta Rating-Scale Audit",
        "input_path": str(arguments.input),
        "output_directory": str(output_directory),
        "record_count": len(dataframe),
        "calibration_performed": False,
    }

    (
        output_directory
        / "audit_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_markdown_report(
        overall_summary=overall_summary,
        quantile_rows=quantile_rows,
        top_rank_rows=top_rank_rows,
        rank_separation_rows=rank_separation_rows,
        adjacent_gap_table=adjacent_gap_table,
        output_path=(
            output_directory
            / "rating_scale_audit.md"
        ),
    )

    print("Study 042 – Opta Rating-Scale Audit")
    print("====================================")
    print(f"Input records: {len(dataframe)}")
    print(
        "Rating range: "
        f"{overall_summary['minimum_rating']:.4f}–"
        f"{overall_summary['maximum_rating']:.4f}"
    )
    print(
        "Mean rating: "
        f"{overall_summary['mean_rating']:.4f}"
    )
    print(
        "Median rating: "
        f"{overall_summary['median_rating']:.4f}"
    )
    print(
        "Standard deviation: "
        f"{overall_summary['standard_deviation']:.4f}"
    )
    print()
    print(f"Outputs: {output_directory}")
    print()
    print(
        "Phase 1 scale audit completed. "
        "No calibration was performed."
    )


if __name__ == "__main__":
    main()
#build_competition_feature_manifest.py

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COVERAGE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "competition_stat_coverage.csv"
)

DEFAULT_OUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "competition_feature_manifest.csv"
)
MIN_COVERAGE = 0.50

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a competition-feature availability manifest "
            "from competition-stat coverage."
        )
    )

    parser.add_argument(
        "--coverage-file",
        type=Path,
        default=DEFAULT_COVERAGE_FILE,
        help=(
            "Competition-stat coverage input CSV. Defaults to "
            "outputs/competition_stat_coverage.csv."
        ),
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUT_FILE,
        help=(
            "Competition-feature manifest output CSV. Defaults "
            "to the canonical competition_feature_manifest.csv."
        ),
    )

    return parser.parse_args()

def main() -> None:
    arguments = parse_arguments()

    coverage_file = arguments.coverage_file
    output_file = arguments.output_file

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage = pd.read_csv(
        coverage_file
    )

    rows = []

    id_cols = [
        "competition",
        "season_year",
        "rows",
        "players",
    ]

    coverage_cols = [
        col
        for col in coverage.columns
        if col.endswith("_coverage")
    ]

    for _, row in coverage.iterrows():
        for col in coverage_cols:
            feature = col.replace(
                "_coverage",
                "",
            )

            coverage_value = row[col]

            rows.append(
                {
                    "competition": row["competition"],
                    "season_year": row["season_year"],
                    "feature": feature,
                    "coverage": coverage_value,
                    "available": (
                        coverage_value >= MIN_COVERAGE
                    ),
                }
            )

    manifest = pd.DataFrame(
        rows
    )

    manifest.to_csv(
        output_file,
        index=False,
    )

    print(manifest.head(50))
    print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()
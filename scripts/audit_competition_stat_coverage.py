#audit_competition_stat_coverage.py

import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STATS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_stats.csv"
)

DEFAULT_OUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "competition_stat_coverage.csv"
)
CHECK_STATS = [
    "minutesPlayed",
    "rating",
    "goals",
    "assists",
    "keyPasses",
    "tackles",
    "interceptions",
    "clearances",
    "saves",
    "cleanSheet",
    "expectedGoals",
    "expectedAssists",
    "shotsOnTarget",
    "accurateCrosses",
    "successfulDribbles",
    "ballRecovery",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Sofascore competition-season stat coverage."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_STATS_FILE,
        help=(
            "Player-stat input CSV. Defaults to the canonical "
            "sofascore_player_stats.csv."
        ),
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUT_FILE,
        help=(
            "Competition-stat coverage output CSV. Defaults to "
            "outputs/competition_stat_coverage.csv."
        ),
    )

    return parser.parse_args()

def main() -> None:
    arguments = parse_arguments()

    input_file = arguments.input_file
    output_file = arguments.output_file

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        input_file,
        dtype={"season_year": str},
    )

    rows = []

    for (
        competition,
        season_year,
    ), group in df.groupby(
        ["competition", "season_year"],
        dropna=False,
    ):
        row = {
            "competition": competition,
            "season_year": season_year,
            "rows": len(group),
            "players": group["player_id"].nunique(),
        }

        for stat in CHECK_STATS:
            if stat in group.columns:
                row[f"{stat}_coverage"] = (
                    group[stat]
                    .notna()
                    .mean()
                )
            else:
                row[f"{stat}_coverage"] = 0.0

        rows.append(row)

    out = (
        pd.DataFrame(rows)
        .sort_values(
            ["rating_coverage", "rows"],
            ascending=[True, False],
        )
    )

    out.to_csv(
        output_file,
        index=False,
    )

    print(out.head(50))
    print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()
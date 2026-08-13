#build_reliability_tables.py

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "probability_calibration_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
)

OUTCOMES = {
    "home_win": "p_home_win",
    "draw": "p_draw",
    "away_win": "p_away_win",
}

N_BINS = 10


def build_table(
    df: pd.DataFrame,
    outcome: str,
    probability_column: str,
) -> pd.DataFrame:

    working = df.copy()

    working["predicted_probability"] = working[probability_column]

    working["observed"] = (
        working["outcome"] == outcome
    ).astype(int)

    working["bin"] = pd.cut(
        working["predicted_probability"],
        bins=N_BINS,
        labels=False,
        include_lowest=True,
    )

    table = (
        working
        .groupby("bin", observed=True)
        .agg(
            matches=(
                "observed",
                "count",
            ),
            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            observed_frequency=(
                "observed",
                "mean",
            ),
        )
        .reset_index()
    )

    table["calibration_error"] = (
        table["observed_frequency"]
        - table["mean_predicted_probability"]
    )

    table["absolute_calibration_error"] = (
        table["calibration_error"].abs()
    )

    return table


def main() -> None:

    df = pd.read_csv(DATASET_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for outcome, probability_column in OUTCOMES.items():

        table = build_table(
            df,
            outcome,
            probability_column,
        )

        output_path = (
            OUTPUT_DIR
            / f"probability_reliability_{outcome}.csv"
        )

        table.to_csv(
            output_path,
            index=False,
        )

        print()
        print(outcome)
        print("-" * len(outcome))
        print(table.to_string(index=False))
        print()
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
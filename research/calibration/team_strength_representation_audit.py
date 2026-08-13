#team_strength_representation_audit.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


TEAM_STRENGTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wc_2026_team_strength.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scoreline_first_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "team_strength_representation_summary.csv"


FEATURE_COLUMNS = [
    "att_composite",
    "mid_composite",
    "def_composite",
    "gk_composite",
    "poisson_attack_adj",
    "poisson_defense_adj",
]


def summarize_feature(df: pd.DataFrame, feature: str) -> dict:
    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    mean = values.mean()
    std = values.std()
    minimum = values.min()
    maximum = values.max()

    return {
        "feature": feature,
        "count": len(values),
        "mean": mean,
        "std": std,
        "min": minimum,
        "p25": values.quantile(0.25),
        "median": values.quantile(0.50),
        "p75": values.quantile(0.75),
        "max": maximum,
        "range": maximum - minimum,
        "coefficient_of_variation": (
            std / abs(mean)
            if mean != 0
            else None
        ),
    }


def main() -> None:
    df = pd.read_csv(TEAM_STRENGTH_PATH)

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required team-strength columns: "
            + ", ".join(missing)
        )

    rows = [
        summarize_feature(df, feature)
        for feature in FEATURE_COLUMNS
    ]

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("Team Strength Representation Audit")
    print("----------------------------------")
    print(result.round(6).to_string(index=False))
    print()
    print(f"Wrote representation audit -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
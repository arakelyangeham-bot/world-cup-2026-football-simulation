#analyze_core_observables.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.football_observatory.observatory_schema import (
    match_observation_from_row,
)
from research.football_observatory.observables import CORE_OBSERVABLES


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research" / "football_observatory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "historical_core_observables.csv"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    observations = [
        match_observation_from_row(row)
        for _, row in df.iterrows()
    ]

    rows = []

    for observable in CORE_OBSERVABLES:
        values = [
            observable.evaluate(observation)
            for observation in observations
        ]

        count = sum(values)
        rate = count / len(values)

        rows.append(
            {
                "observable": observable.name,
                "description": observable.description,
                "count": count,
                "total": len(values),
                "rate": rate,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "rate",
        ascending=False,
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print("Historical Core Football Observables")
    print("------------------------------------")
    print(result.round(4).to_string(index=False))
    print()
    print(f"Wrote observables -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
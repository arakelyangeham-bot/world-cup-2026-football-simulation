#smoke_test_prototype_match_engine.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.prototypes.prototype_match_engine import (
    simulate_match_score_research,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)


def row_to_team_dict(row: pd.Series, prefix: str) -> dict[str, float]:
    return {
        "attack": row[f"{prefix}_attack"],
        "midfield": row[f"{prefix}_midfield"],
        "defense": row[f"{prefix}_defense"],
        "gk": row[f"{prefix}_gk"],
        "poisson_attack": row[f"{prefix}_poisson_attack"],
        "poisson_defense": row[f"{prefix}_poisson_defense"],
        "fifa_points": row[f"{prefix}_fifa_points"],
    }


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    row = df.iloc[0]

    home = row_to_team_dict(row, "home")
    away = row_to_team_dict(row, "away")

    goals_home, goals_away = simulate_match_score_research(
        home,
        away,
    )

    print("Prototype Match Engine Smoke Test")
    print("---------------------------------")
    print(f"Home team: {row.get('home_team')}")
    print(f"Away team: {row.get('away_team')}")
    print(f"Generated scoreline: {goals_home}-{goals_away}")

    assert isinstance(goals_home, int)
    assert isinstance(goals_away, int)
    assert goals_home >= 0
    assert goals_away >= 0

    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
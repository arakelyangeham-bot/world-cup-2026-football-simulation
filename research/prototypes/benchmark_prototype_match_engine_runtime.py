#benchmark_prototype_match_engine_runtime.py

from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.match_engine_adapter import simulate_match_score
from research.prototypes.prototype_match_engine import (
    simulate_match_score_research,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research" / "prototypes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "prototype_match_engine_runtime.csv"

SAMPLE_SIZES = [10, 50, 100]


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


def build_matchups(df: pd.DataFrame, n: int) -> list[tuple[dict, dict]]:
    subset = df.head(n)

    return [
        (
            row_to_team_dict(row, "home"),
            row_to_team_dict(row, "away"),
        )
        for _, row in subset.iterrows()
    ]


def time_engine(
    engine_name: str,
    matchups: list[tuple[dict, dict]],
) -> dict:
    if engine_name == "production":
        engine = simulate_match_score
    elif engine_name == "research_wrapper":
        engine = simulate_match_score_research
    else:
        raise ValueError(f"Unknown engine: {engine_name}")

    start = time.perf_counter()

    for home, away in matchups:
        goals_home, goals_away = engine(
            home,
            away,
            mode="ml",
        )

        assert isinstance(goals_home, int)
        assert isinstance(goals_away, int)
        assert goals_home >= 0
        assert goals_away >= 0

    elapsed = time.perf_counter() - start

    return {
        "engine": engine_name,
        "matches": len(matchups),
        "total_seconds": elapsed,
        "seconds_per_match": elapsed / len(matchups),
    }


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    rows = []

    for n in SAMPLE_SIZES:
        matchups = build_matchups(df, n)

        for engine_name in ["production", "research_wrapper"]:
            rows.append(
                time_engine(
                    engine_name=engine_name,
                    matchups=matchups,
                )
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("Prototype Match Engine Runtime Benchmark")
    print("----------------------------------------")
    print(result.round(6).to_string(index=False))
    print()
    print(f"Wrote runtime benchmark -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
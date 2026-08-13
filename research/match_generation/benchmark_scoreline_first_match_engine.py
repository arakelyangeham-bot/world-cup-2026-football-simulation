#benchmark_scoreline_first_match_engine.py

from __future__ import annotations

from pathlib import Path
from collections import Counter
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.match_engine_adapter import simulate_match_score
from research.match_generation.scoreline_first_match_engine import (
    simulate_scoreline_first_match,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research" / "match_generation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N = 100
MAX_GOALS = 6
OTHER_BUCKET = "other"


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


def bucket_scoreline(home_goals: int, away_goals: int) -> str:
    if home_goals > MAX_GOALS or away_goals > MAX_GOALS:
        return OTHER_BUCKET

    return f"{home_goals}-{away_goals}"


def total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)

    return 0.5 * sum(
        abs(p.get(key, 0.0) - q.get(key, 0.0))
        for key in keys
    )


def summarize_scores(label: str, home_scores: list[int], away_scores: list[int]) -> dict:
    home = np.asarray(home_scores)
    away = np.asarray(away_scores)
    total = home + away

    return {
        "model": label,
        "samples": len(home_scores),
        "avg_total_goals": total.mean(),
        "draw_rate": np.mean(home == away),
        "home_win_rate": np.mean(home > away),
        "away_win_rate": np.mean(home < away),
        "zero_zero_rate": np.mean((home == 0) & (away == 0)),
        "one_one_rate": np.mean((home == 1) & (away == 1)),
        "one_zero_rate": np.mean((home == 1) & (away == 0)),
        "two_one_rate": np.mean((home == 2) & (away == 1)),
        "five_plus_total_rate": np.mean(total >= 5),
    }


def scoreline_distribution(home_scores: list[int], away_scores: list[int]) -> dict[str, float]:
    counts = Counter(
        bucket_scoreline(h, a)
        for h, a in zip(home_scores, away_scores)
    )

    total = sum(counts.values())

    return {
        scoreline: count / total
        for scoreline, count in counts.items()
    }


def simulate_engine(df: pd.DataFrame, engine_name: str) -> tuple[list[int], list[int]]:
    home_scores = []
    away_scores = []

    for _, row in df.iterrows():
        home = row_to_team_dict(row, "home")
        away = row_to_team_dict(row, "away")

        for _ in range(N):
            if engine_name == "production_hybrid":
                goals_home, goals_away = simulate_match_score(
                    home,
                    away,
                    mode="ml",
                )
            elif engine_name == "scoreline_first":
                goals_home, goals_away = simulate_scoreline_first_match(
                    home,
                    away,
                )
            else:
                raise ValueError(f"Unknown engine: {engine_name}")

            home_scores.append(goals_home)
            away_scores.append(goals_away)

    return home_scores, away_scores


def main() -> None:
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)

    historical_home = df["home_score"].astype(int).tolist()
    historical_away = df["away_score"].astype(int).tolist()
    historical_distribution = scoreline_distribution(
        historical_home,
        historical_away,
    )

    rows = [
        summarize_scores("historical", historical_home, historical_away)
    ]

    distributions = {
        "historical": historical_distribution,
    }

    for engine_name in ["production_hybrid", "scoreline_first"]:
        print(f"Simulating {engine_name}...")

        home_scores, away_scores = simulate_engine(df, engine_name)

        rows.append(
            summarize_scores(engine_name, home_scores, away_scores)
        )

        distributions[engine_name] = scoreline_distribution(
            home_scores,
            away_scores,
        )

    summary = pd.DataFrame(rows)

    tvd_rows = []

    for model_name, distribution in distributions.items():
        tvd_rows.append(
            {
                "model": model_name,
                "scoreline_tvd": total_variation_distance(
                    historical_distribution,
                    distribution,
                ),
            }
        )

    tvd = pd.DataFrame(tvd_rows)

    result = summary.merge(tvd, on="model", how="left")

    output_path = OUTPUT_DIR / "scoreline_first_match_engine_benchmark.csv"
    result.to_csv(output_path, index=False)

    print()
    print("Scoreline-First Match Engine Benchmark")
    print("--------------------------------------")
    print(result.round(6).to_string(index=False))
    print()
    print(f"Wrote benchmark -> {output_path}")


if __name__ == "__main__":
    main()
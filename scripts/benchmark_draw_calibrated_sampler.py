#benchmark_draw_calibrated_sampler.py

"""
Benchmark draw-calibrated sampler.

Phase 1:
- Validate that draw_strength=0.0 reproduces the hierarchical stochastic
  lambda sampler baseline.

Phase 2:
- Sweep draw_strength values after the zero-strength validation passes.
"""

from __future__ import annotations

import csv
from pathlib import Path

from simulation.goal_samplers import (
    draw_calibrated_hierarchical_sampler
)


OUTPUT_PATH = Path("outputs/benchmarks/draw_calibrated_sampler_benchmark.csv")


def summarize_scores(results: list[tuple[int, int]]) -> dict[str, float]:
    n = len(results)

    home_goals = sum(h for h, _ in results)
    away_goals = sum(a for _, a in results)

    draws = sum(1 for h, a in results if h == a)
    home_wins = sum(1 for h, a in results if h > a)
    away_wins = sum(1 for h, a in results if h < a)

    return {
        "n": n,
        "mean_goals": (home_goals + away_goals) / n,
        "home_goals": home_goals / n,
        "away_goals": away_goals / n,
        "draw_rate": draws / n,
        "home_win_rate": home_wins / n,
        "away_win_rate": away_wins / n,
    }


def run_sampler_benchmark(
    draw_strength: float,
    n_samples: int = 100_000,
    home_lambda: float = 1.35,
    away_lambda: float = 1.05,
) -> dict[str, float]:
    results = [
        draw_calibrated_hierarchical_sampler(
            home_lambda,
            away_lambda,
            tempo_cv=0.60,
            team_cv=0.10,
            draw_strength=draw_strength,
        )
        for _ in range(n_samples)
    ]

    summary = summarize_scores(results)
    summary["draw_strength"] = draw_strength
    summary["home_lambda"] = home_lambda
    summary["away_lambda"] = away_lambda

    return summary


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    draw_strength_values = [
        0.0,
    ]

    rows = [
        run_sampler_benchmark(draw_strength=value)
        for value in draw_strength_values
    ]

    fieldnames = [
        "draw_strength",
        "home_lambda",
        "away_lambda",
        "n",
        "mean_goals",
        "home_goals",
        "away_goals",
        "draw_rate",
        "home_win_rate",
        "away_win_rate",
    ]

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote benchmark to {OUTPUT_PATH}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
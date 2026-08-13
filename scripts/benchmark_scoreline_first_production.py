#benchmark_scoreline_first_production.py

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.team_strength_loader import load_team_repository
from simulation.match_engine_adapter import simulate_match_score


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scoreline_first_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "scoreline_first_production_benchmark.csv"

N_SAMPLES_PER_MATCH = 100
MAX_GOALS = 6
OTHER_BUCKET = "other"


def bucket_scoreline(home_goals: int, away_goals: int) -> str:
    if home_goals > MAX_GOALS or away_goals > MAX_GOALS:
        return OTHER_BUCKET

    return f"{home_goals}-{away_goals}"


def total_variation_distance(
    p: dict[str, float],
    q: dict[str, float],
) -> float:
    keys = set(p) | set(q)

    return 0.5 * sum(
        abs(p.get(key, 0.0) - q.get(key, 0.0))
        for key in keys
    )


def scoreline_distribution(
    home_scores: list[int],
    away_scores: list[int],
) -> dict[str, float]:
    counts = Counter(
        bucket_scoreline(home, away)
        for home, away in zip(home_scores, away_scores)
    )

    total = sum(counts.values())

    return {
        scoreline: count / total
        for scoreline, count in counts.items()
    }


def summarize_scores(
    label: str,
    home_scores: list[int],
    away_scores: list[int],
) -> dict:
    home = np.asarray(home_scores)
    away = np.asarray(away_scores)
    total = home + away
    abs_diff = np.abs(home - away)

    return {
        "model": label,
        "samples": len(home_scores),
        "avg_total_goals": total.mean(),
        "draw_rate": np.mean(home == away),
        "home_win_rate": np.mean(home > away),
        "away_win_rate": np.mean(home < away),
        "one_goal_rate": np.mean(abs_diff == 1),
        "clean_sheet_rate": np.mean((home == 0) | (away == 0)),
        "both_teams_scored_rate": np.mean((home > 0) & (away > 0)),
        "high_scoring_rate": np.mean(total >= 5),
        "blowout_rate": np.mean(abs_diff >= 3),
        "zero_zero_rate": np.mean((home == 0) & (away == 0)),
        "one_zero_rate": np.mean((home == 1) & (away == 0)),
        "zero_one_rate": np.mean((home == 0) & (away == 1)),
        "one_one_rate": np.mean((home == 1) & (away == 1)),
        "two_one_rate": np.mean((home == 2) & (away == 1)),
        "one_two_rate": np.mean((home == 1) & (away == 2)),
    }


def row_to_team_name(row: pd.Series, prefix: str) -> str:
    return str(row[f"{prefix}_team"])


def simulate_production(
    df: pd.DataFrame,
    team_repository: dict[str, dict],
) -> tuple[list[int], list[int]]:
    home_scores = []
    away_scores = []

    for _, row in df.iterrows():
        home_team = row_to_team_name(row, "home")
        away_team = row_to_team_name(row, "away")

        if home_team not in team_repository:
            raise ValueError(f"Missing home team in repository: {home_team}")

        if away_team not in team_repository:
            raise ValueError(f"Missing away team in repository: {away_team}")

        home = team_repository[home_team]
        away = team_repository[away_team]

        for _ in range(N_SAMPLES_PER_MATCH):
            goals_home, goals_away = simulate_match_score(
                home,
                away,
            )

            home_scores.append(goals_home)
            away_scores.append(goals_away)

    return home_scores, away_scores

def run_benchmark(
    team_repository: dict[str, dict],
    label: str,
) -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)

    historical_home = df["home_score"].astype(int).tolist()
    historical_away = df["away_score"].astype(int).tolist()

    generated_home, generated_away = simulate_production(
        df=df,
        team_repository=team_repository,
    )

    historical_summary = summarize_scores(
        "historical",
        historical_home,
        historical_away,
    )

    generated_summary = summarize_scores(
        label,
        generated_home,
        generated_away,
    )

    historical_dist = scoreline_distribution(
        historical_home,
        historical_away,
    )

    generated_dist = scoreline_distribution(
        generated_home,
        generated_away,
    )

    historical_summary["scoreline_tvd"] = 0.0

    generated_summary["scoreline_tvd"] = total_variation_distance(
        historical_dist,
        generated_dist,
    )

    return pd.DataFrame(
        [
            historical_summary,
            generated_summary,
        ]
    )

def main() -> None:
    np.random.seed(42)

    team_repository = load_team_repository()

    result = run_benchmark(
        team_repository=team_repository,
        label="scoreline_first_production",
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Scoreline-First Production Benchmark")
    print("------------------------------------")
    print(result.round(6).to_string(index=False))
    print()
    print(f"Wrote benchmark -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
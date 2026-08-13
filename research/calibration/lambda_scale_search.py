#lambda_scale_search.py

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.team_strength_loader import load_team_repository
from simulation.goal_samplers import dixon_coles_hierarchical_sampler_fast
from simulation.lambda_models import expected_goals
from simulation.match_engine_adapter import repository_entry_to_poisson_features
from simulation.simulation_config import (
    GOAL_SAMPLER_CONFIG,
    LAMBDA_MODEL,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scoreline_first_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "lambda_scale_search.csv"

LAMBDA_SCALES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

N_SAMPLES_PER_MATCH = 100
MAX_GOALS = 6
OTHER_BUCKET = "other"


def bucket_scoreline(home_goals: int, away_goals: int) -> str:
    if home_goals > MAX_GOALS or away_goals > MAX_GOALS:
        return OTHER_BUCKET

    return f"{home_goals}-{away_goals}"


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


def total_variation_distance(
    p: dict[str, float],
    q: dict[str, float],
) -> float:
    keys = set(p) | set(q)

    return 0.5 * sum(
        abs(p.get(key, 0.0) - q.get(key, 0.0))
        for key in keys
    )


def summarize_scores(
    label: str,
    scale: float,
    home_scores: list[int],
    away_scores: list[int],
) -> dict:
    home = np.asarray(home_scores)
    away = np.asarray(away_scores)
    total = home + away
    abs_diff = np.abs(home - away)

    return {
        "model": label,
        "lambda_scale": scale,
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


def simulate_with_scale(
    df: pd.DataFrame,
    team_repository: dict[str, dict],
    scale: float,
) -> tuple[list[int], list[int]]:
    home_scores = []
    away_scores = []

    for _, row in df.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        home_entry = team_repository[home_team]
        away_entry = team_repository[away_team]

        home_features = repository_entry_to_poisson_features(home_entry)
        away_features = repository_entry_to_poisson_features(away_entry)

        lambda_home, lambda_away = expected_goals(
            home_features,
            away_features,
            lambda_model=LAMBDA_MODEL,
        )

        lambda_home *= scale
        lambda_away *= scale

        for _ in range(N_SAMPLES_PER_MATCH):
            goals_home, goals_away = dixon_coles_hierarchical_sampler_fast(
                lambda_home,
                lambda_away,
                **GOAL_SAMPLER_CONFIG,
            )

            home_scores.append(int(goals_home))
            away_scores.append(int(goals_away))

    return home_scores, away_scores


def main() -> None:
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)
    team_repository = load_team_repository()

    historical_home = df["home_score"].astype(int).tolist()
    historical_away = df["away_score"].astype(int).tolist()

    historical_dist = scoreline_distribution(
        historical_home,
        historical_away,
    )

    historical_summary = summarize_scores(
        label="historical",
        scale=1.0,
        home_scores=historical_home,
        away_scores=historical_away,
    )
    historical_summary["scoreline_tvd"] = 0.0
    historical_summary["avg_goals_error"] = 0.0

    rows = [historical_summary]

    target_avg_goals = historical_summary["avg_total_goals"]

    for scale in LAMBDA_SCALES:
        print(f"Testing lambda scale {scale:.2f}...")

        home_scores, away_scores = simulate_with_scale(
            df=df,
            team_repository=team_repository,
            scale=scale,
        )

        summary = summarize_scores(
            label="scoreline_first_scaled",
            scale=scale,
            home_scores=home_scores,
            away_scores=away_scores,
        )

        dist = scoreline_distribution(home_scores, away_scores)

        summary["scoreline_tvd"] = total_variation_distance(
            historical_dist,
            dist,
        )
        summary["avg_goals_error"] = abs(
            summary["avg_total_goals"] - target_avg_goals
        )

        rows.append(summary)

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Lambda Scale Search")
    print("-------------------")
    print(
        result.sort_values(
            ["avg_goals_error", "scoreline_tvd"],
            ascending=[True, True],
        )
        .round(6)
        .to_string(index=False)
    )
    print()
    print(f"Wrote lambda scale search -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
#home_away_bias_audit.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.team_strength_loader import load_team_repository
from simulation.lambda_models import expected_goals
from simulation.match_engine_adapter import repository_entry_to_poisson_features
from simulation.simulation_config import LAMBDA_MODEL, LAMBDA_SCALE


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scoreline_first_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCH_LEVEL_OUTPUT_PATH = OUTPUT_DIR / "lambda_match_level.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "lambda_home_away_summary.csv"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    team_repository = load_team_repository()

    rows = []

    for _, row in df.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        home_entry = team_repository[home_team]
        away_entry = team_repository[away_team]

        lambda_home_raw, lambda_away_raw = expected_goals(
            repository_entry_to_poisson_features(home_entry),
            repository_entry_to_poisson_features(away_entry),
            lambda_model=LAMBDA_MODEL,
        )

        lambda_home = lambda_home_raw * LAMBDA_SCALE
        lambda_away = lambda_away_raw * LAMBDA_SCALE

        actual_home_goals = int(row["home_score"])
        actual_away_goals = int(row["away_score"])

        rows.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "lambda_home_raw": lambda_home_raw,
                "lambda_away_raw": lambda_away_raw,
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "lambda_total": lambda_home + lambda_away,
                "lambda_diff_home_minus_away": lambda_home - lambda_away,
                "actual_home_goals": actual_home_goals,
                "actual_away_goals": actual_away_goals,
                "actual_total_goals": actual_home_goals + actual_away_goals,
                "actual_diff_home_minus_away": (
                    actual_home_goals - actual_away_goals
                ),
            }
        )

    match_df = pd.DataFrame(rows)
    match_df.to_csv(MATCH_LEVEL_OUTPUT_PATH, index=False)

    summary = {
        "matches": len(match_df),
        "mean_lambda_home": match_df["lambda_home"].mean(),
        "mean_lambda_away": match_df["lambda_away"].mean(),
        "mean_lambda_total": match_df["lambda_total"].mean(),
        "mean_lambda_diff_home_minus_away": (
            match_df["lambda_diff_home_minus_away"].mean()
        ),
        "mean_actual_home_goals": match_df["actual_home_goals"].mean(),
        "mean_actual_away_goals": match_df["actual_away_goals"].mean(),
        "mean_actual_total_goals": match_df["actual_total_goals"].mean(),
        "mean_actual_diff_home_minus_away": (
            match_df["actual_diff_home_minus_away"].mean()
        ),
        "lambda_home_share": (
            match_df["lambda_home"].sum()
            / match_df["lambda_total"].sum()
        ),
        "actual_home_goal_share": (
            match_df["actual_home_goals"].sum()
            / match_df["actual_total_goals"].sum()
        ),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("Home/Away Lambda Bias Audit")
    print("---------------------------")
    print(summary_df.round(6).to_string(index=False))
    print()
    print(f"Wrote match-level lambdas -> {MATCH_LEVEL_OUTPUT_PATH}")
    print(f"Wrote summary            -> {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
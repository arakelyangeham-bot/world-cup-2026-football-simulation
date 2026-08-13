#team_repository_calibration_audit.py

from __future__ import annotations

from pathlib import Path
import sys
from itertools import permutations

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.team_strength_loader import load_team_repository
from simulation.lambda_models import expected_goals
from simulation.match_engine_adapter import repository_entry_to_poisson_features
from simulation.simulation_config import LAMBDA_MODEL, LAMBDA_SCALE


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scoreline_first_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEAM_FEATURE_OUTPUT_PATH = OUTPUT_DIR / "team_repository_feature_summary.csv"
PAIRWISE_OUTPUT_PATH = OUTPUT_DIR / "team_repository_pairwise_lambda_summary.csv"


def summarize_series(name: str, values: pd.Series) -> dict:
    return {
        "feature": name,
        "count": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "p25": values.quantile(0.25),
        "median": values.quantile(0.50),
        "p75": values.quantile(0.75),
        "max": values.max(),
    }


def main() -> None:
    repository = load_team_repository()

    team_rows = []

    for team, values in repository.items():
        team_rows.append(
            {
                "team": team,
                "attack": values["attack"],
                "midfield": values["midfield"],
                "defense": values["defense"],
                "gk": values["gk"],
                "poisson_attack": values["poisson_attack"],
                "poisson_defense": values["poisson_defense"],
                "fifa_points": values["fifa_points"],
            }
        )

    team_df = pd.DataFrame(team_rows)

    feature_rows = []

    for feature in [
        "attack",
        "midfield",
        "defense",
        "gk",
        "poisson_attack",
        "poisson_defense",
        "fifa_points",
    ]:
        feature_rows.append(
            summarize_series(
                feature,
                pd.to_numeric(team_df[feature], errors="coerce").dropna(),
            )
        )

    feature_summary = pd.DataFrame(feature_rows)
    feature_summary.to_csv(TEAM_FEATURE_OUTPUT_PATH, index=False)

    pairwise_rows = []

    teams = sorted(repository.keys())

    for home_team, away_team in permutations(teams, 2):
        home_entry = repository[home_team]
        away_entry = repository[away_team]

        lambda_home_raw, lambda_away_raw = expected_goals(
            repository_entry_to_poisson_features(home_entry),
            repository_entry_to_poisson_features(away_entry),
            lambda_model=LAMBDA_MODEL,
        )

        lambda_home = lambda_home_raw * LAMBDA_SCALE
        lambda_away = lambda_away_raw * LAMBDA_SCALE

        pairwise_rows.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "lambda_total": lambda_home + lambda_away,
                "lambda_diff_home_minus_away": lambda_home - lambda_away,
                "lambda_abs_diff": abs(lambda_home - lambda_away),
                "home_lambda_share": (
                    lambda_home / (lambda_home + lambda_away)
                    if (lambda_home + lambda_away) > 0
                    else None
                ),
                "fifa_points_diff": (
                    home_entry["fifa_points"] - away_entry["fifa_points"]
                ),
                "poisson_attack_diff": (
                    home_entry["poisson_attack"]
                    - away_entry["poisson_attack"]
                ),
                "poisson_defense_diff": (
                    home_entry["poisson_defense"]
                    - away_entry["poisson_defense"]
                ),
            }
        )

    pairwise_df = pd.DataFrame(pairwise_rows)

    pairwise_summary_rows = []

    for feature in [
        "lambda_home",
        "lambda_away",
        "lambda_total",
        "lambda_diff_home_minus_away",
        "lambda_abs_diff",
        "home_lambda_share",
        "fifa_points_diff",
        "poisson_attack_diff",
        "poisson_defense_diff",
    ]:
        pairwise_summary_rows.append(
            summarize_series(
                feature,
                pd.to_numeric(pairwise_df[feature], errors="coerce").dropna(),
            )
        )

    pairwise_summary = pd.DataFrame(pairwise_summary_rows)
    pairwise_summary.to_csv(PAIRWISE_OUTPUT_PATH, index=False)

    print("Team Repository Calibration Audit")
    print("---------------------------------")
    print()
    print("Team feature summary")
    print(feature_summary.round(6).to_string(index=False))
    print()
    print("Pairwise lambda summary")
    print(pairwise_summary.round(6).to_string(index=False))
    print()
    print(f"Wrote team feature summary -> {TEAM_FEATURE_OUTPUT_PATH}")
    print(f"Wrote pairwise summary     -> {PAIRWISE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
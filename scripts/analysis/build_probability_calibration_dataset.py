#build_probability_calibration_dataset.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from simulation.lambda_models import expected_goals
from simulation.probability_adapter import ProbabilityAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "analysis" / "probability_calibration_dataset.csv"


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


def match_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    adapter = ProbabilityAdapter(mode="ml")

    rows = []

    for _, row in df.iterrows():
        home_team = row_to_team_dict(row, "home")
        away_team = row_to_team_dict(row, "away")

        probabilities = adapter.predict_match(home_team, away_team)

        lambda_home, lambda_away = expected_goals(
            home_team,
            away_team,
            lambda_model="calibrated",
        )

        rows.append(
            {
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "competition": row.get("competition"),
                "tournament": row.get("tournament"),
                "date": row.get("date"),
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "outcome": match_outcome(row["home_score"], row["away_score"]),
                "p_home_win": probabilities["home_win"],
                "p_draw": probabilities["draw"],
                "p_away_win": probabilities["away_win"],
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "lambda_total": lambda_home + lambda_away,
                "lambda_diff": lambda_home - lambda_away,
                "fifa_points_diff": home_team["fifa_points"] - away_team["fifa_points"],
                "home_attack": home_team["attack"],
                "away_attack": away_team["attack"],
                "home_defense": home_team["defense"],
                "away_defense": away_team["defense"],
            }
        )

    output = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote calibration dataset: {OUTPUT_PATH}")
    print(f"Rows: {len(output)}")
    print(f"Columns: {len(output.columns)}")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
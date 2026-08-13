#audit_outcome_probability_calibration.py

from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss

from inference.match_predictor import MatchPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)


LABELS = [
    "home_win",
    "draw",
    "away_win",
]


def row_to_team_dict(row, prefix):
    return {
        "attack": row[f"{prefix}_attack"],
        "midfield": row[f"{prefix}_midfield"],
        "defense": row[f"{prefix}_defense"],
        "gk": row[f"{prefix}_gk"],
        "poisson_attack": row[f"{prefix}_poisson_attack"],
        "poisson_defense": row[f"{prefix}_poisson_defense"],
        "fifa_points": row[f"{prefix}_fifa_points"],
    }


def main():
    df = pd.read_csv(DATASET_PATH)

    predictor = MatchPredictor()

    rows = []

    for _, row in df.iterrows():

        home = row_to_team_dict(row, "home")
        away = row_to_team_dict(row, "away")

        probs = predictor.predict_match(home, away)

        rows.append(
            {
                "actual": row["result"],
                "home_win": probs["home_win"],
                "draw": probs["draw"],
                "away_win": probs["away_win"],
            }
        )

    prediction_df = pd.DataFrame(rows)

    print("Outcome Probability Calibration")
    print("-------------------------------")
    print()

    grouped = (
        prediction_df
        .groupby("actual")
        .agg(
            avg_home=("home_win", "mean"),
            avg_draw=("draw", "mean"),
            avg_away=("away_win", "mean"),
            matches=("actual", "count"),
        )
    )

    print(grouped.round(3))
    print()

    print("Overall mean probabilities")
    print("--------------------------")

    print(
        prediction_df[
            ["home_win", "draw", "away_win"]
        ].mean().round(3)
    )

    print()

    print("Brier Scores")
    print("------------")

    for label in LABELS:

        truth = (
            prediction_df["actual"] == label
        ).astype(int)

        score = brier_score_loss(
            truth,
            prediction_df[label],
        )

        print(f"{label:<10} {score:.4f}")


if __name__ == "__main__":
    main()
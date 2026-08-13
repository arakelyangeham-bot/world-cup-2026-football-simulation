# benchmark_match_generator_probability_engine.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.match_predictor import MatchPredictor
from research.match_generation.match_generator_probability_engine import (
    MatchGeneratorProbabilityEngine,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research" / "match_generation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["away_win", "draw", "home_win"]


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


def multiclass_brier(actual: list[str], probabilities: list[list[float]]) -> float:
    total = 0.0

    for actual_label, probs in zip(actual, probabilities):
        for label, prob in zip(LABELS, probs):
            target = 1.0 if label == actual_label else 0.0
            total += (prob - target) ** 2

    return total / len(actual)


def evaluate_engine(name: str, engine, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    actual = []
    predicted = []
    probabilities = []
    rows = []

    for idx, row in df.iterrows():
        home = row_to_team_dict(row, "home")
        away = row_to_team_dict(row, "away")

        probs = engine.predict_match(home, away)

        actual_label = row["result"]
        predicted_label = max(probs, key=probs.get)

        actual.append(actual_label)
        predicted.append(predicted_label)
        probabilities.append([probs[label] for label in LABELS])

        rows.append(
            {
                "row_index": idx,
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "actual": actual_label,
                "predicted": predicted_label,
                "p_away_win": probs["away_win"],
                "p_draw": probs["draw"],
                "p_home_win": probs["home_win"],
            }
        )

    summary = {
        "model": name,
        "rows": len(df),
        "accuracy": accuracy_score(actual, predicted),
        "brier": multiclass_brier(actual, probabilities),
        "log_loss": log_loss(actual, probabilities, labels=LABELS),
        "actual_home_win_rate": actual.count("home_win") / len(actual),
        "actual_draw_rate": actual.count("draw") / len(actual),
        "actual_away_win_rate": actual.count("away_win") / len(actual),
        "predicted_home_win_rate": predicted.count("home_win") / len(predicted),
        "predicted_draw_rate": predicted.count("draw") / len(predicted),
        "predicted_away_win_rate": predicted.count("away_win") / len(predicted),
        "mean_p_home_win": sum(p[2] for p in probabilities) / len(probabilities),
        "mean_p_draw": sum(p[1] for p in probabilities) / len(probabilities),
        "mean_p_away_win": sum(p[0] for p in probabilities) / len(probabilities),
    }

    return summary, pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    production_ml = MatchPredictor()

    match_generator = MatchGeneratorProbabilityEngine(
        lambda_model="calibrated",
        tempo_cv=0.60,
        team_cv=0.10,
        rho=0.30,
        samples=500,
        seed=None,
    )

    summaries = []
    prediction_frames = []

    for name, engine in [
        ("production_ml", production_ml),
        ("match_generator_mc", match_generator),
    ]:
        print(f"Evaluating {name}...")
        summary, predictions = evaluate_engine(name, engine, df)

        summaries.append(summary)
        predictions["model"] = name
        prediction_frames.append(predictions)

    summary_df = pd.DataFrame(summaries)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    summary_path = OUTPUT_DIR / "match_generator_probability_benchmark_summary.csv"
    predictions_path = OUTPUT_DIR / "match_generator_probability_predictions.csv"

    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print()
    print("Match Generator Probability Benchmark")
    print("-------------------------------------")
    print(summary_df.round(6).to_string(index=False))
    print()
    print(f"Wrote summary     -> {summary_path}")
    print(f"Wrote predictions -> {predictions_path}")


if __name__ == "__main__":
    main()
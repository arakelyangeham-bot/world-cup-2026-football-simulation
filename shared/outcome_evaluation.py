#outcome_evaluation.py

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


LABELS = ["away_win", "draw", "home_win"]


@dataclass(frozen=True)
class OutcomeEvaluationResult:
    summary: dict[str, object]
    confusion_matrix: pd.DataFrame
    predictions: pd.DataFrame


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


def evaluate_predictor(
    predictor,
    df: pd.DataFrame,
    model_name: str,
) -> OutcomeEvaluationResult:
    actual = []
    predicted = []
    probabilities = []
    prediction_rows = []

    for row_index, row in df.iterrows():
        home_team = row_to_team_dict(row, "home")
        away_team = row_to_team_dict(row, "away")

        probs = predictor.predict_match(home_team, away_team)

        actual_label = row["result"]
        predicted_label = max(probs, key=probs.get)

        actual.append(actual_label)
        predicted.append(predicted_label)
        probabilities.append([probs[label] for label in LABELS])

        prediction_rows.append(
            {
                "row_index": row_index,
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "actual": actual_label,
                "predicted": predicted_label,
                "p_away_win": probs["away_win"],
                "p_draw": probs["draw"],
                "p_home_win": probs["home_win"],
            }
        )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=LABELS,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )

    summary = {
        "model": model_name,
        "rows": len(df),
        "accuracy": accuracy_score(actual, predicted),
        "log_loss": log_loss(actual, probabilities, labels=LABELS),
        "actual_distribution": (
            pd.Series(actual)
            .value_counts(normalize=True)
            .round(6)
            .to_dict()
        ),
        "predicted_distribution": (
            pd.Series(predicted)
            .value_counts(normalize=True)
            .round(6)
            .to_dict()
        ),
    }

    predictions_df = pd.DataFrame(prediction_rows)

    return OutcomeEvaluationResult(
        summary=summary,
        confusion_matrix=matrix_df,
        predictions=predictions_df,
    )
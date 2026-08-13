#audit_decision_policy.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from inference.match_predictor import MatchPredictor
from shared.outcome_evaluation import LABELS, row_to_team_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "decision_policy_audit.csv"
)

CONFUSION_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "decision_policy_confusion_matrices.csv"
)


def predict_argmax(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


def predict_draw_threshold(
    probs: dict[str, float],
    threshold: float,
) -> str:
    if probs["draw"] >= threshold:
        return "draw"

    return max(
        {
            "away_win": probs["away_win"],
            "home_win": probs["home_win"],
        },
        key=lambda label: probs[label],
    )


def predict_margin_draw(
    probs: dict[str, float],
    margin: float,
) -> str:
    home = probs["home_win"]
    away = probs["away_win"]

    if abs(home - away) <= margin:
        return "draw"

    return "home_win" if home > away else "away_win"


def evaluate_policy(
    df: pd.DataFrame,
    policy_name: str,
    predictions: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    actual = df["result"].tolist()

    accuracy = accuracy_score(
        actual,
        predictions,
    )

    predicted_distribution = (
        pd.Series(predictions)
        .value_counts(normalize=True)
        .reindex(LABELS, fill_value=0.0)
        .to_dict()
    )

    matrix = confusion_matrix(
        actual,
        predictions,
        labels=LABELS,
    )

    matrix_rows = []

    for actual_idx, actual_label in enumerate(LABELS):
        for predicted_idx, predicted_label in enumerate(LABELS):
            matrix_rows.append(
                {
                    "policy": policy_name,
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[actual_idx, predicted_idx]),
                }
            )

    summary = {
        "policy": policy_name,
        "accuracy": accuracy,
        "pred_away_win_share": predicted_distribution["away_win"],
        "pred_draw_share": predicted_distribution["draw"],
        "pred_home_win_share": predicted_distribution["home_win"],
    }

    return summary, pd.DataFrame(matrix_rows)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    predictor = MatchPredictor()

    probability_rows = []

    for _, row in df.iterrows():
        home_team = row_to_team_dict(row, "home")
        away_team = row_to_team_dict(row, "away")

        probs = predictor.predict_match(home_team, away_team)

        probability_rows.append(
            {
                "p_away_win": probs["away_win"],
                "p_draw": probs["draw"],
                "p_home_win": probs["home_win"],
            }
        )

    probability_df = pd.DataFrame(probability_rows)

    policies: list[tuple[str, list[str]]] = []

    policies.append(
        (
            "argmax",
            [
                predict_argmax(
                    {
                        "away_win": row.p_away_win,
                        "draw": row.p_draw,
                        "home_win": row.p_home_win,
                    }
                )
                for row in probability_df.itertuples(index=False)
            ],
        )
    )

    for threshold in np.round(np.arange(0.18, 0.361, 0.01), 2):
        policies.append(
            (
                f"draw_threshold_{threshold:.2f}",
                [
                    predict_draw_threshold(
                        {
                            "away_win": row.p_away_win,
                            "draw": row.p_draw,
                            "home_win": row.p_home_win,
                        },
                        threshold=float(threshold),
                    )
                    for row in probability_df.itertuples(index=False)
                ],
            )
        )

    for margin in np.round(np.arange(0.02, 0.301, 0.02), 2):
        policies.append(
            (
                f"margin_draw_{margin:.2f}",
                [
                    predict_margin_draw(
                        {
                            "away_win": row.p_away_win,
                            "draw": row.p_draw,
                            "home_win": row.p_home_win,
                        },
                        margin=float(margin),
                    )
                    for row in probability_df.itertuples(index=False)
                ],
            )
        )

    summaries = []
    confusion_frames = []

    for policy_name, predictions in policies:
        summary, matrix_df = evaluate_policy(
            df=df,
            policy_name=policy_name,
            predictions=predictions,
        )

        summaries.append(summary)
        confusion_frames.append(matrix_df)

    summary_df = pd.DataFrame(summaries).sort_values(
        "accuracy",
        ascending=False,
    )

    confusion_df = pd.concat(
        confusion_frames,
        ignore_index=True,
    )

    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    confusion_df.to_csv(CONFUSION_OUTPUT, index=False)

    print("Decision Policy Audit")
    print("---------------------")
    print(summary_df.head(20).to_string(index=False))
    print()
    print(f"Wrote {SUMMARY_OUTPUT}")
    print(f"Wrote {CONFUSION_OUTPUT}")


if __name__ == "__main__":
    main()
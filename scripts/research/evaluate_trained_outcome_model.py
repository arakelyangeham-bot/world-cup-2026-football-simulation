#evaluate_trained_outcome_model.py

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

from scripts.research.calibration.common import (
    evaluate_probabilities,
    multiclass_brier_score,
)
from shared.dataset import prepare_train_test_data
from shared.feature_sets import get_feature_set
from shared.label_encoding import encode_labels


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "models"
    / "lightgbm_v2_reproduction.joblib"
)

SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "trained_model_evaluation.csv"
)

CONFUSION_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "trained_model_confusion_matrix.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained outcome model on the historical dataset."
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
    )
    parser.add_argument(
        "--feature-set",
        default="v2",
    )
    parser.add_argument(
        "--experiment-name",
        default="lightgbm_v2_reproduction",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    model = joblib.load(model_path)

    features = get_feature_set(args.feature_set)

    X, y, X_train, X_test, y_train, y_test, labels, metadata = (
        prepare_train_test_data(features)
    )

    (
        y_encoded,
        y_train_encoded,
        y_test_encoded,
        encoded_labels,
        label_mapping,
        label_encoder,
    ) = encode_labels(y, y_train, y_test)

    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)

    outcome_series = pd.Series(
        [label_mapping[int(label)] for label in y_encoded]
    )

    probability_metrics = evaluate_probabilities(
        outcomes=outcome_series,
        probabilities=probabilities,
    )

    accuracy = accuracy_score(
        y_encoded,
        predictions,
    )

    cm = confusion_matrix(
        y_encoded,
        predictions,
        labels=encoded_labels,
    )

    summary = pd.DataFrame(
        [
            {
                "experiment_name": args.experiment_name,
                "model_path": str(model_path),
                "feature_set": args.feature_set,
                "rows": len(X),
                "accuracy": accuracy,
                **probability_metrics,
            }
        ]
    )

    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    confusion_rows = []

    for actual_idx, actual_label in enumerate(encoded_labels):
        for predicted_idx, predicted_label in enumerate(encoded_labels):
            confusion_rows.append(
                {
                    "actual_label": label_mapping[int(actual_label)],
                    "predicted_label": label_mapping[int(predicted_label)],
                    "count": int(cm[actual_idx, predicted_idx]),
                }
            )

    with CONFUSION_OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "actual_label",
                "predicted_label",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(confusion_rows)

    print("Trained Outcome Model Evaluation")
    print("--------------------------------")
    print(summary.to_string(index=False))
    print()
    print("Confusion matrix")
    print(pd.DataFrame(
        cm,
        index=[label_mapping[int(label)] for label in encoded_labels],
        columns=[label_mapping[int(label)] for label in encoded_labels],
    ).to_string())
    print()
    print(f"Wrote {SUMMARY_OUTPUT}")
    print(f"Wrote {CONFUSION_OUTPUT}")


if __name__ == "__main__":
    main()
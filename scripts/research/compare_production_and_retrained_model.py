#compare_production_and_retrained_model.py

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from inference.model_paths import PRODUCTION_MODEL_PATH
from scripts.research.calibration.common import evaluate_probabilities
from shared.dataset import prepare_train_test_data
from shared.feature_sets import get_feature_set
from shared.label_encoding import encode_labels


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRAINED_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "models"
    / "lightgbm_v2_reproduction.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "production_vs_retrained_model_comparison.csv"
)


def summarize_model(name, model, X, y_encoded, outcome_series):
    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)

    metrics = evaluate_probabilities(
        outcomes=outcome_series,
        probabilities=probabilities,
    )

    return {
        "model": name,
        "accuracy": accuracy_score(y_encoded, predictions),
        **metrics,
    }, probabilities, predictions


def main() -> None:
    features = get_feature_set("v2")

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

    outcome_series = pd.Series(
        [label_mapping[int(label)] for label in y_encoded]
    )

    production_model = joblib.load(PRODUCTION_MODEL_PATH)
    retrained_model = joblib.load(RETRAINED_MODEL_PATH)

    production_summary, production_probs, production_preds = summarize_model(
        "production",
        production_model,
        X,
        y_encoded,
        outcome_series,
    )

    retrained_summary, retrained_probs, retrained_preds = summarize_model(
        "retrained",
        retrained_model,
        X,
        y_encoded,
        outcome_series,
    )

    probability_abs_diff = np.abs(production_probs - retrained_probs)

    comparison_summary = pd.DataFrame(
        [
            production_summary,
            retrained_summary,
            {
                "model": "difference",
                "accuracy": retrained_summary["accuracy"] - production_summary["accuracy"],
                "multiclass_brier_score": (
                    retrained_summary["multiclass_brier_score"]
                    - production_summary["multiclass_brier_score"]
                ),
                "multiclass_log_loss": (
                    retrained_summary["multiclass_log_loss"]
                    - production_summary["multiclass_log_loss"]
                ),
                "ece_mean": (
                    retrained_summary["ece_mean"]
                    - production_summary["ece_mean"]
                ),
            },
            {
                "model": "probability_abs_diff",
                "accuracy": np.mean(production_preds != retrained_preds),
                "multiclass_brier_score": probability_abs_diff.mean(),
                "multiclass_log_loss": probability_abs_diff.max(),
                "ece_mean": np.percentile(probability_abs_diff, 95),
            },
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_summary.to_csv(OUTPUT_PATH, index=False)

    print("Production vs Retrained Model Comparison")
    print("----------------------------------------")
    print(comparison_summary.to_string(index=False))
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
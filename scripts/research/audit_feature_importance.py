#audit_feature_importance.py

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from inference.feature_vector_builder import get_feature_order
from inference.model_paths import PRODUCTION_MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "production_feature_importance.csv"
)


def extract_lightgbm_estimators(model):
    estimators = []

    if hasattr(model, "calibrated_classifiers_"):
        for calibrated_classifier in model.calibrated_classifiers_:
            if hasattr(calibrated_classifier, "estimator"):
                estimators.append(calibrated_classifier.estimator)
            elif hasattr(calibrated_classifier, "base_estimator"):
                estimators.append(calibrated_classifier.base_estimator)

    return estimators


def main() -> None:
    model = joblib.load(PRODUCTION_MODEL_PATH)
    feature_order = get_feature_order(model)

    estimators = extract_lightgbm_estimators(model)

    if not estimators:
        raise RuntimeError("No underlying calibrated estimators found")

    importance_rows = []

    for fold_idx, estimator in enumerate(estimators, start=1):
        if not hasattr(estimator, "feature_importances_"):
            raise RuntimeError(
                f"Estimator for fold {fold_idx} has no feature_importances_"
            )

        importances = estimator.feature_importances_

        if len(importances) != len(feature_order):
            raise ValueError(
                f"Importance length mismatch for fold {fold_idx}: "
                f"{len(importances)} importances vs {len(feature_order)} features"
            )

        for feature, importance in zip(feature_order, importances):
            importance_rows.append(
                {
                    "fold": fold_idx,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

    fold_importance = pd.DataFrame(importance_rows)

    summary = (
        fold_importance
        .groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            min_importance=("importance", "min"),
            max_importance=("importance", "max"),
        )
        .reset_index()
    )

    total_importance = summary["mean_importance"].sum()

    summary["importance_share"] = (
        summary["mean_importance"] / total_importance
        if total_importance != 0
        else 0.0
    )

    summary = summary.sort_values(
        "mean_importance",
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(0, "rank", np.arange(1, len(summary) + 1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("Production Feature Importance")
    print("-----------------------------")
    print(f"Model: {type(model).__name__}")
    print(f"Underlying estimators: {len(estimators)}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
#train_outcome_model.py

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from scripts.research.training_config import ResearchTrainingConfig
from shared.dataset import prepare_train_test_data
from shared.feature_sets import get_feature_set
from shared.label_encoding import encode_labels
from shared.model_config import LIGHTGBM_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research" / "models"
SUMMARY_OUTPUT = PROJECT_ROOT / "outputs" / "research" / "training_summary.csv"
METADATA_OUTPUT = PROJECT_ROOT / "outputs" / "research" / "training_metadata.json"


def train_model(config: ResearchTrainingConfig):
    features = get_feature_set(config.feature_set)

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

    model_config = dict(LIGHTGBM_CONFIG)
    model_config["random_state"] = config.random_state

    base_model = LGBMClassifier(**model_config)

    model = CalibratedClassifierCV(
        estimator=base_model,
        method=config.calibration_method,
        cv=config.calibration_cv,
    )

    start = time.perf_counter()
    model.fit(X, y_encoded)
    elapsed = time.perf_counter() - start

    return {
        "model": model,
        "features": features,
        "X": X,
        "y_encoded": y_encoded,
        "encoded_labels": encoded_labels,
        "label_mapping": label_mapping,
        "training_rows": len(X),
        "training_seconds": elapsed,
    }


def main() -> None:
    config = ResearchTrainingConfig()

    result = train_model(config)

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_OUTPUT_DIR / f"{config.experiment_name}.joblib"

    if config.save_model:
        joblib.dump(result["model"], model_path)

    summary = pd.DataFrame(
        [
            {
                "experiment_name": config.experiment_name,
                "feature_set": config.feature_set,
                "model_class": type(result["model"]).__name__,
                "base_estimator": "LGBMClassifier",
                "calibration_method": config.calibration_method,
                "calibration_cv": config.calibration_cv,
                "n_features": len(result["features"]),
                "training_rows": result["training_rows"],
                "training_seconds": result["training_seconds"],
                "model_path": str(model_path) if config.save_model else None,
            }
        ]
    )

    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    metadata = {
        "experiment_name": config.experiment_name,
        "feature_set": config.feature_set,
        "features": result["features"],
        "encoded_labels": result["encoded_labels"],
        "label_mapping": result["label_mapping"],
        "training_rows": result["training_rows"],
        "training_seconds": result["training_seconds"],
        "lightgbm_config": dict(LIGHTGBM_CONFIG),
        "calibration_method": config.calibration_method,
        "calibration_cv": config.calibration_cv,
        "model_path": str(model_path) if config.save_model else None,
    }

    with METADATA_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Research Outcome Model Training")
    print("-------------------------------")
    print(summary.to_string(index=False))
    print()
    print(f"Wrote {SUMMARY_OUTPUT}")
    print(f"Wrote {METADATA_OUTPUT}")

    if config.save_model:
        print(f"Wrote {model_path}")


if __name__ == "__main__":
    main()
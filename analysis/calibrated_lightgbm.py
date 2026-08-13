# analysis/calibrated_lightgbm.py

import json

import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

from shared.calibration import multiclass_brier_score
from shared.config import ML_OUTPUT_DIR
from shared.dataset import prepare_train_test_data
from shared.feature_sets import FEATURE_SETS
from shared.label_encoding import encode_labels
from shared.model_config import LIGHTGBM_CONFIG


OUTPUT_DIR = ML_OUTPUT_DIR / "calibration" / "calibrated_lightgbm"


def evaluate_calibrated_lightgbm(feature_set_name, features, method):
    X, y, X_train, X_test, y_train, y_test, labels, metadata = (
        prepare_train_test_data(features)
    )

    (
        y_encoded,
        y_train_encoded,
        y_test_encoded,
        encoded_labels,
        label_mapping,
        _,
    ) = encode_labels(y, y_train, y_test)

    base_model = LGBMClassifier(**LIGHTGBM_CONFIG)

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=5,
    )

    calibrated_model.fit(X_train, y_train_encoded)

    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)

    row = {
        "model": "CalibratedLightGBM",
        "feature_set": feature_set_name,
        "calibration_method": method,
        "accuracy": accuracy_score(y_test_encoded, y_pred),
        "log_loss": log_loss(y_test_encoded, y_proba, labels=encoded_labels),
        "brier_score": multiclass_brier_score(
            y_test_encoded,
            y_proba,
            encoded_labels,
        ),
        "label_mapping": label_mapping,
    }

    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for feature_set_name, features in FEATURE_SETS.items():
        for method in ["sigmoid", "isotonic"]:
            print(
                f"Evaluating calibrated LightGBM: "
                f"{feature_set_name}, method={method}"
            )

            rows.append(
                evaluate_calibrated_lightgbm(
                    feature_set_name=feature_set_name,
                    features=features,
                    method=method,
                )
            )

    results_df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "calibrated_lightgbm_results.csv"
    results_df.drop(columns=["label_mapping"]).to_csv(csv_path, index=False)

    json_path = OUTPUT_DIR / "calibrated_lightgbm_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print()
    print("Calibrated LightGBM results:")
    print(
        results_df.drop(columns=["label_mapping"])
        .sort_values("log_loss")
        .to_string(index=False)
    )
    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
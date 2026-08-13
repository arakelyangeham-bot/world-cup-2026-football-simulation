# validation/cv_calibrated_lightgbm.py

import json

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

from shared.calibration import multiclass_brier_score
from shared.config import PROJECT_ROOT, RANDOM_STATE
from shared.dataset import prepare_train_test_data
from shared.feature_sets import FEATURE_SETS
from shared.label_encoding import encode_labels
from shared.model_config import LIGHTGBM_CONFIG


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation"


def evaluate_cv_calibrated_lightgbm(feature_set_name, features):
    print(f"\nCV validating calibrated LightGBM: {feature_set_name}")

    X, y, *_rest = prepare_train_test_data(features)
    y_train = _rest[3]
    y_test = _rest[4]

    (
        y_encoded,
        _,
        _,
        encoded_labels,
        label_mapping,
        _,
    ) = encode_labels(y, y_train, y_test)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    accuracy_scores = []
    log_loss_scores = []
    brier_scores = []

    for train_idx, test_idx in cv.split(X, y_encoded):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]

        y_train_fold = y_encoded[train_idx]
        y_test_fold = y_encoded[test_idx]

        base_model = LGBMClassifier(**LIGHTGBM_CONFIG)

        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=3,
        )

        calibrated_model.fit(X_train_fold, y_train_fold)

        y_pred = calibrated_model.predict(X_test_fold)
        y_proba = calibrated_model.predict_proba(X_test_fold)

        accuracy_scores.append(accuracy_score(y_test_fold, y_pred))
        log_loss_scores.append(
            log_loss(y_test_fold, y_proba, labels=encoded_labels)
        )
        brier_scores.append(
            multiclass_brier_score(
                y_test_fold,
                y_proba,
                encoded_labels,
            )
        )

    return {
        "model": "CalibratedLightGBM",
        "feature_set": feature_set_name,
        "calibration_method": "sigmoid",
        "cv_accuracy_mean": float(np.mean(accuracy_scores)),
        "cv_accuracy_std": float(np.std(accuracy_scores)),
        "cv_log_loss_mean": float(np.mean(log_loss_scores)),
        "cv_log_loss_std": float(np.std(log_loss_scores)),
        "cv_brier_score_mean": float(np.mean(brier_scores)),
        "cv_brier_score_std": float(np.std(brier_scores)),
        "label_mapping": label_mapping,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for feature_set_name, features in FEATURE_SETS.items():
        rows.append(
            evaluate_cv_calibrated_lightgbm(
                feature_set_name,
                features,
            )
        )

    results_df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "cv_calibrated_lightgbm.csv"
    json_path = OUTPUT_DIR / "cv_calibrated_lightgbm.json"

    results_df.drop(columns=["label_mapping"]).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print()
    print("Cross-validated calibrated LightGBM results:")
    print(
        results_df.drop(columns=["label_mapping"])
        .sort_values("cv_log_loss_mean")
        .to_string(index=False)
    )

    print()
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
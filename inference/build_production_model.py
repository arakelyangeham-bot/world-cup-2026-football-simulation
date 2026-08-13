# build_production_model.py

import json
import joblib

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from inference.model_paths import (
    PRODUCTION_MODEL_DIR,
    PRODUCTION_MODEL_PATH,
    PRODUCTION_METADATA_PATH,
)

from shared.dataset import prepare_train_test_data
from shared.feature_sets import get_feature_set
from shared.label_encoding import encode_labels
from shared.model_config import LIGHTGBM_CONFIG


# ---------------------------------------------------------------------
# Production Experiment Configuration
# ---------------------------------------------------------------------

PRODUCTION_FEATURE_SET = "v2"


def main():
    PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    features = get_feature_set(PRODUCTION_FEATURE_SET)

    X, y, X_train, X_test, y_train, y_test, labels, metadata = (
        prepare_train_test_data(features)
    )

    (
        y_encoded,
        _,
        _,
        encoded_labels,
        label_mapping,
        _,
    ) = encode_labels(y, y_train, y_test)

    base_model = LGBMClassifier(**LIGHTGBM_CONFIG)

    production_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )

    production_model.fit(X, y_encoded)

    joblib.dump(production_model, PRODUCTION_MODEL_PATH)

    production_metadata = {
        "model": "CalibratedLightGBM",
        "calibration_method": "sigmoid",
        "feature_set": PRODUCTION_FEATURE_SET,
        "features": features,
        "label_mapping": label_mapping,
        "training_rows": len(X),
        "encoded_labels": encoded_labels,
    }

    with open(PRODUCTION_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(production_metadata, f, indent=2)

    print("Production model saved.")
    print(f"Model: {PRODUCTION_MODEL_PATH}")
    print(f"Metadata: {PRODUCTION_METADATA_PATH}")


if __name__ == "__main__":
    main()
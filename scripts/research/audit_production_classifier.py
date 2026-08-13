#audit_production_classifier.py

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from inference.model_paths import PRODUCTION_MODEL_PATH
from inference.feature_vector_builder import get_feature_order


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "production_classifier_audit.csv"
)


def main() -> None:
    model = joblib.load(PRODUCTION_MODEL_PATH)
    feature_order = get_feature_order(model)

    rows = [
        {
            "production_model_path": str(PRODUCTION_MODEL_PATH),
            "model_class": type(model).__name__,
            "n_features": len(feature_order),
            "features": ";".join(feature_order),
            "has_feature_importances": hasattr(model, "feature_importances_"),
            "has_coef": hasattr(model, "coef_"),
            "has_predict_proba": hasattr(model, "predict_proba"),
            "has_classes": hasattr(model, "classes_"),
            "classes": (
                ";".join(str(c) for c in model.classes_)
                if hasattr(model, "classes_")
                else None
            ),
        }
    ]

    output = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("Production Classifier Audit")
    print("---------------------------")
    print(output.to_string(index=False))
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
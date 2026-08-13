# tune_lightgbm.py

import json

import pandas as pd

from lightgbm import LGBMClassifier

from shared.config import ML_OUTPUT_DIR, RANDOM_STATE
from shared.dataset import prepare_train_test_data
from shared.feature_sets import FEATURE_SETS
from shared.hyperparameter_search import run_random_search
from shared.label_encoding import encode_labels
from shared.model_config import LIGHTGBM_CONFIG, LIGHTGBM_PARAM_DISTRIBUTIONS


OUTPUT_DIR = ML_OUTPUT_DIR / "tuning" / "lightgbm"


def tune_feature_set(feature_set_name, features):
    print(f"\nTuning LightGBM using feature set: {feature_set_name}")

    X, y, *_rest = prepare_train_test_data(features)
    y_train = _rest[3]
    y_test = _rest[4]

    (
        y_encoded,
        _,
        _,
        _,
        label_mapping,
        _,
    ) = encode_labels(y, y_train, y_test)

    model = LGBMClassifier(**LIGHTGBM_CONFIG)

    results = run_random_search(
        model=model,
        param_distributions=LIGHTGBM_PARAM_DISTRIBUTIONS,
        X=X,
        y=y_encoded,
        scoring="neg_log_loss",
        random_state=RANDOM_STATE,
        n_iter=50,
    )

    feature_output_dir = OUTPUT_DIR / feature_set_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    results["cv_results"].to_csv(
        feature_output_dir / "random_search_results.csv",
        index=False,
    )

    with open(feature_output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(results["best_params"], f, indent=2)

    print(f"Best params: {results['best_params']}")
    print(f"Best CV log loss: {-results['best_score']:.4f}")

    return {
        "feature_set": feature_set_name,
        "best_score_neg_log_loss": results["best_score"],
        "best_cv_log_loss": -results["best_score"],
        "best_params": results["best_params"],
        "label_mapping": label_mapping,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for feature_set_name, features in FEATURE_SETS.items():
        summary_rows.append(tune_feature_set(feature_set_name, features))

    summary_df = pd.DataFrame(summary_rows)

    csv_path = OUTPUT_DIR / "lightgbm_tuning_summary.csv"
    json_path = OUTPUT_DIR / "lightgbm_tuning_summary.json"

    summary_df.drop(columns=["label_mapping"]).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print()
    print("LightGBM tuning summary:")
    print(
        summary_df.drop(columns=["label_mapping"])
        .sort_values("best_cv_log_loss")
        .to_string(index=False)
    )

    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
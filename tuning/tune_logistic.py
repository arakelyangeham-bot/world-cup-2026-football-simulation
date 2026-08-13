# tune_logistic.py

import json

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.config import ML_OUTPUT_DIR, RANDOM_STATE
from shared.dataset import prepare_train_test_data
from shared.feature_sets import FEATURE_SETS
from shared.hyperparameter_search import run_grid_search
from shared.model_config import LOGISTIC_CONFIG, LOGISTIC_PARAM_GRID


OUTPUT_DIR = ML_OUTPUT_DIR / "tuning" / "logistic"


def tune_feature_set(feature_set_name, features):
    print(f"\nTuning Logistic Regression using feature set: {feature_set_name}")

    X, y, *_ = prepare_train_test_data(features)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logistic_regression", LogisticRegression(**LOGISTIC_CONFIG)),
        ]
    )

    results = run_grid_search(
        model=model,
        param_grid=LOGISTIC_PARAM_GRID,
        X=X,
        y=y,
        scoring="neg_log_loss",
        random_state=RANDOM_STATE,
    )

    feature_output_dir = OUTPUT_DIR / feature_set_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    results["cv_results"].to_csv(
        feature_output_dir / "grid_search_results.csv",
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
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for feature_set_name, features in FEATURE_SETS.items():
        summary_rows.append(tune_feature_set(feature_set_name, features))

    summary_path = OUTPUT_DIR / "logistic_tuning_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print()
    print(f"Saved tuning summary: {summary_path}")


if __name__ == "__main__":
    main()
# experiment.py

import joblib
import time

from shared.evaluation import evaluate_model
from shared.results_summary import build_summary_row
from shared.cross_validation import evaluate_cross_validation
from shared.feature_importance import save_feature_importance


def run_experiment(
    model_name,
    model,
    X,
    y,
    X_train,
    X_test,
    y_train,
    y_test,
    labels,
    feature_set_name,
    features,
    output_dir,
    metadata,
):
    """
    Fit, evaluate, save, and summarize one ML experiment.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = output_dir / f"{model_name.lower()}_{feature_set_name}"

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - train_start

    metadata = dict(metadata)
    metadata["train_seconds"] = train_seconds

    cv_results = evaluate_cross_validation(
        model=model,
        X=X,
        y=y,
        labels=labels,
        random_state=metadata["random_state"],
    )

    metadata.update(cv_results)

    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        output_prefix=output_prefix,
        metadata=metadata,
    )

    model_path = output_dir / f"{model_name.lower()}_{feature_set_name}.joblib"
    joblib.dump(model, model_path)

    save_feature_importance(
        model=model,
        feature_names=features,
        output_prefix=output_prefix,
    )

    print(f"Saved model: {model_path}")
    print(f"Training time: {train_seconds:.4f} seconds")

    print("=" * 60)
    print(f"Results ({feature_set_name})")
    print("=" * 60)

    for metric_name, metric_value in results["metrics"].items():
        print(f"{metric_name:20s}: {metric_value:.4f}")

    print()

    return build_summary_row(
        model_name=model_name,
        feature_set_name=feature_set_name,
        metrics=results["metrics"],
        metadata=metadata,
    )
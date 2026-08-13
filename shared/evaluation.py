# evaluation.py

import json
import pandas as pd

from shared.metrics import calculate_classification_metrics


def evaluate_model(
    model,
    X_test,
    y_test,
    labels,
    output_prefix,
    metadata=None,
):
    """
    Evaluate a fitted classification model and save reproducible outputs.

    Saves:
        - predictions CSV
        - metrics CSV
        - confusion matrix CSV
        - classification report CSV
        - metadata JSON
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics, confusion, report = calculate_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels,
    )

    predictions_df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred,
    })

    for idx, label in enumerate(model.classes_):
        predictions_df[f"prob_{label}"] = y_proba[:, idx]

    metrics_df = pd.DataFrame([metrics])

    confusion_df = pd.DataFrame(
        confusion,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    report_df = pd.DataFrame(report).transpose()

    predictions_df.to_csv(f"{output_prefix}_predictions.csv", index=False)
    metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
    confusion_df.to_csv(f"{output_prefix}_confusion_matrix.csv")
    report_df.to_csv(f"{output_prefix}_classification_report.csv")

    if metadata is not None:
        with open(f"{output_prefix}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_df,
        "classification_report": report_df,
        "predictions": predictions_df,
    }
# metrics.py

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

from shared.calibration import multiclass_brier_score


def calculate_classification_metrics(y_true, y_pred, y_proba, labels):
    """
    Calculate reusable classification metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=labels),
        "brier_score": multiclass_brier_score(y_true, y_proba, labels),
    }

    confusion = confusion_matrix(y_true, y_pred, labels=labels)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return metrics, confusion, report
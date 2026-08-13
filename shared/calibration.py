#calibration.py

from sklearn.metrics import brier_score_loss


def multiclass_brier_score(y_true, y_proba, labels):
    """
    Compute multiclass Brier score using one-vs-rest averaging.
    Lower is better.
    """

    scores = []

    for idx, label in enumerate(labels):
        binary_true = (y_true == label).astype(int)
        scores.append(
            brier_score_loss(binary_true, y_proba[:, idx])
        )

    return sum(scores) / len(scores)
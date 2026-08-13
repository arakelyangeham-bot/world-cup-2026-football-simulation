# cross_validation.py

import numpy as np

from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold


def evaluate_cross_validation(model, X, y, labels, cv_splits=5, random_state=42):
    """
    Evaluate a classification model using stratified K-fold CV.
    """

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    accuracy_scores = []
    balanced_accuracy_scores = []
    log_loss_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        if hasattr(y, "iloc"):
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
        else:
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)

        y_pred = fold_model.predict(X_test_fold)
        y_proba = fold_model.predict_proba(X_test_fold)

        accuracy_scores.append(accuracy_score(y_test_fold, y_pred))
        balanced_accuracy_scores.append(
            balanced_accuracy_score(y_test_fold, y_pred)
        )
        log_loss_scores.append(
            log_loss(y_test_fold, y_proba, labels=labels)
        )

    return {
        "cv_accuracy_mean": np.mean(accuracy_scores),
        "cv_accuracy_std": np.std(accuracy_scores),
        "cv_balanced_accuracy_mean": np.mean(balanced_accuracy_scores),
        "cv_balanced_accuracy_std": np.std(balanced_accuracy_scores),
        "cv_log_loss_mean": np.mean(log_loss_scores),
        "cv_log_loss_std": np.std(log_loss_scores),
        "cv_splits": cv_splits,
    }
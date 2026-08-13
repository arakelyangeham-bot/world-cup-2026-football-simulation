# hyperparameter_search.py

import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


def run_grid_search(
    model,
    param_grid,
    X,
    y,
    scoring="neg_log_loss",
    cv_splits=5,
    random_state=42,
    n_jobs=-1,
):
    """
    Run reusable stratified grid search.

    Returns:
        best_model
        best_params
        best_score
        cv_results_df
    """

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )

    search.fit(X, y)

    cv_results_df = pd.DataFrame(search.cv_results_)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": cv_results_df,
    }

def run_random_search(
    model,
    param_distributions,
    X,
    y,
    scoring="neg_log_loss",
    cv_splits=5,
    random_state=42,
    n_iter=50,
    n_jobs=-1,
):
    """
    Run reusable stratified randomized search.
    """

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        return_train_score=True,
    )

    search.fit(X, y)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": pd.DataFrame(search.cv_results_),
    }
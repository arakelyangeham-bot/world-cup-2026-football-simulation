# model_config.py

from shared.config import RANDOM_STATE


LOGISTIC_CONFIG = {
    "C": 0.01,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}


RANDOM_FOREST_CONFIG = {
    "n_estimators": 500,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGBOOST_CONFIG = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
}

LIGHTGBM_CONFIG = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
}

MLP_CONFIG = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.001,
    "learning_rate_init": 0.001,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

LOGISTIC_PARAM_GRID = {
    "logistic_regression__C": [0.001, 0.01, 0.1, 1.0],
    "logistic_regression__solver": ["lbfgs"],
}

LIGHTGBM_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "num_leaves": [7, 15, 31, 63],
    "max_depth": [2, 3, 4, 5, 6, -1],
    "min_child_samples": [5, 10, 20, 30],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.0, 0.01, 0.1, 1.0],
}
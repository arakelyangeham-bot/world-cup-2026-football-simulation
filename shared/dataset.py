#dataset.py

import pandas as pd

from sklearn.model_selection import train_test_split
from shared.feature_sets import get_feature_set

from shared.config import (
    HISTORICAL_TRAINING_DATASET_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

FEATURES = get_feature_set("v2")

TARGET_COLUMN = "result"


def load_training_dataset():
    return pd.read_csv(HISTORICAL_TRAINING_DATASET_PATH)


def validate_columns(df, features, target_column=TARGET_COLUMN):
    missing = [col for col in features + [target_column] if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_train_test_data(features, target_column=TARGET_COLUMN):
    df = load_training_dataset()

    validate_columns(df, features, target_column)

    model_df = df[features + [target_column]].dropna()

    X = model_df[features]
    y = model_df[target_column]

    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    metadata = {
        "target_column": target_column,
        "rows_used": len(model_df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "labels": labels,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    return X, y, X_train, X_test, y_train, y_test, labels, metadata
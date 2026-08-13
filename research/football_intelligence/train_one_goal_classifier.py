#train_one_goal_classifier.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "football_intelligence"
    / "football_population_training_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "football_intelligence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "one_goal_classifier_metrics.csv"

TARGET_COLUMN = "is_one_goal_match"

FEATURE_COLUMNS = [
    "home_attack",
    "home_midfield",
    "home_defense",
    "home_gk",
    "away_attack",
    "away_midfield",
    "away_defense",
    "away_gk",
    "attack_diff",
    "midfield_diff",
    "defense_diff",
    "gk_diff",
    "home_poisson_attack",
    "home_poisson_defense",
    "away_poisson_attack",
    "away_poisson_defense",
    "poisson_attack_diff",
    "poisson_defense_diff",
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
]


def evaluate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "model": name,
        "target": TARGET_COLUMN,
        "rows": len(y),
        "positive_rate": y.mean(),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "brier_score": brier_score_loss(y, probabilities),
    }


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    missing = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    models = [
        (
            "majority_baseline",
            DummyClassifier(
                strategy="most_frequent",
            ),
        ),
        (
            "logistic_regression",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]

    rows = [
        evaluate_model(
            name=name,
            model=model,
            X=X,
            y=y,
        )
        for name, model in models
    ]

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("Football Intelligence — One-Goal Classifier")
    print("-------------------------------------------")
    print(result.round(4).to_string(index=False))
    print()
    print(f"Wrote metrics -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
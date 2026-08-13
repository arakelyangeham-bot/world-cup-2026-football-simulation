from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

from scripts.research.calibration.class_multiplier import ClassMultiplierCalibration
from scripts.research.calibration.common import evaluate_probabilities
from scripts.research.calibration.identity import IdentityCalibration
from scripts.research.calibration.temperature import TemperatureScalingCalibration


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "probability_calibration_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "probability_calibration_cv_results.csv"
)


def base_probabilities(df: pd.DataFrame):
    return df[
        [
            "p_home_win",
            "p_draw",
            "p_away_win",
        ]
    ].to_numpy(dtype=float)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    probs = base_probabilities(df)

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(df), start=1):
        train_outcomes = df.iloc[train_idx]["outcome"]
        test_outcomes = df.iloc[test_idx]["outcome"]

        train_probs = probs[train_idx]
        test_probs = probs[test_idx]

        methods = [
            IdentityCalibration(),
            ClassMultiplierCalibration(),
            TemperatureScalingCalibration(),
        ]

        for method in methods:
            method.fit(
                outcomes=train_outcomes,
                probabilities=train_probs,
            )

            calibrated_test_probs = method.transform(test_probs)

            metrics = evaluate_probabilities(
                test_outcomes,
                calibrated_test_probs,
            )

            rows.append(
                {
                    "model": method.name,
                    "fold": fold,
                    "selected_on": (
                        "none"
                        if method.name == "identity"
                        else "training_fold"
                    ),
                    "parameters": method.parameters(),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("Probability Calibration Cross-Validation")
    print("----------------------------------------")
    print(results.to_string(index=False))
    print()
    print("Mean metrics")
    print(
        results
        .drop(columns=["fold"])
        .groupby("model")
        .mean(numeric_only=True)
        .to_string()
    )
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
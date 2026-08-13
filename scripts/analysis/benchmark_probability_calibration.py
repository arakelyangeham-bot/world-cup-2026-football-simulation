#benchmark_probability_calibration.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "probability_calibration_dataset.csv"
)

SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "probability_calibration_summary.csv"
)


OUTCOME_ORDER = [
    "home_win",
    "draw",
    "away_win",
]

MODEL_NAME = "v5.1_dixon_coles_hierarchical"
CALIBRATION_DATASET = "probability_calibration_dataset.csv"

def multiclass_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Mean multiclass Brier score.
    """

    one_hot = np.zeros_like(y_prob)

    one_hot[np.arange(len(y_true)), y_true] = 1.0

    return np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))

def expected_calibration_error(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """
    Compute Expected Calibration Error (ECE) and
    Maximum Calibration Error (MCE) for a binary event.
    """

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0

    total = len(observed)

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (predicted >= bins[i])
                & (predicted <= bins[i + 1])
            )
        else:
            mask = (
                (predicted >= bins[i])
                & (predicted < bins[i + 1])
            )

        if not np.any(mask):
            continue

        observed_frequency = observed[mask].mean()
        predicted_mean = predicted[mask].mean()

        error = abs(
            observed_frequency
            - predicted_mean
        )

        weight = mask.sum() / total

        ece += weight * error
        mce = max(mce, error)

    return ece, mce


def main() -> None:

    df = pd.read_csv(DATASET_PATH)

    outcome_to_index = {
        outcome: idx
        for idx, outcome in enumerate(OUTCOME_ORDER)
    }

    y_true = (
        df["outcome"]
        .map(outcome_to_index)
        .to_numpy(dtype=int)
    )

    y_prob = df[
        [
            "p_home_win",
            "p_draw",
            "p_away_win",
        ]
    ].to_numpy(dtype=float)

    brier = multiclass_brier_score(
        y_true,
        y_prob,
    )

    ll = log_loss(
        y_true,
        y_prob,
        labels=[0, 1, 2],
    )

    ece_home, mce_home = expected_calibration_error(
        (df["outcome"] == "home_win").to_numpy(dtype=int),
        df["p_home_win"].to_numpy(dtype=float),
    )

    ece_draw, mce_draw = expected_calibration_error(
        (df["outcome"] == "draw").to_numpy(dtype=int),
        df["p_draw"].to_numpy(dtype=float),
    )

    ece_away, mce_away = expected_calibration_error(
        (df["outcome"] == "away_win").to_numpy(dtype=int),
        df["p_away_win"].to_numpy(dtype=float),
    )

    summary = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "calibration_dataset": CALIBRATION_DATASET,
                "matches": len(df),
                "multiclass_brier_score": brier,
                "multiclass_log_loss": ll,
                "ece_home_win": ece_home,
                "ece_draw": ece_draw,
                "ece_away_win": ece_away,
                "ece_mean": np.mean(
                    [
                        ece_home,
                        ece_draw,
                        ece_away,
                    ]
                ),
                "mce_home_win": mce_home,
                "mce_draw": mce_draw,
                "mce_away_win": mce_away,
                "mce_max": max(
                    mce_home,
                    mce_draw,
                    mce_away,
                ),
            }
        ]
    )
    SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("Probability Calibration Summary")
    print("--------------------------------")
    print(summary.to_string(index=False))
    print()
    print(f"Wrote {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
#common.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


OUTCOME_ORDER = ["home_win", "draw", "away_win"]


@dataclass(frozen=True)
class CalibrationResult:
    model: str
    parameters: str
    probabilities: np.ndarray


class CalibrationMethod:
    name = "base"

    def fit(
        self,
        outcomes: pd.Series,
        probabilities: np.ndarray,
    ) -> "CalibrationMethod":
        return self

    def transform(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> str:
        return "none"


def multiclass_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(observed)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (predicted >= bins[i]) & (predicted <= bins[i + 1])
        else:
            mask = (predicted >= bins[i]) & (predicted < bins[i + 1])

        if not np.any(mask):
            continue

        observed_frequency = observed[mask].mean()
        predicted_mean = predicted[mask].mean()

        ece += (mask.sum() / total) * abs(
            observed_frequency - predicted_mean
        )

    return float(ece)


def mean_ece(
    outcomes: pd.Series,
    probabilities: np.ndarray,
) -> float:
    return float(
        np.mean(
            [
                expected_calibration_error(
                    (outcomes == "home_win").to_numpy(dtype=int),
                    probabilities[:, 0],
                ),
                expected_calibration_error(
                    (outcomes == "draw").to_numpy(dtype=int),
                    probabilities[:, 1],
                ),
                expected_calibration_error(
                    (outcomes == "away_win").to_numpy(dtype=int),
                    probabilities[:, 2],
                ),
            ]
        )
    )


def evaluate_probabilities(
    outcomes: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    outcome_to_index = {
        outcome: idx
        for idx, outcome in enumerate(OUTCOME_ORDER)
    }

    y_true = outcomes.map(outcome_to_index).to_numpy(dtype=int)

    return {
        "multiclass_brier_score": multiclass_brier_score(
            y_true,
            probabilities,
        ),
        "multiclass_log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1, 2],
            )
        ),
        "ece_mean": mean_ece(
            outcomes,
            probabilities,
        ),
    }
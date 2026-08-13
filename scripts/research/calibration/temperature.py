#temperature.py

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.calibration.common import (
    CalibrationMethod,
    evaluate_probabilities,
)


class TemperatureScalingCalibration(CalibrationMethod):
    name = "temperature_scaling"

    def __init__(self, temperature_values=None):
        self.temperature_values = temperature_values
        self.temperature = 1.0

    def fit(
        self,
        outcomes: pd.Series,
        probabilities: np.ndarray,
    ) -> "TemperatureScalingCalibration":
        temperature_values = self.temperature_values or np.round(
            np.arange(0.70, 1.501, 0.01),
            2,
        )

        best = None

        for temperature in temperature_values:
            adjusted = self._apply(
                probabilities,
                float(temperature),
            )

            metrics = evaluate_probabilities(
                outcomes,
                adjusted,
            )

            candidate_key = (
                metrics["multiclass_log_loss"],
                metrics["multiclass_brier_score"],
                metrics["ece_mean"],
            )

            if best is None or candidate_key < best["key"]:
                best = {
                    "key": candidate_key,
                    "temperature": float(temperature),
                }

        if best is None:
            raise RuntimeError("No temperature candidates evaluated")

        self.temperature = best["temperature"]

        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return self._apply(
            probabilities,
            self.temperature,
        )

    def parameters(self) -> str:
        return f"temperature={self.temperature:.2f}"

    @staticmethod
    def _apply(
        probabilities: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        eps = 1e-15

        clipped = np.clip(
            probabilities,
            eps,
            1.0,
        )

        logits = np.log(clipped)
        scaled_logits = logits / temperature

        scaled_logits = scaled_logits - scaled_logits.max(
            axis=1,
            keepdims=True,
        )

        exp_logits = np.exp(scaled_logits)

        return exp_logits / exp_logits.sum(
            axis=1,
            keepdims=True,
        )
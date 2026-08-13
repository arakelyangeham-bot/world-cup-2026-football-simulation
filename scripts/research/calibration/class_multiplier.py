#class_multiplier.py

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.calibration.common import (
    CalibrationMethod,
    evaluate_probabilities,
)


class ClassMultiplierCalibration(CalibrationMethod):
    name = "class_multiplier_cv"

    def __init__(
        self,
        home_values=None,
        draw_values=None,
        away_values=None,
    ):
        self.home_values = home_values
        self.draw_values = draw_values
        self.away_values = away_values

        self.home_multiplier = 1.0
        self.draw_multiplier = 1.0
        self.away_multiplier = 1.0

    def fit(
        self,
        outcomes: pd.Series,
        probabilities: np.ndarray,
    ) -> "ClassMultiplierCalibration":
        home_values = self.home_values or np.round(np.arange(0.95, 1.051, 0.01), 2)
        draw_values = self.draw_values or np.round(np.arange(0.90, 1.211, 0.01), 2)
        away_values = self.away_values or np.round(np.arange(0.95, 1.051, 0.01), 2)

        best = None

        for home_multiplier in home_values:
            for draw_multiplier in draw_values:
                for away_multiplier in away_values:
                    adjusted = self._apply(
                        probabilities,
                        float(home_multiplier),
                        float(draw_multiplier),
                        float(away_multiplier),
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
                            "home_multiplier": float(home_multiplier),
                            "draw_multiplier": float(draw_multiplier),
                            "away_multiplier": float(away_multiplier),
                        }

        if best is None:
            raise RuntimeError("No class multiplier candidates evaluated")

        self.home_multiplier = best["home_multiplier"]
        self.draw_multiplier = best["draw_multiplier"]
        self.away_multiplier = best["away_multiplier"]

        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return self._apply(
            probabilities,
            self.home_multiplier,
            self.draw_multiplier,
            self.away_multiplier,
        )

    def parameters(self) -> str:
        return (
            f"home={self.home_multiplier:.2f};"
            f"draw={self.draw_multiplier:.2f};"
            f"away={self.away_multiplier:.2f}"
        )

    @staticmethod
    def _apply(
        probabilities: np.ndarray,
        home_multiplier: float,
        draw_multiplier: float,
        away_multiplier: float,
    ) -> np.ndarray:
        multipliers = np.array(
            [home_multiplier, draw_multiplier, away_multiplier],
            dtype=float,
        )

        adjusted = probabilities * multipliers
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)

        return adjusted
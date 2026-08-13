#identity.py

from __future__ import annotations

import numpy as np

from scripts.research.calibration.common import CalibrationMethod


class IdentityCalibration(CalibrationMethod):
    name = "identity"

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return probabilities.copy()
#training_config.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchTrainingConfig:
    experiment_name: str = "lightgbm_v2_reproduction"
    feature_set: str = "v2"
    calibration_method: str = "sigmoid"
    calibration_cv: int = 5
    random_state: int = 42
    save_model: bool = True
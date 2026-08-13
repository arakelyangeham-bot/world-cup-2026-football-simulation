#club_goal_model_artifact

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GoalModelTargetArtifact:
    """
    Frozen coefficients for one Poisson goal target.
    """

    target: str
    features: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]

    def validate(self) -> None:
        if not self.target.strip():
            raise ValueError(
                "Target name cannot be empty."
            )

        if not self.features:
            raise ValueError(
                "Target artifact must contain features."
            )

        if len(self.features) != len(
            self.coefficients
        ):
            raise ValueError(
                "Feature and coefficient counts differ."
            )

        if len(self.features) != len(
            set(self.features)
        ):
            raise ValueError(
                "Target artifact contains duplicate "
                "features."
            )

        numeric_values = np.array(
            [
                self.intercept,
                *self.coefficients,
            ],
            dtype=float,
        )

        if not np.isfinite(
            numeric_values
        ).all():
            raise ValueError(
                "Target artifact contains non-finite "
                "parameters."
            )

    def predict_expected_goals(
        self,
        feature_values: Mapping[str, float],
    ) -> float:
        """
        Apply the fitted Poisson log-link model directly.
        """
        self.validate()

        missing = (
            set(self.features)
            - set(feature_values)
        )

        if missing:
            raise KeyError(
                "Prediction input is missing features: "
                f"{sorted(missing)}"
            )

        values = np.array(
            [
                float(feature_values[feature])
                for feature in self.features
            ],
            dtype=float,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Prediction input contains non-finite "
                "feature values."
            )

        linear_predictor = float(
            self.intercept
            + np.dot(
                values,
                np.asarray(
                    self.coefficients,
                    dtype=float,
                ),
            )
        )

        expected_goals = math.exp(
            linear_predictor
        )

        if (
            not math.isfinite(expected_goals)
            or expected_goals <= 0.0
        ):
            raise AssertionError(
                "Production model generated an invalid "
                "expected-goals value."
            )

        return expected_goals


@dataclass(frozen=True)
class ClubGoalModelArtifact:
    """
    Portable fitted artifact for the production club
    goal model.

    This records both fitted parameters and the exact
    research contract under which they were produced.
    """

    artifact_name: str
    artifact_version: str

    baseline_name: str
    baseline_version: str
    feature_specification: str

    model_family: str
    alpha: float

    training_dataset: str
    training_match_count: int
    training_start_date: str
    training_end_date: str

    fitted_at: str

    home_model: GoalModelTargetArtifact
    away_model: GoalModelTargetArtifact

    def validate(self) -> None:
        if not self.artifact_name.strip():
            raise ValueError(
                "Artifact name cannot be empty."
            )

        if not self.artifact_version.strip():
            raise ValueError(
                "Artifact version cannot be empty."
            )

        if self.model_family != "poisson_log_link":
            raise ValueError(
                "Unsupported model family: "
                f"{self.model_family!r}"
            )

        if self.alpha < 0.0:
            raise ValueError(
                "Artifact alpha cannot be negative."
            )

        if self.training_match_count < 1:
            raise ValueError(
                "Training match count must be positive."
            )

        training_start = date.fromisoformat(
            self.training_start_date
        )

        training_end = date.fromisoformat(
            self.training_end_date
        )

        if training_start > training_end:
            raise ValueError(
                "Training start date occurs after the "
                "training end date."
            )

        datetime.fromisoformat(
            self.fitted_at
        )

        self.home_model.validate()
        self.away_model.validate()

        if self.home_model.target != "home_score":
            raise ValueError(
                "Home target must be 'home_score'."
            )

        if self.away_model.target != "away_score":
            raise ValueError(
                "Away target must be 'away_score'."
            )

    def predict(
        self,
        feature_values: Mapping[str, float],
    ) -> tuple[float, float]:
        self.validate()

        return (
            self.home_model.predict_expected_goals(
                feature_values
            ),
            self.away_model.predict_expected_goals(
                feature_values
            ),
        )


def write_club_goal_model_artifact(
    artifact: ClubGoalModelArtifact,
    path: Path,
) -> None:
    artifact.validate()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(artifact),
            indent=2,
        ),
        encoding="utf-8",
    )


def load_club_goal_model_artifact(
    path: Path,
) -> ClubGoalModelArtifact:
    if not path.exists():
        raise FileNotFoundError(
            f"Club goal-model artifact does not exist: "
            f"{path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    artifact = ClubGoalModelArtifact(
        artifact_name=payload[
            "artifact_name"
        ],
        artifact_version=payload[
            "artifact_version"
        ],
        baseline_name=payload[
            "baseline_name"
        ],
        baseline_version=payload[
            "baseline_version"
        ],
        feature_specification=payload[
            "feature_specification"
        ],
        model_family=payload[
            "model_family"
        ],
        alpha=float(
            payload["alpha"]
        ),
        training_dataset=payload[
            "training_dataset"
        ],
        training_match_count=int(
            payload["training_match_count"]
        ),
        training_start_date=payload[
            "training_start_date"
        ],
        training_end_date=payload[
            "training_end_date"
        ],
        fitted_at=payload[
            "fitted_at"
        ],
        home_model=GoalModelTargetArtifact(
            target=payload[
                "home_model"
            ]["target"],
            features=tuple(
                payload[
                    "home_model"
                ]["features"]
            ),
            intercept=float(
                payload[
                    "home_model"
                ]["intercept"]
            ),
            coefficients=tuple(
                float(value)
                for value in payload[
                    "home_model"
                ]["coefficients"]
            ),
        ),
        away_model=GoalModelTargetArtifact(
            target=payload[
                "away_model"
            ]["target"],
            features=tuple(
                payload[
                    "away_model"
                ]["features"]
            ),
            intercept=float(
                payload[
                    "away_model"
                ]["intercept"]
            ),
            coefficients=tuple(
                float(value)
                for value in payload[
                    "away_model"
                ]["coefficients"]
            ),
        ),
    )

    artifact.validate()

    return artifact
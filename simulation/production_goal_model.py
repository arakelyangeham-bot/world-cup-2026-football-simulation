#production_goal_model

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from research.production.club_goal_model_artifact import (
    ClubGoalModelArtifact,
    load_club_goal_model_artifact,
)


@dataclass(frozen=True)
class ProductionGoalPrediction:
    """
    Scalar expected-goal prediction returned by a frozen
    production goal model.
    """

    lambda_home: float
    lambda_away: float

    @property
    def pred_home_goals(self) -> float:
        return self.lambda_home

    @property
    def pred_away_goals(self) -> float:
        return self.lambda_away

    @property
    def pred_total_goals(self) -> float:
        return self.lambda_home + self.lambda_away

    @property
    def pred_goal_diff(self) -> float:
        return self.lambda_home - self.lambda_away


class ProductionGoalModel:
    """
    Runtime wrapper around a frozen club goal-model artifact.

    This class performs no fitting. It validates and evaluates
    an already-fitted production artifact.
    """

    def __init__(
        self,
        artifact: ClubGoalModelArtifact,
    ) -> None:
        artifact.validate()
        self._artifact = artifact

    @classmethod
    def from_path(
        cls,
        artifact_path: Path,
    ) -> "ProductionGoalModel":
        artifact = load_club_goal_model_artifact(
            artifact_path
        )

        return cls(artifact)

    @property
    def artifact(self) -> ClubGoalModelArtifact:
        return self._artifact

    @property
    def artifact_name(self) -> str:
        return self._artifact.artifact_name

    @property
    def artifact_version(self) -> str:
        return self._artifact.artifact_version

    @property
    def baseline_name(self) -> str:
        return self._artifact.baseline_name

    @property
    def baseline_version(self) -> str:
        return self._artifact.baseline_version

    @property
    def feature_specification(self) -> str:
        return self._artifact.feature_specification

    @property
    def training_end_date(self) -> str:
        return self._artifact.training_end_date

    @property
    def required_features(self) -> tuple[str, ...]:
        """
        Return the ordered union of home- and away-model
        features.
        """

        ordered_features: list[str] = []

        for feature in (
            *self._artifact.home_model.features,
            *self._artifact.away_model.features,
        ):
            if feature not in ordered_features:
                ordered_features.append(feature)

        return tuple(ordered_features)

    def validate_feature_values(
        self,
        feature_values: Mapping[str, float],
    ) -> None:
        missing = (
            set(self.required_features)
            - set(feature_values)
        )

        if missing:
            raise KeyError(
                "Production prediction is missing required "
                f"features: {sorted(missing)}"
            )

    def predict(
        self,
        feature_values: Mapping[str, float],
    ) -> ProductionGoalPrediction:
        self.validate_feature_values(
            feature_values
        )

        lambda_home, lambda_away = (
            self._artifact.predict(
                feature_values
            )
        )

        return ProductionGoalPrediction(
            lambda_home=float(lambda_home),
            lambda_away=float(lambda_away),
        )
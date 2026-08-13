#integrated_club_goal_predictor

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

from simulation.production_goal_model import (
    ProductionGoalModel,
    ProductionGoalPrediction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLUB_GOAL_MODEL_ARTIFACT = (
    PROJECT_ROOT
    / "outputs"
    / "study_069_production_club_goal_model_v1"
    / "integrated_club_goal_model_v1.json"
)


@dataclass(frozen=True)
class ClubGoalPrediction:
    """
    Expected-goal prediction with production-model provenance.
    """

    lambda_home: float
    lambda_away: float

    artifact_name: str
    artifact_version: str
    baseline_version: str
    feature_specification: str
    training_end_date: str

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


class IntegratedClubGoalPredictor:
    """
    Production interface for Integrated Club Goal Model v1.

    Study 070 accepts an already-assembled feature mapping.
    Live feature assembly will be introduced separately.
    """

    def __init__(
        self,
        artifact_path: Path | None = None,
        enforce_post_training_prediction_date: bool = True,
    ) -> None:
        selected_path = (
            artifact_path
            or DEFAULT_CLUB_GOAL_MODEL_ARTIFACT
        )

        self._model = (
            ProductionGoalModel.from_path(
                selected_path
            )
        )

        self._artifact_path = selected_path
        self._enforce_post_training_prediction_date = (
            enforce_post_training_prediction_date
        )

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    @property
    def model(self) -> ProductionGoalModel:
        return self._model

    @property
    def required_features(self) -> tuple[str, ...]:
        return self._model.required_features

    def _parse_prediction_date(
        self,
        prediction_date: str | date | datetime,
    ) -> date:
        if isinstance(
            prediction_date,
            datetime,
        ):
            return prediction_date.date()

        if isinstance(
            prediction_date,
            date,
        ):
            return prediction_date

        return date.fromisoformat(
            prediction_date
        )

    def _validate_prediction_date(
        self,
        prediction_date: str | date | datetime | None,
    ) -> None:
        if prediction_date is None:
            return

        parsed_prediction_date = (
            self._parse_prediction_date(
                prediction_date
            )
        )

        training_end_date = date.fromisoformat(
            self._model.training_end_date
        )

        if (
            self._enforce_post_training_prediction_date
            and parsed_prediction_date
            <= training_end_date
        ):
            raise ValueError(
                "Production artifact cannot be used for a "
                "leakage-sensitive prediction on or before "
                "its training cutoff. "
                f"Prediction date: {parsed_prediction_date}. "
                f"Training cutoff: {training_end_date}."
            )

    def predict_features(
        self,
        feature_values: Mapping[str, float],
        prediction_date: (
            str | date | datetime | None
        ) = None,
    ) -> ClubGoalPrediction:
        """
        Predict expected goals from a complete Version 1
        feature mapping.
        """

        self._validate_prediction_date(
            prediction_date
        )

        prediction: ProductionGoalPrediction = (
            self._model.predict(
                feature_values
            )
        )

        return ClubGoalPrediction(
            lambda_home=prediction.lambda_home,
            lambda_away=prediction.lambda_away,
            artifact_name=(
                self._model.artifact_name
            ),
            artifact_version=(
                self._model.artifact_version
            ),
            baseline_version=(
                self._model.baseline_version
            ),
            feature_specification=(
                self._model.feature_specification
            ),
            training_end_date=(
                self._model.training_end_date
            ),
        )
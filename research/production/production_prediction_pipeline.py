#production_prediction_pipeline

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Iterable, Mapping

import pandas as pd

from simulation.production_goal_model import (
    ProductionGoalModel,
)
from research.production.scoreline_probability_calculator import (
    outcome_probabilities,
)

# Adjust only if this module lives elsewhere.
from simulation.live_match_observation_builder import (
    LiveMatchObservation,
    LiveMatchObservationBuilder,
)


@dataclass(frozen=True)
class ProductionFixturePrediction:
    """
    Complete deterministic production prediction for one fixture.

    This object combines:

    - resolved fixture identity;
    - football-intelligence features;
    - prediction-date ClubElo information;
    - expected-goal predictions;
    - normalized match-outcome probabilities;
    - production artifact provenance.
    """

    requested_home_team: str
    requested_away_team: str

    home_team: str
    away_team: str
    prediction_date: date

    home_attack: float
    away_attack: float

    home_defense: float
    away_defense: float

    home_attack_depth: float
    away_attack_depth: float
    attack_depth_diff: float

    home_rating_prior: float
    away_rating_prior: float
    rating_prior_diff: float

    lambda_home: float
    lambda_away: float

    pred_total_goals: float
    pred_goal_diff: float

    home_win_probability: float
    draw_probability: float
    away_win_probability: float

    home_rating_effective_from: date
    home_rating_effective_to: date

    away_rating_effective_from: date
    away_rating_effective_to: date

    rating_prior_source: str

    repository_version: str | None
    repository_scope: str | None

    goal_model_artifact_name: str
    goal_model_artifact_version: str
    goal_model_baseline_name: str
    goal_model_baseline_version: str
    goal_model_feature_specification: str
    goal_model_training_end_date: str

    def to_record(self) -> dict[str, object]:
        """
        Return a flat persistence-ready dictionary.
        """

        record = asdict(self)

        for field_name in (
            "prediction_date",
            "home_rating_effective_from",
            "home_rating_effective_to",
            "away_rating_effective_from",
            "away_rating_effective_to",
        ):
            value = record[field_name]

            if isinstance(value, date):
                record[field_name] = value.isoformat()

        return record


class ProductionPredictionPipeline:
    """
    Public runtime interface for club-match prediction.

    Responsibilities
    ----------------
    - Construct one live match observation.
    - Validate the observation-model feature contract.
    - Predict home and away expected goals.
    - Convert expected goals into normalized outcome
      probabilities.
    - Return a complete immutable prediction object.

    Non-responsibilities
    --------------------
    - Build football-intelligence repositories.
    - Download ClubElo histories.
    - Fit goal models.
    - Sample stochastic scorelines.
    - Evaluate prediction accuracy.
    """

    def __init__(
        self,
        *,
        observation_builder: LiveMatchObservationBuilder,
        goal_model: ProductionGoalModel,
    ) -> None:
        self.observation_builder = observation_builder
        self.goal_model = goal_model

        self._validate_runtime_contract()

    def _validate_runtime_contract(self) -> None:
        model_features = set(
            self.goal_model.required_features
        )

        expected_features = {
            "home_attack",
            "away_attack",
            "home_defense",
            "away_defense",
            "attack_depth_diff",
            "rating_prior_diff",
        }

        if model_features != expected_features:
            raise RuntimeError(
                "Production goal-model feature contract does "
                "not match the live observation contract. "
                f"Expected: {sorted(expected_features)}. "
                f"Model requires: {sorted(model_features)}."
            )

    def predict_fixture(
        self,
        *,
        home_team: str,
        away_team: str,
        prediction_date: str | date | datetime,
    ) -> ProductionFixturePrediction:
        """
        Generate one deterministic production prediction.
        """

        observation = self.observation_builder.build(
            home_team=home_team,
            away_team=away_team,
            prediction_date=prediction_date,
        )

        feature_values = (
            observation.to_feature_mapping()
        )

        goal_prediction = self.goal_model.predict(
            feature_values
        )

        probabilities = outcome_probabilities(
            lambda_home=goal_prediction.lambda_home,
            lambda_away=goal_prediction.lambda_away,
        )

        prediction = ProductionFixturePrediction(
            requested_home_team=(
                observation.requested_home_team
            ),
            requested_away_team=(
                observation.requested_away_team
            ),
            home_team=observation.home_team,
            away_team=observation.away_team,
            prediction_date=observation.prediction_date,

            home_attack=observation.home_attack,
            away_attack=observation.away_attack,

            home_defense=observation.home_defense,
            away_defense=observation.away_defense,

            home_attack_depth=(
                observation.home_attack_depth
            ),
            away_attack_depth=(
                observation.away_attack_depth
            ),
            attack_depth_diff=(
                observation.attack_depth_diff
            ),

            home_rating_prior=(
                observation.home_rating_prior
            ),
            away_rating_prior=(
                observation.away_rating_prior
            ),
            rating_prior_diff=(
                observation.rating_prior_diff
            ),

            lambda_home=goal_prediction.lambda_home,
            lambda_away=goal_prediction.lambda_away,

            pred_total_goals=(
                goal_prediction.pred_total_goals
            ),
            pred_goal_diff=(
                goal_prediction.pred_goal_diff
            ),

            home_win_probability=(
                probabilities.home_win
            ),
            draw_probability=(
                probabilities.draw
            ),
            away_win_probability=(
                probabilities.away_win
            ),

            home_rating_effective_from=(
                observation.home_rating_effective_from
            ),
            home_rating_effective_to=(
                observation.home_rating_effective_to
            ),
            away_rating_effective_from=(
                observation.away_rating_effective_from
            ),
            away_rating_effective_to=(
                observation.away_rating_effective_to
            ),

            rating_prior_source=(
                observation.rating_prior_source
            ),

            repository_version=(
                observation.repository_version
            ),
            repository_scope=(
                observation.repository_scope
            ),

            goal_model_artifact_name=(
                self.goal_model.artifact_name
            ),
            goal_model_artifact_version=(
                self.goal_model.artifact_version
            ),
            goal_model_baseline_name=(
                self.goal_model.baseline_name
            ),
            goal_model_baseline_version=(
                self.goal_model.baseline_version
            ),
            goal_model_feature_specification=(
                self.goal_model.feature_specification
            ),
            goal_model_training_end_date=(
                self.goal_model.training_end_date
            ),
        )

        self._validate_prediction(
            prediction
        )

        return prediction

    def predict_fixtures(
        self,
        fixtures: pd.DataFrame,
        *,
        home_team_column: str = "home_team",
        away_team_column: str = "away_team",
        prediction_date_column: str = "date",
        continue_on_error: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict an entire fixture DataFrame.

        Returns
        -------
        predictions:
            Successful prediction records.

        failures:
            Failed fixture records with exception information.
        """

        required_columns = {
            home_team_column,
            away_team_column,
            prediction_date_column,
        }

        missing = (
            required_columns
            - set(fixtures.columns)
        )

        if missing:
            raise ValueError(
                "Fixture DataFrame is missing required "
                f"columns: {sorted(missing)}"
            )

        prediction_records: list[
            dict[str, object]
        ] = []

        failure_records: list[
            dict[str, object]
        ] = []

        for row_index, row in fixtures.iterrows():
            home_team = str(
                row[home_team_column]
            ).strip()

            away_team = str(
                row[away_team_column]
            ).strip()

            prediction_date = (
                row[prediction_date_column]
            )

            try:
                prediction = self.predict_fixture(
                    home_team=home_team,
                    away_team=away_team,
                    prediction_date=prediction_date,
                )

                record = {
                    column: row[column]
                    for column in fixtures.columns
                }

                record.update(
                    prediction.to_record()
                )

                record["prediction_status"] = "PASS"
                record["runtime_error_type"] = None
                record["runtime_error"] = None

                prediction_records.append(
                    record
                )

            except Exception as error:
                failure_record = {
                    column: row[column]
                    for column in fixtures.columns
                }

                failure_record.update(
                    {
                        "fixture_row_index": int(
                            row_index
                        ),
                        "requested_home_team": (
                            home_team
                        ),
                        "requested_away_team": (
                            away_team
                        ),
                        "requested_prediction_date": (
                            str(prediction_date)
                        ),
                        "prediction_status": "FAILED",
                        "runtime_error_type": (
                            type(error).__name__
                        ),
                        "runtime_error": str(error),
                    }
                )

                failure_records.append(
                    failure_record
                )

                if not continue_on_error:
                    raise

        predictions = pd.DataFrame(
            prediction_records
        )

        failures = pd.DataFrame(
            failure_records
        )

        return predictions, failures

    @staticmethod
    def _validate_prediction(
        prediction: ProductionFixturePrediction,
    ) -> None:
        numeric_values = {
            "home_attack":
                prediction.home_attack,
            "away_attack":
                prediction.away_attack,
            "home_defense":
                prediction.home_defense,
            "away_defense":
                prediction.away_defense,
            "home_attack_depth":
                prediction.home_attack_depth,
            "away_attack_depth":
                prediction.away_attack_depth,
            "attack_depth_diff":
                prediction.attack_depth_diff,
            "home_rating_prior":
                prediction.home_rating_prior,
            "away_rating_prior":
                prediction.away_rating_prior,
            "rating_prior_diff":
                prediction.rating_prior_diff,
            "lambda_home":
                prediction.lambda_home,
            "lambda_away":
                prediction.lambda_away,
            "pred_total_goals":
                prediction.pred_total_goals,
            "pred_goal_diff":
                prediction.pred_goal_diff,
            "home_win_probability":
                prediction.home_win_probability,
            "draw_probability":
                prediction.draw_probability,
            "away_win_probability":
                prediction.away_win_probability,
        }

        for field_name, value in (
            numeric_values.items()
        ):
            if not math.isfinite(value):
                raise AssertionError(
                    "Production prediction contains a "
                    "non-finite value. "
                    f"Field={field_name!r}, "
                    f"value={value!r}."
                )

        if prediction.lambda_home <= 0.0:
            raise AssertionError(
                "lambda_home must be positive."
            )

        if prediction.lambda_away <= 0.0:
            raise AssertionError(
                "lambda_away must be positive."
            )

        expected_total = (
            prediction.lambda_home
            + prediction.lambda_away
        )

        if not math.isclose(
            prediction.pred_total_goals,
            expected_total,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                "Predicted total goals are inconsistent."
            )

        expected_goal_diff = (
            prediction.lambda_home
            - prediction.lambda_away
        )

        if not math.isclose(
            prediction.pred_goal_diff,
            expected_goal_diff,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                "Predicted goal difference is inconsistent."
            )

        probabilities = (
            prediction.home_win_probability,
            prediction.draw_probability,
            prediction.away_win_probability,
        )

        if any(
            value < 0.0 or value > 1.0
            for value in probabilities
        ):
            raise AssertionError(
                "Outcome probabilities must be between "
                "zero and one."
            )

        if not math.isclose(
            sum(probabilities),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                "Outcome probabilities do not sum to one."
            )

        if not (
            prediction.home_rating_effective_from
            <= prediction.prediction_date
            <= prediction.home_rating_effective_to
        ):
            raise AssertionError(
                "Home ClubElo interval is not valid on the "
                "prediction date."
            )

        if not (
            prediction.away_rating_effective_from
            <= prediction.prediction_date
            <= prediction.away_rating_effective_to
        ):
            raise AssertionError(
                "Away ClubElo interval is not valid on the "
                "prediction date."
            )
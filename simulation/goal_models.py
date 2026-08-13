#goal_models.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor


@dataclass
class GoalPrediction:
    pred_home_goals: np.ndarray
    pred_away_goals: np.ndarray

    @property
    def pred_total_goals(self) -> np.ndarray:
        return self.pred_home_goals + self.pred_away_goals

    @property
    def pred_goal_diff(self) -> np.ndarray:
        return self.pred_home_goals - self.pred_away_goals
    
    @property
    def lambda_home(self) -> np.ndarray:
        return self.pred_home_goals

    @property
    def lambda_away(self) -> np.ndarray:
        return self.pred_away_goals


class GoalModel(ABC):
    name: str

    @abstractmethod
    def fit(self, training_df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, evaluation_df: pd.DataFrame) -> GoalPrediction:
        pass


class PoissonGoalModel(GoalModel):
    def __init__(
        self,
        name: str,
        home_features: list[str],
        away_features: list[str],
        alpha: float = 0.0,
    ):
        self.name = name
        self.home_features = home_features
        self.away_features = away_features
        self.alpha = alpha

        self.home_model = PoissonRegressor(
            alpha=alpha,
            max_iter=1000,
        )
        self.away_model = PoissonRegressor(
            alpha=alpha,
            max_iter=1000,
        )

    def fit(self, training_df: pd.DataFrame) -> None:
        home_df = training_df[
            self.home_features + ["home_score"]
        ].dropna()

        away_df = training_df[
            self.away_features + ["away_score"]
        ].dropna()

        self.home_model.fit(
            home_df[self.home_features],
            home_df["home_score"],
        )

        self.away_model.fit(
            away_df[self.away_features],
            away_df["away_score"],
        )

    def predict(self, evaluation_df: pd.DataFrame) -> GoalPrediction:
        return GoalPrediction(
            pred_home_goals=self.home_model.predict(
                evaluation_df[self.home_features],
            ),
            pred_away_goals=self.away_model.predict(
                evaluation_df[self.away_features],
            ),
        )
# match_predictor.py

import joblib
import pandas as pd

from inference.model_paths import PRODUCTION_MODEL_PATH
from inference.feature_builder import (
    build_engineered_features,
    validate_feature_row,
)
from inference.feature_vector_builder import (
    get_feature_order,
    build_feature_vector,
)


OUTCOME_LABELS = {
    0: "away_win",
    1: "draw",
    2: "home_win",
}


class MatchPredictor:
    def __init__(self, model_path=PRODUCTION_MODEL_PATH):
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.feature_order = get_feature_order(self.model)

    def predict_proba_from_features(self, feature_row):
        """
        Predict outcome probabilities from one engineered feature row.
        """

        if isinstance(feature_row, dict):
            X = pd.DataFrame([feature_row])
        else:
            X = feature_row

        probabilities = self.model.predict_proba(X)[0]

        return {
            OUTCOME_LABELS[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
        }
    
    def predict_proba_from_vector(self, feature_vector):
        probabilities = self.model.predict_proba(feature_vector)[0]

        return {
            OUTCOME_LABELS[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
        }
    
    def predict_match(self, home_team, away_team):
        """
        Predict match probabilities from home/away team rating dictionaries.
        """

        feature_row = build_engineered_features(home_team, away_team)
        validate_feature_row(feature_row)

        return self.predict_proba_from_features(feature_row)
    
    def predict_match_fast(self, home_team, away_team):
        feature_vector = build_feature_vector(
            home_team,
            away_team,
            self.feature_order,
        )

        return self.predict_proba_from_vector(feature_vector)
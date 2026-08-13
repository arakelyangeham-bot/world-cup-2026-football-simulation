# probability_engine.py

from inference.match_predictor import MatchPredictor


class ProbabilityEngine:
    """
    Unified probability interface for the tournament simulator.

    Supported modes:
        ml
    """

    def __init__(self, mode="ml"):
        self.mode = mode

        if self.mode == "ml":
            self.predictor = MatchPredictor()
        else:
            raise ValueError(f"Unsupported probability engine mode: {mode}")

    def predict_match(self, home_team, away_team):
        if self.mode == "ml":
            return self.predictor.predict_match(home_team, away_team)

        raise ValueError(f"Unsupported probability engine mode: {self.mode}")
# probability_adapter.py

from inference.probability_engine import ProbabilityEngine


class ProbabilityAdapter:
    """
    Adapter between the tournament simulator and probability engines.

    Current supported mode:
        ml
    """

    def __init__(self, mode="ml"):
        self.mode = mode
        self.engine = ProbabilityEngine(mode=mode)

    def predict_match_probabilities(self, home_team, away_team):
        """
        Return simulator-ready probabilities.

        Output:
            {
                "home_win": float,
                "draw": float,
                "away_win": float,
            }
        """

        probabilities = self.engine.predict_match(home_team, away_team)

        return {
            "home_win": probabilities["home_win"],
            "draw": probabilities["draw"],
            "away_win": probabilities["away_win"],
        }
    
    def predict_match(self, home_team, away_team):
        """
        Return the complete production prediction.

        This method exposes the underlying production prediction object
        without removing any fields. Analysis and benchmarking code should
        use this interface when additional prediction metadata is needed.

        Returns
        -------
        dict
        Prediction dictionary produced by the production
        ProbabilityEngine.
        """
        return self.engine.predict_match(home_team, away_team)
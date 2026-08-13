# match_sampler.py

import random

from simulation.probability_adapter import ProbabilityAdapter


class MatchSampler:
    """
    Convert match outcome probabilities into sampled outcomes.
    """

    def __init__(self, mode="ml"):
        self.adapter = ProbabilityAdapter(mode=mode)

    def sample_outcome(self, home_team, away_team):
        probabilities = self.adapter.predict_match_probabilities(
            home_team,
            away_team,
        )

        outcomes = ["home_win", "draw", "away_win"]
        weights = [probabilities[outcome] for outcome in outcomes]

        sampled = random.choices(outcomes, weights=weights, k=1)[0]

        return {
            "outcome": sampled,
            "probabilities": probabilities,
        }

    def sample_knockout_winner(self, home_team, away_team):
        """
        Knockout matches cannot end in a draw.

        If draw is sampled, resolve winner using normalized home/away
        win probabilities.
        """

        result = self.sample_outcome(home_team, away_team)
        outcome = result["outcome"]
        probabilities = result["probabilities"]

        if outcome == "home_win":
            winner = "home"
        elif outcome == "away_win":
            winner = "away"
        else:
            home_p = probabilities["home_win"]
            away_p = probabilities["away_win"]
            total = home_p + away_p

            winner = random.choices(
                ["home", "away"],
                weights=[home_p / total, away_p / total],
                k=1,
            )[0]

        return {
            "winner": winner,
            "sampled_outcome": outcome,
            "probabilities": probabilities,
        }
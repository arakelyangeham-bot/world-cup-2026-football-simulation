#match_generator_probability_engine.py

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import numpy as np

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import dixon_coles_hierarchical_sampler_fast


@dataclass(frozen=True)
class MatchGeneratorProbabilityResult:
    home_win: float
    draw: float
    away_win: float
    lambda_home: float
    lambda_away: float
    samples: int


class MatchGeneratorProbabilityEngine:
    """
    Research-only probability engine.

    Derives home/draw/away probabilities from the production-style
    scoreline generator instead of from the ML outcome classifier.
    """

    def __init__(
        self,
        lambda_model: str = "calibrated",
        tempo_cv: float = 0.60,
        team_cv: float = 0.10,
        rho: float = 0.30,
        samples: int = 5000,
        seed: int | None = None,
    ) -> None:
        self.lambda_model = lambda_model
        self.tempo_cv = tempo_cv
        self.team_cv = team_cv
        self.rho = rho
        self.samples = samples
        self.seed = seed

    def predict_match(self, home_team: dict, away_team: dict) -> dict[str, float]:
        result = self.predict_match_with_metadata(home_team, away_team)

        return {
            "home_win": result.home_win,
            "draw": result.draw,
            "away_win": result.away_win,
        }

    def predict_match_with_metadata(
        self,
        home_team: dict,
        away_team: dict,
    ) -> MatchGeneratorProbabilityResult:
        if self.seed is not None:
            np.random.seed(self.seed)

        lambda_home, lambda_away = expected_goals(
            home_team,
            away_team,
            lambda_model=self.lambda_model,
        )

        outcomes = Counter()

        for _ in range(self.samples):
            home_goals, away_goals = dixon_coles_hierarchical_sampler_fast(
                lambda_home,
                lambda_away,
                tempo_cv=self.tempo_cv,
                team_cv=self.team_cv,
                rho=self.rho,
            )

            if home_goals > away_goals:
                outcomes["home_win"] += 1
            elif home_goals < away_goals:
                outcomes["away_win"] += 1
            else:
                outcomes["draw"] += 1

        total = sum(outcomes.values())

        return MatchGeneratorProbabilityResult(
            home_win=outcomes["home_win"] / total,
            draw=outcomes["draw"] / total,
            away_win=outcomes["away_win"] / total,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            samples=total,
        )
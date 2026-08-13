#scoreline_probability_calculator

from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_MAX_GOALS = 15


@dataclass(frozen=True)
class MatchOutcomeProbabilities:
    """
    Normalized home-win, draw, and away-win probabilities
    implied by independent Poisson goal distributions.
    """

    home_win: float
    draw: float
    away_win: float

    @property
    def total(self) -> float:
        return (
            self.home_win
            + self.draw
            + self.away_win
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "home_win_probability": self.home_win,
            "draw_probability": self.draw,
            "away_win_probability": self.away_win,
        }


def poisson_probability(
    goals: int,
    expected_goals: float,
) -> float:
    """
    Return P(X = goals) for a Poisson random variable.
    """

    if goals < 0:
        raise ValueError(
            "goals must not be negative."
        )

    expected_goals = float(
        expected_goals
    )

    if (
        not math.isfinite(expected_goals)
        or expected_goals <= 0.0
    ):
        raise ValueError(
            "expected_goals must be positive and finite."
        )

    log_probability = (
        -expected_goals
        + goals * math.log(expected_goals)
        - math.lgamma(goals + 1)
    )

    return math.exp(
        log_probability
    )


def outcome_probabilities(
    *,
    lambda_home: float,
    lambda_away: float,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> MatchOutcomeProbabilities:
    """
    Calculate normalized match-outcome probabilities from
    independent Poisson home and away goal distributions.

    The finite score grid is normalized so that the returned
    probabilities sum to one.
    """

    lambda_home = float(
        lambda_home
    )

    lambda_away = float(
        lambda_away
    )

    if (
        not math.isfinite(lambda_home)
        or lambda_home <= 0.0
    ):
        raise ValueError(
            "lambda_home must be positive and finite."
        )

    if (
        not math.isfinite(lambda_away)
        or lambda_away <= 0.0
    ):
        raise ValueError(
            "lambda_away must be positive and finite."
        )

    if max_goals < 1:
        raise ValueError(
            "max_goals must be at least 1."
        )

    home_goal_probabilities = [
        poisson_probability(
            goals,
            lambda_home,
        )
        for goals in range(
            max_goals + 1
        )
    ]

    away_goal_probabilities = [
        poisson_probability(
            goals,
            lambda_away,
        )
        for goals in range(
            max_goals + 1
        )
    ]

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for home_goals, home_probability in enumerate(
        home_goal_probabilities
    ):
        for away_goals, away_probability in enumerate(
            away_goal_probabilities
        ):
            probability = (
                home_probability
                * away_probability
            )

            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability

    total = (
        home_win
        + draw
        + away_win
    )

    if (
        not math.isfinite(total)
        or total <= 0.0
    ):
        raise ValueError(
            "Outcome probability grid produced zero or "
            "non-finite probability mass."
        )

    result = MatchOutcomeProbabilities(
        home_win=home_win / total,
        draw=draw / total,
        away_win=away_win / total,
    )

    if not math.isclose(
        result.total,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            "Normalized outcome probabilities do not sum "
            "to one."
        )

    return result
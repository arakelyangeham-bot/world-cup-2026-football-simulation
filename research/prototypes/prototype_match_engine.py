from __future__ import annotations

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import dixon_coles_hierarchical_sampler_fast

from research.prototypes.football_realization import (
    football_realization_adjustment,
)


def simulate_match_score_research(
    team1_data: dict,
    team2_data: dict,
    lambda_model: str = "calibrated",
    tempo_cv: float = 0.60,
    team_cv: float = 0.10,
    rho: float = 0.30,
    context: dict | None = None,
) -> tuple[int, int]:
    """
    Research prototype match engine.

    Version 1 routes expected goals through the football-realization hook.
    The hook currently returns lambdas unchanged, so this should behave like
    the scoreline-first Dixon-Coles research engine.
    """

    lambda_home, lambda_away = expected_goals(
        team1_data,
        team2_data,
        lambda_model=lambda_model,
    )

    lambda_home, lambda_away = football_realization_adjustment(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        context=context,
    )

    goals_home, goals_away = dixon_coles_hierarchical_sampler_fast(
        lambda_home,
        lambda_away,
        tempo_cv=tempo_cv,
        team_cv=team_cv,
        rho=rho,
    )

    return int(goals_home), int(goals_away)
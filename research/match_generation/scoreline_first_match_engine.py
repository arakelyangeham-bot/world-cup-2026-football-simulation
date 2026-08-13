#scoreline_first_match_engine.py

from __future__ import annotations

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import dixon_coles_hierarchical_sampler_fast


def simulate_scoreline_first_match(
    team1_data: dict,
    team2_data: dict,
    lambda_model: str = "calibrated",
    tempo_cv: float = 0.60,
    team_cv: float = 0.10,
    rho: float = 0.30,
) -> tuple[int, int]:
    """
    Research-only scoreline-first match engine.

    Generates the scoreline directly from the expected-goals model and
    production Dixon-Coles hierarchical sampler.

    No W/D/L outcome is sampled first.
    """

    lambda1, lambda2 = expected_goals(
        team1_data,
        team2_data,
        lambda_model=lambda_model,
    )

    goals1, goals2 = dixon_coles_hierarchical_sampler_fast(
        lambda1,
        lambda2,
        tempo_cv=tempo_cv,
        team_cv=team_cv,
        rho=rho,
    )

    return int(goals1), int(goals2)
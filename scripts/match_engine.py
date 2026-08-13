# match_engine.py

import numpy as np

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import (
    hierarchical_stochastic_lambda_sampler,
    dixon_coles_hierarchical_sampler_fast,
)
from simulation.simulation_config import (
    LAMBDA_MODEL,
    GOAL_SAMPLER,
    GOAL_SAMPLER_CONFIG,
)

def rating_gap_to_probs(team_a_rating: float, team_b_rating: float) -> dict[str, float]:
    """
    Temporary bridge model:
    converts team strength gap into W/D/L probabilities.
    Later, the real match predictor can replace this function.
    """
    gap = team_a_rating - team_b_rating

    draw_prob = 0.26
    win_a = 1 / (1 + 10 ** (-gap / 400))

    non_draw = 1 - draw_prob
    p_a = win_a * non_draw
    p_b = (1 - win_a) * non_draw

    return {
        "team_a_win": p_a,
        "draw": draw_prob,
        "team_b_win": p_b,
    }

def poisson_expected_goals(
    team_a: dict[str, float],
    team_b: dict[str, float],
) -> tuple[float, float]:
    return expected_goals(
        team_a,
        team_b,
        lambda_model=LAMBDA_MODEL,
    )

def simulate_poisson_score(
    team_a_strength: dict[str, float],
    team_b_strength: dict[str, float],
) -> tuple[int, int]:
    """
    Production score-generation entry point.

    Expected goals are produced by the configured lambda model.
    Goals are generated using the configured production sampler.

    The function name is retained for backward compatibility with
    existing tournament simulation code.
    """
    lambda_a, lambda_b = poisson_expected_goals(
        team_a_strength,
        team_b_strength,
    )

    if GOAL_SAMPLER == "poisson":
        goals_a = np.random.poisson(lambda_a)
        goals_b = np.random.poisson(lambda_b)
        return int(goals_a), int(goals_b)

    if GOAL_SAMPLER == "hierarchical_stochastic_lambda":
        goals_a, goals_b = hierarchical_stochastic_lambda_sampler(
            lambda_a,
            lambda_b,
            tempo_cv=GOAL_SAMPLER_CONFIG["tempo_cv"],
            team_cv=GOAL_SAMPLER_CONFIG["team_cv"],
        )

        return int(goals_a), int(goals_b)
    
    if GOAL_SAMPLER == "dixon_coles_hierarchical":
        goals_a, goals_b = dixon_coles_hierarchical_sampler_fast(
            lambda_a,
            lambda_b,
            tempo_cv=GOAL_SAMPLER_CONFIG["tempo_cv"],
            team_cv=GOAL_SAMPLER_CONFIG["team_cv"],
            rho=GOAL_SAMPLER_CONFIG["rho"],
        )

        return int(goals_a), int(goals_b)

    raise ValueError(f"Unknown goal sampler: {GOAL_SAMPLER}")

def simulate_many(team_a, team_b, n=10000):
    a_wins = 0
    draws = 0
    b_wins = 0

    goals_a = 0
    goals_b = 0

    for _ in range(n):
        ga, gb = simulate_poisson_score(team_a, team_b)

        goals_a += ga
        goals_b += gb

        if ga > gb:
            a_wins += 1
        elif gb > ga:
            b_wins += 1
        else:
            draws += 1

    print(f"Matches: {n}")
    print(f"A wins : {a_wins/n:.3f}")
    print(f"Draws  : {draws/n:.3f}")
    print(f"B wins : {b_wins/n:.3f}")
    print(f"Avg goals A: {goals_a/n:.3f}")
    print(f"Avg goals B: {goals_b/n:.3f}")

'''
if __name__ == "__main__":
    strong = {
        "attack": 0.8,
        "defense": -0.2,
    }

    weak = {
        "attack": 0.0,
        "defense": 1.0,
    }

    simulate_many(strong, weak)
'''
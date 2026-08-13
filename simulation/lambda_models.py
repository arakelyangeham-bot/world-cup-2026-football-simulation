#lambda_models.py

import math

from simulation.poisson_calibration import load_poisson_goal_coefficients


def heuristic_expected_goals(
    team_a: dict[str, float],
    team_b: dict[str, float],
) -> tuple[float, float]:
    """
    Original hand-tuned lambda model.

    Expects:
        team["attack"]
        team["defense"]
    """

    MEAN_ATTACK = 0.295
    MEAN_DEFENSE = 0.636

    base_goals = 1.35

    attack_a = team_a["attack"] - MEAN_ATTACK
    attack_b = team_b["attack"] - MEAN_ATTACK

    defense_a = MEAN_DEFENSE - team_a["defense"]
    defense_b = MEAN_DEFENSE - team_b["defense"]

    lambda_a = (
        base_goals
        + 1.2 * attack_a
        - 0.8 * defense_b
    )

    lambda_b = (
        base_goals
        + 1.2 * attack_b
        - 0.8 * defense_a
    )

    return max(0.2, lambda_a), max(0.2, lambda_b)


def calibrated_expected_goals(
    team_a: dict[str, float],
    team_b: dict[str, float],
) -> tuple[float, float]:
    """
    Statistically fitted Poisson GLM lambda model.

    Expects:
        team["poisson_attack"]
        team["poisson_defense"]
        team["rating_prior"]

    The fitted regression coefficients were originally estimated using
    FIFA-points differences for national teams. The public model interface
    now treats this input as a generic external team-strength prior.

    Uses log-link Poisson regression:
        lambda = exp(intercept + beta*x + ...)
    """

    coefficients = load_poisson_goal_coefficients()

    home = coefficients["home_goal_model"]
    away = coefficients["away_goal_model"]

    rating_prior_diff = (
        team_a["rating_prior"]
        - team_b["rating_prior"]
    )

    home_linear = (
        home["intercept"]
        + home["home_poisson_attack"] * team_a["poisson_attack"]
        + home["away_poisson_defense"] * team_b["poisson_defense"]
        + home["fifa_points_diff"] * rating_prior_diff
    )

    away_linear = (
        away["intercept"]
        + away["away_poisson_attack"] * team_b["poisson_attack"]
        + away["home_poisson_defense"] * team_a["poisson_defense"]
        + away["fifa_points_diff"] * rating_prior_diff
    )

    lambda_a = math.exp(home_linear)
    lambda_b = math.exp(away_linear)

    return max(0.2, lambda_a), max(0.2, lambda_b)

def expected_goals(
    team_a: dict[str, float],
    team_b: dict[str, float],
    lambda_model: str = "heuristic",
) -> tuple[float, float]:
    if lambda_model == "heuristic":
        return heuristic_expected_goals(team_a, team_b)

    if lambda_model == "calibrated":
        return calibrated_expected_goals(team_a, team_b)

    raise ValueError(f"Unsupported lambda model: {lambda_model}")
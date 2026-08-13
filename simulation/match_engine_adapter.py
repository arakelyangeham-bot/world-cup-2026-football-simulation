from __future__ import annotations

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import dixon_coles_hierarchical_sampler_fast
from simulation.simulation_config import (
    GOAL_SAMPLER,
    GOAL_SAMPLER_CONFIG,
    LAMBDA_MODEL,
    LAMBDA_SCALE,
    MATCH_ENGINE_MODE,
)
from simulation.match_sampler import MatchSampler
import random


_PERSISTENT_MATCH_SAMPLER = MatchSampler(mode="ml")
_MATCH_PROBABILITY_CACHE = {}


def repository_entry_to_poisson_features(team_entry):
    rating_prior = team_entry.get(
        "rating_prior",
        team_entry.get("fifa_points"),
    )

    if rating_prior is None:
        raise ValueError(
            "Repository entry is missing the required external "
            "team-strength prior: 'rating_prior'."
        )

    return {
        "attack": team_entry["poisson_attack"],
        "defense": team_entry["poisson_defense"],
        "poisson_attack": team_entry["poisson_attack"],
        "poisson_defense": team_entry["poisson_defense"],

        # Generic interface
        "rating_prior": float(rating_prior),

        # Temporary compatibility for the current lambda model
        "fifa_points": float(rating_prior),
    }



def simulate_match_score(
    team1_data,
    team2_data,
    mode=None,
):
    """
    Return a simulated scoreline using the selected match engine.

    Supported modes:
        scoreline_first - expected goals -> configured goal sampler -> scoreline
        poisson         - legacy alias for scoreline_first
        ml              - legacy ML-guided outcome-first engine
    """

    selected_mode = mode or MATCH_ENGINE_MODE

    if selected_mode in {"scoreline_first", "poisson"}:
        return simulate_scoreline_first_score(team1_data, team2_data)

    if selected_mode == "ml":
        return simulate_ml_guided_score(team1_data, team2_data)

    raise ValueError(f"Unsupported match engine mode: {selected_mode}")


def simulate_scoreline_first_score(team1_data, team2_data):
    lambda1, lambda2 = expected_goals(
        repository_entry_to_poisson_features(team1_data),
        repository_entry_to_poisson_features(team2_data),
        lambda_model=LAMBDA_MODEL,
    )

    return simulate_scoreline_from_lambdas(
        lambda_home=lambda1,
        lambda_away=lambda2,
    )

def simulate_scoreline_from_lambdas(
    lambda_home: float,
    lambda_away: float,
) -> tuple[int, int]:
    """
    Sample a scoreline from externally supplied expected
    goals using the configured production goal sampler.

    This preserves the existing scoreline-sampling layer while
    allowing a production predictor to replace only the source
    of expected goals.
    """

    lambda_home = float(lambda_home)
    lambda_away = float(lambda_away)

    if lambda_home <= 0.0:
        raise ValueError(
            "lambda_home must be positive."
        )

    if lambda_away <= 0.0:
        raise ValueError(
            "lambda_away must be positive."
        )

    scaled_home = (
        lambda_home
        * LAMBDA_SCALE
    )

    scaled_away = (
        lambda_away
        * LAMBDA_SCALE
    )

    if GOAL_SAMPLER != "dixon_coles_hierarchical":
        raise ValueError(
            "Unsupported GOAL_SAMPLER for production "
            f"lambda route: {GOAL_SAMPLER}"
        )

    goals_home, goals_away = (
        dixon_coles_hierarchical_sampler_fast(
            scaled_home,
            scaled_away,
            **GOAL_SAMPLER_CONFIG,
        )
    )

    return (
        int(goals_home),
        int(goals_away),
    )

def simulate_ml_guided_score(team1_data, team2_data):
    probabilities = get_cached_probabilities(team1_data, team2_data)

    outcomes = ["home_win", "draw", "away_win"]
    weights = [probabilities[outcome] for outcome in outcomes]

    desired_outcome = random.choices(outcomes, weights=weights, k=1)[0]

    team1_poisson = repository_entry_to_poisson_features(team1_data)
    team2_poisson = repository_entry_to_poisson_features(team2_data)

    # Legacy fallback path kept only for explicit mode="ml".
    from scripts.match_engine import simulate_poisson_score

    for _ in range(100):
        goals1, goals2 = simulate_poisson_score(
            team1_poisson,
            team2_poisson,
        )

        if desired_outcome == "home_win" and goals1 > goals2:
            return goals1, goals2

        if desired_outcome == "draw" and goals1 == goals2:
            return goals1, goals2

        if desired_outcome == "away_win" and goals2 > goals1:
            return goals1, goals2

    if desired_outcome == "home_win":
        return 1, 0

    if desired_outcome == "draw":
        return 1, 1

    return 0, 1


def team_cache_key(team_data):
    return (
        team_data.get("team")
        or team_data.get("name")
        or tuple(sorted(team_data.items()))
    )


def get_cached_probabilities(team1_data, team2_data):
    key = (
        team_cache_key(team1_data),
        team_cache_key(team2_data),
    )

    if key not in _MATCH_PROBABILITY_CACHE:
        _MATCH_PROBABILITY_CACHE[key] = (
            _PERSISTENT_MATCH_SAMPLER.adapter.predict_match_probabilities(
                team1_data,
                team2_data,
            )
        )

    return _MATCH_PROBABILITY_CACHE[key]
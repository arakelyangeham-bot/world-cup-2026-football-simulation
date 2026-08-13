#goal_samplers.py

import math
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class MatchState:
    probability: float
    home_multiplier: float
    away_multiplier: float


def configurable_mixture_goal_sampler(
    lambda_home: float,
    lambda_away: float,
    states: list[MatchState],
) -> tuple[int, int]:
    total_probability = sum(state.probability for state in states)

    if not np.isclose(total_probability, 1.0):
        raise ValueError(
            f"State probabilities must sum to 1.0, got {total_probability}"
        )

    avg_home_multiplier = sum(
        state.probability * state.home_multiplier
        for state in states
    )
    avg_away_multiplier = sum(
        state.probability * state.away_multiplier
        for state in states
    )

    draw = np.random.random()
    cumulative = 0.0

    for state in states:
        cumulative += state.probability

        if draw <= cumulative:
            adjusted_home = (
                lambda_home
                * state.home_multiplier
                / avg_home_multiplier
            )
            adjusted_away = (
                lambda_away
                * state.away_multiplier
                / avg_away_multiplier
            )

            return (
                int(np.random.poisson(adjusted_home)),
                int(np.random.poisson(adjusted_away)),
            )

    raise RuntimeError("Failed to sample mixture state")

def poisson_sampler(lambda_value: float) -> int:
    return int(np.random.poisson(lambda_value))


def negative_binomial_sampler(
    lambda_value: float,
    k: float,
) -> int:
    """
    Negative Binomial parameterization using:

        mean = lambda
        variance = lambda + lambda² / k

    Larger k -> approaches Poisson.

    Smaller k -> greater overdispersion.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    p = k / (k + lambda_value)

    return int(
        np.random.negative_binomial(
            k,
            p,
        )
    )

def hybrid_volatility_sampler(
    lambda_value: float,
    volatility_probability: float,
    volatility_multiplier: float,
) -> int:
    """
    Hybrid sampler.

    Most matches use the original Poisson lambda.

    A fraction of samples enter a high-volatility mode where lambda is
    multiplied upward. This preserves the calibrated mean approximately
    while allowing more high-scoring outcomes.

    This is experimental.
    """

    if volatility_probability < 0 or volatility_probability > 1:
        raise ValueError("volatility_probability must be between 0 and 1")

    if volatility_multiplier <= 0:
        raise ValueError("volatility_multiplier must be positive")

    adjusted_lambda = lambda_value

    if np.random.random() < volatility_probability:
        adjusted_lambda = lambda_value * volatility_multiplier

    return int(np.random.poisson(adjusted_lambda))

def shared_tempo_poisson_sampler(
    lambda_home: float,
    lambda_away: float,
    tempo_variance: float,
) -> tuple[int, int]:
    """
    Shared-tempo Poisson sampler.

    A match-level tempo multiplier is sampled once, then applied to both teams.

        tempo mean = 1
        tempo variance = tempo_variance

    This preserves expected goals on average while increasing total-goal variance.
    """

    if tempo_variance < 0:
        raise ValueError("tempo_variance must be non-negative")

    if tempo_variance == 0:
        tempo = 1.0
    else:
        shape = 1.0 / tempo_variance
        scale = tempo_variance
        tempo = np.random.gamma(shape, scale)

    adjusted_home = lambda_home * tempo
    adjusted_away = lambda_away * tempo

    return (
        int(np.random.poisson(adjusted_home)),
        int(np.random.poisson(adjusted_away)),
    )

def mixture_goal_sampler(
    lambda_home: float,
    lambda_away: float,
) -> tuple[int, int]:
    states = [
        (0.70, 1.00, 1.00),
        (0.10, 0.70, 0.70),
        (0.10, 1.25, 1.25),
        (0.05, 1.80, 0.90),
        (0.05, 0.90, 1.80),
    ]

    avg_home_multiplier = sum(p * h for p, h, _ in states)
    avg_away_multiplier = sum(p * a for p, _, a in states)

    draw = np.random.random()
    cumulative = 0.0

    for probability, home_multiplier, away_multiplier in states:
        cumulative += probability

        if draw <= cumulative:
            adjusted_home = (
                lambda_home
                * home_multiplier
                / avg_home_multiplier
            )
            adjusted_away = (
                lambda_away
                * away_multiplier
                / avg_away_multiplier
            )

            return (
                int(np.random.poisson(adjusted_home)),
                int(np.random.poisson(adjusted_away)),
            )

    return (
        int(np.random.poisson(lambda_home)),
        int(np.random.poisson(lambda_away)),
    )

def stochastic_lambda_poisson_sampler(
    lambda_home: float,
    lambda_away: float,
    lambda_cv: float,
) -> tuple[int, int]:
    """
    Poisson-lognormal sampler.

    Instead of treating lambda as fixed, sample match-specific lambdas
    from a lognormal distribution with the same mean.

    lambda_cv = coefficient of variation for lambda uncertainty.
    """

    if lambda_cv < 0:
        raise ValueError("lambda_cv must be non-negative")

    if lambda_cv == 0:
        sampled_home_lambda = lambda_home
        sampled_away_lambda = lambda_away
    else:
        sigma2 = np.log(1 + lambda_cv ** 2)
        sigma = np.sqrt(sigma2)

        home_mu = np.log(lambda_home) - 0.5 * sigma2
        away_mu = np.log(lambda_away) - 0.5 * sigma2

        sampled_home_lambda = np.random.lognormal(home_mu, sigma)
        sampled_away_lambda = np.random.lognormal(away_mu, sigma)

    return (
        int(np.random.poisson(sampled_home_lambda)),
        int(np.random.poisson(sampled_away_lambda)),
    )

def _lognormal_mean_one(cv: float) -> float:
    if cv < 0:
        raise ValueError("cv must be non-negative")

    if cv == 0:
        return 1.0

    sigma2 = np.log(1 + cv ** 2)
    sigma = np.sqrt(sigma2)
    mu = -0.5 * sigma2

    return float(np.random.lognormal(mu, sigma))


def hierarchical_stochastic_lambda_sampler(
    lambda_home,
    lambda_away,
    tempo_cv,
    team_cv,
):
    adjusted_home, adjusted_away = hierarchical_stochastic_lambdas(
        lambda_home,
        lambda_away,
        tempo_cv,
        team_cv,
    )

    goals_home = poisson_sampler(adjusted_home)
    goals_away = poisson_sampler(adjusted_away)

    return goals_home, goals_away

def draw_calibrated_hierarchical_sampler(
    lambda_home,
    lambda_away,
    tempo_cv,
    team_cv,
    draw_strength=0.0,
    max_draw_goal=4,
):
    goals_home, goals_away = hierarchical_stochastic_lambda_sampler(
        lambda_home,
        lambda_away,
        tempo_cv,
        team_cv,
    )

    if draw_strength <= 0.0:
        return goals_home, goals_away

    if goals_home == goals_away:
        return goals_home, goals_away

    goal_gap = abs(goals_home - goals_away)

    if goal_gap != 1:
        return goals_home, goals_away

    lower_goal = min(goals_home, goals_away)

    if lower_goal > max_draw_goal:
        return goals_home, goals_away

    if np.random.random() >= draw_strength:
        return goals_home, goals_away

    return lower_goal, lower_goal

def draw_tempered_lambda_sampler(
    lambda_home,
    lambda_away,
    tempo_cv,
    team_cv,
    draw_strength=0.0,
):
    adjusted_home, adjusted_away = hierarchical_stochastic_lambdas(
        lambda_home,
        lambda_away,
        tempo_cv,
        team_cv,
    )

    if draw_strength > 0.0:
        average_lambda = 0.5 * (adjusted_home + adjusted_away)

        adjusted_home = (
            (1.0 - draw_strength) * adjusted_home
            + draw_strength * average_lambda
        )

        adjusted_away = (
            (1.0 - draw_strength) * adjusted_away
            + draw_strength * average_lambda
        )

    goals_home = poisson_sampler(adjusted_home)
    goals_away = poisson_sampler(adjusted_away)

    return goals_home, goals_away

def hierarchical_stochastic_lambdas(
    lambda_home,
    lambda_away,
    tempo_cv,
    team_cv,
):
    match_tempo = np.random.lognormal(
        mean=-0.5 * tempo_cv**2,
        sigma=tempo_cv,
    )

    home_team_effect = np.random.lognormal(
        mean=-0.5 * team_cv**2,
        sigma=team_cv,
    )

    away_team_effect = np.random.lognormal(
        mean=-0.5 * team_cv**2,
        sigma=team_cv,
    )

    adjusted_home = lambda_home * match_tempo * home_team_effect
    adjusted_away = lambda_away * match_tempo * away_team_effect

    return adjusted_home, adjusted_away

def bivariate_poisson_sampler(
    lambda_home,
    lambda_away,
    shared_lambda,
):
    shared_lambda = max(0.0, shared_lambda)

    max_shared = min(lambda_home, lambda_away)

    if shared_lambda > max_shared:
        shared_lambda = max_shared

    home_only_lambda = lambda_home - shared_lambda
    away_only_lambda = lambda_away - shared_lambda

    shared_goals = poisson_sampler(shared_lambda)
    home_only_goals = poisson_sampler(home_only_lambda)
    away_only_goals = poisson_sampler(away_only_lambda)

    goals_home = shared_goals + home_only_goals
    goals_away = shared_goals + away_only_goals

    return goals_home, goals_away

def hierarchical_bivariate_poisson_sampler(
    lambda_home,
    lambda_away,
    tempo_cv,
    team_cv,
    shared_fraction,
):
    adjusted_home, adjusted_away = hierarchical_stochastic_lambdas(
        lambda_home,
        lambda_away,
        tempo_cv,
        team_cv,
    )

    shared_fraction = min(max(shared_fraction, 0.0), 1.0)

    shared_lambda = shared_fraction * min(adjusted_home, adjusted_away)

    return bivariate_poisson_sampler(
        adjusted_home,
        adjusted_away,
        shared_lambda,
    )

def zero_zero_deflated_hierarchical_sampler(
    lambda_home,
    lambda_away,
    tempo_cv,
    team_cv,
    zero_zero_resample_prob=0.0,
):
    goals_home, goals_away = hierarchical_stochastic_lambda_sampler(
        lambda_home,
        lambda_away,
        tempo_cv,
        team_cv,
    )

    if goals_home == 0 and goals_away == 0:
        if np.random.random() < zero_zero_resample_prob:
            goals_home, goals_away = hierarchical_stochastic_lambda_sampler(
                lambda_home,
                lambda_away,
                tempo_cv,
                team_cv,
            )

    return goals_home, goals_away

DIXON_COLES_MAX_GOALS = 10
DIXON_COLES_GOAL_VALUES = np.arange(DIXON_COLES_MAX_GOALS + 1)

DIXON_COLES_HOME_GRID, DIXON_COLES_AWAY_GRID = np.meshgrid(
    DIXON_COLES_GOAL_VALUES,
    DIXON_COLES_GOAL_VALUES,
    indexing="ij",
)

DIXON_COLES_SCORELINES = list(
    zip(
        DIXON_COLES_HOME_GRID.ravel(),
        DIXON_COLES_AWAY_GRID.ravel(),
    )
)

DIXON_COLES_FACTORIALS = np.array(
    [math.factorial(k) for k in DIXON_COLES_GOAL_VALUES],
    dtype=float,
)

def poisson_pmf(k, lam):
    return np.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_pmf_vector(lam):
    return (
        np.exp(-lam)
        * (lam ** DIXON_COLES_GOAL_VALUES)
        / DIXON_COLES_FACTORIALS
    )


def dixon_coles_hierarchical_sampler_fast(
    lambda_home,
    lambda_away,
    tempo_cv,
    team_cv,
    rho,
):
    adjusted_home, adjusted_away = hierarchical_stochastic_lambdas(
        lambda_home,
        lambda_away,
        tempo_cv,
        team_cv,
    )

    home_probs = poisson_pmf_vector(adjusted_home)
    away_probs = poisson_pmf_vector(adjusted_away)

    probabilities = np.outer(home_probs, away_probs)

    # Dixon-Coles low-score correction.
    probabilities[0, 0] *= 1.0 - adjusted_home * adjusted_away * rho
    probabilities[0, 1] *= 1.0 + adjusted_home * rho
    probabilities[1, 0] *= 1.0 + adjusted_away * rho
    probabilities[1, 1] *= 1.0 - rho

    probabilities = np.maximum(probabilities, 0.0)

    flat_probs = probabilities.ravel()
    total = flat_probs.sum()

    if total <= 0.0:
        return hierarchical_stochastic_lambda_sampler(
            lambda_home,
            lambda_away,
            tempo_cv,
            team_cv,
        )

    flat_probs = flat_probs / total

    sampled_index = np.random.choice(
        len(DIXON_COLES_SCORELINES),
        p=flat_probs,
    )

    return DIXON_COLES_SCORELINES[sampled_index]

def dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    if home_goals == 0 and away_goals == 0:
        return 1.0 - (lambda_home * lambda_away * rho)

    if home_goals == 0 and away_goals == 1:
        return 1.0 + (lambda_home * rho)

    if home_goals == 1 and away_goals == 0:
        return 1.0 + (lambda_away * rho)

    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho

    return 1.0

def dixon_coles_hierarchical_sampler(
    lambda_home,
    lambda_away,
    tempo_cv,
    team_cv,
    rho,
    max_goals=10,
):
    adjusted_home, adjusted_away = hierarchical_stochastic_lambdas(
        lambda_home,
        lambda_away,
        tempo_cv,
        team_cv,
    )

    scorelines = []
    probabilities = []

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            base_prob = (
                poisson_pmf(home_goals, adjusted_home)
                * poisson_pmf(away_goals, adjusted_away)
            )

            tau = dixon_coles_tau(
                home_goals,
                away_goals,
                adjusted_home,
                adjusted_away,
                rho,
            )

            prob = base_prob * tau

            if prob < 0.0:
                prob = 0.0

            scorelines.append((home_goals, away_goals))
            probabilities.append(prob)

    probabilities = np.asarray(probabilities, dtype=float)
    total = probabilities.sum()

    if total <= 0.0:
        return hierarchical_stochastic_lambda_sampler(
            lambda_home,
            lambda_away,
            tempo_cv,
            team_cv,
        )

    probabilities = probabilities / total

    sampled_index = np.random.choice(
        len(scorelines),
        p=probabilities,
    )

    return scorelines[sampled_index]
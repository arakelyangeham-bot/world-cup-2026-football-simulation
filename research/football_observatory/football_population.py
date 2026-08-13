#football_population.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from research.football_observatory.football_profile import (
    FootballProfile,
    build_football_profile,
    compare_profiles,
    profile_to_dataframe,
)
from research.football_observatory.observatory_schema import MatchObservation


PopulationSelector = Callable[[MatchObservation], bool]


@dataclass(frozen=True)
class FootballPopulation:
    name: str
    description: str
    selector: PopulationSelector

    def contains(self, observation: MatchObservation) -> bool:
        return bool(self.selector(observation))


@dataclass(frozen=True)
class FootballPopulationAnalysis:
    population: FootballPopulation
    baseline_profile: FootballProfile
    population_profile: FootballProfile
    comparison: pd.DataFrame


def select_population(
    observations: list[MatchObservation],
    population: FootballPopulation,
) -> list[MatchObservation]:
    return [
        observation
        for observation in observations
        if population.contains(observation)
    ]


def analyze_population(
    observations: list[MatchObservation],
    population: FootballPopulation,
    baseline_label: str = "all_matches",
) -> FootballPopulationAnalysis:
    baseline_profile = build_football_profile(
        observations=observations,
        label=baseline_label,
    )

    population_observations = select_population(
        observations=observations,
        population=population,
    )

    population_profile = build_football_profile(
        observations=population_observations,
        label=population.name,
    )

    comparison = compare_profiles(
        baseline=baseline_profile,
        subset=population_profile,
    )

    return FootballPopulationAnalysis(
        population=population,
        baseline_profile=baseline_profile,
        population_profile=population_profile,
        comparison=comparison,
    )


def population_profiles_to_dataframe(
    analysis: FootballPopulationAnalysis,
) -> pd.DataFrame:
    return pd.concat(
        [
            profile_to_dataframe(analysis.baseline_profile),
            profile_to_dataframe(analysis.population_profile),
        ],
        ignore_index=True,
    )


CORE_POPULATIONS: list[FootballPopulation] = [
    FootballPopulation(
        name="one_goal_matches",
        description="Matches decided by exactly one goal.",
        selector=lambda obs: obs.outcome.is_one_goal_match,
    ),
    FootballPopulation(
        name="draws",
        description="Matches ending level.",
        selector=lambda obs: obs.outcome.is_draw,
    ),
    FootballPopulation(
        name="clean_sheets",
        description="Matches where at least one team kept a clean sheet.",
        selector=lambda obs: obs.outcome.is_clean_sheet,
    ),
    FootballPopulation(
        name="both_teams_scored",
        description="Matches where both teams scored.",
        selector=lambda obs: obs.outcome.both_teams_scored,
    ),
    FootballPopulation(
        name="high_scoring",
        description="Matches with five or more total goals.",
        selector=lambda obs: obs.outcome.is_high_scoring,
    ),
    FootballPopulation(
        name="blowouts",
        description="Matches decided by three or more goals.",
        selector=lambda obs: obs.outcome.is_blowout,
    ),
    FootballPopulation(
        name="scoreless_equilibrium",
        description="Matches ending 0–0.",
        selector=lambda obs: (
            obs.outcome.home_score == 0
            and obs.outcome.away_score == 0
        ),
    ),

    FootballPopulation(
        name="scoring_equilibrium",
        description="Draws in which both teams scored.",
        selector=lambda obs: (
            obs.outcome.is_draw
            and obs.outcome.both_teams_scored
        ),
    ),
]
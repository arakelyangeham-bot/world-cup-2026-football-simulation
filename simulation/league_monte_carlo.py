#league_monte_carlo

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from competition_catalog import (
    CompetitionBuilder,
    CompetitionDefinition,
)
from fixture_generation import ScheduledFixture
from simulation.competition import CompetitionEngine
from simulation.league_match_simulator import LeagueMatchSimulator


@dataclass(frozen=True)
class LeagueMonteCarloResult:
    club_rows: list[dict[str, Any]]
    simulation_count: int


class LeagueMonteCarloRunner:
    """
    Generic Monte Carlo runner for a league competition.

    The runner is independent of the underlying football model.
    It requires only a football_model exposing:

        simulate_match(
            home_team,
            away_team,
            prediction_date=None,
        ) -> tuple[int, int]
    """

    def __init__(
        self,
        *,
        definition: CompetitionDefinition,
        participants: Iterable[str],
        fixtures: list[ScheduledFixture],
        football_model: Any,
        top_four_count: int = 4,
        top_six_count: int = 6,
        relegation_count: int = 3,
    ) -> None:
        self.definition = definition
        self.participants = list(participants)
        self.fixtures = list(fixtures)
        self.football_model = football_model
        self.top_four_count = top_four_count
        self.top_six_count = top_six_count
        self.relegation_count = relegation_count

        if not self.participants:
            raise ValueError(
                "participants cannot be empty."
            )

        if not self.fixtures:
            raise ValueError(
                "fixtures cannot be empty."
            )

        participant_count = len(
            self.participants
        )

        if not (
            1
            <= self.top_four_count
            <= participant_count
        ):
            raise ValueError(
                "top_four_count must be between "
                "1 and participant count."
            )

        if not (
            1
            <= self.top_six_count
            <= participant_count
        ):
            raise ValueError(
                "top_six_count must be between "
                "1 and participant count."
            )

        if (
            self.top_four_count
            > self.top_six_count
        ):
            raise ValueError(
                "top_four_count cannot exceed "
                "top_six_count."
            )

        if not (
            1
            <= self.relegation_count
            < participant_count
        ):
            raise ValueError(
                "relegation_count must be between "
                "1 and participant count - 1."
            )

    def _simulate_one_season(
        self,
        *,
        seed: int,
    ) -> list[dict[str, Any]]:
        random.seed(seed)
        np.random.seed(seed)

        competition = CompetitionBuilder().build(
            definition=self.definition,
            participants=self.participants,
        )

        league_stage = competition.stages[0]

        league_stage.matches = (
            LeagueMatchSimulator()
            .simulate_fixtures(
                fixtures=self.fixtures,
                football_model=self.football_model,
                stage_name=league_stage.name,
                competition_name=competition.name,
            )
        )

        if len(league_stage.matches) != len(
            self.fixtures
        ):
            raise RuntimeError(
                "Season simulation did not produce "
                "the expected number of matches."
            )

        result = CompetitionEngine().resolve(
            competition
        )

        if not result.stage_results:
            raise RuntimeError(
                "Competition produced no stage results."
            )

        stage_result = result.stage_results[0]

        if stage_result.standings is None:
            raise RuntimeError(
                "League stage produced no standings."
            )

        standings = (
            stage_result
            .standings
            .as_rows()
        )

        if len(standings) != len(
            self.participants
        ):
            raise RuntimeError(
                "League standings contain an unexpected "
                "number of clubs."
            )

        return standings

    def run(
        self,
        *,
        simulation_count: int,
        base_seed: int = 202627,
    ) -> LeagueMonteCarloResult:
        if simulation_count < 1:
            raise ValueError(
                "simulation_count must be at least 1."
            )

        champion_counts = defaultdict(int)
        top_four_counts = defaultdict(int)
        top_six_counts = defaultdict(int)
        relegation_counts = defaultdict(int)

        points_totals = defaultdict(float)
        position_totals = defaultdict(float)
        goals_for_totals = defaultdict(float)
        goals_against_totals = defaultdict(float)

        position_counts = {
            team: defaultdict(int)
            for team in self.participants
        }

        for simulation_index in range(
            simulation_count
        ):
            seed = (
                base_seed
                + simulation_index
            )

            standings = (
                self._simulate_one_season(
                    seed=seed,
                )
            )

            champion_counts[
                standings[0]["team"]
            ] += 1

            for row in standings[
                : self.top_four_count
            ]:
                top_four_counts[
                    row["team"]
                ] += 1

            for row in standings[
                : self.top_six_count
            ]:
                top_six_counts[
                    row["team"]
                ] += 1

            for row in standings[
                -self.relegation_count :
            ]:
                relegation_counts[
                    row["team"]
                ] += 1

            for row in standings:
                team = row["team"]
                rank = int(row["rank"])

                position_counts[
                    team
                ][rank] += 1

                points_totals[
                    team
                ] += float(
                    row["points"]
                )

                position_totals[
                    team
                ] += float(
                    rank
                )

                goals_for_totals[
                    team
                ] += float(
                    row["goals_for"]
                )

                goals_against_totals[
                    team
                ] += float(
                    row["goals_against"]
                )

        club_rows: list[
            dict[str, Any]
        ] = []

        for team in self.participants:
            row: dict[str, Any] = {
                "team": team,
                "simulation_count":
                    simulation_count,
                "champion_probability": (
                    champion_counts[team]
                    / simulation_count
                ),
                "top_four_probability": (
                    top_four_counts[team]
                    / simulation_count
                ),
                "top_six_probability": (
                    top_six_counts[team]
                    / simulation_count
                ),
                "relegation_probability": (
                    relegation_counts[team]
                    / simulation_count
                ),
                "average_points": (
                    points_totals[team]
                    / simulation_count
                ),
                "average_position": (
                    position_totals[team]
                    / simulation_count
                ),
                "average_goals_for": (
                    goals_for_totals[team]
                    / simulation_count
                ),
                "average_goals_against": (
                    goals_against_totals[team]
                    / simulation_count
                ),
            }

            for position in range(
                1,
                len(self.participants) + 1,
            ):
                row[
                    f"position_{position}_probability"
                ] = (
                    position_counts[
                        team
                    ][position]
                    / simulation_count
                )

            club_rows.append(
                row
            )

        club_rows.sort(
            key=lambda row: (
                row["average_position"],
                -row["average_points"],
            )
        )

        return LeagueMonteCarloResult(
            club_rows=club_rows,
            simulation_count=simulation_count,
        )
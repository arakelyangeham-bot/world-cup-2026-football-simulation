#statistics_observer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simulation.observers.base_observer import TournamentObserver


@dataclass
class SimulationStatistics:
    tournaments: int = 0
    total_matches: int = 0
    total_goals: int = 0
    extra_time_matches: int = 0
    penalty_shootouts: int = 0


class StatisticsObserver(TournamentObserver):
    """
    Observer that tracks aggregate Monte Carlo tournament statistics.

    This replaces the old update_statistics(stats, result) pattern with
    the same logic inside the observer framework.
    """

    def __init__(self) -> None:
        self.stats = SimulationStatistics()

    def observe(
        self,
        result: Any,
        simulation_id: int,
        team_repository: dict[str, dict] | None = None,
    ) -> None:
        self.stats.tournaments += 1

        group_stage_goals = sum(
            row.goals_for
            for group_rows in result.standings.values()
            for row in group_rows
        )

        self.stats.total_goals += group_stage_goals
        self.stats.total_matches += 72

        knockout_stages = (
            result.r32_results,
            result.r16_results,
            result.qf_results,
            result.sf_results,
            result.third_place_results,
            result.final_results,
        )

        for stage in knockout_stages:
            for match in stage:
                self.stats.total_matches += 1
                self.stats.total_goals += match.goals_team1 + match.goals_team2

                if match.went_to_extra_time:
                    self.stats.extra_time_matches += 1

                if match.went_to_penalties:
                    self.stats.penalty_shootouts += 1

    def finalize(self) -> dict[str, Any]:
        tournaments = self.stats.tournaments
        total_matches = self.stats.total_matches

        if tournaments == 0 or total_matches == 0:
            return {
                "statistics": self.stats,
                "summary": {},
            }

        knockout_matches = 32 * tournaments

        summary = {
            "tournaments": tournaments,
            "total_matches": total_matches,
            "total_goals": self.stats.total_goals,
            "avg_goals_per_tournament": self.stats.total_goals / tournaments,
            "avg_goals_per_match": self.stats.total_goals / total_matches,
            "extra_time_matches": self.stats.extra_time_matches,
            "penalty_shootouts": self.stats.penalty_shootouts,
            "extra_time_rate_all_matches": self.stats.extra_time_matches / total_matches,
            "penalty_rate_all_matches": self.stats.penalty_shootouts / total_matches,
            "extra_time_rate_knockout_matches": self.stats.extra_time_matches / knockout_matches,
            "penalty_rate_knockout_matches": self.stats.penalty_shootouts / knockout_matches,
        }

        return {
            "statistics": self.stats,
            "summary": summary,
        }
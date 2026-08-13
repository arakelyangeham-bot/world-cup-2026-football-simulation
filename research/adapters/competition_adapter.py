#competition_adapter.py

from __future__ import annotations

from typing import Any

from research.experiments import ExperimentRunResult
from simulation.competition import Competition, MatchResult


class CompetitionAdapter:
    """
    Adapter between the football competition framework and the Version 3
    research framework.

    The adapter converts completed football competitions into
    ExperimentRunResult objects.
    """

    def __init__(self, team_strengths: dict[str, float]) -> None:
        self.team_strengths = team_strengths

    def to_experiment_run_result(
        self,
        competition_result: Any,
        experiment_name: str,
        format_name: str,
        run_id: int,
    ) -> ExperimentRunResult:
        champion = competition_result.champion

        champion_strength = (
            self.team_strengths.get(champion)
            if champion is not None
            else None
        )

        return ExperimentRunResult(
            experiment_name=experiment_name,
            format_name=format_name,
            run_id=run_id,
            champion=champion,
            champion_strength=champion_strength,
            match_results=self.collect_match_results(competition_result),
            metadata={
                "competition_name": competition_result.competition_name,
            },
        )

    def collect_match_results(self, competition_result: Any) -> list[MatchResult]:
        match_results: list[MatchResult] = []

        for stage_result in competition_result.stage_results:
            for item in stage_result.match_results:
                if isinstance(item, MatchResult):
                    match_results.append(item)

                elif hasattr(item, "match_results"):
                    match_results.extend(item.match_results)

        return match_results
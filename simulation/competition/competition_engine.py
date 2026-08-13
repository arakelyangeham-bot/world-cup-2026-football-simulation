#competition_engine.py

from __future__ import annotations

from simulation.competition.advancement import AdvancementResult
from simulation.competition.competition import Competition, CompetitionResult
from simulation.competition.stage_resolver import StageResolver


class CompetitionEngine:
    """
    Minimal generic competition engine.

    This engine resolves stages in order using StageResolver, then applies
    each stage's advancement rule if one is defined.

    Version 1 intentionally does not generate future stages dynamically.
    It validates the core competition-resolution pipeline.
    """

    def __init__(self, stage_resolver: StageResolver | None = None) -> None:
        self.stage_resolver = stage_resolver or StageResolver()

    def resolve(self, competition: Competition) -> CompetitionResult:
        competition_result = CompetitionResult(
            competition_name=competition.name,
            participants=competition.participants,
            metadata={
                "engine": "CompetitionEngine",
                "stage_count": len(competition.stages),
            },
        )

        for stage in competition.stages:
            stage_result = self.stage_resolver.resolve(stage)
            competition_result.stage_results.append(stage_result)

            if stage.advancement_rule is not None:
                advancement_result = stage.advancement_rule.apply(stage_result)
            else:
                advancement_result = AdvancementResult(
                    qualifiers=[],
                    eliminated=[],
                    metadata={
                        "rule": None,
                    },
                )

            competition_result.advancement_results.append(advancement_result)

        self._infer_simple_champion(competition_result)

        return competition_result

    def _infer_simple_champion(
        self,
        competition_result: CompetitionResult,
    ) -> None:
        """
        Infer a simple champion from the final resolved stage.

        For standings-based competitions, the champion is the top-ranked team.
        For knockout/final stages, the champion is the final stage winner.
        """

        if not competition_result.stage_results:
            return

        last_stage_result = competition_result.stage_results[-1]

        if last_stage_result.winner is not None:
            competition_result.champion = last_stage_result.winner
            competition_result.runner_up = last_stage_result.runner_up
            return

        if last_stage_result.standings is None:
            return

        ranked_rows = last_stage_result.standings.ranked_rows()

        if not ranked_rows:
            return

        competition_result.champion = ranked_rows[0].team

        if len(ranked_rows) > 1:
            competition_result.runner_up = ranked_rows[1].team
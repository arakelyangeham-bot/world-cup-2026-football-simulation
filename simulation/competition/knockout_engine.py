#knockout_engine.py

from __future__ import annotations

from simulation.competition.match_result import MatchResult
from simulation.competition.stage import Stage, StageType
from simulation.competition.stage_result import StageResult
from simulation.competition.tie import Tie, TieResult


class KnockoutEngine:
    """
    Generic engine for resolving single-match knockout stages.

    Version 1 expects the stage.matches list to contain Tie objects.
    Each Tie should contain exactly one completed MatchResult.
    """

    SUPPORTED_STAGE_TYPES = {
        StageType.KNOCKOUT,
        StageType.PLAYOFF,
        StageType.FINAL,
    }

    def resolve(
        self,
        stage: Stage,
        match_results: list[Tie],
    ) -> StageResult:
        if stage.stage_type not in self.SUPPORTED_STAGE_TYPES:
            raise ValueError(
                f"KnockoutEngine does not support stage type: {stage.stage_type}"
            )

        tie_results = [
            self._resolve_tie(tie)
            for tie in match_results
        ]

        winners = [
            tie_result.winner
            for tie_result in tie_results
        ]

        losers = [
            tie_result.loser
            for tie_result in tie_results
        ]

        return StageResult(
            stage_name=stage.name,
            stage_type=stage.stage_type.value,
            participants=stage.participants,
            match_results=tie_results,
            qualifiers=winners,
            eliminated=losers,
            winner=winners[0] if len(winners) == 1 else None,
            runner_up=losers[0] if len(losers) == 1 else None,
            metadata={
                "engine": "KnockoutEngine",
                "tie_count": len(tie_results),
            },
        )

    def _resolve_tie(self, tie: Tie) -> TieResult:
        if len(tie.match_results) != 1:
            raise ValueError(
                "KnockoutEngine v1 expects each tie to contain exactly one match result."
            )

        match = tie.match_results[0]

        if not isinstance(match, MatchResult):
            raise TypeError("Tie match_results must contain MatchResult objects.")

        if match.winner is None or match.loser is None:
            raise ValueError(
                "KnockoutEngine v1 requires a non-drawn match result."
            )

        return TieResult(
            team1=tie.team1,
            team2=tie.team2,
            match_results=tie.match_results,
            winner=match.winner,
            loser=match.loser,
            metadata={
                "total_goals": match.total_goals,
                "goal_margin": match.goal_margin,
            },
        )
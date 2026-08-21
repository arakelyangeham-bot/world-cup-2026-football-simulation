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
        StageType.TWO_LEG_KNOCKOUT,
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
        if len(tie.match_results) == 1:
            return self._resolve_single_match_tie(tie)

        if len(tie.match_results) == 2:
            return self._resolve_two_leg_tie(tie)

        raise ValueError(
            "KnockoutEngine expects each tie to contain "
            "either one or two match results."
            )

    def _resolve_single_match_tie(
        self,
        tie: Tie,
    ) -> TieResult:
        match = tie.match_results[0]

        if not isinstance(match, MatchResult):
            raise TypeError(
                "Tie match_results must contain MatchResult objects."
            )

        if match.winner is None or match.loser is None:
            raise ValueError(
                "KnockoutEngine requires a non-drawn "
                "single-match result."
            )

        return TieResult(
            team1=tie.team1,
            team2=tie.team2,
            match_results=tie.match_results,
            winner=match.winner,
            loser=match.loser,
            metadata={
                **tie.metadata,
                "total_goals": match.total_goals,
                "goal_margin": match.goal_margin,
            },
        )

    def _resolve_two_leg_tie(
        self,
        tie: Tie,
    ) -> TieResult:
        for match in tie.match_results:
            if not isinstance(match, MatchResult):
                raise TypeError(
                    "Tie match_results must contain MatchResult objects."
                )

            if {
                match.team1,
                match.team2,
            } != {
                tie.team1,
                tie.team2,
            }:
                raise ValueError(
                    "Two-leg tie contains a match involving "
                    "unexpected participants."
                )

        aggregate_team1 = 0
        aggregate_team2 = 0

        for match in tie.match_results:
            if match.team1 == tie.team1:
                aggregate_team1 += match.goals_team1
                aggregate_team2 += match.goals_team2
            else:
                aggregate_team1 += match.goals_team2
                aggregate_team2 += match.goals_team1

        if aggregate_team1 == aggregate_team2:
            raise ValueError(
                "Two-leg tie is level on aggregate and requires "
                "extra-time or penalty resolution."
            )

        if aggregate_team1 > aggregate_team2:
            winner = tie.team1
            loser = tie.team2
        else:
            winner = tie.team2
            loser = tie.team1

        return TieResult(
            team1=tie.team1,
            team2=tie.team2,
            match_results=tie.match_results,
            winner=winner,
            loser=loser,
            metadata={
                **tie.metadata,
                "aggregate_team1": aggregate_team1,
                "aggregate_team2": aggregate_team2,
                "aggregate_margin": abs(
                    aggregate_team1 - aggregate_team2
                ),
            },
        )
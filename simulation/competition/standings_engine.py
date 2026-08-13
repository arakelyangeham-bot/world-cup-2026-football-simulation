#standings_engine.py

from __future__ import annotations

from simulation.competition.match_result import MatchResult
from simulation.competition.stage import Stage, StageType
from simulation.competition.stage_result import StageResult
from simulation.competition.standings import StandingsTable


class StandingsEngine:
    """
    Generic engine for resolving standings-based competition stages.

    This engine can support group stages, domestic leagues, league phases,
    and other table-based football formats.
    """

    SUPPORTED_STAGE_TYPES = {
        StageType.GROUP,
        StageType.LEAGUE,
        StageType.SWISS,
    }

    def resolve(
        self,
        stage: Stage,
        match_results: list[MatchResult],
    ) -> StageResult:
        if stage.stage_type not in self.SUPPORTED_STAGE_TYPES:
            raise ValueError(
                f"StandingsEngine does not support stage type: {stage.stage_type}"
            )

        table = StandingsTable(teams=stage.participants)

        for match in match_results:
            table.record_match(
                team1=match.team1,
                team2=match.team2,
                goals_team1=match.goals_team1,
                goals_team2=match.goals_team2,
            )

        return StageResult(
            stage_name=stage.name,
            stage_type=stage.stage_type.value,
            participants=stage.participants,
            match_results=match_results,
            standings=table,
            metadata={
                "engine": "StandingsEngine",
                "match_count": len(match_results),
            },
        )
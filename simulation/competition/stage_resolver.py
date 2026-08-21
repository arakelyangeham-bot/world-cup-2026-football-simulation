#stage_resolver.py

from __future__ import annotations

from typing import Protocol

from simulation.competition.stage import Stage, StageType
from simulation.competition.stage_result import StageResult
from simulation.competition.standings_engine import StandingsEngine
from simulation.competition.knockout_engine import KnockoutEngine


class StageEngine(Protocol):
    def resolve(self, stage: Stage, match_results: list) -> StageResult:
        ...


class StageResolver:
    """
    Resolves stages by dispatching them to the appropriate stage engine.

    This keeps CompetitionEngine from needing hardcoded logic for every
    possible stage type.
    """

    def __init__(self) -> None:
        self.engines: dict[StageType, StageEngine] = {}
        self.register_default_engines()

    def register_default_engines(self) -> None:
        standings_engine = StandingsEngine()
        knockout_engine = KnockoutEngine()

        self.register(StageType.GROUP, standings_engine)
        self.register(StageType.LEAGUE, standings_engine)
        self.register(StageType.SWISS, standings_engine)

        self.register(StageType.KNOCKOUT, knockout_engine)
        self.register(StageType.PLAYOFF, knockout_engine)
        self.register(StageType.FINAL, knockout_engine)
        self.register(
            StageType.TWO_LEG_KNOCKOUT,
            knockout_engine,
        )
        
    def register(self, stage_type: StageType, engine: StageEngine) -> None:
        self.engines[stage_type] = engine

    def resolve(self, stage: Stage) -> StageResult:
        engine = self.engines.get(stage.stage_type)

        if engine is None:
            raise ValueError(
                f"No engine registered for stage type: {stage.stage_type}"
            )

        return engine.resolve(
            stage=stage,
            match_results=stage.matches,
        )
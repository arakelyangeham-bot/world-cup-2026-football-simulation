#competition.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulation.competition.stage import Stage
from simulation.competition.stage_result import StageResult
from simulation.competition.advancement import AdvancementResult


@dataclass
class Competition:
    """
    Generic definition of a football competition.

    A Competition describes structure. It does not simulate matches directly.
    Engines consume Competition objects and produce CompetitionResult objects.
    """

    name: str
    participants: list[str]
    stages: list[Stage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage: Stage) -> None:
        self.stages.append(stage)


@dataclass
class CompetitionResult:
    """
    Completed output of a simulated or resolved football competition.
    """

    competition_name: str
    participants: list[str]
    stage_results: list[StageResult] = field(default_factory=list)
    advancement_results: list[AdvancementResult] = field(default_factory=list)
    champion: str | None = None
    runner_up: str | None = None
    final_standings: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
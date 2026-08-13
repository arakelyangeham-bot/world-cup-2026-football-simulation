#competition_definition.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageDefinition:
    """
    Static definition of one stage inside a football competition.

    This describes the structure of a stage, not the simulated results.
    """

    name: str
    stage_type: str
    participant_count: int | None = None
    competition_format: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompetitionDefinition:
    """
    Static definition of a football competition.

    This is a catalog/configuration object. It does not run simulations.
    """

    name: str
    competition_type: str
    region: str | None = None
    governing_body: str | None = None
    participant_count: int | None = None
    stages: list[StageDefinition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def stage_names(self) -> list[str]:
        return [stage.name for stage in self.stages]
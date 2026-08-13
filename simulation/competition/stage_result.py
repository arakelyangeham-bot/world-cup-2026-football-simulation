#stage_result.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    """
    Generic output produced by simulating or resolving a competition stage.

    A StageResult should contain what happened in the stage, not how the
    stage was simulated.
    """

    stage_name: str
    stage_type: str
    participants: list[str]
    match_results: list[Any] = field(default_factory=list)
    standings: Any | None = None
    qualifiers: list[str] = field(default_factory=list)
    eliminated: list[str] = field(default_factory=list)
    winner: str | None = None
    runner_up: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
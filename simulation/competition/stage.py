#stage.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageType(str, Enum):
    GROUP = "group"
    LEAGUE = "league"
    KNOCKOUT = "knockout"
    TWO_LEG_KNOCKOUT = "two_leg_knockout"
    SWISS = "swiss"
    PLAYOFF = "playoff"
    FINAL = "final"


@dataclass
class Stage:
    """
    Generic competition stage definition.

    A Stage describes one phase of a competition. It does not simulate
    matches directly. Engines consume Stage objects and produce StageResult
    objects.
    """

    name: str
    stage_type: StageType
    participants: list[str]
    matches: list[Any] = field(default_factory=list)
    advancement_rule: Any | None = None
    result: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_match(self, match: Any) -> None:
        self.matches.append(match)

    def set_result(self, result: Any) -> None:
        self.result = result
#match_result.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MatchResult:
    """
    Generic completed football match result used by competition engines.

    This object intentionally does not depend on the internal match engine's
    native result format. It provides a stable interface for standings,
    stages, observers, and future competition engines.
    """

    team1: str
    team2: str
    goals_team1: int
    goals_team2: int
    stage: str | None = None
    match_id: str | int | None = None
    metadata: dict[str, Any] | None = None

    @property
    def total_goals(self) -> int:
        return self.goals_team1 + self.goals_team2

    @property
    def goal_margin(self) -> int:
        return abs(self.goals_team1 - self.goals_team2)

    @property
    def is_draw(self) -> bool:
        return self.goals_team1 == self.goals_team2

    @property
    def winner(self) -> str | None:
        if self.goals_team1 > self.goals_team2:
            return self.team1

        if self.goals_team2 > self.goals_team1:
            return self.team2

        return None

    @property
    def loser(self) -> str | None:
        if self.goals_team1 > self.goals_team2:
            return self.team2

        if self.goals_team2 > self.goals_team1:
            return self.team1

        return None
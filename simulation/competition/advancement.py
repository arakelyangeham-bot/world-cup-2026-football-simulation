#advancement.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdvancementResult:
    """
    Output produced by applying an advancement rule to a completed stage.
    """

    qualifiers: list[str] = field(default_factory=list)
    eliminated: list[str] = field(default_factory=list)
    placements: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AdvancementRule(ABC):
    """
    Base class for advancement rules.

    Advancement rules interpret completed stage results. They should not
    simulate matches or modify standings.
    """

    name: str = "advancement_rule"
    description: str = ""

    @abstractmethod
    def apply(self, stage_result: Any) -> AdvancementResult:
        raise NotImplementedError


class TopNAdvanceRule(AdvancementRule):
    """
    Advances the top N teams from a standings-based stage.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.name = f"top_{n}_advance"
        self.description = f"Top {n} teams advance."

    def apply(self, stage_result: Any) -> AdvancementResult:
        if stage_result.standings is None:
            raise ValueError("TopNAdvanceRule requires standings.")

        ranked_rows = stage_result.standings.ranked_rows()

        qualifiers = [
            row.team
            for row in ranked_rows[: self.n]
        ]

        eliminated = [
            row.team
            for row in ranked_rows[self.n :]
        ]

        placements = {
            row.team: rank
            for rank, row in enumerate(ranked_rows, start=1)
        }

        return AdvancementResult(
            qualifiers=qualifiers,
            eliminated=eliminated,
            placements=placements,
            metadata={
                "rule": self.name,
                "n": self.n,
            },
        )


class BottomNEliminatedRule(AdvancementRule):
    """
    Marks the bottom N teams from a standings-based stage as eliminated.

    This is useful for relegation-style logic or cutoff analysis.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.name = f"bottom_{n}_eliminated"
        self.description = f"Bottom {n} teams are eliminated."

    def apply(self, stage_result: Any) -> AdvancementResult:
        if stage_result.standings is None:
            raise ValueError("BottomNEliminatedRule requires standings.")

        ranked_rows = stage_result.standings.ranked_rows()

        eliminated_rows = ranked_rows[-self.n :] if self.n > 0 else []
        qualified_rows = ranked_rows[: -self.n] if self.n > 0 else ranked_rows

        qualifiers = [
            row.team
            for row in qualified_rows
        ]

        eliminated = [
            row.team
            for row in eliminated_rows
        ]

        placements = {
            row.team: rank
            for rank, row in enumerate(ranked_rows, start=1)
        }

        return AdvancementResult(
            qualifiers=qualifiers,
            eliminated=eliminated,
            placements=placements,
            metadata={
                "rule": self.name,
                "n": self.n,
            },
        )
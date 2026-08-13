#tie.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulation.competition.match_result import MatchResult


@dataclass
class TieResult:
    """
    Completed result of a knockout tie.

    Version 1 supports single-match ties only.
    Future versions can support two-leg aggregate ties.
    """

    team1: str
    team2: str
    match_results: list[MatchResult]
    winner: str
    loser: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tie:
    """
    Generic knockout tie definition.

    A tie is a competition unit that resolves one team advancing over another.
    """

    team1: str
    team2: str
    match_results: list[MatchResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_match_result(self, match_result: MatchResult) -> None:
        self.match_results.append(match_result)
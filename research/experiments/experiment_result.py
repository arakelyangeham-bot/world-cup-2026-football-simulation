from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulation.competition import MatchResult


@dataclass
class ExperimentRunResult:
    """
    Result of one simulated competition run inside a Version 3 experiment.
    """

    experiment_name: str
    format_name: str
    run_id: int
    champion: str | None
    champion_strength: float | None
    match_results: list[MatchResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """
    Collection of repeated simulation runs for one experimental condition.

    Example:
    - all league-format runs
    - all knockout-format runs
    """

    experiment_name: str
    format_name: str
    strongest_team: str
    team_strengths: dict[str, float]
    runs: list[ExperimentRunResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_run(self, run: ExperimentRunResult) -> None:
        self.runs.append(run)
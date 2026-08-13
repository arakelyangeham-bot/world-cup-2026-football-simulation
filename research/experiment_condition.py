#experiment_condition.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentCondition:
    """
    One experimental condition within a Version 3 experiment.

    An Experiment compares one or more ExperimentConditions while keeping
    the research question fixed.
    """

    name: str

    # Competition
    competition_format: str

    # Football model
    repository_source: str
    match_engine: str

    # Simulation
    simulation_count: int
    random_seed: int | None = None

    # Optional experiment-specific settings
    parameters: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "competition_format": self.competition_format,
            "repository_source": self.repository_source,
            "match_engine": self.match_engine,
            "simulation_count": self.simulation_count,
            "random_seed": self.random_seed,
            **self.parameters,
        }
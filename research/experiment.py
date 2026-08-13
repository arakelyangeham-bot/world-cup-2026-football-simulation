#experiment.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.metrics import Metric


@dataclass
class Experiment:
    """
    Research-level definition of a Version 3 computational football experiment.

    An Experiment describes the scientific question, hypothesis, fixed variables,
    independent variable, and metrics. It does not run simulations directly.
    """

    experiment_id: str
    title: str
    research_question: str
    hypothesis: str
    fixed_variables: dict[str, Any]
    independent_variable: str
    dependent_variables: list[str]
    metrics: list[Metric]
    metadata: dict[str, Any] = field(default_factory=dict)

    def metric_names(self) -> list[str]:
        return [
            metric.name
            for metric in self.metrics
        ]
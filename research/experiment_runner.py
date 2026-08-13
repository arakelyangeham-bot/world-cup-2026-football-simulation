#experiment_runner.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.experiments import ExperimentResult
from research.metrics import Metric


@dataclass
class MetricResult:
    """
    Output of one metric computed for one experimental condition.
    """

    metric_name: str
    value: Any
    description: str = ""


@dataclass
class ExperimentReport:
    """
    Metric outputs for one experimental condition.

    Example:
    - League condition
    - Knockout condition
    """

    experiment_name: str
    format_name: str
    simulation_count: int
    metric_results: list[MetricResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "experiment_name": self.experiment_name,
                "format_name": self.format_name,
                "simulation_count": self.simulation_count,
                "metric_name": metric.metric_name,
                "metric_value": metric.value,
                "metric_description": metric.description,
            }
            for metric in self.metric_results
        ]


class ExperimentRunner:
    """
    Runs reusable Version 3 research metrics against experiment results.

    This class does not simulate football competitions. It assumes simulations
    have already produced ExperimentResult objects.
    """

    def __init__(self, metrics: list[Metric]) -> None:
        self.metrics = metrics

    def evaluate(
        self,
        experiment_result: ExperimentResult,
    ) -> ExperimentReport:
        metric_results = []

        for metric in self.metrics:
            metric_results.append(
                MetricResult(
                    metric_name=metric.name,
                    value=metric.compute(experiment_result),
                    description=metric.description,
                )
            )

        return ExperimentReport(
            experiment_name=experiment_result.experiment_name,
            format_name=experiment_result.format_name,
            simulation_count=len(experiment_result.runs),
            metric_results=metric_results,
            metadata={
                "runner": "ExperimentRunner",
            },
        )
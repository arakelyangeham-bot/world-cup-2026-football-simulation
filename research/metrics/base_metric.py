#base_metric.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """
    Base class for Version 3 research metrics.

    Metrics compute interpretable measurements from experiment outputs.
    They should not run simulations, alter results, or depend on a specific
    competition format unless explicitly documented.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def compute(self, experiment_result: Any) -> Any:
        raise NotImplementedError
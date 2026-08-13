#observer_manager.py

from __future__ import annotations

from typing import Any

from simulation.observers.base_observer import TournamentObserver


class ObserverManager:
    """
    Coordinates a collection of tournament observers.

    The Monte Carlo driver should pass each completed TournamentResult
    to this manager. The manager then forwards the result to each observer.
    """

    def __init__(self, observers: list[TournamentObserver] | None = None):
        self.observers = observers or []

    def add_observer(self, observer: TournamentObserver) -> None:
        self.observers.append(observer)

    def observe(
        self,
        result: Any,
        simulation_id: int,
        team_repository: dict[str, dict] | None = None,
    ) -> None:
        for observer in self.observers:
            observer.observe(
                result=result,
                simulation_id=simulation_id,
                team_repository=team_repository,
            )

    def finalize(self) -> dict[str, Any]:
        outputs = {}

        for observer in self.observers:
            observer_name = observer.__class__.__name__
            outputs[observer_name] = observer.finalize()

        return outputs
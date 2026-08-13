#base_observer.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TournamentObserver(ABC):
    """
    Base class for objects that observe completed tournament simulations.

    Observers should never modify tournament results, simulate matches,
    alter standings, or affect tournament progression. They only consume
    completed TournamentResult objects and produce summaries or outputs.
    """

    @abstractmethod
    def observe(
        self,
        result: Any,
        simulation_id: int,
        team_repository: dict[str, dict] | None = None,
    ) -> None:
        """
        Observe one completed tournament result.
        """
        raise NotImplementedError

    def finalize(self) -> dict[str, Any]:
        """
        Return final observer outputs after all simulations have been observed.

        Subclasses may override this if they need to expose summaries,
        records, statistics, or export-ready data.
        """
        return {}
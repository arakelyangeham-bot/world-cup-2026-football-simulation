from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ScheduledFixture:
    """
    A scheduled football fixture before the match has been
    simulated.

    `match_date` may be omitted for legacy or purely
    structural schedules. Calendar-aware production
    simulations should always provide it.
    """

    fixture_id: str
    matchday: int
    home_team: str
    away_team: str
    leg: int = 1
    match_date: date | None = None

    @property
    def teams(self) -> tuple[str, str]:
        return self.home_team, self.away_team

    @property
    def is_calendar_aware(self) -> bool:
        return self.match_date is not None
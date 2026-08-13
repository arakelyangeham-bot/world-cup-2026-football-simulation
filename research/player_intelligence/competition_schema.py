#competition_schema.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Competition:
    """
    Canonical competition object.

    This should represent any football competition:
    World Cup, Champions League, Premier League, MLS, etc.
    """

    competition_id: str
    name: str
    season: str | None = None
    competition_type: str | None = None


@dataclass(frozen=True)
class TeamEntry:
    """
    A team participating in a competition.
    """

    team_id: str
    name: str
    competition_id: str

    country: str | None = None


@dataclass(frozen=True)
class CompetitionRoster:
    """
    Roster membership for one team inside one competition.

    This is intentionally separate from Squad because a roster is
    competition-specific, while a Squad is the football object consumed
    by Player Intelligence.
    """

    competition_id: str
    team_id: str
    team_name: str

    player_ids: tuple[str, ...]
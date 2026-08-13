#representation_provider

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
)


@dataclass(frozen=True)
class RepresentationRequest:
    """
    Identifies the team representation required for one prediction.

    `prediction_season_start_year` is the season containing the target
    match.

    `representation_season_start_year` is the historical season from
    which team intelligence may safely be drawn.
    """

    competition_key: str
    prediction_season_start_year: int
    representation_season_start_year: int

    team_id: int
    team_name: str | None = None


@dataclass(frozen=True)
class RepresentationResult:
    """
    One resolved pre-match team representation with provenance.
    """

    request: RepresentationRequest
    representation: TeamRepresentation

    representation_type: str
    formation: str | None

    competition_id: int
    season_id: int

    source: str
    temporal_validity_pass: bool


class RepresentationProvider(Protocol):
    """
    Supplies a temporally valid TeamRepresentation for an observation.

    Implementations may use:

    - full squads;
    - expected starting XIs;
    - confirmed starting XIs;
    - rolling representations;
    - future learned representations.

    The observation builder depends only on this interface.
    """

    @property
    def provider_name(self) -> str:
        ...

    def get_representation(
        self,
        request: RepresentationRequest,
    ) -> RepresentationResult:
        ...
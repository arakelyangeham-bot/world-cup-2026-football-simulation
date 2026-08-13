from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
)


@dataclass(frozen=True)
class ProductionClubRecord:
    """
    Serializable production snapshot for one club.

    This record is distinct from TeamRepresentation because it
    includes artifact-level metadata required for persistence and
    runtime traceability.
    """

    club: str

    attack: float
    midfield: float
    defense: float
    goalkeeper: float

    attack_depth: float
    midfield_depth: float
    defense_depth: float

    squad_quality: float
    evidence_score: float

    representation_type: str
    aggregation_profile: str

    player_count: int
    available_player_count: int

    repository_version: str
    repository_scope: str
    representation_season_id: str

    @classmethod
    def from_team_representation(
        cls,
        *,
        club: str,
        representation: TeamRepresentation,
        repository_version: str,
        repository_scope: str,
        representation_season_id: str,
    ) -> ProductionClubRecord:
        """
        Convert a domain TeamRepresentation into a production
        persistence record.
        """

        return cls(
            club=club,
            attack=representation.attack,
            midfield=representation.midfield,
            defense=representation.defense,
            goalkeeper=representation.goalkeeper,
            attack_depth=representation.attack_depth,
            midfield_depth=representation.midfield_depth,
            defense_depth=representation.defense_depth,
            squad_quality=representation.squad_quality,
            evidence_score=representation.evidence_score,
            representation_type=representation.representation_type,
            aggregation_profile=representation.aggregation_profile,
            player_count=representation.player_count,
            available_player_count=(
                representation.available_player_count
            ),
            repository_version=repository_version,
            repository_scope=repository_scope,
            representation_season_id=(
                representation_season_id
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
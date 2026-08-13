from __future__ import annotations

from dataclasses import dataclass

from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
)


@dataclass(frozen=True)
class TeamRepositoryEntry:
    team: str

    attack: float
    midfield: float
    defense: float
    gk: float

    poisson_attack: float
    poisson_defense: float

    representation_type: str
    aggregation_profile: str
    player_count: int
    available_player_count: int

    fifa_points: float | None = None


def project_representation_to_repository_entry(
    representation: TeamRepresentation,
    fifa_points: float | None = None,
) -> TeamRepositoryEntry:
    return TeamRepositoryEntry(
        team=representation.national_team,
        attack=representation.attack,
        midfield=representation.midfield,
        defense=representation.defense,
        gk=representation.goalkeeper,
        poisson_attack=representation.attack,
        poisson_defense=representation.defense,
        representation_type=representation.representation_type,
        aggregation_profile=representation.aggregation_profile,
        player_count=representation.player_count,
        available_player_count=representation.available_player_count,
        fifa_points=fifa_points,
    )


def repository_entry_to_dict(
    entry: TeamRepositoryEntry,
) -> dict:
    return {
        "attack": entry.attack,
        "midfield": entry.midfield,
        "defense": entry.defense,
        "gk": entry.gk,
        "poisson_attack": entry.poisson_attack,
        "poisson_defense": entry.poisson_defense,
        "representation_type": entry.representation_type,
        "aggregation_profile": entry.aggregation_profile,
        "player_count": entry.player_count,
        "available_player_count": entry.available_player_count,
        "fifa_points": entry.fifa_points,
    }
#test_contribution_team_representation_parity

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pytest

from research.player_intelligence.player_schema import (
    LineupAssignment,
    Player,
    PlayerAvailability,
    PlayerIdentity,
    PlayerRatings,
    RoleRatings,
    StartingXI,
)
from research.player_intelligence.team_representation_builder import (
    build_team_representation_from_starting_xi,
    build_team_representation_from_starting_xi_contributions,
)


def make_player(
    player_id: str,
    *,
    overall: float,
    cb: float | None = None,
    fb: float | None = None,
    dm: float | None = None,
    cm: float | None = None,
    am: float | None = None,
    w: float | None = None,
    st: float | None = None,
    available: bool = True,
) -> Player:
    return Player(
        identity=PlayerIdentity(
            player_id=player_id,
            name=f"Player {player_id}",
            national_team="Test Team",
        ),
        ratings=PlayerRatings(
            overall=overall,
            attack=overall,
            midfield=overall,
            defense=overall,
            goalkeeper=0.0,
        ),
        role_ratings=RoleRatings(
            CB=cb,
            FB=fb,
            DM=dm,
            CM=cm,
            AM=am,
            W=w,
            ST=st,
        ),
        availability=PlayerAvailability(
            available=available,
        ),
    )


def make_starting_xi() -> StartingXI:
    players = (
        make_player(
            "1",
            overall=0.80,
            cb=0.90,
        ),
        make_player(
            "2",
            overall=0.75,
            fb=0.85,
        ),
        make_player(
            "3",
            overall=0.88,
            dm=0.82,
            cm=0.88,
        ),
        make_player(
            "4",
            overall=0.92,
            cm=0.92,
            am=0.87,
        ),
        make_player(
            "5",
            overall=0.95,
            w=0.95,
        ),
        make_player(
            "6",
            overall=1.00,
            st=1.00,
            available=False,
        ),
    )

    assignments = (
        LineupAssignment(
            slot="CB1",
            tactical_role="CB",
            player=players[0],
            selection_rating=0.90,
        ),
        LineupAssignment(
            slot="FB1",
            tactical_role="FB",
            player=players[1],
            selection_rating=0.85,
        ),
        LineupAssignment(
            slot="DM1",
            tactical_role="DM",
            player=players[2],
            selection_rating=0.82,
        ),
        LineupAssignment(
            slot="CM1",
            tactical_role="CM",
            player=players[3],
            selection_rating=0.92,
        ),
        LineupAssignment(
            slot="W1",
            tactical_role="W",
            player=players[4],
            selection_rating=0.95,
        ),
        LineupAssignment(
            slot="ST1",
            tactical_role="ST",
            player=players[5],
            selection_rating=1.00,
        ),
    )

    return StartingXI(
        national_team="Test Team",
        formation="test-formation",
        players=players,
        assignments=assignments,
    )


def test_contribution_path_matches_legacy_default_exactly() -> None:
    starting_xi = make_starting_xi()

    legacy = (
        build_team_representation_from_starting_xi(
            starting_xi
        )
    )

    contribution_based = (
        build_team_representation_from_starting_xi_contributions(
            starting_xi
        )
    )

    assert contribution_based == legacy


@dataclass(frozen=True)
class TopKSpecification:
    aggregation_family: str = "top_k_mean"
    output_type: str = "scalar"
    parameters: Mapping[str, object] = None  # type: ignore[assignment]
    specification_id: str = "test_top_3"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            {
                "k": 3,
            },
        )

    def validate(self) -> None:
        if self.parameters["k"] != 3:
            raise ValueError(
                "Test specification must use k=3."
            )


def test_contribution_path_matches_explicit_aggregation() -> None:
    starting_xi = make_starting_xi()
    specification = TopKSpecification()

    legacy = (
        build_team_representation_from_starting_xi(
            starting_xi,
            aggregation_specification=specification,
        )
    )

    contribution_based = (
        build_team_representation_from_starting_xi_contributions(
            starting_xi,
            aggregation_specification=specification,
        )
    )

    assert contribution_based == legacy


def test_contribution_path_requires_assignments() -> None:
    starting_xi = make_starting_xi()

    legacy_only = StartingXI(
        national_team=starting_xi.national_team,
        formation=starting_xi.formation,
        players=starting_xi.players,
    )

    with pytest.raises(
        ValueError,
        match="requires preserved lineup assignments",
    ):
        build_team_representation_from_starting_xi_contributions(
            legacy_only
        )
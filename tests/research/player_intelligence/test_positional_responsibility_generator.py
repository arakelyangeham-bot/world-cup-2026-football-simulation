#test_positional_responsibility_generator

from __future__ import annotations

import pytest

from research.player_intelligence.football_responsibility import (
    ResponsibilityType,
)
from research.player_intelligence.formation_geometry import (
    FormationGeometry,
    FormationPosition,
)
from research.player_intelligence.player_schema import (
    LineupAssignment,
    Player,
    PlayerIdentity,
    PlayerRatings,
    StartingXI,
)
from research.player_intelligence.positional_responsibility_generator import (
    FootballResponsibility,
    broad_corridor,
    generate_positional_responsibilities,
)


def make_player(
    player_id: str,
) -> Player:
    return Player(
        identity=PlayerIdentity(
            player_id=player_id,
            name=f"Player {player_id}",
            national_team="Test Team",
        ),
        ratings=PlayerRatings(
            overall=0.5,
            attack=0.5,
            midfield=0.5,
            defense=0.5,
            goalkeeper=0.5,
        ),
    )


def make_position(
    *,
    slot: str,
    role: str,
    tactical_line: int,
    side: str,
    x: float,
    y: float,
) -> FormationPosition:
    return FormationPosition(
        formation="test-formation",
        slot=slot,
        role=role,
        x=x,
        y=y,
        tactical_line=tactical_line,
        side=side,
    )


def make_lineup() -> StartingXI:
    specifications = (
        ("CB1", "CB"),
        ("DM1", "DM"),
        ("CM1", "CM"),
        ("W1", "W"),
    )

    players = tuple(
        make_player(str(index))
        for index in range(
            len(specifications)
        )
    )

    assignments = tuple(
        LineupAssignment(
            slot=slot,
            tactical_role=role,
            player=player,
            selection_rating=0.5,
        )
        for (
            slot,
            role,
        ), player in zip(
            specifications,
            players,
        )
    )

    return StartingXI(
        national_team="Test Team",
        formation="test-formation",
        players=players,
        assignments=assignments,
    )


def make_geometry() -> FormationGeometry:
    return FormationGeometry(
        formation="test-formation",
        positions=(
            make_position(
                slot="CB1",
                role="CB",
                tactical_line=1,
                side="left_center",
                x=-0.5,
                y=1.0,
            ),
            make_position(
                slot="DM1",
                role="DM",
                tactical_line=2,
                side="center",
                x=0.0,
                y=2.0,
            ),
            make_position(
                slot="CM1",
                role="CM",
                tactical_line=3,
                side="left",
                x=-0.8,
                y=3.0,
            ),
            make_position(
                slot="W1",
                role="W",
                tactical_line=4,
                side="left",
                x=-1.8,
                y=4.0,
            ),
        ),
    )


def test_symmetric_relationship_requires_canonical_order() -> None:
    with pytest.raises(
        ValueError,
        match="canonical",
    ):
        FootballResponsibility(
            source_slot="CM1",
            target_slot="CB1",
            responsibility_type=(
                ResponsibilityType.SAME_LINE
            ),
        )


def test_same_slot_relationship_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="itself",
    ):
        FootballResponsibility(
            source_slot="CM1",
            target_slot="CM1",
            responsibility_type=(
                ResponsibilityType.SAME_LINE
            ),
        )


def test_broad_corridor_mapping() -> None:
    left_center = make_position(
        slot="CB1",
        role="CB",
        tactical_line=1,
        side="left_center",
        x=-0.5,
        y=1.0,
    )

    assert (
        broad_corridor(
            left_center
        )
        == "left"
    )


def test_generator_creates_adjacent_line_relationships() -> None:
    result = generate_positional_responsibilities(
        starting_xi=make_lineup(),
        geometry=make_geometry(),
    )

    adjacent_pairs = {
        (
            relationship.source_slot,
            relationship.target_slot,
        )
        for relationship
        in result.relationships_of_type(
            ResponsibilityType.ADJACENT_LINE
        )
    }

    assert adjacent_pairs == {
        ("CB1", "DM1"),
        ("CM1", "DM1"),
        ("CM1", "W1"),
    }


def test_generator_creates_same_corridor_relationships() -> None:
    result = generate_positional_responsibilities(
        starting_xi=make_lineup(),
        geometry=make_geometry(),
    )

    corridor_pairs = {
        (
            relationship.source_slot,
            relationship.target_slot,
        )
        for relationship
        in result.relationships_of_type(
            ResponsibilityType.SAME_CORRIDOR
        )
    }

    assert corridor_pairs == {
        ("CB1", "CM1"),
        ("CB1", "W1"),
        ("CM1", "W1"),
    }


def test_generation_is_deterministic() -> None:
    lineup = make_lineup()
    geometry = make_geometry()

    first = generate_positional_responsibilities(
        starting_xi=lineup,
        geometry=geometry,
    )

    second = generate_positional_responsibilities(
        starting_xi=lineup,
        geometry=geometry,
    )

    assert first == second


def test_team_identity_does_not_change_topology() -> None:
    lineup = make_lineup()

    renamed = StartingXI(
        national_team="Another Team",
        formation=lineup.formation,
        players=lineup.players,
        assignments=lineup.assignments,
    )

    geometry = make_geometry()

    first = generate_positional_responsibilities(
        starting_xi=lineup,
        geometry=geometry,
    )

    second = generate_positional_responsibilities(
        starting_xi=renamed,
        geometry=geometry,
    )

    first_keys = tuple(
        relationship.canonical_key
        for relationship
        in first.responsibilities
    )

    second_keys = tuple(
        relationship.canonical_key
        for relationship
        in second.responsibilities
    )

    assert first_keys == second_keys


def test_role_mismatch_is_rejected() -> None:
    lineup = make_lineup()

    invalid_geometry = FormationGeometry(
        formation="test-formation",
        positions=(
            make_position(
                slot="CB1",
                role="DM",
                tactical_line=1,
                side="left_center",
                x=-0.5,
                y=1.0,
            ),
            *make_geometry().positions[1:],
        ),
    )

    with pytest.raises(
        AssertionError,
        match="roles differ",
    ):
        generate_positional_responsibilities(
            starting_xi=lineup,
            geometry=invalid_geometry,
        )
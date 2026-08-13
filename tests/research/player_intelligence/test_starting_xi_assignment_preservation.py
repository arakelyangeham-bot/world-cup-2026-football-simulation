#test_starting_xi_assignment_preservation

from __future__ import annotations

import pandas as pd
import pytest

from research.player_intelligence.player_schema import (
    LineupAssignment,
    Player,
    PlayerIdentity,
    PlayerRatings,
    RoleRatings,
    Squad,
    StartingXI,
)
from research.player_intelligence.starting_xi_builder import (
    StartingXIBuilder,
)


def make_player(
    player_id: str,
    *,
    cb: float | None = None,
    cm: float | None = None,
    st: float | None = None,
) -> Player:
    return Player(
        identity=PlayerIdentity(
            player_id=player_id,
            name=f"Player {player_id}",
            national_team="Test Team",
        ),
        ratings=PlayerRatings(
            overall=1.0,
            attack=1.0,
            midfield=1.0,
            defense=1.0,
            goalkeeper=0.0,
        ),
        role_ratings=RoleRatings(
            CB=cb,
            CM=cm,
            ST=st,
        ),
    )


def test_builder_preserves_slot_role_and_rating() -> None:
    players = (
        make_player(
            "1",
            cb=0.8,
        ),
        make_player(
            "2",
            cm=0.9,
        ),
        make_player(
            "3",
            st=1.1,
        ),
    )

    squad = Squad(
        national_team="Test Team",
        players=players,
    )

    formation = pd.DataFrame(
        [
            {
                "slot": "CB1",
                "role": "CB",
            },
            {
                "slot": "CM1",
                "role": "CM",
            },
            {
                "slot": "ST1",
                "role": "ST",
            },
        ]
    )

    starting_xi = (
        StartingXIBuilder(
            formation="test-formation"
        )
        .build_for_squad(
            squad=squad,
            formation_df=formation,
        )
    )

    assert starting_xi.formation == (
        "test-formation"
    )

    assert len(
        starting_xi.players
    ) == 3

    assert len(
        starting_xi.assignments
    ) == 3

    assert [
        assignment.slot
        for assignment
        in starting_xi.assignments
    ] == [
        "CB1",
        "CM1",
        "ST1",
    ]

    assert [
        assignment.tactical_role
        for assignment
        in starting_xi.assignments
    ] == [
        "CB",
        "CM",
        "ST",
    ]

    assert [
        assignment.selection_rating
        for assignment
        in starting_xi.assignments
    ] == pytest.approx(
        [
            0.8,
            0.9,
            1.1,
        ]
    )

    assert tuple(
        assignment.player
        for assignment
        in starting_xi.assignments
    ) == starting_xi.players


def test_starting_xi_remains_backward_compatible() -> None:
    player = make_player(
        "1",
        cm=0.8,
    )

    starting_xi = StartingXI(
        national_team="Test Team",
        formation="4-3-3",
        players=(player,),
    )

    assert starting_xi.players == (
        player,
    )

    assert starting_xi.assignments == ()


def test_starting_xi_rejects_misaligned_assignments() -> None:
    first = make_player(
        "1",
        cb=0.8,
    )

    second = make_player(
        "2",
        cm=0.9,
    )

    assignment = LineupAssignment(
        slot="CB1",
        tactical_role="CB",
        player=second,
        selection_rating=0.9,
    )

    with pytest.raises(
        ValueError,
        match="not aligned",
    ):
        StartingXI(
            national_team="Test Team",
            formation="4-3-3",
            players=(first,),
            assignments=(assignment,),
        )
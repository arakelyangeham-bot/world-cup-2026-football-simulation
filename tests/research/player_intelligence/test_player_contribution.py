#test_player_contribution

from __future__ import annotations

import pytest

from research.player_intelligence.player_contribution import (
    build_player_contribution,
    build_starting_xi_contributions,
)
from research.player_intelligence.player_schema import (
    LineupAssignment,
    Player,
    PlayerIdentity,
    PlayerRatings,
    RoleRatings,
)
from research.player_intelligence.role_projection import (
    project_attack,
    project_defense,
    project_goalkeeper,
    project_midfield,
)


def make_player() -> Player:
    return Player(
        identity=PlayerIdentity(
            player_id="1",
            name="Test Player",
            national_team="Test Team",
        ),
        ratings=PlayerRatings(
            overall=0.8,
            attack=0.7,
            midfield=0.8,
            defense=0.5,
            goalkeeper=0.0,
        ),
        role_ratings=RoleRatings(
            DM=0.70,
            CM=0.90,
            AM=0.80,
        ),
    )


def test_contribution_reproduces_legacy_projection() -> None:
    player = make_player()

    assignment = LineupAssignment(
        slot="DM1",
        tactical_role="DM",
        player=player,
        selection_rating=0.70,
    )

    contribution = (
        build_player_contribution(
            assignment
        )
    )

    assert contribution.attack == pytest.approx(
        project_attack(
            player.role_ratings
        )
    )

    assert contribution.midfield == pytest.approx(
        project_midfield(
            player.role_ratings
        )
    )

    assert contribution.defense == pytest.approx(
        project_defense(
            player.role_ratings
        )
    )

    assert contribution.goalkeeper == pytest.approx(
        project_goalkeeper(
            player.role_ratings
        )
    )

    assert (
        contribution.contextual_adjustment_applied
        is False
    )


def test_contribution_preserves_role_diagnostics() -> None:
    player = make_player()

    assignment = LineupAssignment(
        slot="DM1",
        tactical_role="DM",
        player=player,
        selection_rating=0.70,
    )

    contribution = (
        build_player_contribution(
            assignment
        )
    )

    assert contribution.assigned_role == "DM"

    assert (
        contribution.assigned_role_rating
        == pytest.approx(0.70)
    )

    assert contribution.strongest_role == "CM"

    assert (
        contribution.strongest_role_rating
        == pytest.approx(0.90)
    )

    assert contribution.assigned_role_rank == 3

    assert (
        contribution.role_rating_difference
        == pytest.approx(0.20)
    )

    assert contribution.role_fit_ratio == pytest.approx(
        0.70 / 0.90
    )


def test_non_positive_best_rating_has_no_ratio() -> None:
    player = Player(
        identity=PlayerIdentity(
            player_id="2",
            name="Negative Scale Player",
            national_team="Test Team",
        ),
        ratings=PlayerRatings(
            overall=-0.1,
            attack=-0.1,
            midfield=-0.1,
            defense=-0.1,
            goalkeeper=0.0,
        ),
        role_ratings=RoleRatings(
            DM=-0.20,
            CM=-0.10,
        ),
    )

    assignment = LineupAssignment(
        slot="DM1",
        tactical_role="DM",
        player=player,
        selection_rating=-0.20,
    )

    contribution = (
        build_player_contribution(
            assignment
        )
    )

    assert contribution.strongest_role == "CM"
    assert contribution.assigned_role_rank == 2
    assert contribution.role_fit_ratio is None

    assert (
        contribution.role_rating_difference
        == pytest.approx(0.10)
    )


def test_lineup_contribution_order_is_preserved() -> None:
    first = make_player()

    second = Player(
        identity=PlayerIdentity(
            player_id="2",
            name="Second Player",
            national_team="Test Team",
        ),
        ratings=first.ratings,
        role_ratings=RoleRatings(
            ST=1.0,
        ),
    )

    assignments = (
        LineupAssignment(
            slot="DM1",
            tactical_role="DM",
            player=first,
            selection_rating=0.70,
        ),
        LineupAssignment(
            slot="ST1",
            tactical_role="ST",
            player=second,
            selection_rating=1.0,
        ),
    )

    contributions = (
        build_starting_xi_contributions(
            assignments
        )
    )

    assert tuple(
        contribution.player
        for contribution in contributions
    ) == (
        first,
        second,
    )

    assert tuple(
        contribution.assignment.slot
        for contribution in contributions
    ) == (
        "DM1",
        "ST1",
    )
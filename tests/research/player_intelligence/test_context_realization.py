#test_context_realization

from __future__ import annotations

import pytest

from research.player_intelligence.context_realization import (
    ContextRealizationPolicy,
    context_multiplier,
)
from research.player_intelligence.player_contribution import (
    build_player_contribution,
)
from research.player_intelligence.player_schema import (
    LineupAssignment,
    Player,
    PlayerIdentity,
    PlayerRatings,
    RoleRatings,
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
            attack=0.8,
            midfield=0.8,
            defense=0.8,
            goalkeeper=0.0,
        ),
        role_ratings=RoleRatings(
            CM=0.90,
            DM=0.70,
        ),
    )


def test_zero_strength_reproduces_legacy_exactly() -> None:
    player = make_player()

    assignment = LineupAssignment(
        slot="DM1",
        tactical_role="DM",
        player=player,
        selection_rating=0.70,
    )

    contribution = build_player_contribution(
        assignment,
        realization_policy=(
            ContextRealizationPolicy(
                adjustment_strength=0.0
            )
        ),
    )

    assert contribution.attack == contribution.intrinsic_attack
    assert contribution.midfield == contribution.intrinsic_midfield
    assert contribution.defense == contribution.intrinsic_defense
    assert contribution.goalkeeper == contribution.intrinsic_goalkeeper
    assert contribution.context_multiplier == pytest.approx(1.0)
    assert contribution.contextual_adjustment_applied is False


def test_strongest_role_receives_no_adjustment() -> None:
    player = make_player()

    assignment = LineupAssignment(
        slot="CM1",
        tactical_role="CM",
        player=player,
        selection_rating=0.90,
    )

    contribution = build_player_contribution(
        assignment,
        realization_policy=(
            ContextRealizationPolicy(
                adjustment_strength=0.05
            )
        ),
    )

    assert contribution.context_gap == pytest.approx(0.0)
    assert contribution.context_multiplier == pytest.approx(1.0)
    assert contribution.attack == contribution.intrinsic_attack
    assert contribution.contextual_adjustment_applied is False


def test_out_of_role_assignment_reduces_outfield_contribution() -> None:
    player = make_player()

    assignment = LineupAssignment(
        slot="DM1",
        tactical_role="DM",
        player=player,
        selection_rating=0.70,
    )

    contribution = build_player_contribution(
        assignment,
        realization_policy=(
            ContextRealizationPolicy(
                adjustment_strength=0.05
            )
        ),
    )

    expected_gap = (
        0.20
        / 1.60
    )

    expected_multiplier = (
        1.0
        - 0.05
        * expected_gap
    )

    assert contribution.context_gap == pytest.approx(
        expected_gap
    )

    assert contribution.context_multiplier == pytest.approx(
        expected_multiplier
    )

    assert contribution.attack == pytest.approx(
        contribution.intrinsic_attack
        * expected_multiplier
    )

    assert contribution.midfield == pytest.approx(
        contribution.intrinsic_midfield
        * expected_multiplier
    )

    assert contribution.defense == pytest.approx(
        contribution.intrinsic_defense
        * expected_multiplier
    )

    assert contribution.goalkeeper == pytest.approx(
        contribution.intrinsic_goalkeeper
    )

    assert contribution.contextual_adjustment_applied is True


def test_unresolved_gap_receives_neutral_multiplier() -> None:
    assert context_multiplier(
        gap=None,
        policy=ContextRealizationPolicy(
            adjustment_strength=0.05
        ),
    ) == pytest.approx(1.0)


def test_invalid_strength_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must lie within",
    ):
        ContextRealizationPolicy(
            adjustment_strength=1.1
        )
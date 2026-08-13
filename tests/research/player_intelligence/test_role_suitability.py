#test_role_suitability

from __future__ import annotations

import pytest

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
from research.player_intelligence.role_suitability import (
    build_role_suitability_signals,
    symmetric_relative_gap,
)


def make_player(
    *,
    player_id: str = "1",
    dm: float | None = None,
    cm: float | None = None,
    am: float | None = None,
) -> Player:
    return Player(
        identity=PlayerIdentity(
            player_id=player_id,
            name=f"Player {player_id}",
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
            DM=dm,
            CM=cm,
            AM=am,
        ),
    )


def test_strongest_role_assignment_has_zero_gap() -> None:
    player = make_player(
        dm=0.90,
        cm=0.80,
    )

    assignment = LineupAssignment(
        slot="DM1",
        tactical_role="DM",
        player=player,
        selection_rating=0.90,
    )

    contribution = build_player_contribution(
        assignment
    )

    signals = build_role_suitability_signals(
        contribution
    )

    assert signals.assigned_role_rank == 1
    assert signals.raw_rating_gap == pytest.approx(
        0.0
    )
    assert (
        signals.symmetric_relative_gap
        == pytest.approx(0.0)
    )
    assert (
        signals.reciprocal_rank_score
        == pytest.approx(1.0)
    )
    assert signals.strongest_role_indicator == 1.0
    assert signals.contextual_adjustment_applied is False


def test_second_best_assignment_has_interpretable_signals() -> None:
    player = make_player(
        dm=0.70,
        cm=0.90,
        am=0.80,
    )

    assignment = LineupAssignment(
        slot="DM1",
        tactical_role="DM",
        player=player,
        selection_rating=0.70,
    )

    signals = build_role_suitability_signals(
        build_player_contribution(
            assignment
        )
    )

    assert signals.assigned_role_rank == 3
    assert signals.raw_rating_gap == pytest.approx(
        0.20
    )
    assert signals.absolute_rating_gap == pytest.approx(
        0.20
    )
    assert (
        signals.symmetric_relative_gap
        == pytest.approx(
            0.20 / 1.60
        )
    )
    assert (
        signals.reciprocal_rank_score
        == pytest.approx(
            1.0 / 3.0
        )
    )
    assert signals.strongest_role_indicator == 0.0
    assert signals.positive_scale_ratio == pytest.approx(
        0.70 / 0.90
    )


def test_negative_ratings_remain_stable() -> None:
    player = make_player(
        dm=-0.20,
        cm=-0.10,
    )

    assignment = LineupAssignment(
        slot="DM1",
        tactical_role="DM",
        player=player,
        selection_rating=-0.20,
    )

    signals = build_role_suitability_signals(
        build_player_contribution(
            assignment
        )
    )

    assert signals.assigned_role_rank == 2
    assert signals.raw_rating_gap == pytest.approx(
        0.10
    )
    assert (
        signals.symmetric_relative_gap
        == pytest.approx(
            0.10 / 0.30
        )
    )
    assert signals.positive_scale_ratio is None


def test_opposite_sign_ratings_produce_bounded_gap() -> None:
    gap = symmetric_relative_gap(
        assigned_rating=-0.10,
        strongest_rating=0.05,
    )

    assert gap == pytest.approx(
        1.0
    )


def test_zero_ratings_produce_zero_gap() -> None:
    gap = symmetric_relative_gap(
        assigned_rating=0.0,
        strongest_rating=0.0,
    )

    assert gap == pytest.approx(
        0.0
    )


def test_missing_assigned_rating_remains_unresolved() -> None:
    player = make_player(
        cm=0.80,
    )

    assignment = LineupAssignment(
        slot="DM1",
        tactical_role="DM",
        player=player,
        selection_rating=0.0,
    )

    signals = build_role_suitability_signals(
        build_player_contribution(
            assignment
        )
    )

    assert signals.assigned_role_rating is None
    assert signals.assigned_role_rank is None
    assert signals.raw_rating_gap is None
    assert signals.symmetric_relative_gap is None
    assert signals.positive_scale_ratio is None
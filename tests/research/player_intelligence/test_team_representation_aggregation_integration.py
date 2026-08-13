# test_team_representation_aggregation_integration.py

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import research.player_intelligence.team_representation_builder as builder_module

from research.player_intelligence.aggregation_functions import (
    power_mean_top_k,
    rank_weighted_top_k,
    softmax_weighted_top_k,
    star_influence_top_k,
    top_k_mean,
)

from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
    build_team_representation_from_players,
    build_team_representation_from_squad,
    build_team_representation_from_starting_xi,
)

from research.studies.study_089_aggregation_mathematics.aggregation_specifications import (
    AggregationSpecification,
    build_aggregation_specifications,
)


def _specification_by_id(
    specification_id: str,
) -> AggregationSpecification:
    return next(
        specification
        for specification in build_aggregation_specifications()
        if specification.specification_id
        == specification_id
    )


def _player(
    *,
    attack: float,
    midfield: float,
    defense: float,
    goalkeeper: float,
    overall: float,
    available: bool = True,
) -> Any:
    """
    Minimal player-like object satisfying the builder contract.

    The role projection functions are patched below to read the
    synthetic role-rating mapping directly.
    """

    return SimpleNamespace(
        role_ratings={
            "attack": attack,
            "midfield": midfield,
            "defense": defense,
            "goalkeeper": goalkeeper,
        },
        ratings=SimpleNamespace(
            overall=overall,
        ),
        availability=SimpleNamespace(
            available=available,
        ),
    )


@pytest.fixture(autouse=True)
def patch_role_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Isolate aggregation integration from role-projection mathematics.
    """

    monkeypatch.setattr(
        builder_module,
        "project_attack",
        lambda ratings: float(
            ratings["attack"]
        ),
    )

    monkeypatch.setattr(
        builder_module,
        "project_midfield",
        lambda ratings: float(
            ratings["midfield"]
        ),
    )

    monkeypatch.setattr(
        builder_module,
        "project_defense",
        lambda ratings: float(
            ratings["defense"]
        ),
    )

    monkeypatch.setattr(
        builder_module,
        "project_goalkeeper",
        lambda ratings: float(
            ratings["goalkeeper"]
        ),
    )


@pytest.fixture
def players() -> tuple[Any, ...]:
    return (
        _player(
            attack=0.95,
            midfield=0.74,
            defense=0.51,
            goalkeeper=0.10,
            overall=0.88,
        ),
        _player(
            attack=0.90,
            midfield=0.79,
            defense=0.56,
            goalkeeper=0.20,
            overall=0.86,
        ),
        _player(
            attack=0.85,
            midfield=0.84,
            defense=0.61,
            goalkeeper=0.30,
            overall=0.84,
        ),
        _player(
            attack=0.80,
            midfield=0.89,
            defense=0.66,
            goalkeeper=0.40,
            overall=0.82,
        ),
        _player(
            attack=0.75,
            midfield=0.94,
            defense=0.71,
            goalkeeper=0.50,
            overall=0.80,
        ),
        _player(
            attack=0.70,
            midfield=0.69,
            defense=0.76,
            goalkeeper=0.60,
            overall=0.78,
        ),
        _player(
            attack=0.65,
            midfield=0.64,
            defense=0.81,
            goalkeeper=0.70,
            overall=0.76,
        ),
        _player(
            attack=0.60,
            midfield=0.59,
            defense=0.86,
            goalkeeper=0.80,
            overall=0.74,
            available=False,
        ),
        _player(
            attack=0.55,
            midfield=0.54,
            defense=0.91,
            goalkeeper=0.90,
            overall=0.72,
        ),
        _player(
            attack=0.50,
            midfield=0.49,
            defense=0.96,
            goalkeeper=1.00,
            overall=0.70,
        ),
    )


def _attack_values(
    players: tuple[Any, ...],
) -> tuple[float, ...]:
    return tuple(
        float(
            player.role_ratings[
                "attack"
            ]
        )
        for player in players
    )


def _midfield_values(
    players: tuple[Any, ...],
) -> tuple[float, ...]:
    return tuple(
        float(
            player.role_ratings[
                "midfield"
            ]
        )
        for player in players
    )


def _defense_values(
    players: tuple[Any, ...],
) -> tuple[float, ...]:
    return tuple(
        float(
            player.role_ratings[
                "defense"
            ]
        )
        for player in players
    )


def _goalkeeper_values(
    players: tuple[Any, ...],
) -> tuple[float, ...]:
    return tuple(
        float(
            player.role_ratings[
                "goalkeeper"
            ]
        )
        for player in players
    )


def _overall_values(
    players: tuple[Any, ...],
) -> tuple[float, ...]:
    return tuple(
        float(
            player.ratings.overall
        )
        for player in players
    )


# ---------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------


def test_legacy_builder_uses_top_five_mean(
    players: tuple[Any, ...],
) -> None:
    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
        )
    )

    assert representation.attack == pytest.approx(
        top_k_mean(
            _attack_values(players),
            k=5,
        )
    )

    assert representation.midfield == pytest.approx(
        top_k_mean(
            _midfield_values(players),
            k=5,
        )
    )

    assert representation.defense == pytest.approx(
        top_k_mean(
            _defense_values(players),
            k=5,
        )
    )


def test_legacy_builder_preserves_default_metadata(
    players: tuple[Any, ...],
) -> None:
    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
        )
    )

    assert (
        representation.national_team
        == "Test Team"
    )

    assert (
        representation.representation_type
        == "full_squad"
    )

    assert (
        representation.aggregation_profile
        == "legacy_top_5"
    )


def test_legacy_depth_fields_remain_full_population_means(
    players: tuple[Any, ...],
) -> None:
    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
        )
    )

    assert representation.attack_depth == pytest.approx(
        sum(
            _attack_values(players)
        ) / len(players)
    )

    assert representation.midfield_depth == pytest.approx(
        sum(
            _midfield_values(players)
        ) / len(players)
    )

    assert representation.defense_depth == pytest.approx(
        sum(
            _defense_values(players)
        ) / len(players)
    )


def test_legacy_goalkeeper_remains_maximum(
    players: tuple[Any, ...],
) -> None:
    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
        )
    )

    assert representation.goalkeeper == pytest.approx(
        max(
            _goalkeeper_values(players)
        )
    )


def test_legacy_squad_quality_remains_overall_mean(
    players: tuple[Any, ...],
) -> None:
    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
        )
    )

    assert representation.squad_quality == pytest.approx(
        sum(
            _overall_values(players)
        ) / len(players)
    )


def test_legacy_evidence_and_player_counts_are_preserved(
    players: tuple[Any, ...],
) -> None:
    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
        )
    )

    assert representation.player_count == 10
    assert representation.available_player_count == 9
    assert representation.evidence_score == pytest.approx(
        1.0
    )


def test_custom_legacy_profile_name_is_preserved_without_specification(
    players: tuple[Any, ...],
) -> None:
    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
            aggregation_profile=(
                "custom_legacy_profile"
            ),
        )
    )

    assert (
        representation.aggregation_profile
        == "custom_legacy_profile"
    )


# ---------------------------------------------------------------------
# Top-five arithmetic equivalence
# ---------------------------------------------------------------------


def test_top5_arithmetic_reproduces_legacy_primary_dimensions(
    players: tuple[Any, ...],
) -> None:
    legacy = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
        )
    )

    specification = _specification_by_id(
        "top5_arithmetic"
    )

    delegated = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
            aggregation_specification=(
                specification
            ),
        )
    )

    assert delegated.attack == pytest.approx(
        legacy.attack
    )

    assert delegated.midfield == pytest.approx(
        legacy.midfield
    )

    assert delegated.defense == pytest.approx(
        legacy.defense
    )


def test_top5_arithmetic_uses_specification_profile_name(
    players: tuple[Any, ...],
) -> None:
    specification = _specification_by_id(
        "top5_arithmetic"
    )

    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
            aggregation_specification=(
                specification
            ),
        )
    )

    assert (
        representation.aggregation_profile
        == "top5_arithmetic"
    )


def test_explicit_specification_overrides_manual_profile_name(
    players: tuple[Any, ...],
) -> None:
    specification = _specification_by_id(
        "top5_arithmetic"
    )

    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
            aggregation_profile=(
                "manual_name_should_not_win"
            ),
            aggregation_specification=(
                specification
            ),
        )
    )

    assert (
        representation.aggregation_profile
        == "top5_arithmetic"
    )


# ---------------------------------------------------------------------
# Alternative primary aggregations
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "specification_id",
        "expected_attack",
        "expected_midfield",
        "expected_defense",
    ),
    [
        (
            "top5_rank_moderate",
            lambda values: rank_weighted_top_k(
                values,
                weights=(
                    0.30,
                    0.25,
                    0.20,
                    0.15,
                    0.10,
                ),
            ),
            lambda values: rank_weighted_top_k(
                values,
                weights=(
                    0.30,
                    0.25,
                    0.20,
                    0.15,
                    0.10,
                ),
            ),
            lambda values: rank_weighted_top_k(
                values,
                weights=(
                    0.30,
                    0.25,
                    0.20,
                    0.15,
                    0.10,
                ),
            ),
        ),
        (
            "top5_star_alpha_0_20",
            lambda values: star_influence_top_k(
                values,
                k=5,
                alpha=0.20,
            ),
            lambda values: star_influence_top_k(
                values,
                k=5,
                alpha=0.20,
            ),
            lambda values: star_influence_top_k(
                values,
                k=5,
                alpha=0.20,
            ),
        ),
        (
            "top5_power_1_50",
            lambda values: power_mean_top_k(
                values,
                k=5,
                power=1.50,
            ),
            lambda values: power_mean_top_k(
                values,
                k=5,
                power=1.50,
            ),
            lambda values: power_mean_top_k(
                values,
                k=5,
                power=1.50,
            ),
        ),
        (
            "top5_softmax_beta_3",
            lambda values: softmax_weighted_top_k(
                values,
                k=5,
                beta=3.0,
            ),
            lambda values: softmax_weighted_top_k(
                values,
                k=5,
                beta=3.0,
            ),
            lambda values: softmax_weighted_top_k(
                values,
                k=5,
                beta=3.0,
            ),
        ),
    ],
)
def test_alternative_specification_matches_frozen_mathematics(
    players: tuple[Any, ...],
    specification_id: str,
    expected_attack: Any,
    expected_midfield: Any,
    expected_defense: Any,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
            aggregation_specification=(
                specification
            ),
        )
    )

    assert representation.attack == pytest.approx(
        expected_attack(
            _attack_values(players)
        )
    )

    assert representation.midfield == pytest.approx(
        expected_midfield(
            _midfield_values(players)
        )
    )

    assert representation.defense == pytest.approx(
        expected_defense(
            _defense_values(players)
        )
    )

    assert (
        representation.aggregation_profile
        == specification_id
    )


@pytest.mark.parametrize(
    "specification_id",
    [
        "top5_rank_moderate",
        "top5_star_alpha_0_20",
        "top5_power_1_50",
        "top5_softmax_beta_3",
    ],
)
def test_alternative_aggregation_changes_only_primary_contract(
    players: tuple[Any, ...],
    specification_id: str,
) -> None:
    legacy = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
        )
    )

    specification = _specification_by_id(
        specification_id
    )

    alternative = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
            aggregation_specification=(
                specification
            ),
        )
    )

    assert alternative.goalkeeper == pytest.approx(
        legacy.goalkeeper
    )

    assert alternative.attack_depth == pytest.approx(
        legacy.attack_depth
    )

    assert alternative.midfield_depth == pytest.approx(
        legacy.midfield_depth
    )

    assert alternative.defense_depth == pytest.approx(
        legacy.defense_depth
    )

    assert alternative.squad_quality == pytest.approx(
        legacy.squad_quality
    )

    assert alternative.evidence_score == pytest.approx(
        legacy.evidence_score
    )

    assert (
        alternative.player_count
        == legacy.player_count
    )

    assert (
        alternative.available_player_count
        == legacy.available_player_count
    )

    assert (
        alternative.representation_type
        == legacy.representation_type
    )

    assert (
        alternative.national_team
        == legacy.national_team
    )


# ---------------------------------------------------------------------
# Strict population behavior
# ---------------------------------------------------------------------


def test_top_five_specification_rejects_four_players() -> None:
    short_population = (
        _player(
            attack=0.90,
            midfield=0.90,
            defense=0.90,
            goalkeeper=0.50,
            overall=0.85,
        ),
        _player(
            attack=0.80,
            midfield=0.80,
            defense=0.80,
            goalkeeper=0.40,
            overall=0.75,
        ),
        _player(
            attack=0.70,
            midfield=0.70,
            defense=0.70,
            goalkeeper=0.30,
            overall=0.65,
        ),
        _player(
            attack=0.60,
            midfield=0.60,
            defense=0.60,
            goalkeeper=0.20,
            overall=0.55,
        ),
    )

    specification = _specification_by_id(
        "top5_arithmetic"
    )

    with pytest.raises(
        ValueError,
        match="required 5, received 4",
    ):
        build_team_representation_from_players(
            national_team="Short Team",
            players=short_population,
            aggregation_specification=(
                specification
            ),
        )


def test_replacement_specification_requires_ten_players() -> None:
    five_players = tuple(
        _player(
            attack=value,
            midfield=value,
            defense=value,
            goalkeeper=value,
            overall=value,
        )
        for value in (
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
        )
    )

    specification = _specification_by_id(
        "replacement_mean_5_5"
    )

    with pytest.raises(
        ValueError,
        match="required 10, received 5",
    ):
        build_team_representation_from_players(
            national_team="Short Team",
            players=five_players,
            aggregation_specification=(
                specification
            ),
        )


def test_replacement_specification_accepts_exactly_ten_players(
    players: tuple[Any, ...],
) -> None:
    specification = _specification_by_id(
        "replacement_mean_5_5"
    )

    representation = (
        build_team_representation_from_players(
            national_team="Test Team",
            players=players,
            aggregation_specification=(
                specification
            ),
        )
    )

    assert isinstance(
        representation,
        TeamRepresentation,
    )

    assert (
        representation.aggregation_profile
        == "replacement_mean_5_5"
    )


# ---------------------------------------------------------------------
# Wrapper propagation
# ---------------------------------------------------------------------


def test_squad_wrapper_preserves_legacy_behavior(
    players: tuple[Any, ...],
) -> None:
    squad = SimpleNamespace(
        national_team="Squad Team",
        players=players,
    )

    direct = (
        build_team_representation_from_players(
            national_team="Squad Team",
            players=players,
            representation_type="full_squad",
            aggregation_profile="legacy_top_5",
        )
    )

    wrapped = (
        build_team_representation_from_squad(
            squad
        )
    )

    assert wrapped == direct


def test_squad_wrapper_passes_specification_through(
    players: tuple[Any, ...],
) -> None:
    squad = SimpleNamespace(
        national_team="Squad Team",
        players=players,
    )

    specification = _specification_by_id(
        "top5_softmax_beta_3"
    )

    direct = (
        build_team_representation_from_players(
            national_team="Squad Team",
            players=players,
            representation_type="full_squad",
            aggregation_profile="legacy_top_5",
            aggregation_specification=(
                specification
            ),
        )
    )

    wrapped = (
        build_team_representation_from_squad(
            squad,
            aggregation_specification=(
                specification
            ),
        )
    )

    assert wrapped == direct


def test_starting_xi_wrapper_preserves_legacy_behavior(
    players: tuple[Any, ...],
) -> None:
    starting_xi = SimpleNamespace(
        national_team="XI Team",
        players=players,
    )

    direct = (
        build_team_representation_from_players(
            national_team="XI Team",
            players=players,
            representation_type=(
                "expected_starting_xi"
            ),
            aggregation_profile="legacy_top_5",
        )
    )

    wrapped = (
        build_team_representation_from_starting_xi(
            starting_xi
        )
    )

    assert wrapped == direct


def test_starting_xi_wrapper_passes_specification_through(
    players: tuple[Any, ...],
) -> None:
    starting_xi = SimpleNamespace(
        national_team="XI Team",
        players=players,
    )

    specification = _specification_by_id(
        "top5_rank_moderate"
    )

    direct = (
        build_team_representation_from_players(
            national_team="XI Team",
            players=players,
            representation_type=(
                "expected_starting_xi"
            ),
            aggregation_profile="legacy_top_5",
            aggregation_specification=(
                specification
            ),
        )
    )

    wrapped = (
        build_team_representation_from_starting_xi(
            starting_xi,
            aggregation_specification=(
                specification
            ),
        )
    )

    assert wrapped == direct


# ---------------------------------------------------------------------
# Empty-population legacy behavior
# ---------------------------------------------------------------------


def test_empty_population_preserves_legacy_zero_behavior() -> None:
    representation = (
        build_team_representation_from_players(
            national_team="Empty Team",
            players=(),
        )
    )

    assert representation.attack == 0.0
    assert representation.midfield == 0.0
    assert representation.defense == 0.0
    assert representation.goalkeeper == 0.0

    assert representation.attack_depth == 0.0
    assert representation.midfield_depth == 0.0
    assert representation.defense_depth == 0.0

    assert representation.squad_quality == 0.0
    assert representation.evidence_score == 0.0

    assert representation.player_count == 0
    assert representation.available_player_count == 0
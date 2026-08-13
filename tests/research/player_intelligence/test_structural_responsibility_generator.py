#test_structural_responsibility_generator

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
    generate_positional_responsibilities,
)
from research.player_intelligence.structural_responsibility_generator import (
    INITIAL_STRUCTURAL_HYPOTHESES,
    StructuralHypothesisScope,
    StructuralHypothesisStatus,
    StructuralRule,
    StructuralUnitType,
    generate_structural_responsibilities,
    hypothesis_scope_matches,
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
        ("FB1", "FB"),
        ("CB1", "CB"),
        ("CB2", "CB"),
        ("FB2", "FB"),
        ("DM1", "DM"),
        ("CM1", "CM"),
        ("CM2", "CM"),
        ("W1", "W"),
        ("W2", "W"),
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

def make_4231_lineup() -> StartingXI:
    specifications = (
        ("FB1", "FB"),
        ("CB1", "CB"),
        ("CB2", "CB"),
        ("FB2", "FB"),
        ("DM1", "DM"),
        ("DM2", "DM"),
        ("W1", "W"),
        ("AM1", "AM"),
        ("W2", "W"),
        ("ST1", "ST"),
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
        formation="4-2-3-1",
        players=players,
        assignments=assignments,
    )

def make_4231_geometry() -> FormationGeometry:
    return FormationGeometry(
        formation="4-2-3-1",
        positions=(
            FormationPosition(
                formation="4-2-3-1",
                slot="FB1",
                role="FB",
                x=-1.8,
                y=1.5,
                tactical_line=1,
                side="left",
            ),
            FormationPosition(
                formation="4-2-3-1",
                slot="CB1",
                role="CB",
                x=-0.6,
                y=1.3,
                tactical_line=1,
                side="left_center",
            ),
            FormationPosition(
                formation="4-2-3-1",
                slot="CB2",
                role="CB",
                x=0.6,
                y=1.3,
                tactical_line=1,
                side="right_center",
            ),
            FormationPosition(
                formation="4-2-3-1",
                slot="FB2",
                role="FB",
                x=1.8,
                y=1.5,
                tactical_line=1,
                side="right",
            ),
            FormationPosition(
                formation="4-2-3-1",
                slot="DM1",
                role="DM",
                x=-0.6,
                y=2.4,
                tactical_line=2,
                side="left_center",
            ),
            FormationPosition(
                formation="4-2-3-1",
                slot="DM2",
                role="DM",
                x=0.6,
                y=2.4,
                tactical_line=2,
                side="right_center",
            ),
            FormationPosition(
                formation="4-2-3-1",
                slot="W1",
                role="W",
                x=-1.9,
                y=3.8,
                tactical_line=3,
                side="left",
            ),
            FormationPosition(
                formation="4-2-3-1",
                slot="AM1",
                role="AM",
                x=0.0,
                y=3.7,
                tactical_line=3,
                side="center",
            ),
            FormationPosition(
                formation="4-2-3-1",
                slot="W2",
                role="W",
                x=1.9,
                y=3.8,
                tactical_line=3,
                side="right",
            ),
            FormationPosition(
                formation="4-2-3-1",
                slot="ST1",
                role="ST",
                x=0.0,
                y=4.7,
                tactical_line=4,
                side="center",
            ),
        ),
    )

def make_4231_positional_set():
    return generate_positional_responsibilities(
        starting_xi=make_4231_lineup(),
        geometry=make_4231_geometry(),
    )

def make_geometry() -> FormationGeometry:
    return FormationGeometry(
        formation="test-formation",
        positions=(
            make_position(
                slot="FB1",
                role="FB",
                tactical_line=1,
                side="left",
                x=-1.8,
                y=1.5,
            ),
            make_position(
                slot="CB1",
                role="CB",
                tactical_line=1,
                side="left_center",
                x=-0.6,
                y=1.3,
            ),
            make_position(
                slot="CB2",
                role="CB",
                tactical_line=1,
                side="right_center",
                x=0.6,
                y=1.3,
            ),
            make_position(
                slot="FB2",
                role="FB",
                tactical_line=1,
                side="right",
                x=1.8,
                y=1.5,
            ),
            make_position(
                slot="DM1",
                role="DM",
                tactical_line=2,
                side="center",
                x=0.0,
                y=2.4,
            ),
            make_position(
                slot="CM1",
                role="CM",
                tactical_line=3,
                side="left",
                x=-0.8,
                y=3.1,
            ),
            make_position(
                slot="CM2",
                role="CM",
                tactical_line=3,
                side="right",
                x=0.8,
                y=3.1,
            ),
            make_position(
                slot="W1",
                role="W",
                tactical_line=4,
                side="left",
                x=-1.9,
                y=4.3,
            ),
            make_position(
                slot="W2",
                role="W",
                tactical_line=4,
                side="right",
                x=1.9,
                y=4.3,
            ),
        ),
    )


def make_positional_set():
    return generate_positional_responsibilities(
        starting_xi=make_lineup(),
        geometry=make_geometry(),
    )

def test_dm_protects_center_backs() -> None:
    result = generate_structural_responsibilities(
        starting_xi=make_lineup(),
        geometry=make_geometry(),
        positional_set=make_positional_set(),
    )

    protections = {
        (
            relationship.source_slot,
            relationship.target_slot,
            relationship.rule_id,
        )
        for relationship
        in result.relationships_of_type(
            ResponsibilityType.PROTECTION
        )
    }

    assert protections == {
        (
            "DM1",
            "CB1",
            "dm_protects_cb_v1",
        ),
        (
            "DM1",
            "CB2",
            "dm_protects_cb_v1",
        ),
    }

def test_hypothesis_rule_ids_are_unique() -> None:
    rule_ids = tuple(
        rule.rule_id
        for rule in INITIAL_STRUCTURAL_HYPOTHESES
    )

    assert len(rule_ids) == len(
        set(rule_ids)
    )

def test_audited_hypothesis_statuses() -> None:
    status_by_rule = {
        rule.rule_id: rule.status
        for rule in INITIAL_STRUCTURAL_HYPOTHESES
    }

    assert status_by_rule == {
        "dm_protects_cb_v1":
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
        "dm_supports_cm_v1":
            StructuralHypothesisStatus.REVISION_REQUIRED,
        "cb_covers_fb_v1":
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
        "cm_supports_w_v1":
            StructuralHypothesisStatus.REVISION_REQUIRED,
        "dm_connects_cb_cm_v1":
            StructuralHypothesisStatus.DEFERRED,
    }

def test_every_hypothesis_has_scope() -> None:
    for hypothesis in (
        INITIAL_STRUCTURAL_HYPOTHESES
    ):
        assert isinstance(
            hypothesis.scope,
            StructuralHypothesisScope,
        )

def test_dm_protection_scope_is_single_pivot_only() -> None:
    hypothesis = next(
        item
        for item in INITIAL_STRUCTURAL_HYPOTHESES
        if item.rule_id == "dm_protects_cb_v1"
    )

    assert (
        hypothesis.scope.supported_formations
        == (
            "4-3-3",
        )
    )

    assert (
        hypothesis.scope
        .requires_single_source_role
        is True
    )

    assert (
        hypothesis.scope
        .supported_source_role_counts
        == (
            1,
        )
    )

def test_cb_coverage_scope_supports_both_formations() -> None:
    hypothesis = next(
        item
        for item in INITIAL_STRUCTURAL_HYPOTHESES
        if item.rule_id == "cb_covers_fb_v1"
    )

    assert (
        hypothesis.scope.supported_formations
        == (
            "4-2-3-1",
            "4-3-3",
        )
    )

    assert (
        hypothesis.scope.formation_general
        is True
    )

def test_single_pivot_dm_rule_does_not_match_4231_scope() -> None:
    hypothesis = next(
        item
        for item in INITIAL_STRUCTURAL_HYPOTHESES
        if item.rule_id == "dm_protects_cb_v1"
    )

    lineup = make_4231_lineup()

    assert (
        hypothesis_scope_matches(
            rule=hypothesis,
            starting_xi=lineup,
        )
        is False
    )

def test_cb_coverage_rule_matches_4231_scope() -> None:
    hypothesis = next(
        item
        for item in INITIAL_STRUCTURAL_HYPOTHESES
        if item.rule_id == "cb_covers_fb_v1"
    )

    lineup = make_4231_lineup()

    assert (
        hypothesis_scope_matches(
            rule=hypothesis,
            starting_xi=lineup,
        )
        is True
    )

def test_formation_general_requires_multiple_formations() -> None:
    with pytest.raises(
        ValueError,
        match="multiple supported formations",
    ):
        StructuralHypothesisScope(
            supported_formations=(
                "4-3-3",
            ),
            supported_source_role_counts=(
                2,
            ),
            supported_target_role_counts=(
                2,
            ),
            unit_type=(
                StructuralUnitType.DEFENSIVE
            ),
            formation_general=True,
        )

def test_active_diagnostic_filter_excludes_other_statuses() -> None:
    result = generate_structural_responsibilities(
        starting_xi=make_lineup(),
        geometry=make_geometry(),
        positional_set=make_positional_set(),
        included_statuses=(
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
        ),
    )

    assert result.responsibilities

    assert {
        relationship.hypothesis_status
        for relationship in result.responsibilities
    } == {
        StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC
    }

    assert {
        relationship.rule_id
        for relationship in result.responsibilities
    }.issubset(
        {
            "dm_protects_cb_v1",
            "cb_covers_fb_v1",
        }
    )

def test_default_generation_retains_all_diagnostic_states() -> None:
    result = generate_structural_responsibilities(
        starting_xi=make_lineup(),
        geometry=make_geometry(),
        positional_set=make_positional_set(),
    )

    statuses = {
        relationship.hypothesis_status
        for relationship in result.responsibilities
    }

    status_by_rule = {
        rule.rule_id: rule.status
        for rule in INITIAL_STRUCTURAL_HYPOTHESES
    }

    for relationship in result.responsibilities:
        assert (
            relationship.hypothesis_status
            == status_by_rule[
                relationship.rule_id
            ]
        )

def test_corridor_flag_requires_same_corridor_evidence() -> None:
    with pytest.raises(
        ValueError,
        match="must cite same-corridor",
    ):
        StructuralRule(
            rule_id="invalid_corridor_rule",
            source_roles=("CB",),
            target_roles=("FB",),
            responsibility_type=(
                ResponsibilityType.COVERAGE
            ),
            required_positional_types=(
                ResponsibilityType.SAME_LINE,
            ),
            status=(
                StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC
            ),
            scope=StructuralHypothesisScope(
                supported_formations=(
                    "4-3-3",
                ),
                supported_source_role_counts=(
                    2,
                ),
                supported_target_role_counts=(
                    2,
                ),
                unit_type=(
                    StructuralUnitType.DEFENSIVE
                ),
            ),
            same_broad_corridor_required=True,
        )

def test_empty_status_filter_generates_nothing() -> None:
    result = generate_structural_responsibilities(
        starting_xi=make_lineup(),
        geometry=make_geometry(),
        positional_set=make_positional_set(),
        included_statuses=(),
    )

    assert result.responsibilities == ()

def test_legacy_generation_retains_unscoped_4231_protection() -> None:
    result = generate_structural_responsibilities(
        starting_xi=make_4231_lineup(),
        geometry=make_4231_geometry(),
        positional_set=make_4231_positional_set(),
        included_statuses=(
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
        ),
        enforce_hypothesis_scope=False,
    )

    protection_pairs = {
        (
            relationship.source_slot,
            relationship.target_slot,
        )
        for relationship in result.responsibilities
        if (
            relationship.rule_id
            == "dm_protects_cb_v1"
        )
    }

    assert protection_pairs == {
        ("DM1", "CB1"),
        ("DM1", "CB2"),
        ("DM2", "CB1"),
        ("DM2", "CB2"),
    }

def test_scoped_generation_excludes_4231_dm_protection() -> None:
    result = generate_structural_responsibilities(
        starting_xi=make_4231_lineup(),
        geometry=make_4231_geometry(),
        positional_set=make_4231_positional_set(),
        included_statuses=(
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
        ),
        enforce_hypothesis_scope=True,
    )

    assert all(
        relationship.rule_id
        != "dm_protects_cb_v1"
        for relationship
        in result.responsibilities
    )

def test_scoped_generation_retains_4231_cb_coverage() -> None:
    result = generate_structural_responsibilities(
        starting_xi=make_4231_lineup(),
        geometry=make_4231_geometry(),
        positional_set=make_4231_positional_set(),
        included_statuses=(
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
        ),
        enforce_hypothesis_scope=True,
    )

    observed = {
        (
            relationship.source_slot,
            relationship.target_slot,
            relationship.rule_id,
        )
        for relationship
        in result.responsibilities
    }

    assert observed == {
        (
            "CB1",
            "FB1",
            "cb_covers_fb_v1",
        ),
        (
            "CB2",
            "FB2",
            "cb_covers_fb_v1",
        ),
    }

def make_433_lineup() -> StartingXI:
    base = make_lineup()

    return StartingXI(
        national_team=base.national_team,
        formation="4-3-3",
        players=base.players,
        assignments=base.assignments,
    )

def make_433_geometry() -> FormationGeometry:
    base = make_geometry()

    return FormationGeometry(
        formation="4-3-3",
        positions=tuple(
            FormationPosition(
                formation="4-3-3",
                slot=position.slot,
                role=position.role,
                x=position.x,
                y=position.y,
                tactical_line=position.tactical_line,
                side=position.side,
            )
            for position in base.positions
        ),
    )

def test_scope_enforcement_preserves_433_active_output() -> None:
    lineup = make_433_lineup()
    geometry = make_433_geometry()

    positional_set = (
        generate_positional_responsibilities(
            starting_xi=lineup,
            geometry=geometry,
        )
    )

    legacy = generate_structural_responsibilities(
        starting_xi=lineup,
        geometry=geometry,
        positional_set=positional_set,
        included_statuses=(
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
        ),
        enforce_hypothesis_scope=False,
    )

    scoped = generate_structural_responsibilities(
        starting_xi=lineup,
        geometry=geometry,
        positional_set=positional_set,
        included_statuses=(
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
        ),
        enforce_hypothesis_scope=True,
    )

    assert legacy == scoped

def test_scope_enforcement_requires_boolean() -> None:
    with pytest.raises(
        TypeError,
        match="must be a boolean",
    ):
        generate_structural_responsibilities(
            starting_xi=make_lineup(),
            geometry=make_geometry(),
            positional_set=make_positional_set(),
            enforce_hypothesis_scope="yes",
        )
from __future__ import annotations

import pytest

from research.player_intelligence.defensive_structural_graph import (
    DefensiveStructuralEdge,
    DefensiveStructuralGraph,
    build_defensive_structural_graph,
)
from research.player_intelligence.football_responsibility import (
    ResponsibilityType,
)
from research.player_intelligence.structural_responsibility_generator import (
    StructuralHypothesisStatus,
    generate_structural_responsibilities,
)

# Replace this import with the actual location of your existing
# make_lineup, make_geometry, and make_positional_set helpers.
from tests.research.player_intelligence.test_structural_responsibility_generator import (
    make_geometry,
    make_lineup,
    make_positional_set,
)


def make_graph() -> DefensiveStructuralGraph:
    lineup = make_lineup()

    structural_set = generate_structural_responsibilities(
        starting_xi=lineup,
        geometry=make_geometry(),
        positional_set=make_positional_set(),
        included_statuses=(
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
        ),
    )

    return build_defensive_structural_graph(
        starting_xi=lineup,
        structural_set=structural_set,
    )


def test_graph_contains_expected_defensive_slots() -> None:
    graph = make_graph()

    assert {
        node.slot
        for node in graph.nodes
    } == {
        "FB1",
        "CB1",
        "CB2",
        "FB2",
        "DM1",
    }


def test_graph_contains_expected_edges() -> None:
    graph = make_graph()

    observed = {
        (
            edge.source_slot,
            edge.target_slot,
            edge.responsibility_type,
            edge.rule_id,
        )
        for edge in graph.edges
    }

    assert observed == {
        (
            "DM1",
            "CB1",
            ResponsibilityType.PROTECTION,
            "dm_protects_cb_v1",
        ),
        (
            "DM1",
            "CB2",
            ResponsibilityType.PROTECTION,
            "dm_protects_cb_v1",
        ),
        (
            "CB1",
            "FB1",
            ResponsibilityType.COVERAGE,
            "cb_covers_fb_v1",
        ),
        (
            "CB2",
            "FB2",
            ResponsibilityType.COVERAGE,
            "cb_covers_fb_v1",
        ),
    }


def test_graph_records_excluded_lineup_slots() -> None:
    graph = make_graph()

    expected = {
        assignment.slot
        for assignment in make_lineup().assignments
    } - {
        "FB1",
        "CB1",
        "CB2",
        "FB2",
        "DM1",
    }

    assert set(
        graph.excluded_lineup_slots
    ) == expected


def test_graph_is_partial_and_unweighted() -> None:
    graph = make_graph()

    assert graph.partial_graph is True
    assert graph.weighted is False


def test_non_active_edge_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="active-diagnostic",
    ):
        DefensiveStructuralEdge(
            source_slot="DM1",
            target_slot="CM1",
            responsibility_type=(
                ResponsibilityType.SUPPORT
            ),
            rule_id="dm_supports_cm_v1",
            hypothesis_status=(
                StructuralHypothesisStatus.REVISION_REQUIRED
            ),
            supporting_positional_types=(
                ResponsibilityType.ADJACENT_LINE,
            ),
        )


def test_graph_generation_is_deterministic() -> None:
    assert make_graph() == make_graph()


def test_graph_preserves_rule_traceability() -> None:
    graph = make_graph()

    assert {
        edge.rule_id
        for edge in graph.edges
    } == {
        "dm_protects_cb_v1",
        "cb_covers_fb_v1",
    }

    assert all(
        edge.supporting_positional_types
        for edge in graph.edges
    )


def test_node_lookup_and_edge_queries() -> None:
    graph = make_graph()

    assert graph.node_by_slot(
        "DM1"
    ).tactical_role == "DM"

    assert len(
        graph.outgoing_edges(
            "DM1"
        )
    ) == 2

    assert len(
        graph.incoming_edges(
            "CB1"
        )
    ) == 1
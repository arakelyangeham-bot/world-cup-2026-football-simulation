#test_defensive_unit_diagnostics

from __future__ import annotations

from research.player_intelligence.defensive_structural_graph import (
    DefensiveStructuralGraph,
    build_defensive_structural_graph,
)
from research.player_intelligence.defensive_unit_diagnostics import (
    build_undirected_adjacency,
    connected_components,
    diagnose_defensive_unit,
)
from research.player_intelligence.structural_responsibility_generator import (
    StructuralHypothesisStatus,
    generate_structural_responsibilities,
)

from tests.research.player_intelligence.test_structural_responsibility_generator import (
    make_geometry,
    make_lineup,
    make_positional_set,
)



def make_graph() -> DefensiveStructuralGraph:
    lineup = make_lineup()

    structural_set = (
        generate_structural_responsibilities(
            starting_xi=lineup,
            geometry=make_geometry(),
            positional_set=make_positional_set(),
            included_statuses=(
                StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
            ),
        )
    )

    return build_defensive_structural_graph(
        starting_xi=lineup,
        structural_set=structural_set,
    )

def test_defensive_graph_is_one_weak_component() -> None:
    graph = make_graph()

    adjacency = build_undirected_adjacency(
        graph
    )

    components = connected_components(
        adjacency
    )

    assert len(components) == 1

    assert set(
        components[0]
    ) == {
        "FB1",
        "CB1",
        "DM1",
        "CB2",
        "FB2",
    }

def test_diagnostic_population_matches_graph() -> None:
    graph = make_graph()

    diagnostics = diagnose_defensive_unit(
        graph
    )

    assert diagnostics.node_count == 5
    assert diagnostics.edge_count == 4
    assert diagnostics.weak_component_count == 1
    assert diagnostics.largest_component_size == 5
    assert diagnostics.isolated_slots == ()

def test_dm_is_an_articulation_point() -> None:
    diagnostics = diagnose_defensive_unit(
        make_graph()
    )

    assert "DM1" in (
        diagnostics.articulation_slots
    )

    dm = diagnostics.diagnostic_by_slot(
        "DM1"
    )

    assert dm.articulation_point is True
    assert (
        dm.component_count_after_removal
        == 2
    )
    assert (
        dm.largest_component_size_after_removal
        == 2
    )

def test_center_backs_are_also_articulation_points() -> None:
    diagnostics = diagnose_defensive_unit(
        make_graph()
    )

    assert {
        "CB1",
        "CB2",
    }.issubset(
        set(
            diagnostics.articulation_slots
        )
    )

def test_fullbacks_are_not_articulation_points() -> None:
    diagnostics = diagnose_defensive_unit(
        make_graph()
    )

    assert (
        diagnostics
        .diagnostic_by_slot(
            "FB1"
        )
        .articulation_point
        is False
    )

    assert (
        diagnostics
        .diagnostic_by_slot(
            "FB2"
        )
        .articulation_point
        is False
    )

def test_degree_diagnostics_are_correct() -> None:
    diagnostics = diagnose_defensive_unit(
        make_graph()
    )

    dm = diagnostics.diagnostic_by_slot(
        "DM1"
    )

    assert dm.out_degree == 2
    assert dm.in_degree == 0
    assert dm.total_degree == 2

    cb1 = diagnostics.diagnostic_by_slot(
        "CB1"
    )

    assert cb1.out_degree == 1
    assert cb1.in_degree == 1
    assert cb1.total_degree == 2

    fb1 = diagnostics.diagnostic_by_slot(
        "FB1"
    )

    assert fb1.out_degree == 0
    assert fb1.in_degree == 1
    assert fb1.total_degree == 1

def test_responsibility_coverage_is_complete() -> None:
    diagnostics = diagnose_defensive_unit(
        make_graph()
    )

    coverage = (
        diagnostics
        .responsibility_coverage
    )

    assert coverage.center_back_slots == (
        "CB1",
        "CB2",
    )

    assert (
        coverage
        .protected_center_back_slots
        == (
            "CB1",
            "CB2",
        )
    )

    assert coverage.fullback_slots == (
        "FB1",
        "FB2",
    )

    assert (
        coverage.covered_fullback_slots
        == (
            "FB1",
            "FB2",
        )
    )

    assert (
        coverage
        .center_back_protection_fraction
        == 1.0
    )

    assert (
        coverage
        .fullback_coverage_fraction
        == 1.0
    )

    assert (
        coverage
        .center_back_protection_balanced
        is True
    )

    assert (
        coverage
        .fullback_coverage_balanced
        is True
    )

def test_diagnostics_are_deterministic() -> None:
    graph = make_graph()

    first = diagnose_defensive_unit(
        graph
    )

    second = diagnose_defensive_unit(
        graph
    )

    assert first == second
#audit_defensive_unit_diagnostics

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research.player_intelligence.defensive_structural_graph import (
    build_defensive_structural_graph,
)
from research.player_intelligence.defensive_unit_diagnostics import (
    diagnose_defensive_unit,
)
from research.player_intelligence.formation_geometry import (
    load_formation_geometry,
)
from research.player_intelligence.positional_responsibility_generator import (
    generate_positional_responsibilities,
)
from research.player_intelligence.starting_xi_builder import (
    StartingXIBuilder,
)
from research.player_intelligence.structural_responsibility_generator import (
    StructuralHypothesisStatus,
    generate_structural_responsibilities,
)
from scripts.build_player_intelligence_team_repository import (
    create_default_roster_builder,
    load_formation,
)
from scripts.wc2026_data import GROUPS
from shared.team_name_normalizer import (
    normalize_team_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

GEOMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "formation_geometry.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_103_structural_responsibility"
    / "study_103c"
)

NODE_DIAGNOSTICS_PATH = (
    OUTPUT_DIRECTORY
    / "world_cup_defensive_node_diagnostics.csv"
)

UNIT_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "world_cup_defensive_unit_summary.csv"
)

ARTICULATION_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "defensive_articulation_summary.csv"
)

COVERAGE_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "defensive_responsibility_coverage_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_103c_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_103C_REPORT.md"
)


FORMATION = "4-3-3"
EXPECTED_TEAM_COUNT = 48
EXPECTED_LINEUP_SLOT_COUNT = 11
EXPECTED_GRAPH_NODE_COUNT = 5
EXPECTED_GRAPH_EDGE_COUNT = 4

EXPECTED_ARTICULATION_SLOTS = (
    "CB1",
    "CB2",
    "DM1",
)

WORLD_CUP_TEAMS = {
    normalize_team_name(team)
    for group in GROUPS.values()
    for team in group
}

def build_diagnostic_population() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    geometry = load_formation_geometry(
        path=GEOMETRY_PATH,
        formation=FORMATION,
    )

    formation_df = load_formation(
        formation=FORMATION
    )

    roster_builder = (
        create_default_roster_builder()
    )

    lineup_builder = StartingXIBuilder(
        formation=FORMATION
    )

    node_rows: list[
        dict[str, Any]
    ] = []

    unit_rows: list[
        dict[str, Any]
    ] = []

    coverage_rows: list[
        dict[str, Any]
    ] = []

    for source_team in (
        roster_builder.list_teams()
    ):
        nation = normalize_team_name(
            source_team
        )

        if nation not in WORLD_CUP_TEAMS:
            continue

        squad = roster_builder.get_squad(
            source_team
        )

        starting_xi = (
            lineup_builder.build_for_squad(
                squad=squad,
                formation_df=formation_df,
            )
        )

        if len(
            starting_xi.assignments
        ) != EXPECTED_LINEUP_SLOT_COUNT:
            raise AssertionError(
                f"{nation!r} has an unexpected lineup "
                f"assignment count: "
                f"{len(starting_xi.assignments)}."
            )

        positional_set = (
            generate_positional_responsibilities(
                starting_xi=starting_xi,
                geometry=geometry,
            )
        )

        structural_set = (
            generate_structural_responsibilities(
                starting_xi=starting_xi,
                geometry=geometry,
                positional_set=positional_set,
                included_statuses=(
                    StructuralHypothesisStatus
                    .ACTIVE_DIAGNOSTIC,
                ),
            )
        )

        graph = (
            build_defensive_structural_graph(
                starting_xi=starting_xi,
                structural_set=structural_set,
            )
        )

        diagnostics = (
            diagnose_defensive_unit(
                graph
            )
        )

        assignments_by_slot = {
            assignment.slot:
                assignment
            for assignment
            in starting_xi.assignments
        }

        for node_diagnostic in (
            diagnostics.node_diagnostics
        ):
            assignment = (
                assignments_by_slot[
                    node_diagnostic.slot
                ]
            )

            node_rows.append(
                {
                    "nation": nation,
                    "formation": FORMATION,
                    "slot":
                        node_diagnostic.slot,
                    "tactical_role":
                        node_diagnostic
                        .tactical_role,
                    "player_id":
                        assignment
                        .player.identity.player_id,
                    "player_name":
                        assignment
                        .player.identity.name,
                    "out_degree":
                        node_diagnostic.out_degree,
                    "in_degree":
                        node_diagnostic.in_degree,
                    "total_degree":
                        node_diagnostic.total_degree,
                    "articulation_point":
                        node_diagnostic
                        .articulation_point,
                    "component_count_after_removal":
                        node_diagnostic
                        .component_count_after_removal,
                    "largest_component_size_after_removal":
                        node_diagnostic
                        .largest_component_size_after_removal,
                }
            )

        coverage = (
            diagnostics
            .responsibility_coverage
        )

        articulation_slots = tuple(
            diagnostics.articulation_slots
        )

        unit_rows.append(
            {
                "nation": nation,
                "formation": FORMATION,
                "lineup_slot_count":
                    len(
                        starting_xi.assignments
                    ),
                "graph_node_count":
                    diagnostics.node_count,
                "graph_edge_count":
                    diagnostics.edge_count,
                "weak_component_count":
                    diagnostics
                    .weak_component_count,
                "largest_component_size":
                    diagnostics
                    .largest_component_size,
                "isolated_slot_count":
                    len(
                        diagnostics
                        .isolated_slots
                    ),
                "isolated_slots":
                    "|".join(
                        diagnostics
                        .isolated_slots
                    ),
                "articulation_slot_count":
                    len(
                        articulation_slots
                    ),
                "articulation_slots":
                    "|".join(
                        articulation_slots
                    ),
                "partial_graph":
                    diagnostics.partial_graph,
                "weighted":
                    diagnostics.weighted,
            }
        )

        coverage_rows.append(
            {
                "nation": nation,
                "formation": FORMATION,
                "center_back_slot_count":
                    len(
                        coverage
                        .center_back_slots
                    ),
                "protected_center_back_count":
                    len(
                        coverage
                        .protected_center_back_slots
                    ),
                "center_back_slots":
                    "|".join(
                        coverage
                        .center_back_slots
                    ),
                "protected_center_back_slots":
                    "|".join(
                        coverage
                        .protected_center_back_slots
                    ),
                "center_back_protection_fraction":
                    coverage
                    .center_back_protection_fraction,
                "center_back_protection_balanced":
                    coverage
                    .center_back_protection_balanced,
                "fullback_slot_count":
                    len(
                        coverage
                        .fullback_slots
                    ),
                "covered_fullback_count":
                    len(
                        coverage
                        .covered_fullback_slots
                    ),
                "fullback_slots":
                    "|".join(
                        coverage
                        .fullback_slots
                    ),
                "covered_fullback_slots":
                    "|".join(
                        coverage
                        .covered_fullback_slots
                    ),
                "fullback_coverage_fraction":
                    coverage
                    .fullback_coverage_fraction,
                "fullback_coverage_balanced":
                    coverage
                    .fullback_coverage_balanced,
            }
        )

    node_frame = (
        pd.DataFrame(
            node_rows
        )
        .sort_values(
            [
                "nation",
                "slot",
            ]
        )
        .reset_index(drop=True)
    )

    unit_frame = (
        pd.DataFrame(
            unit_rows
        )
        .sort_values(
            "nation"
        )
        .reset_index(drop=True)
    )

    coverage_frame = (
        pd.DataFrame(
            coverage_rows
        )
        .sort_values(
            "nation"
        )
        .reset_index(drop=True)
    )

    return (
        node_frame,
        unit_frame,
        coverage_frame,
    )

def validate_population(
    *,
    node_frame: pd.DataFrame,
    unit_frame: pd.DataFrame,
    coverage_frame: pd.DataFrame,
) -> None:
    if (
        node_frame.empty
        or unit_frame.empty
        or coverage_frame.empty
    ):
        raise AssertionError(
            "At least one Study 103C output is empty."
        )

    observed_teams = set(
        unit_frame[
            "nation"
        ].astype(str)
    )

    if observed_teams != WORLD_CUP_TEAMS:
        missing = sorted(
            WORLD_CUP_TEAMS
            - observed_teams
        )

        extra = sorted(
            observed_teams
            - WORLD_CUP_TEAMS
        )

        raise AssertionError(
            "Defensive diagnostic population does not "
            "match the World Cup field. "
            f"Missing={missing}, extra={extra}"
        )

    if len(
        observed_teams
    ) != EXPECTED_TEAM_COUNT:
        raise AssertionError(
            "Unexpected diagnosed team count."
        )

    if not unit_frame[
        "lineup_slot_count"
    ].eq(
        EXPECTED_LINEUP_SLOT_COUNT
    ).all():
        raise AssertionError(
            "At least one team does not account for "
            "all 11 lineup slots."
        )

    if not unit_frame[
        "graph_node_count"
    ].eq(
        EXPECTED_GRAPH_NODE_COUNT
    ).all():
        raise AssertionError(
            "At least one defensive graph does not "
            "contain five nodes."
        )

    if not unit_frame[
        "graph_edge_count"
    ].eq(
        EXPECTED_GRAPH_EDGE_COUNT
    ).all():
        raise AssertionError(
            "At least one defensive graph does not "
            "contain four edges."
        )

    if not unit_frame[
        "weak_component_count"
    ].eq(1).all():
        raise AssertionError(
            "At least one defensive graph is not "
            "weakly connected."
        )

    if not unit_frame[
        "largest_component_size"
    ].eq(
        EXPECTED_GRAPH_NODE_COUNT
    ).all():
        raise AssertionError(
            "At least one graph has an unexpected "
            "largest component."
        )

    if not unit_frame[
        "isolated_slot_count"
    ].eq(0).all():
        raise AssertionError(
            "At least one defensive graph contains "
            "an isolated included node."
        )

    expected_articulation_text = "|".join(
        EXPECTED_ARTICULATION_SLOTS
    )

    if not unit_frame[
        "articulation_slots"
    ].eq(
        expected_articulation_text
    ).all():
        raise AssertionError(
            "At least one team has an unexpected "
            "articulation-slot population."
        )

    if not unit_frame[
        "articulation_slot_count"
    ].eq(
        len(
            EXPECTED_ARTICULATION_SLOTS
        )
    ).all():
        raise AssertionError(
            "Unexpected articulation-slot count."
        )

    if not unit_frame[
        "partial_graph"
    ].all():
        raise AssertionError(
            "At least one defensive graph is not "
            "marked partial."
        )

    if unit_frame[
        "weighted"
    ].any():
        raise AssertionError(
            "At least one defensive graph is weighted."
        )

    per_team_node_count = (
        node_frame
        .groupby(
            "nation"
        )
        .size()
    )

    if not per_team_node_count.eq(
        EXPECTED_GRAPH_NODE_COUNT
    ).all():
        raise AssertionError(
            "Node-diagnostic rows do not reconcile "
            "with graph node counts."
        )

    if not coverage_frame[
        "center_back_slot_count"
    ].eq(2).all():
        raise AssertionError(
            "At least one team has an unexpected "
            "center-back population."
        )

    if not coverage_frame[
        "protected_center_back_count"
    ].eq(2).all():
        raise AssertionError(
            "At least one center-back is unprotected."
        )

    if not coverage_frame[
        "center_back_protection_fraction"
    ].eq(1.0).all():
        raise AssertionError(
            "Center-back protection coverage is incomplete."
        )

    if not coverage_frame[
        "center_back_protection_balanced"
    ].all():
        raise AssertionError(
            "Center-back protection is not balanced."
        )

    if not coverage_frame[
        "fullback_slot_count"
    ].eq(2).all():
        raise AssertionError(
            "At least one team has an unexpected "
            "fullback population."
        )

    if not coverage_frame[
        "covered_fullback_count"
    ].eq(2).all():
        raise AssertionError(
            "At least one fullback is uncovered."
        )

    if not coverage_frame[
        "fullback_coverage_fraction"
    ].eq(1.0).all():
        raise AssertionError(
            "Fullback coverage is incomplete."
        )

    if not coverage_frame[
        "fullback_coverage_balanced"
    ].all():
        raise AssertionError(
            "Fullback coverage is not balanced."
        )

    topology_columns = [
        "graph_node_count",
        "graph_edge_count",
        "weak_component_count",
        "largest_component_size",
        "isolated_slot_count",
        "articulation_slot_count",
        "articulation_slots",
        "partial_graph",
        "weighted",
    ]

    if unit_frame[
        topology_columns
    ].drop_duplicates().shape[0] != 1:
        raise AssertionError(
            "Equal formations produced different "
            "defensive topology diagnostics."
        )

    node_topology_columns = [
        "slot",
        "tactical_role",
        "out_degree",
        "in_degree",
        "total_degree",
        "articulation_point",
        "component_count_after_removal",
        "largest_component_size_after_removal",
    ]

    distinct_slot_topologies = (
        node_frame[
            node_topology_columns
        ]
        .drop_duplicates()
    )

    if len(
        distinct_slot_topologies
    ) != EXPECTED_GRAPH_NODE_COUNT:
        raise AssertionError(
            "Player identities affected slot-level "
            "structural diagnostics."
        )

def build_articulation_summary(
    node_frame: pd.DataFrame,
) -> pd.DataFrame:
    return (
        node_frame
        .groupby(
            [
                "slot",
                "tactical_role",
            ],
            as_index=False,
        )
        .agg(
            team_count=(
                "nation",
                "nunique",
            ),
            articulation_team_count=(
                "articulation_point",
                "sum",
            ),
            mean_out_degree=(
                "out_degree",
                "mean",
            ),
            mean_in_degree=(
                "in_degree",
                "mean",
            ),
            mean_total_degree=(
                "total_degree",
                "mean",
            ),
            mean_component_count_after_removal=(
                "component_count_after_removal",
                "mean",
            ),
            mean_largest_component_after_removal=(
                "largest_component_size_after_removal",
                "mean",
            ),
        )
        .sort_values(
            "slot"
        )
        .reset_index(drop=True)
    )

def write_report(
    *,
    unit_frame: pd.DataFrame,
    articulation_summary: pd.DataFrame,
    coverage_frame: pd.DataFrame,
) -> None:
    topology_summary = (
        unit_frame[
            [
                "graph_node_count",
                "graph_edge_count",
                "weak_component_count",
                "largest_component_size",
                "isolated_slot_count",
                "articulation_slot_count",
                "articulation_slots",
                "partial_graph",
                "weighted",
            ]
        ]
        .drop_duplicates()
    )

    coverage_summary = (
        coverage_frame[
            [
                "center_back_slot_count",
                "protected_center_back_count",
                "center_back_protection_fraction",
                "center_back_protection_balanced",
                "fullback_slot_count",
                "covered_fullback_count",
                "fullback_coverage_fraction",
                "fullback_coverage_balanced",
            ]
        ]
        .drop_duplicates()
    )

    report = f"""# Study 103C — Defensive Unit Diagnostics Audit

## Status

**PASS**

## Purpose

Validate the first topology-based interpretation of the active
4-3-3 defensive structural subgraph across all 48 World Cup teams.

## Topology summary

{topology_summary.to_markdown(index=False)}

## Node and articulation summary

{articulation_summary.to_markdown(index=False)}

## Responsibility coverage summary

{coverage_summary.to_markdown(index=False)}

## Permitted conclusions

- The active defensive subgraph is weakly connected.
- The subgraph contains five included nodes and four edges.
- CB1, CB2, and DM1 are articulation points.
- Both center-backs receive protection under the retained hypothesis.
- Both fullbacks receive coverage under the retained hypothesis.
- Equal formation structures produce equal topology diagnostics.

## Prohibited conclusions

- The teams have equal defensive quality.
- The 4-3-3 is defensively superior.
- The graph predicts goals conceded.
- The retained hypotheses are empirically validated.
- Player quality is irrelevant to real defensive performance.

## Interpretation boundary

These diagnostics describe one partial, unweighted, expert-defined
defensive structural subgraph. They do not measure actual defensive
ability, chemistry, tactical execution, or match performance.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 103C — DEFENSIVE UNIT "
        "DIAGNOSTICS AUDIT"
    )
    print("=" * 88)

    (
        node_frame,
        unit_frame,
        coverage_frame,
    ) = build_diagnostic_population()

    validate_population(
        node_frame=node_frame,
        unit_frame=unit_frame,
        coverage_frame=coverage_frame,
    )

    articulation_summary = (
        build_articulation_summary(
            node_frame
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    node_frame.to_csv(
        NODE_DIAGNOSTICS_PATH,
        index=False,
    )

    unit_frame.to_csv(
        UNIT_SUMMARY_PATH,
        index=False,
    )

    articulation_summary.to_csv(
        ARTICULATION_SUMMARY_PATH,
        index=False,
    )

    coverage_frame.to_csv(
        COVERAGE_SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "study_id": "103C",
        "study_name": (
            "Defensive Unit Diagnostics Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "formation": FORMATION,
        "team_count": int(
            unit_frame[
                "nation"
            ].nunique()
        ),
        "lineup_slots_per_team":
            EXPECTED_LINEUP_SLOT_COUNT,
        "graph_nodes_per_team":
            EXPECTED_GRAPH_NODE_COUNT,
        "graph_edges_per_team":
            EXPECTED_GRAPH_EDGE_COUNT,
        "weak_components_per_team": 1,
        "largest_component_size":
            EXPECTED_GRAPH_NODE_COUNT,
        "isolated_defensive_nodes_per_team": 0,
        "articulation_slots": list(
            EXPECTED_ARTICULATION_SLOTS
        ),
        "center_back_protection_fraction": 1.0,
        "fullback_coverage_fraction": 1.0,
        "diagnostics_identical_across_equal_formations":
            True,
        "player_identity_affects_topology":
            False,
        "graph_modified": False,
        "weights_created": False,
        "team_strength_changed": False,
        "repository_changed": False,
        "simulation_run": False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "Diagnostics describe one partial, unweighted, "
            "expert-defined defensive subgraph. They do not "
            "measure real defensive quality or predictive value."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        unit_frame=unit_frame,
        articulation_summary=(
            articulation_summary
        ),
        coverage_frame=coverage_frame,
    )

    print()
    print("Diagnostic audit summary")
    print("-" * 88)
    print(
        f"  Teams diagnosed: "
        f"{metadata['team_count']}"
    )
    print(
        "  Defensive nodes per team: "
        f"{metadata['graph_nodes_per_team']}"
    )
    print(
        "  Structural edges per team: "
        f"{metadata['graph_edges_per_team']}"
    )
    print(
        "  Weak components per team: 1"
    )
    print(
        "  Isolated defensive nodes per team: 0"
    )
    print(
        "  Articulation slots: "
        + "|".join(
            EXPECTED_ARTICULATION_SLOTS
        )
    )
    print(
        "  Center-back protection complete: PASS"
    )
    print(
        "  Fullback coverage complete: PASS"
    )
    print(
        "  Equal-formation diagnostic parity: PASS"
    )
    print(
        "  Player identity topology independence: PASS"
    )
    print(
        "  Graph modified: NO"
    )
    print(
        "  Weights created: NO"
    )
    print(
        "  Team strength changed: NO"
    )
    print(
        "  Simulation run: NO"
    )

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
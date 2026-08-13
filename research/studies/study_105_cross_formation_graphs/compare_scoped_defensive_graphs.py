#compare_scoped_defensive_graphs

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research.player_intelligence.defensive_structural_graph import (
    build_defensive_structural_graph,
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
    / "study_105_cross_formation_graphs"
    / "study_105a"
)

NODE_PATH = (
    OUTPUT_DIRECTORY
    / "scoped_defensive_graph_nodes.csv"
)

EDGE_PATH = (
    OUTPUT_DIRECTORY
    / "scoped_defensive_graph_edges.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "scoped_defensive_graph_summary.csv"
)

COMPONENT_PATH = (
    OUTPUT_DIRECTORY
    / "scoped_defensive_graph_components.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_105a_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_105A_REPORT.md"
)


FORMATIONS = (
    "4-3-3",
    "4-2-3-1",
)

EXPECTED = {
    "4-3-3": {
        "included_nodes": 5,
        "excluded_slots": 6,
        "edges": 4,
        "components": 1,
        "largest_component": 5,
        "isolated_included_nodes": 0,
    },
    "4-2-3-1": {
        "included_nodes": 4,
        "excluded_slots": 7,
        "edges": 2,
        "components": 2,
        "largest_component": 2,
        "isolated_included_nodes": 0,
    },
}

def find_compatible_squad(
    *,
    roster_builder,
    formation_frames: dict[str, pd.DataFrame],
) -> tuple[str, object]:
    for team in roster_builder.list_teams():
        squad = roster_builder.get_squad(
            team
        )

        compatible = True

        for formation in FORMATIONS:
            builder = StartingXIBuilder(
                formation=formation
            )

            try:
                builder.build_for_squad(
                    squad=squad,
                    formation_df=(
                        formation_frames[
                            formation
                        ]
                    ),
                )

            except ValueError:
                compatible = False
                break

        if compatible:
            return team, squad

    raise AssertionError(
        "No squad could populate both formations."
    )

def build_adjacency(
    *,
    node_slots: tuple[str, ...],
    edges,
) -> dict[str, set[str]]:
    adjacency = {
        slot: set()
        for slot in node_slots
    }

    for edge in edges:
        adjacency[
            edge.source_slot
        ].add(
            edge.target_slot
        )

        adjacency[
            edge.target_slot
        ].add(
            edge.source_slot
        )

    return adjacency

def connected_components(
    adjacency: dict[str, set[str]],
) -> tuple[tuple[str, ...], ...]:
    unvisited = set(
        adjacency
    )

    components: list[
        tuple[str, ...]
    ] = []

    while unvisited:
        start = min(
            unvisited
        )

        queue: deque[str] = deque(
            [start]
        )

        visited: set[str] = set()

        while queue:
            current = queue.popleft()

            if current in visited:
                continue

            visited.add(
                current
            )

            for neighbor in sorted(
                adjacency[current]
            ):
                if neighbor not in visited:
                    queue.append(
                        neighbor
                    )

        unvisited -= visited

        components.append(
            tuple(
                sorted(
                    visited
                )
            )
        )

    return tuple(
        sorted(
            components,
            key=lambda component: (
                -len(component),
                component,
            ),
        )
    )

def build_graph_population() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    roster_builder = (
        create_default_roster_builder()
    )

    formation_frames = {
        formation: load_formation(
            formation=formation
        )
        for formation in FORMATIONS
    }

    source_team, squad = find_compatible_squad(
        roster_builder=roster_builder,
        formation_frames=formation_frames,
    )

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for formation in FORMATIONS:
        geometry = load_formation_geometry(
            path=GEOMETRY_PATH,
            formation=formation,
        )

        lineup_builder = StartingXIBuilder(
            formation=formation
        )

        starting_xi = (
            lineup_builder.build_for_squad(
                squad=squad,
                formation_df=(
                    formation_frames[
                        formation
                    ]
                ),
            )
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
                enforce_hypothesis_scope=True,
            )
        )

        graph = (
            build_defensive_structural_graph(
                starting_xi=starting_xi,
                structural_set=structural_set,
            )
        )

        included_slots = tuple(
            node.slot
            for node in graph.nodes
        )

        adjacency = build_adjacency(
            node_slots=included_slots,
            edges=graph.edges,
        )

        components = connected_components(
            adjacency
        )

        isolated_included = tuple(
            sorted(
                slot
                for slot, neighbors
                in adjacency.items()
                if not neighbors
            )
        )

        for node in graph.nodes:
            node_rows.append(
                {
                    "source_team": source_team,
                    "formation": formation,
                    "slot": node.slot,
                    "tactical_role":
                        node.tactical_role,
                    "player_id":
                        node.player_id,
                    "player_name":
                        node.player_name,
                    "included_in_graph": True,
                }
            )

        assignments_by_slot = {
            assignment.slot:
                assignment
            for assignment
            in starting_xi.assignments
        }

        for slot in graph.excluded_lineup_slots:
            assignment = assignments_by_slot[
                slot
            ]

            node_rows.append(
                {
                    "source_team": source_team,
                    "formation": formation,
                    "slot": slot,
                    "tactical_role":
                        assignment.tactical_role,
                    "player_id":
                        assignment
                        .player.identity.player_id,
                    "player_name":
                        assignment
                        .player.identity.name,
                    "included_in_graph": False,
                }
            )

        for edge in graph.edges:
            edge_rows.append(
                {
                    "source_team": source_team,
                    "formation": formation,
                    "source_slot":
                        edge.source_slot,
                    "target_slot":
                        edge.target_slot,
                    "responsibility_type":
                        edge
                        .responsibility_type.value,
                    "rule_id":
                        edge.rule_id,
                    "hypothesis_status":
                        edge
                        .hypothesis_status.value,
                    "supporting_positional_types":
                        "|".join(
                            item.value
                            for item
                            in edge
                            .supporting_positional_types
                        ),
                    "directional": True,
                    "weighted": False,
                }
            )

        for index, component in enumerate(
            components,
            start=1,
        ):
            component_rows.append(
                {
                    "source_team": source_team,
                    "formation": formation,
                    "component_id": index,
                    "component_size":
                        len(component),
                    "component_slots":
                        "|".join(component),
                }
            )

        summary_rows.append(
            {
                "source_team": source_team,
                "formation": formation,
                "lineup_slot_count":
                    len(
                        starting_xi.assignments
                    ),
                "included_node_count":
                    len(graph.nodes),
                "excluded_slot_count":
                    len(
                        graph.excluded_lineup_slots
                    ),
                "edge_count":
                    len(graph.edges),
                "weak_component_count":
                    len(components),
                "largest_component_size":
                    max(
                        len(component)
                        for component
                        in components
                    ),
                "isolated_included_node_count":
                    len(isolated_included),
                "isolated_included_slots":
                    "|".join(
                        isolated_included
                    ),
                "partial_graph":
                    graph.partial_graph,
                "weighted":
                    graph.weighted,
            }
        )

    return (
        pd.DataFrame(node_rows),
        pd.DataFrame(edge_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(component_rows),
    )

def validate_graph_population(
    *,
    node_frame: pd.DataFrame,
    edge_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    component_frame: pd.DataFrame,
) -> None:
    if (
        node_frame.empty
        or edge_frame.empty
        or summary_frame.empty
        or component_frame.empty
    ):
        raise AssertionError(
            "At least one graph output is empty."
        )

    if set(
        summary_frame[
            "formation"
        ].astype(str)
    ) != set(FORMATIONS):
        raise AssertionError(
            "Graph summary does not contain both formations."
        )

    for formation in FORMATIONS:
        expected = EXPECTED[
            formation
        ]

        row = summary_frame.loc[
            summary_frame[
                "formation"
            ].eq(
                formation
            )
        ].iloc[0]

        if int(
            row["lineup_slot_count"]
        ) != 11:
            raise AssertionError(
                f"{formation} does not account for 11 slots."
            )

        checks = {
            "included_node_count":
                "included_nodes",
            "excluded_slot_count":
                "excluded_slots",
            "edge_count":
                "edges",
            "weak_component_count":
                "components",
            "largest_component_size":
                "largest_component",
            "isolated_included_node_count":
                "isolated_included_nodes",
        }

        for observed_column, expected_key in (
            checks.items()
        ):
            if int(
                row[observed_column]
            ) != int(
                expected[expected_key]
            ):
                raise AssertionError(
                    f"{formation} has unexpected "
                    f"{observed_column}: "
                    f"{row[observed_column]}."
                )

        if not bool(
            row["partial_graph"]
        ):
            raise AssertionError(
                f"{formation} graph is not marked partial."
            )

        if bool(
            row["weighted"]
        ):
            raise AssertionError(
                f"{formation} graph is unexpectedly weighted."
            )

    scoped_433_rules = set(
        edge_frame.loc[
            edge_frame[
                "formation"
            ].eq(
                "4-3-3"
            ),
            "rule_id",
        ].astype(str)
    )

    if scoped_433_rules != {
        "dm_protects_cb_v1",
        "cb_covers_fb_v1",
    }:
        raise AssertionError(
            "Unexpected scoped 4-3-3 defensive rules."
        )

    scoped_4231_rules = set(
        edge_frame.loc[
            edge_frame[
                "formation"
            ].eq(
                "4-2-3-1"
            ),
            "rule_id",
        ].astype(str)
    )

    if scoped_4231_rules != {
        "cb_covers_fb_v1",
    }:
        raise AssertionError(
            "Unexpected scoped 4-2-3-1 defensive rules."
        )

    if edge_frame[
        "weighted"
    ].any():
        raise AssertionError(
            "At least one graph edge is weighted."
        )

    if not edge_frame[
        "directional"
    ].all():
        raise AssertionError(
            "At least one graph edge is not directional."
        )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 105A — CROSS-FORMATION "
        "SCOPED DEFENSIVE GRAPHS"
    )
    print("=" * 88)

    (
        node_frame,
        edge_frame,
        summary_frame,
        component_frame,
    ) = build_graph_population()

    validate_graph_population(
        node_frame=node_frame,
        edge_frame=edge_frame,
        summary_frame=summary_frame,
        component_frame=component_frame,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    node_frame.to_csv(
        NODE_PATH,
        index=False,
    )

    edge_frame.to_csv(
        EDGE_PATH,
        index=False,
    )

    summary_frame.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    component_frame.to_csv(
        COMPONENT_PATH,
        index=False,
    )

    metadata = {
        "study_id": "105A",
        "study_name": (
            "Cross-Formation Scoped Defensive Graphs"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "formations_compared": list(
            FORMATIONS
        ),
        "scope_enforced": True,
        "formation_433": EXPECTED[
            "4-3-3"
        ],
        "formation_4231": EXPECTED[
            "4-2-3-1"
        ],
        "double_pivot_hypothesis_promoted":
            False,
        "graphs_partial": True,
        "graphs_weighted": False,
        "team_strength_changed": False,
        "repository_changed": False,
        "simulation_run": False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "The graphs contain only active hypotheses "
            "whose declared scope matches each formation. "
            "Graph completeness does not establish tactical "
            "quality or predictive value."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = f"""# Study 105A — Cross-Formation Scoped Defensive Graphs

## Status

**PASS**

## Graph summary

{summary_frame.to_markdown(index=False)}

## Connected components

{component_frame.to_markdown(index=False)}

## Main result

The scoped 4-3-3 defensive graph forms one connected five-node unit.

The scoped 4-2-3-1 defensive graph contains two disconnected
center-back/fullback pairs because no double-pivot protection
hypothesis has been promoted.

## Interpretation boundary

The sparse 4-2-3-1 graph is not evidence that the formation is
defensively inferior. It records the current limits of the accepted
hypothesis register.

No weights, team-strength adjustments, repositories, or simulations
were created or changed.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    summary_by_formation = (
        summary_frame
        .set_index("formation")
    )

    row_433 = summary_by_formation.loc[
        "4-3-3"
    ]

    row_4231 = summary_by_formation.loc[
        "4-2-3-1"
    ]

    print()
    print("Scoped graph summary")
    print("-" * 88)
    print(
        "  4-3-3 included nodes: "
        f"{int(row_433['included_node_count'])}"
    )
    print(
        "  4-3-3 edges: "
        f"{int(row_433['edge_count'])}"
    )
    print(
        "  4-3-3 weak components: "
        f"{int(row_433['weak_component_count'])}"
    )
    print(
        "  4-2-3-1 included nodes: "
        f"{int(row_4231['included_node_count'])}"
    )
    print(
        "  4-2-3-1 edges: "
        f"{int(row_4231['edge_count'])}"
    )
    print(
        "  4-2-3-1 weak components: "
        f"{int(row_4231['weak_component_count'])}"
    )
    print(
        "  Scope enforcement active: PASS"
    )
    print(
        "  Double-pivot hypothesis promoted: NO"
    )
    print(
        "  Graphs partial: PASS"
    )
    print(
        "  Graphs unweighted: PASS"
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
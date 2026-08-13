#compare_double_pivot_hypotheses

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from research.player_intelligence.formation_geometry import (
    FormationGeometry,
    load_formation_geometry,
)
from research.player_intelligence.starting_xi_builder import (
    StartingXIBuilder,
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
    / "study_105b"
)

RELATIONSHIP_PATH = (
    OUTPUT_DIRECTORY
    / "double_pivot_candidate_relationships.csv"
)

GRAPH_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "double_pivot_candidate_graph_summary.csv"
)

COMPONENT_PATH = (
    OUTPUT_DIRECTORY
    / "double_pivot_candidate_components.csv"
)

NODE_DIAGNOSTIC_PATH = (
    OUTPUT_DIRECTORY
    / "double_pivot_candidate_node_diagnostics.csv"
)

DIFFERENCE_PATH = (
    OUTPUT_DIRECTORY
    / "double_pivot_hypothesis_difference_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_105b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_105B_REPORT.md"
)


FORMATION = "4-2-3-1"

DEFENSIVE_SLOTS = (
    "FB1",
    "CB1",
    "DM1",
    "DM2",
    "CB2",
    "FB2",
)

CENTER_BACK_SLOTS = (
    "CB1",
    "CB2",
)

FULLBACK_SLOTS = (
    "FB1",
    "FB2",
)

DOUBLE_PIVOT_SLOTS = (
    "DM1",
    "DM2",
)

class CandidateResearchStatus(str, Enum):
    """
    Research state for a candidate football hypothesis that has not
    entered the canonical structural hypothesis register.
    """

    DIAGNOSTIC_CANDIDATE = "diagnostic_candidate"
    EMPIRICAL_VALIDATION_REQUIRED = (
        "empirical_validation_required"
    )


@dataclass(frozen=True)
class CandidateRelationship:
    source_slot: str
    target_slot: str

    responsibility_type: str
    candidate_rule_id: str

    positional_basis: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_slot.strip():
            raise ValueError(
                "Candidate source slot must not be empty."
            )

        if not self.target_slot.strip():
            raise ValueError(
                "Candidate target slot must not be empty."
            )

        if self.source_slot == self.target_slot:
            raise ValueError(
                "Candidate relationship cannot be a self-edge."
            )

        if not self.candidate_rule_id.strip():
            raise ValueError(
                "Candidate rule ID must not be empty."
            )

        if not self.positional_basis:
            raise ValueError(
                "Candidate relationship must preserve its "
                "positional basis."
            )

    @property
    def canonical_key(
        self,
    ) -> tuple[str, str, str, str]:
        return (
            self.source_slot,
            self.target_slot,
            self.responsibility_type,
            self.candidate_rule_id,
        )


@dataclass(frozen=True)
class CompetingStructuralHypothesis:
    hypothesis_id: str
    name: str
    description: str

    formation: str

    relationships: tuple[
        CandidateRelationship,
        ...,
    ]

    research_status: CandidateResearchStatus

    empirically_validated: bool = False
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError(
                "Candidate hypothesis ID must not be empty."
            )

        if not self.name.strip():
            raise ValueError(
                "Candidate hypothesis name must not be empty."
            )

        if not self.description.strip():
            raise ValueError(
                "Candidate hypothesis description must not be empty."
            )

        if self.formation != FORMATION:
            raise ValueError(
                "Study 105B candidates must target 4-2-3-1."
            )

        if not self.relationships:
            raise ValueError(
                "Candidate hypothesis must contain relationships."
            )

        relationship_keys = tuple(
            relationship.canonical_key
            for relationship
            in self.relationships
        )

        if len(relationship_keys) != len(
            set(relationship_keys)
        ):
            raise ValueError(
                "Candidate hypothesis contains duplicate "
                "relationships."
            )

        if relationship_keys != tuple(
            sorted(relationship_keys)
        ):
            raise ValueError(
                "Candidate relationships must use deterministic "
                "ordering."
            )

        if self.empirically_validated:
            raise ValueError(
                "Study 105B candidates cannot be marked "
                "empirically validated."
            )

        if self.production_eligible:
            raise ValueError(
                "Study 105B candidates cannot be marked "
                "production eligible."
            )

SHARED_SCREEN_HYPOTHESIS = CompetingStructuralHypothesis(
    hypothesis_id="double_pivot_shared_screen_v1",
    name="Shared double-pivot screen",
    description=(
        "Both defensive midfielders share responsibility for "
        "protecting both center-backs."
    ),
    formation=FORMATION,
    relationships=tuple(
        sorted(
            (
                CandidateRelationship(
                    source_slot="DM1",
                    target_slot="CB1",
                    responsibility_type="protection",
                    candidate_rule_id=(
                        "double_pivot_shared_screen_v1"
                    ),
                    positional_basis=(
                        "adjacent_line",
                    ),
                ),
                CandidateRelationship(
                    source_slot="DM1",
                    target_slot="CB2",
                    responsibility_type="protection",
                    candidate_rule_id=(
                        "double_pivot_shared_screen_v1"
                    ),
                    positional_basis=(
                        "adjacent_line",
                    ),
                ),
                CandidateRelationship(
                    source_slot="DM2",
                    target_slot="CB1",
                    responsibility_type="protection",
                    candidate_rule_id=(
                        "double_pivot_shared_screen_v1"
                    ),
                    positional_basis=(
                        "adjacent_line",
                    ),
                ),
                CandidateRelationship(
                    source_slot="DM2",
                    target_slot="CB2",
                    responsibility_type="protection",
                    candidate_rule_id=(
                        "double_pivot_shared_screen_v1"
                    ),
                    positional_basis=(
                        "adjacent_line",
                    ),
                ),
            ),
            key=lambda relationship:
                relationship.canonical_key,
        )
    ),
    research_status=(
        CandidateResearchStatus
        .EMPIRICAL_VALIDATION_REQUIRED
    ),
)


SIDE_SPECIFIC_HYPOTHESIS = CompetingStructuralHypothesis(
    hypothesis_id="double_pivot_side_specific_v1",
    name="Side-specific double-pivot protection",
    description=(
        "Each defensive midfielder protects the center-back "
        "occupying the same broad corridor."
    ),
    formation=FORMATION,
    relationships=tuple(
        sorted(
            (
                CandidateRelationship(
                    source_slot="DM1",
                    target_slot="CB1",
                    responsibility_type="protection",
                    candidate_rule_id=(
                        "double_pivot_side_specific_v1"
                    ),
                    positional_basis=(
                        "adjacent_line",
                        "same_corridor",
                    ),
                ),
                CandidateRelationship(
                    source_slot="DM2",
                    target_slot="CB2",
                    responsibility_type="protection",
                    candidate_rule_id=(
                        "double_pivot_side_specific_v1"
                    ),
                    positional_basis=(
                        "adjacent_line",
                        "same_corridor",
                    ),
                ),
            ),
            key=lambda relationship:
                relationship.canonical_key,
        )
    ),
    research_status=(
        CandidateResearchStatus
        .EMPIRICAL_VALIDATION_REQUIRED
    ),
)


CANDIDATE_HYPOTHESES = (
    SHARED_SCREEN_HYPOTHESIS,
    SIDE_SPECIFIC_HYPOTHESIS,
)

FORMATION_GENERAL_COVERAGE = tuple(
    sorted(
        (
            CandidateRelationship(
                source_slot="CB1",
                target_slot="FB1",
                responsibility_type="coverage",
                candidate_rule_id="cb_covers_fb_v1",
                positional_basis=(
                    "same_line",
                    "same_corridor",
                ),
            ),
            CandidateRelationship(
                source_slot="CB2",
                target_slot="FB2",
                responsibility_type="coverage",
                candidate_rule_id="cb_covers_fb_v1",
                positional_basis=(
                    "same_line",
                    "same_corridor",
                ),
            ),
        ),
        key=lambda relationship:
            relationship.canonical_key,
    )
)

def build_adjacency(
    *,
    node_slots: tuple[str, ...],
    relationships: tuple[
        CandidateRelationship,
        ...,
    ],
) -> dict[str, set[str]]:
    adjacency = {
        slot: set()
        for slot in node_slots
    }

    for relationship in relationships:
        if relationship.source_slot not in adjacency:
            raise AssertionError(
                "Candidate relationship references an unknown "
                f"source slot: {relationship.source_slot!r}."
            )

        if relationship.target_slot not in adjacency:
            raise AssertionError(
                "Candidate relationship references an unknown "
                f"target slot: {relationship.target_slot!r}."
            )

        adjacency[
            relationship.source_slot
        ].add(
            relationship.target_slot
        )

        adjacency[
            relationship.target_slot
        ].add(
            relationship.source_slot
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

def adjacency_without_slot(
    *,
    adjacency: dict[str, set[str]],
    removed_slot: str,
) -> dict[str, set[str]]:
    return {
        slot: {
            neighbor
            for neighbor in neighbors
            if neighbor != removed_slot
        }
        for slot, neighbors
        in adjacency.items()
        if slot != removed_slot
    }

def articulation_slots(
    adjacency: dict[str, set[str]],
) -> tuple[str, ...]:
    baseline_component_count = len(
        connected_components(
            adjacency
        )
    )

    articulation: list[str] = []

    for slot in sorted(
        adjacency
    ):
        reduced = adjacency_without_slot(
            adjacency=adjacency,
            removed_slot=slot,
        )

        if not reduced:
            continue

        reduced_component_count = len(
            connected_components(
                reduced
            )
        )

        if (
            reduced_component_count
            > baseline_component_count
        ):
            articulation.append(
                slot
            )

    return tuple(
        articulation
    )

def validate_candidate_geometry(
    geometry: FormationGeometry,
) -> None:
    positions_by_slot = {
        position.slot: position
        for position in geometry.positions
    }

    missing_slots = sorted(
        set(DEFENSIVE_SLOTS)
        - set(positions_by_slot)
    )

    if missing_slots:
        raise AssertionError(
            "Registered 4-2-3-1 geometry is missing "
            f"candidate defensive slots: {missing_slots}"
        )

    for relationship in (
        SHARED_SCREEN_HYPOTHESIS.relationships
    ):
        source = positions_by_slot[
            relationship.source_slot
        ]

        target = positions_by_slot[
            relationship.target_slot
        ]

        if (
            source.tactical_line
            - target.tactical_line
        ) != 1:
            raise AssertionError(
                "Shared-screen relationship does not connect "
                "adjacent tactical lines."
            )

    expected_side_pairs = {
        ("DM1", "CB1"),
        ("DM2", "CB2"),
    }

    observed_side_pairs = {
        (
            relationship.source_slot,
            relationship.target_slot,
        )
        for relationship
        in SIDE_SPECIFIC_HYPOTHESIS.relationships
    }

    if observed_side_pairs != expected_side_pairs:
        raise AssertionError(
            "Side-specific candidate contains unexpected pairs."
        )

    for relationship in (
        SIDE_SPECIFIC_HYPOTHESIS.relationships
    ):
        source = positions_by_slot[
            relationship.source_slot
        ]

        target = positions_by_slot[
            relationship.target_slot
        ]

        source_side = str(
            source.side
        ).replace(
            "_center",
            "",
        )

        target_side = str(
            target.side
        ).replace(
            "_center",
            "",
        )

        if source_side != target_side:
            raise AssertionError(
                "Side-specific candidate pairs do not occupy "
                "the same broad side."
            )

def find_compatible_squad(
    *,
    roster_builder,
    formation_frame: pd.DataFrame,
) -> tuple[str, object]:
    lineup_builder = StartingXIBuilder(
        formation=FORMATION
    )

    for team in roster_builder.list_teams():
        squad = roster_builder.get_squad(
            team
        )

        try:
            lineup_builder.build_for_squad(
                squad=squad,
                formation_df=formation_frame,
            )

        except ValueError:
            continue

        return team, squad

    raise AssertionError(
        "No squad could populate the registered 4-2-3-1."
    )

def build_comparison_population() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    geometry = load_formation_geometry(
        path=GEOMETRY_PATH,
        formation=FORMATION,
    )

    validate_candidate_geometry(
        geometry
    )

    roster_builder = (
        create_default_roster_builder()
    )

    formation_frame = load_formation(
        formation=FORMATION
    )

    source_team, squad = find_compatible_squad(
        roster_builder=roster_builder,
        formation_frame=formation_frame,
    )

    lineup_builder = StartingXIBuilder(
        formation=FORMATION
    )

    starting_xi = (
        lineup_builder.build_for_squad(
            squad=squad,
            formation_df=formation_frame,
        )
    )

    assignments_by_slot = {
        assignment.slot:
            assignment
        for assignment
        in starting_xi.assignments
    }

    relationship_rows: list[
        dict[str, Any]
    ] = []

    graph_rows: list[
        dict[str, Any]
    ] = []

    component_rows: list[
        dict[str, Any]
    ] = []

    node_rows: list[
        dict[str, Any]
    ] = []

    for hypothesis in CANDIDATE_HYPOTHESES:
        relationships = tuple(
            sorted(
                (
                    *hypothesis.relationships,
                    *FORMATION_GENERAL_COVERAGE,
                ),
                key=lambda relationship:
                    relationship.canonical_key,
            )
        )

        adjacency = build_adjacency(
            node_slots=DEFENSIVE_SLOTS,
            relationships=relationships,
        )

        components = connected_components(
            adjacency
        )

        candidate_articulation = (
            articulation_slots(
                adjacency
            )
        )

        for relationship in relationships:
            relationship_rows.append(
                {
                    "source_team": source_team,
                    "formation": FORMATION,
                    "hypothesis_id":
                        hypothesis.hypothesis_id,
                    "hypothesis_name":
                        hypothesis.name,
                    "research_status":
                        hypothesis
                        .research_status.value,
                    "source_slot":
                        relationship.source_slot,
                    "target_slot":
                        relationship.target_slot,
                    "responsibility_type":
                        relationship
                        .responsibility_type,
                    "candidate_rule_id":
                        relationship
                        .candidate_rule_id,
                    "positional_basis":
                        "|".join(
                            relationship
                            .positional_basis
                        ),
                    "empirically_validated":
                        False,
                    "production_eligible":
                        False,
                }
            )

        for component_id, component in enumerate(
            components,
            start=1,
        ):
            component_rows.append(
                {
                    "source_team": source_team,
                    "formation": FORMATION,
                    "hypothesis_id":
                        hypothesis.hypothesis_id,
                    "component_id":
                        component_id,
                    "component_size":
                        len(component),
                    "component_slots":
                        "|".join(component),
                }
            )

        for slot in DEFENSIVE_SLOTS:
            assignment = assignments_by_slot[
                slot
            ]

            outgoing = sum(
                relationship.source_slot
                == slot
                for relationship
                in relationships
            )

            incoming = sum(
                relationship.target_slot
                == slot
                for relationship
                in relationships
            )

            node_rows.append(
                {
                    "source_team": source_team,
                    "formation": FORMATION,
                    "hypothesis_id":
                        hypothesis.hypothesis_id,
                    "slot": slot,
                    "tactical_role":
                        assignment.tactical_role,
                    "player_id":
                        assignment
                        .player.identity.player_id,
                    "player_name":
                        assignment
                        .player.identity.name,
                    "out_degree":
                        int(outgoing),
                    "in_degree":
                        int(incoming),
                    "total_degree":
                        int(
                            outgoing + incoming
                        ),
                    "articulation_point":
                        slot
                        in candidate_articulation,
                }
            )

        protected_center_backs = {
            relationship.target_slot
            for relationship in relationships
            if (
                relationship
                .responsibility_type
                == "protection"
                and relationship.target_slot
                in CENTER_BACK_SLOTS
            )
        }

        covered_fullbacks = {
            relationship.target_slot
            for relationship in relationships
            if (
                relationship
                .responsibility_type
                == "coverage"
                and relationship.target_slot
                in FULLBACK_SLOTS
            )
        }

        graph_rows.append(
            {
                "source_team": source_team,
                "formation": FORMATION,
                "hypothesis_id":
                    hypothesis.hypothesis_id,
                "hypothesis_name":
                    hypothesis.name,
                "relationship_count":
                    len(relationships),
                "protection_edge_count":
                    sum(
                        relationship
                        .responsibility_type
                        == "protection"
                        for relationship
                        in relationships
                    ),
                "coverage_edge_count":
                    sum(
                        relationship
                        .responsibility_type
                        == "coverage"
                        for relationship
                        in relationships
                    ),
                "node_count":
                    len(DEFENSIVE_SLOTS),
                "weak_component_count":
                    len(components),
                "largest_component_size":
                    max(
                        len(component)
                        for component
                        in components
                    ),
                "isolated_node_count":
                    sum(
                        not adjacency[slot]
                        for slot in adjacency
                    ),
                "articulation_slot_count":
                    len(
                        candidate_articulation
                    ),
                "articulation_slots":
                    "|".join(
                        candidate_articulation
                    ),
                "protected_center_back_count":
                    len(
                        protected_center_backs
                    ),
                "center_back_protection_fraction":
                    len(
                        protected_center_backs
                    )
                    / len(
                        CENTER_BACK_SLOTS
                    ),
                "covered_fullback_count":
                    len(
                        covered_fullbacks
                    ),
                "fullback_coverage_fraction":
                    len(
                        covered_fullbacks
                    )
                    / len(
                        FULLBACK_SLOTS
                    ),
                "empirically_validated":
                    False,
                "production_eligible":
                    False,
            }
        )

    return (
        pd.DataFrame(
            relationship_rows
        ),
        pd.DataFrame(
            graph_rows
        ),
        pd.DataFrame(
            component_rows
        ),
        pd.DataFrame(
            node_rows
        ),
    )

def build_difference_summary(
    graph_frame: pd.DataFrame,
) -> pd.DataFrame:
    indexed = graph_frame.set_index(
        "hypothesis_id"
    )

    shared = indexed.loc[
        "double_pivot_shared_screen_v1"
    ]

    side_specific = indexed.loc[
        "double_pivot_side_specific_v1"
    ]

    rows = []

    for metric in (
        "relationship_count",
        "protection_edge_count",
        "coverage_edge_count",
        "node_count",
        "weak_component_count",
        "largest_component_size",
        "isolated_node_count",
        "articulation_slot_count",
        "protected_center_back_count",
        "center_back_protection_fraction",
        "covered_fullback_count",
        "fullback_coverage_fraction",
    ):
        shared_value = shared[
            metric
        ]

        side_value = side_specific[
            metric
        ]

        rows.append(
            {
                "metric": metric,
                "shared_screen_value":
                    shared_value,
                "side_specific_value":
                    side_value,
                "side_specific_minus_shared":
                    float(
                        side_value
                    )
                    - float(
                        shared_value
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )

def validate_comparison(
    *,
    relationship_frame: pd.DataFrame,
    graph_frame: pd.DataFrame,
    component_frame: pd.DataFrame,
    node_frame: pd.DataFrame,
) -> None:
    if (
        relationship_frame.empty
        or graph_frame.empty
        or component_frame.empty
        or node_frame.empty
    ):
        raise AssertionError(
            "At least one Study 105B output is empty."
        )

    if set(
        graph_frame[
            "hypothesis_id"
        ].astype(str)
    ) != {
        "double_pivot_shared_screen_v1",
        "double_pivot_side_specific_v1",
    }:
        raise AssertionError(
            "Candidate hypothesis population is invalid."
        )

    graph_by_hypothesis = (
        graph_frame
        .set_index(
            "hypothesis_id"
        )
    )

    shared = graph_by_hypothesis.loc[
        "double_pivot_shared_screen_v1"
    ]

    side_specific = graph_by_hypothesis.loc[
        "double_pivot_side_specific_v1"
    ]

    if int(
        shared["relationship_count"]
    ) != 6:
        raise AssertionError(
            "Shared-screen graph must contain six relationships."
        )

    if int(
        shared["protection_edge_count"]
    ) != 4:
        raise AssertionError(
            "Shared-screen graph must contain four "
            "protection edges."
        )

    if int(
        side_specific["relationship_count"]
    ) != 4:
        raise AssertionError(
            "Side-specific graph must contain four relationships."
        )

    if int(
        side_specific["protection_edge_count"]
    ) != 2:
        raise AssertionError(
            "Side-specific graph must contain two "
            "protection edges."
        )

    for hypothesis_id, row in (
        graph_by_hypothesis.iterrows()
    ):
        if int(
            row["node_count"]
        ) != 6:
            raise AssertionError(
                f"{hypothesis_id} must contain six nodes."
            )

        if int(
            row["isolated_node_count"]
        ) != 0:
            raise AssertionError(
                f"{hypothesis_id} contains an isolated node."
            )

        if float(
            row[
                "center_back_protection_fraction"
            ]
        ) != 1.0:
            raise AssertionError(
                f"{hypothesis_id} does not protect both "
                "center-backs."
            )

        if float(
            row[
                "fullback_coverage_fraction"
            ]
        ) != 1.0:
            raise AssertionError(
                f"{hypothesis_id} does not cover both fullbacks."
            )

    if int(
        shared["weak_component_count"]
    ) != 1:
        raise AssertionError(
            "Shared-screen graph must be weakly connected."
        )

    if int(
        shared["largest_component_size"]
    ) != 6:
        raise AssertionError(
            "Shared-screen graph must contain one six-node "
            "component."
        )

    if int(
        side_specific["weak_component_count"]
    ) != 2:
        raise AssertionError(
            "Side-specific graph must contain two weak "
            "components."
        )

    if int(
        side_specific["largest_component_size"]
    ) != 3:
        raise AssertionError(
            "Side-specific graph must contain two three-node "
            "components."
        )
    
    if relationship_frame[
        "empirically_validated"
    ].any():
        raise AssertionError(
            "A candidate relationship was incorrectly marked "
            "empirically validated."
        )

    if relationship_frame[
        "production_eligible"
    ].any():
        raise AssertionError(
            "A candidate relationship was incorrectly marked "
            "production eligible."
        )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 105B — COMPETING DOUBLE-PIVOT "
        "HYPOTHESES"
    )
    print("=" * 88)

    (
        relationship_frame,
        graph_frame,
        component_frame,
        node_frame,
    ) = build_comparison_population()

    validate_comparison(
        relationship_frame=relationship_frame,
        graph_frame=graph_frame,
        component_frame=component_frame,
        node_frame=node_frame,
    )

    difference_frame = (
        build_difference_summary(
            graph_frame
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    relationship_frame.to_csv(
        RELATIONSHIP_PATH,
        index=False,
    )

    graph_frame.to_csv(
        GRAPH_SUMMARY_PATH,
        index=False,
    )

    component_frame.to_csv(
        COMPONENT_PATH,
        index=False,
    )

    node_frame.to_csv(
        NODE_DIAGNOSTIC_PATH,
        index=False,
    )

    difference_frame.to_csv(
        DIFFERENCE_PATH,
        index=False,
    )

    metadata = {
        "study_id": "105B",
        "study_name": (
            "Competing Double-Pivot Hypotheses"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "formation": FORMATION,
        "candidate_hypothesis_count": 2,
        "candidate_hypotheses": [
            hypothesis.hypothesis_id
            for hypothesis
            in CANDIDATE_HYPOTHESES
        ],
        "shared_screen_protection_edges": 4,
        "side_specific_protection_edges": 2,
        "shared_screen_total_edges": 6,
        "side_specific_total_edges": 4,
        "shared_screen_connected": True,
        "side_specific_connected": False,
        "side_specific_component_count": 2,
        "side_specific_largest_component_size": 3,
        "both_center_back_protection_complete":
            True,
        "both_fullback_coverage_complete":
            True,
        "candidate_promoted": False,
        "canonical_hypothesis_register_changed":
            False,
        "scope_register_changed": False,
        "empirical_validation_performed":
            False,
        "football_graph_production_changed":
            False,
        "weights_created": False,
        "team_strength_changed": False,
        "repository_changed": False,
        "simulation_run": False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "This study formalizes and compares two expert "
            "double-pivot hypotheses. It does not determine "
            "which hypothesis better represents real football."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = f"""# Study 105B — Competing Double-Pivot Hypotheses

## Status

**PASS**

## Purpose

Formalize and compare two candidate explanations of center-back
protection in a 4-2-3-1 double pivot.

## Candidate graph comparison

{graph_frame.to_markdown(index=False)}

## Candidate relationships

{relationship_frame.to_markdown(index=False)}

## Topological differences

{difference_frame.to_markdown(index=False)}

## Interpretation

The shared-screen hypothesis creates four DM-to-CB protection edges.
The side-specific hypothesis creates two DM-to-CB protection edges.

After the accepted CB-to-FB coverage relationships are included, both
candidate defensive subnetworks:

- contain six defensive nodes;
- protect both center-backs;
- cover both fullbacks;
- contain no isolated included nodes.

However, their topology differs:

- The shared-screen candidate forms one connected six-node component.
- The side-specific candidate forms two disconnected three-node
  components: FB1-CB1-DM1 and FB2-CB2-DM2.

The candidates therefore differ not only in redundancy and degree,
but also in whether the left and right defensive structures are
connected.

The candidates therefore differ in responsibility redundancy and node
degree, not basic coverage completeness.

## Research disposition

Both hypotheses remain viable diagnostic candidates.

Neither candidate has sufficient empirical evidence to be promoted
into the canonical structural hypothesis register.

Future club-football validation must compare their explanatory value
against observed defensive organization.

## Production boundary

No canonical hypothesis, scope declaration, graph implementation,
team representation, prediction model, or simulation behavior changed.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    summary_by_hypothesis = (
        graph_frame
        .set_index(
            "hypothesis_id"
        )
    )

    shared = summary_by_hypothesis.loc[
        "double_pivot_shared_screen_v1"
    ]

    side_specific = summary_by_hypothesis.loc[
        "double_pivot_side_specific_v1"
    ]

    print()
    print("Candidate comparison summary")
    print("-" * 88)
    print(
        "  Candidate hypotheses: 2"
    )
    print(
        "  Shared-screen protection edges: "
        f"{int(shared['protection_edge_count'])}"
    )
    print(
        "  Side-specific protection edges: "
        f"{int(side_specific['protection_edge_count'])}"
    )
    print(
        "  Shared-screen total edges: "
        f"{int(shared['relationship_count'])}"
    )
    print(
        "  Side-specific total edges: "
        f"{int(side_specific['relationship_count'])}"
    )
    print(
        "  Shared-screen connected: PASS"
    )
    print(
        "  Side-specific weak components: 2"
    )
    print(
        "  Side-specific largest component: 3"
    )
    print(
        "  Center-back protection complete in both: PASS"
    )
    print(
        "  Fullback coverage complete in both: PASS"
    )
    print(
        "  Candidate promoted: NO"
    )
    print(
        "  Canonical hypothesis register changed: NO"
    )
    print(
        "  Empirical validation performed: NO"
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
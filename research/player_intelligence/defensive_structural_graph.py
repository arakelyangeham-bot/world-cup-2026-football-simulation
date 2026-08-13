#defensive_structural_graph

from __future__ import annotations

from dataclasses import dataclass

from research.player_intelligence.football_responsibility import (
    ResponsibilityType,
)
from research.player_intelligence.player_schema import (
    StartingXI,
)
from research.player_intelligence.structural_responsibility_generator import (
    StructuralHypothesisStatus,
    StructuralResponsibility,
    StructuralResponsibilitySet,
)


@dataclass(frozen=True)
class DefensiveStructuralNode:
    """
    One lineup slot participating in the active defensive
    structural subnetwork.

    The node stores assignment identity only. It contains no
    interaction weight or team-strength adjustment.
    """

    slot: str
    tactical_role: str

    player_id: str
    player_name: str

    def __post_init__(self) -> None:
        if not self.slot.strip():
            raise ValueError(
                "Defensive graph node slot must not be empty."
            )

        if not self.tactical_role.strip():
            raise ValueError(
                "Defensive graph node role must not be empty."
            )

        if not self.player_id.strip():
            raise ValueError(
                "Defensive graph node player ID must not be empty."
            )

        if not self.player_name.strip():
            raise ValueError(
                "Defensive graph node player name must not be empty."
            )

    @property
    def canonical_key(
        self,
    ) -> tuple[str, str]:
        return (
            self.slot,
            self.player_id,
        )


@dataclass(frozen=True)
class DefensiveStructuralEdge:
    """
    One active-diagnostic directional structural relationship.
    """

    source_slot: str
    target_slot: str

    responsibility_type: ResponsibilityType
    rule_id: str

    hypothesis_status: StructuralHypothesisStatus

    supporting_positional_types: tuple[
        ResponsibilityType,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.source_slot.strip():
            raise ValueError(
                "Defensive graph edge source slot must not be empty."
            )

        if not self.target_slot.strip():
            raise ValueError(
                "Defensive graph edge target slot must not be empty."
            )

        if self.source_slot == self.target_slot:
            raise ValueError(
                "Defensive graph edge cannot be a self-edge."
            )

        if (
            self.hypothesis_status
            != StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC
        ):
            raise ValueError(
                "Defensive graph edges must come from "
                "active-diagnostic hypotheses."
            )

        if self.responsibility_type not in {
            ResponsibilityType.PROTECTION,
            ResponsibilityType.COVERAGE,
        }:
            raise ValueError(
                "Defensive graph supports protection and "
                "coverage responsibilities only."
            )

        if not self.rule_id.strip():
            raise ValueError(
                "Defensive graph edge must preserve its rule ID."
            )

        if not self.supporting_positional_types:
            raise ValueError(
                "Defensive graph edge must preserve positional "
                "evidence."
            )

    @property
    def canonical_key(
        self,
    ) -> tuple[str, str, str, str]:
        return (
            self.source_slot,
            self.target_slot,
            self.responsibility_type.value,
            self.rule_id,
        )


@dataclass(frozen=True)
class DefensiveStructuralGraph:
    """
    Immutable partial graph describing the active defensive
    structural subnetwork of one starting XI.

    This object intentionally excludes lineup slots not connected by
    active-diagnostic defensive hypotheses.
    """

    national_team: str
    formation: str

    nodes: tuple[
        DefensiveStructuralNode,
        ...,
    ]

    edges: tuple[
        DefensiveStructuralEdge,
        ...,
    ]

    excluded_lineup_slots: tuple[
        str,
        ...,
    ]

    partial_graph: bool = True
    weighted: bool = False

    def __post_init__(self) -> None:
        if not self.national_team.strip():
            raise ValueError(
                "Defensive graph team must not be empty."
            )

        if not self.formation.strip():
            raise ValueError(
                "Defensive graph formation must not be empty."
            )

        if not self.nodes:
            raise ValueError(
                "Defensive graph must contain nodes."
            )

        if not self.edges:
            raise ValueError(
                "Defensive graph must contain edges."
            )

        node_slots = tuple(
            node.slot
            for node in self.nodes
        )

        if len(node_slots) != len(
            set(node_slots)
        ):
            raise ValueError(
                "Defensive graph contains duplicate node slots."
            )

        node_keys = tuple(
            node.canonical_key
            for node in self.nodes
        )

        if node_keys != tuple(
            sorted(node_keys)
        ):
            raise ValueError(
                "Defensive graph nodes must use deterministic order."
            )

        edge_keys = tuple(
            edge.canonical_key
            for edge in self.edges
        )

        if len(edge_keys) != len(
            set(edge_keys)
        ):
            raise ValueError(
                "Defensive graph contains duplicate edges."
            )

        if edge_keys != tuple(
            sorted(edge_keys)
        ):
            raise ValueError(
                "Defensive graph edges must use deterministic order."
            )

        valid_slots = set(
            node_slots
        )

        for edge in self.edges:
            if edge.source_slot not in valid_slots:
                raise ValueError(
                    "Defensive graph edge references an unknown "
                    f"source slot: {edge.source_slot!r}."
                )

            if edge.target_slot not in valid_slots:
                raise ValueError(
                    "Defensive graph edge references an unknown "
                    f"target slot: {edge.target_slot!r}."
                )

        excluded = tuple(
            self.excluded_lineup_slots
        )

        if len(excluded) != len(
            set(excluded)
        ):
            raise ValueError(
                "Excluded lineup slots contain duplicates."
            )

        if set(excluded) & valid_slots:
            raise ValueError(
                "A lineup slot cannot be both included and excluded."
            )

        if not self.partial_graph:
            raise ValueError(
                "Study 103B graph must remain explicitly partial."
            )

        if self.weighted:
            raise ValueError(
                "Study 103B graph must remain unweighted."
            )

    def node_by_slot(
        self,
        slot: str,
    ) -> DefensiveStructuralNode:
        for node in self.nodes:
            if node.slot == slot:
                return node

        raise KeyError(
            f"Unknown defensive graph slot: {slot!r}"
        )

    def outgoing_edges(
        self,
        slot: str,
    ) -> tuple[
        DefensiveStructuralEdge,
        ...,
    ]:
        return tuple(
            edge
            for edge in self.edges
            if edge.source_slot == slot
        )

    def incoming_edges(
        self,
        slot: str,
    ) -> tuple[
        DefensiveStructuralEdge,
        ...,
    ]:
        return tuple(
            edge
            for edge in self.edges
            if edge.target_slot == slot
        )

def validate_active_defensive_responsibility(
    responsibility: StructuralResponsibility,
) -> None:
    if (
        responsibility.hypothesis_status
        != StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC
    ):
        raise ValueError(
            "Defensive graph received a non-active hypothesis: "
            f"{responsibility.rule_id!r}."
        )

    if responsibility.responsibility_type not in {
        ResponsibilityType.PROTECTION,
        ResponsibilityType.COVERAGE,
    }:
        raise ValueError(
            "Defensive graph received a non-defensive "
            f"responsibility: "
            f"{responsibility.responsibility_type.value!r}."
        )


def build_defensive_structural_graph(
    *,
    starting_xi: StartingXI,
    structural_set: StructuralResponsibilitySet,
) -> DefensiveStructuralGraph:
    """
    Construct the active defensive structural subgraph.

    The builder does not generate responsibilities. It only promotes
    already-audited active defensive relationships into graph form.
    """

    if not starting_xi.assignments:
        raise ValueError(
            "Defensive graph construction requires preserved "
            "lineup assignments."
        )

    if (
        structural_set.national_team
        != starting_xi.national_team
    ):
        raise ValueError(
            "Structural responsibility team does not match "
            "the starting XI."
        )

    if (
        structural_set.formation
        != starting_xi.formation
    ):
        raise ValueError(
            "Structural responsibility formation does not match "
            "the starting XI."
        )

    active_defensive = tuple(
        responsibility
        for responsibility
        in structural_set.responsibilities
        if (
            responsibility.hypothesis_status
            == StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC
            and responsibility.responsibility_type
            in {
                ResponsibilityType.PROTECTION,
                ResponsibilityType.COVERAGE,
            }
        )
    )

    if not active_defensive:
        raise ValueError(
            "No active defensive responsibilities are available."
        )

    for responsibility in active_defensive:
        validate_active_defensive_responsibility(
            responsibility
        )

    included_slots = {
        slot
        for responsibility in active_defensive
        for slot in (
            responsibility.source_slot,
            responsibility.target_slot,
        )
    }

    assignments_by_slot = {
        assignment.slot: assignment
        for assignment in starting_xi.assignments
    }

    unknown_slots = sorted(
        included_slots
        - set(assignments_by_slot)
    )

    if unknown_slots:
        raise ValueError(
            "Defensive responsibilities reference unknown "
            f"lineup slots: {unknown_slots}"
        )

    nodes = tuple(
        sorted(
            (
                DefensiveStructuralNode(
                    slot=slot,
                    tactical_role=(
                        assignments_by_slot[
                            slot
                        ].tactical_role
                    ),
                    player_id=(
                        assignments_by_slot[
                            slot
                        ].player.identity.player_id
                    ),
                    player_name=(
                        assignments_by_slot[
                            slot
                        ].player.identity.name
                    ),
                )
                for slot in included_slots
            ),
            key=lambda node:
                node.canonical_key,
        )
    )

    edges = tuple(
        sorted(
            (
                DefensiveStructuralEdge(
                    source_slot=(
                        responsibility.source_slot
                    ),
                    target_slot=(
                        responsibility.target_slot
                    ),
                    responsibility_type=(
                        responsibility
                        .responsibility_type
                    ),
                    rule_id=(
                        responsibility.rule_id
                    ),
                    hypothesis_status=(
                        responsibility
                        .hypothesis_status
                    ),
                    supporting_positional_types=(
                        responsibility
                        .supporting_positional_types
                    ),
                )
                for responsibility
                in active_defensive
            ),
            key=lambda edge:
                edge.canonical_key,
        )
    )

    lineup_slots = {
        assignment.slot
        for assignment in starting_xi.assignments
    }

    excluded_lineup_slots = tuple(
        sorted(
            lineup_slots
            - included_slots
        )
    )

    return DefensiveStructuralGraph(
        national_team=(
            starting_xi.national_team
        ),
        formation=starting_xi.formation,
        nodes=nodes,
        edges=edges,
        excluded_lineup_slots=(
            excluded_lineup_slots
        ),
    )
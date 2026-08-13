#defensive_unit_diagnostics

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from research.player_intelligence.defensive_structural_graph import (
    DefensiveStructuralGraph,
)
from research.player_intelligence.football_responsibility import (
    ResponsibilityType,
)


@dataclass(frozen=True)
class DefensiveNodeDiagnostic:
    """
    Structural diagnostics for one node in the defensive graph.

    All measurements are topology-based and independent of player
    quality.
    """

    slot: str
    tactical_role: str

    out_degree: int
    in_degree: int
    total_degree: int

    articulation_point: bool

    component_count_after_removal: int
    largest_component_size_after_removal: int

    def __post_init__(self) -> None:
        if not self.slot.strip():
            raise ValueError(
                "Diagnostic slot must not be empty."
            )

        if not self.tactical_role.strip():
            raise ValueError(
                "Diagnostic tactical role must not be empty."
            )

        for field_name, value in (
            ("out_degree", self.out_degree),
            ("in_degree", self.in_degree),
            ("total_degree", self.total_degree),
            (
                "component_count_after_removal",
                self.component_count_after_removal,
            ),
            (
                "largest_component_size_after_removal",
                self.largest_component_size_after_removal,
            ),
        ):
            if value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if (
            self.total_degree
            != self.out_degree + self.in_degree
        ):
            raise ValueError(
                "Total degree must equal out-degree plus "
                "in-degree."
            )


@dataclass(frozen=True)
class DefensiveResponsibilityCoverage:
    """
    Coverage summary for the defensive responsibilities currently
    represented by the graph.
    """

    center_back_slots: tuple[str, ...]
    protected_center_back_slots: tuple[str, ...]

    fullback_slots: tuple[str, ...]
    covered_fullback_slots: tuple[str, ...]

    center_back_protection_fraction: float
    fullback_coverage_fraction: float

    center_back_protection_balanced: bool
    fullback_coverage_balanced: bool

    def __post_init__(self) -> None:
        for field_name, value in (
            (
                "center_back_protection_fraction",
                self.center_back_protection_fraction,
            ),
            (
                "fullback_coverage_fraction",
                self.fullback_coverage_fraction,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must lie inside [0, 1]."
                )

        if not set(
            self.protected_center_back_slots
        ).issubset(
            set(self.center_back_slots)
        ):
            raise ValueError(
                "Protected center-back slots must belong to the "
                "center-back population."
            )

        if not set(
            self.covered_fullback_slots
        ).issubset(
            set(self.fullback_slots)
        ):
            raise ValueError(
                "Covered fullback slots must belong to the "
                "fullback population."
            )


@dataclass(frozen=True)
class DefensiveUnitDiagnostics:
    """
    Immutable diagnostic interpretation of one defensive structural
    graph.

    This object contains measurements only. It does not alter the
    graph, player values, team strength, or simulation behavior.
    """

    national_team: str
    formation: str

    node_count: int
    edge_count: int

    weak_component_count: int
    largest_component_size: int

    isolated_slots: tuple[str, ...]
    articulation_slots: tuple[str, ...]

    node_diagnostics: tuple[
        DefensiveNodeDiagnostic,
        ...,
    ]

    responsibility_coverage: DefensiveResponsibilityCoverage

    partial_graph: bool
    weighted: bool

    def __post_init__(self) -> None:
        if not self.national_team.strip():
            raise ValueError(
                "Diagnostic team must not be empty."
            )

        if not self.formation.strip():
            raise ValueError(
                "Diagnostic formation must not be empty."
            )

        if self.node_count <= 0:
            raise ValueError(
                "Diagnostic node count must be positive."
            )

        if self.edge_count <= 0:
            raise ValueError(
                "Diagnostic edge count must be positive."
            )

        if self.weak_component_count <= 0:
            raise ValueError(
                "Weak component count must be positive."
            )

        if not (
            1
            <= self.largest_component_size
            <= self.node_count
        ):
            raise ValueError(
                "Largest component size is invalid."
            )

        diagnostic_slots = tuple(
            diagnostic.slot
            for diagnostic
            in self.node_diagnostics
        )

        if len(diagnostic_slots) != len(
            set(diagnostic_slots)
        ):
            raise ValueError(
                "Node diagnostics contain duplicate slots."
            )

        if len(diagnostic_slots) != self.node_count:
            raise ValueError(
                "Node diagnostic count does not match node count."
            )

        if diagnostic_slots != tuple(
            sorted(diagnostic_slots)
        ):
            raise ValueError(
                "Node diagnostics must use deterministic order."
            )

        if tuple(
            sorted(self.isolated_slots)
        ) != self.isolated_slots:
            raise ValueError(
                "Isolated slots must use deterministic order."
            )

        if tuple(
            sorted(self.articulation_slots)
        ) != self.articulation_slots:
            raise ValueError(
                "Articulation slots must use deterministic order."
            )

        if not self.partial_graph:
            raise ValueError(
                "Study 103C diagnostics must describe the "
                "partial defensive graph."
            )

        if self.weighted:
            raise ValueError(
                "Study 103C diagnostics must remain unweighted."
            )

    def diagnostic_by_slot(
        self,
        slot: str,
    ) -> DefensiveNodeDiagnostic:
        for diagnostic in self.node_diagnostics:
            if diagnostic.slot == slot:
                return diagnostic

        raise KeyError(
            f"Unknown diagnostic slot: {slot!r}"
        )

def build_undirected_adjacency(
    graph: DefensiveStructuralGraph,
) -> dict[str, set[str]]:
    """
    Build an undirected view for weak-connectivity diagnostics.

    The underlying structural edges remain directional.
    """

    adjacency = {
        node.slot: set()
        for node in graph.nodes
    }

    for edge in graph.edges:
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

def adjacency_without_slot(
    *,
    adjacency: dict[str, set[str]],
    removed_slot: str,
) -> dict[str, set[str]]:
    if removed_slot not in adjacency:
        raise KeyError(
            f"Unknown adjacency slot: {removed_slot!r}"
        )

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

def component_metrics_after_removal(
    *,
    adjacency: dict[str, set[str]],
    removed_slot: str,
) -> tuple[int, int]:
    reduced = adjacency_without_slot(
        adjacency=adjacency,
        removed_slot=removed_slot,
    )

    if not reduced:
        return 0, 0

    components = connected_components(
        reduced
    )

    return (
        len(components),
        max(
            len(component)
            for component in components
        ),
    )

def build_node_diagnostics(
    graph: DefensiveStructuralGraph,
) -> tuple[
    DefensiveNodeDiagnostic,
    ...,
]:
    adjacency = build_undirected_adjacency(
        graph
    )

    baseline_component_count = len(
        connected_components(
            adjacency
        )
    )

    rows: list[
        DefensiveNodeDiagnostic
    ] = []

    for node in graph.nodes:
        outgoing = len(
            graph.outgoing_edges(
                node.slot
            )
        )

        incoming = len(
            graph.incoming_edges(
                node.slot
            )
        )

        (
            component_count_after_removal,
            largest_component_after_removal,
        ) = component_metrics_after_removal(
            adjacency=adjacency,
            removed_slot=node.slot,
        )

        articulation = bool(
            component_count_after_removal
            > baseline_component_count
        )

        rows.append(
            DefensiveNodeDiagnostic(
                slot=node.slot,
                tactical_role=(
                    node.tactical_role
                ),
                out_degree=outgoing,
                in_degree=incoming,
                total_degree=(
                    outgoing + incoming
                ),
                articulation_point=(
                    articulation
                ),
                component_count_after_removal=(
                    component_count_after_removal
                ),
                largest_component_size_after_removal=(
                    largest_component_after_removal
                ),
            )
        )

    return tuple(
        sorted(
            rows,
            key=lambda diagnostic:
                diagnostic.slot,
        )
    )

def build_responsibility_coverage(
    graph: DefensiveStructuralGraph,
) -> DefensiveResponsibilityCoverage:
    center_back_slots = tuple(
        sorted(
            node.slot
            for node in graph.nodes
            if node.tactical_role == "CB"
        )
    )

    fullback_slots = tuple(
        sorted(
            node.slot
            for node in graph.nodes
            if node.tactical_role == "FB"
        )
    )

    protected_center_back_slots = tuple(
        sorted(
            {
                edge.target_slot
                for edge in graph.edges
                if (
                    edge.responsibility_type
                    == ResponsibilityType.PROTECTION
                    and edge.target_slot
                    in center_back_slots
                )
            }
        )
    )

    covered_fullback_slots = tuple(
        sorted(
            {
                edge.target_slot
                for edge in graph.edges
                if (
                    edge.responsibility_type
                    == ResponsibilityType.COVERAGE
                    and edge.target_slot
                    in fullback_slots
                )
            }
        )
    )

    center_back_protection_fraction = (
        len(
            protected_center_back_slots
        )
        / len(center_back_slots)
        if center_back_slots
        else 0.0
    )

    fullback_coverage_fraction = (
        len(
            covered_fullback_slots
        )
        / len(fullback_slots)
        if fullback_slots
        else 0.0
    )

    return DefensiveResponsibilityCoverage(
        center_back_slots=center_back_slots,
        protected_center_back_slots=(
            protected_center_back_slots
        ),
        fullback_slots=fullback_slots,
        covered_fullback_slots=(
            covered_fullback_slots
        ),
        center_back_protection_fraction=float(
            center_back_protection_fraction
        ),
        fullback_coverage_fraction=float(
            fullback_coverage_fraction
        ),
        center_back_protection_balanced=bool(
            (
                not center_back_slots
            )
            or (
                len(
                    protected_center_back_slots
                )
                == len(center_back_slots)
            )
        ),
        fullback_coverage_balanced=bool(
            (
                not fullback_slots
            )
            or (
                len(
                    covered_fullback_slots
                )
                == len(fullback_slots)
            )
        ),
    )

def diagnose_defensive_unit(
    graph: DefensiveStructuralGraph,
) -> DefensiveUnitDiagnostics:
    adjacency = build_undirected_adjacency(
        graph
    )

    components = connected_components(
        adjacency
    )

    node_diagnostics = (
        build_node_diagnostics(
            graph
        )
    )

    isolated_slots = tuple(
        sorted(
            slot
            for slot, neighbors
            in adjacency.items()
            if not neighbors
        )
    )

    articulation_slots = tuple(
        sorted(
            diagnostic.slot
            for diagnostic
            in node_diagnostics
            if diagnostic.articulation_point
        )
    )

    responsibility_coverage = (
        build_responsibility_coverage(
            graph
        )
    )

    return DefensiveUnitDiagnostics(
        national_team=(
            graph.national_team
        ),
        formation=graph.formation,
        node_count=len(
            graph.nodes
        ),
        edge_count=len(
            graph.edges
        ),
        weak_component_count=len(
            components
        ),
        largest_component_size=max(
            len(component)
            for component in components
        ),
        isolated_slots=(
            isolated_slots
        ),
        articulation_slots=(
            articulation_slots
        ),
        node_diagnostics=(
            node_diagnostics
        ),
        responsibility_coverage=(
            responsibility_coverage
        ),
        partial_graph=(
            graph.partial_graph
        ),
        weighted=graph.weighted,
    )
#compare_structural_network_scopes

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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
    INITIAL_STRUCTURAL_HYPOTHESES,
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
    / "study_103a3"
)

ALL_RELATIONSHIPS_PATH = (
    OUTPUT_DIRECTORY
    / "world_cup_structural_relationships_all.csv"
)

ACTIVE_RELATIONSHIPS_PATH = (
    OUTPUT_DIRECTORY
    / "world_cup_structural_relationships_active.csv"
)

RELATIONSHIP_DELTAS_PATH = (
    OUTPUT_DIRECTORY
    / "structural_scope_relationship_deltas.csv"
)

DEGREE_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "structural_scope_degree_comparison.csv"
)

COMPONENT_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "structural_scope_component_summary.csv"
)

SCOPE_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "structural_scope_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_103a3_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_103A3_REPORT.md"
)


FORMATION = "4-3-3"
EXPECTED_TEAM_COUNT = 48
EXPECTED_SLOT_COUNT = 11

SCOPES = {
    "all_hypotheses": None,
    "active_diagnostic_only": (
        StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
    ),
}

WORLD_CUP_TEAMS = {
    normalize_team_name(team)
    for group in GROUPS.values()
    for team in group
}

STATUS_BY_RULE = {
    hypothesis.rule_id:
        hypothesis.status
    for hypothesis
    in INITIAL_STRUCTURAL_HYPOTHESES
}

def build_undirected_adjacency(
    *,
    lineup_slots: tuple[str, ...],
    relationships: pd.DataFrame,
) -> dict[str, set[str]]:
    adjacency = {
        slot: set()
        for slot in lineup_slots
    }

    for row in relationships.itertuples(
        index=False
    ):
        source = str(
            row.source_slot
        )

        target = str(
            row.target_slot
        )

        if source not in adjacency:
            raise AssertionError(
                "Relationship references unknown source slot: "
                f"{source!r}."
            )

        if target not in adjacency:
            raise AssertionError(
                "Relationship references unknown target slot: "
                f"{target!r}."
            )

        adjacency[source].add(
            target
        )

        adjacency[target].add(
            source
        )

    return adjacency

def connected_components(
    adjacency: dict[
        str,
        set[str],
    ],
) -> tuple[
    tuple[str, ...],
    ...,
]:
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

        visited_component: set[
            str
        ] = set()

        while queue:
            current = queue.popleft()

            if current in visited_component:
                continue

            visited_component.add(
                current
            )

            neighbors = sorted(
                adjacency[current]
            )

            for neighbor in neighbors:
                if (
                    neighbor
                    not in visited_component
                ):
                    queue.append(
                        neighbor
                    )

        unvisited -= (
            visited_component
        )

        components.append(
            tuple(
                sorted(
                    visited_component
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

def build_scope_populations() -> tuple[
    dict[str, pd.DataFrame],
    dict[
        tuple[str, str],
        tuple[str, ...],
    ],
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

    rows_by_scope: dict[
        str,
        list[dict[str, Any]],
    ] = {
        scope_name: []
        for scope_name in SCOPES
    }

    lineup_slots_by_team_scope: dict[
        tuple[str, str],
        tuple[str, ...],
    ] = {}

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

        positional_set = (
            generate_positional_responsibilities(
                starting_xi=starting_xi,
                geometry=geometry,
            )
        )

        assignments_by_slot = {
            assignment.slot:
                assignment
            for assignment
            in starting_xi.assignments
        }

        lineup_slots = tuple(
            assignment.slot
            for assignment
            in starting_xi.assignments
        )

        if len(lineup_slots) != EXPECTED_SLOT_COUNT:
            raise AssertionError(
                "Unexpected lineup slot count for "
                f"{nation!r}: {len(lineup_slots)}."
            )

        for (
            scope_name,
            included_statuses,
        ) in SCOPES.items():
            structural_set = (
                generate_structural_responsibilities(
                    starting_xi=starting_xi,
                    geometry=geometry,
                    positional_set=positional_set,
                    included_statuses=(
                        included_statuses
                    ),
                )
            )

            lineup_slots_by_team_scope[
                (
                    nation,
                    scope_name,
                )
            ] = lineup_slots

            for relationship in (
                structural_set.responsibilities
            ):
                source_assignment = (
                    assignments_by_slot[
                        relationship.source_slot
                    ]
                )

                target_assignment = (
                    assignments_by_slot[
                        relationship.target_slot
                    ]
                )

                rows_by_scope[
                    scope_name
                ].append(
                    {
                        "scope":
                            scope_name,
                        "nation":
                            nation,
                        "formation":
                            FORMATION,
                        "source_slot":
                            relationship.source_slot,
                        "source_role":
                            relationship.source_role,
                        "source_player_id":
                            source_assignment
                            .player.identity.player_id,
                        "source_player":
                            source_assignment
                            .player.identity.name,
                        "target_slot":
                            relationship.target_slot,
                        "target_role":
                            relationship.target_role,
                        "target_player_id":
                            target_assignment
                            .player.identity.player_id,
                        "target_player":
                            target_assignment
                            .player.identity.name,
                        "responsibility_type":
                            relationship
                            .responsibility_type.value,
                        "rule_id":
                            relationship.rule_id,
                        "hypothesis_status":
                            relationship
                            .hypothesis_status.value,
                        "supporting_positional_types":
                            "|".join(
                                item.value
                                for item
                                in relationship
                                .supporting_positional_types
                            ),
                        "directional":
                            True,
                        "weighted":
                            False,
                    }
                )

    frames = {
        scope_name: (
            pd.DataFrame(rows)
            .sort_values(
                [
                    "nation",
                    "source_slot",
                    "target_slot",
                    "responsibility_type",
                    "rule_id",
                ]
            )
            .reset_index(drop=True)
        )
        for scope_name, rows
        in rows_by_scope.items()
    }

    return (
        frames,
        lineup_slots_by_team_scope,
    )

def validate_scope_population(
    *,
    scope_name: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        raise AssertionError(
            f"Scope {scope_name!r} is empty."
        )

    observed_teams = set(
        frame["nation"].astype(str)
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
            f"Scope {scope_name!r} has an invalid team "
            f"population. Missing={missing}, extra={extra}"
        )

    if frame[
        "source_slot"
    ].eq(
        frame["target_slot"]
    ).any():
        raise AssertionError(
            f"Scope {scope_name!r} contains "
            "self-relationships."
        )

    duplicate_columns = [
        "nation",
        "source_slot",
        "target_slot",
        "responsibility_type",
        "rule_id",
        "hypothesis_status",
    ]

    if frame.duplicated(
        subset=duplicate_columns
    ).any():
        raise AssertionError(
            f"Scope {scope_name!r} contains duplicate "
            "relationships."
        )

    if not frame[
        "directional"
    ].all():
        raise AssertionError(
            f"Scope {scope_name!r} contains a "
            "non-directional relationship."
        )

    if frame[
        "weighted"
    ].any():
        raise AssertionError(
            f"Scope {scope_name!r} contains a weighted "
            "relationship."
        )

    for row in frame.itertuples(
        index=False
    ):
        registered_status = (
            STATUS_BY_RULE.get(
                str(row.rule_id)
            )
        )

        if registered_status is None:
            raise AssertionError(
                "Relationship references an unregistered "
                f"rule ID: {row.rule_id!r}."
            )

        if (
            str(row.hypothesis_status)
            != registered_status.value
        ):
            raise AssertionError(
                "Relationship lifecycle status does not match "
                f"the registered hypothesis: {row.rule_id!r}."
            )

    if (
        scope_name
        == "active_diagnostic_only"
    ):
        invalid = frame.loc[
            ~frame[
                "hypothesis_status"
            ].eq(
                StructuralHypothesisStatus
                .ACTIVE_DIAGNOSTIC
                .value
            )
        ]

        if not invalid.empty:
            raise AssertionError(
                "Active-only scope contains non-active "
                "hypotheses."
            )

def relationship_identity_columns() -> list[str]:
    return [
        "nation",
        "formation",
        "source_slot",
        "source_role",
        "target_slot",
        "target_role",
        "responsibility_type",
        "rule_id",
        "hypothesis_status",
        "supporting_positional_types",
    ]

def build_relationship_deltas(
    *,
    all_frame: pd.DataFrame,
    active_frame: pd.DataFrame,
) -> pd.DataFrame:
    identity_columns = (
        relationship_identity_columns()
    )

    all_keys = (
        all_frame[
            identity_columns
        ]
        .drop_duplicates()
        .assign(
            present_in_all=True
        )
    )

    active_keys = (
        active_frame[
            identity_columns
        ]
        .drop_duplicates()
        .assign(
            present_in_active=True
        )
    )

    comparison = all_keys.merge(
        active_keys,
        on=identity_columns,
        how="outer",
    )

    comparison[
        "present_in_all"
    ] = (
        comparison[
            "present_in_all"
        ]
        .fillna(False)
        .astype(bool)
    )

    comparison[
        "present_in_active"
    ] = (
        comparison[
            "present_in_active"
        ]
        .fillna(False)
        .astype(bool)
    )

    comparison[
        "retained_in_active"
    ] = (
        comparison[
            "present_in_all"
        ]
        & comparison[
            "present_in_active"
        ]
    )

    comparison[
        "removed_from_active"
    ] = (
        comparison[
            "present_in_all"
        ]
        & ~comparison[
            "present_in_active"
        ]
    )

    comparison[
        "unexpected_active_only"
    ] = (
        comparison[
            "present_in_active"
        ]
        & ~comparison[
            "present_in_all"
        ]
    )

    return (
        comparison
        .sort_values(
            [
                "nation",
                "removed_from_active",
                "source_slot",
                "target_slot",
                "rule_id",
            ],
            ascending=[
                True,
                False,
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

def build_degree_comparison(
    *,
    scope_frames: dict[
        str,
        pd.DataFrame,
    ],
    lineup_slots_by_team_scope: dict[
        tuple[str, str],
        tuple[str, ...],
    ],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scope_name, frame in (
        scope_frames.items()
    ):
        for nation in sorted(
            WORLD_CUP_TEAMS
        ):
            team_frame = frame.loc[
                frame[
                    "nation"
                ].eq(
                    nation
                )
            ]

            lineup_slots = (
                lineup_slots_by_team_scope[
                    (
                        nation,
                        scope_name,
                    )
                ]
            )

            out_degree = (
                team_frame
                .groupby(
                    "source_slot"
                )
                .size()
                .to_dict()
            )

            in_degree = (
                team_frame
                .groupby(
                    "target_slot"
                )
                .size()
                .to_dict()
            )

            for slot in lineup_slots:
                outgoing = int(
                    out_degree.get(
                        slot,
                        0,
                    )
                )

                incoming = int(
                    in_degree.get(
                        slot,
                        0,
                    )
                )

                rows.append(
                    {
                        "nation": nation,
                        "formation": FORMATION,
                        "scope": scope_name,
                        "slot": slot,
                        "out_degree": outgoing,
                        "in_degree": incoming,
                        "total_directed_degree":
                            outgoing + incoming,
                        "isolated_directed":
                            bool(
                                outgoing == 0
                                and incoming == 0
                            ),
                    }
                )

    degree_frame = pd.DataFrame(
        rows
    )

    wide = (
        degree_frame
        .pivot(
            index=[
                "nation",
                "formation",
                "slot",
            ],
            columns="scope",
            values=[
                "out_degree",
                "in_degree",
                "total_directed_degree",
                "isolated_directed",
            ],
        )
    )

    wide.columns = [
        f"{metric}_{scope}"
        for metric, scope
        in wide.columns
    ]

    wide = (
        wide
        .reset_index()
    )

    wide[
        "out_degree_delta_active_minus_all"
    ] = (
        wide[
            "out_degree_active_diagnostic_only"
        ]
        - wide[
            "out_degree_all_hypotheses"
        ]
    )

    wide[
        "in_degree_delta_active_minus_all"
    ] = (
        wide[
            "in_degree_active_diagnostic_only"
        ]
        - wide[
            "in_degree_all_hypotheses"
        ]
    )

    wide[
        "total_degree_delta_active_minus_all"
    ] = (
        wide[
            "total_directed_degree_active_diagnostic_only"
        ]
        - wide[
            "total_directed_degree_all_hypotheses"
        ]
    )

    return wide

def build_component_summary(
    *,
    scope_frames: dict[
        str,
        pd.DataFrame,
    ],
    lineup_slots_by_team_scope: dict[
        tuple[str, str],
        tuple[str, ...],
    ],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scope_name, frame in (
        scope_frames.items()
    ):
        for nation in sorted(
            WORLD_CUP_TEAMS
        ):
            team_frame = frame.loc[
                frame[
                    "nation"
                ].eq(
                    nation
                )
            ]

            lineup_slots = (
                lineup_slots_by_team_scope[
                    (
                        nation,
                        scope_name,
                    )
                ]
            )

            adjacency = (
                build_undirected_adjacency(
                    lineup_slots=lineup_slots,
                    relationships=team_frame,
                )
            )

            components = (
                connected_components(
                    adjacency
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

            connected_slots = (
                len(lineup_slots)
                - len(isolated_slots)
            )

            largest_component_size = max(
                len(component)
                for component in components
            )

            rows.append(
                {
                    "nation": nation,
                    "formation": FORMATION,
                    "scope": scope_name,
                    "lineup_slot_count":
                        len(lineup_slots),
                    "relationship_count":
                        len(team_frame),
                    "weak_component_count":
                        len(components),
                    "largest_component_size":
                        largest_component_size,
                    "connected_slot_count":
                        connected_slots,
                    "isolated_slot_count":
                        len(isolated_slots),
                    "isolated_slots":
                        "|".join(
                            isolated_slots
                        ),
                    "fully_connected_weakly":
                        bool(
                            len(components) == 1
                        ),
                    "component_membership":
                        " || ".join(
                            "|".join(
                                component
                            )
                            for component
                            in components
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )

def build_scope_summary(
    *,
    scope_frames: dict[
        str,
        pd.DataFrame,
    ],
    component_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scope_name, frame in (
        scope_frames.items()
    ):
        per_team_counts = (
            frame
            .groupby(
                "nation"
            )
            .size()
        )

        if per_team_counts.nunique() != 1:
            raise AssertionError(
                f"Scope {scope_name!r} has inconsistent "
                "relationship counts across identical formations."
            )

        scope_components = (
            component_summary.loc[
                component_summary[
                    "scope"
                ].eq(
                    scope_name
                )
            ]
        )

        rows.append(
            {
                "scope": scope_name,
                "team_count":
                    int(
                        frame[
                            "nation"
                        ].nunique()
                    ),
                "relationships_per_team":
                    int(
                        per_team_counts.iloc[0]
                    ),
                "responsibility_type_count":
                    int(
                        frame[
                            "responsibility_type"
                        ].nunique()
                    ),
                "rule_count":
                    int(
                        frame[
                            "rule_id"
                        ].nunique()
                    ),
                "status_count":
                    int(
                        frame[
                            "hypothesis_status"
                        ].nunique()
                    ),
                "unique_directed_slot_pair_count":
                    int(
                        frame[
                            [
                                "source_slot",
                                "target_slot",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),
                "mean_component_count":
                    float(
                        scope_components[
                            "weak_component_count"
                        ].mean()
                    ),
                "mean_largest_component_size":
                    float(
                        scope_components[
                            "largest_component_size"
                        ].mean()
                    ),
                "mean_isolated_slot_count":
                    float(
                        scope_components[
                            "isolated_slot_count"
                        ].mean()
                    ),
                "fully_connected_team_count":
                    int(
                        scope_components[
                            "fully_connected_weakly"
                        ].sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )

def validate_comparison(
    *,
    scope_frames: dict[
        str,
        pd.DataFrame,
    ],
    relationship_deltas: pd.DataFrame,
    degree_comparison: pd.DataFrame,
    component_summary: pd.DataFrame,
    scope_summary: pd.DataFrame,
) -> None:
    all_frame = scope_frames[
        "all_hypotheses"
    ]

    active_frame = scope_frames[
        "active_diagnostic_only"
    ]

    all_identity = set(
        tuple(row)
        for row in all_frame[
            relationship_identity_columns()
        ].itertuples(
            index=False,
            name=None,
        )
    )

    active_identity = set(
        tuple(row)
        for row in active_frame[
            relationship_identity_columns()
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if not active_identity:
        raise AssertionError(
            "Active-diagnostic network is empty."
        )

    if not active_identity < all_identity:
        raise AssertionError(
            "Active-diagnostic relationships are not a "
            "strict subset of all relationships."
        )

    if relationship_deltas[
        "unexpected_active_only"
    ].any():
        raise AssertionError(
            "Active-only scope contains relationships absent "
            "from the all-hypothesis scope."
        )

    removed = relationship_deltas.loc[
        relationship_deltas[
            "removed_from_active"
        ]
    ]

    if removed.empty:
        raise AssertionError(
            "No relationships were removed by the "
            "active-only filter."
        )

    invalid_removed_statuses = (
        set(
            removed[
                "hypothesis_status"
            ].astype(str)
        )
        - {
            StructuralHypothesisStatus
            .REVISION_REQUIRED.value,
            StructuralHypothesisStatus
            .DEFERRED.value,
        }
    )

    if invalid_removed_statuses:
        raise AssertionError(
            "Active relationships were incorrectly removed: "
            f"{sorted(invalid_removed_statuses)}"
        )

    all_relationship_count = len(
        all_frame
    )

    active_relationship_count = len(
        active_frame
    )

    if (
        degree_comparison[
            "out_degree_all_hypotheses"
        ].sum()
        != all_relationship_count
    ):
        raise AssertionError(
            "All-scope out-degree totals do not reconcile."
        )

    if (
        degree_comparison[
            "in_degree_all_hypotheses"
        ].sum()
        != all_relationship_count
    ):
        raise AssertionError(
            "All-scope in-degree totals do not reconcile."
        )

    if (
        degree_comparison[
            "out_degree_active_diagnostic_only"
        ].sum()
        != active_relationship_count
    ):
        raise AssertionError(
            "Active-scope out-degree totals do not reconcile."
        )

    if (
        degree_comparison[
            "in_degree_active_diagnostic_only"
        ].sum()
        != active_relationship_count
    ):
        raise AssertionError(
            "Active-scope in-degree totals do not reconcile."
        )

    if component_summary.empty:
        raise AssertionError(
            "Connectivity summary is empty."
        )

    if not component_summary[
        "largest_component_size"
    ].between(
        1,
        EXPECTED_SLOT_COUNT,
        inclusive="both",
    ).all():
        raise AssertionError(
            "Invalid largest-component size."
        )

    if not scope_summary[
        "team_count"
    ].eq(
        EXPECTED_TEAM_COUNT
    ).all():
        raise AssertionError(
            "Scope summaries do not preserve all 48 teams."
        )

def build_metadata(
    *,
    scope_summary: pd.DataFrame,
    relationship_deltas: pd.DataFrame,
    component_summary: pd.DataFrame,
) -> dict[str, Any]:
    summary_by_scope = (
        scope_summary
        .set_index("scope")
    )

    all_relationships = int(
        summary_by_scope.loc[
            "all_hypotheses",
            "relationships_per_team",
        ]
    )

    active_relationships = int(
        summary_by_scope.loc[
            "active_diagnostic_only",
            "relationships_per_team",
        ]
    )

    active_components = (
        component_summary.loc[
            component_summary[
                "scope"
            ].eq(
                "active_diagnostic_only"
            )
        ]
    )

    return {
        "study_id": "103A3",
        "study_name": (
            "Structural Network Scope Comparison"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "formation": FORMATION,
        "team_count": EXPECTED_TEAM_COUNT,
        "scope_count": len(SCOPES),
        "all_hypothesis_relationships_per_team":
            all_relationships,
        "active_relationships_per_team":
            active_relationships,
        "relationships_removed_per_team": (
            all_relationships
            - active_relationships
        ),
        "active_relationship_fraction": (
            active_relationships
            / all_relationships
            if all_relationships
            else 0.0
        ),
        "active_mean_isolated_slot_count":
            float(
                active_components[
                    "isolated_slot_count"
                ].mean()
            ),
        "active_mean_largest_component_size":
            float(
                active_components[
                    "largest_component_size"
                ].mean()
            ),
        "active_network_fully_connected":
            bool(
                active_components[
                    "fully_connected_weakly"
                ].all()
            ),
        "removed_relationship_record_count":
            int(
                relationship_deltas[
                    "removed_from_active"
                ].sum()
            ),
        "weights_created": False,
        "football_graph_created": False,
        "team_strength_changed": False,
        "repository_changed": False,
        "simulation_run": False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "This study compares diagnostic structural "
            "hypothesis scopes. Connectivity does not establish "
            "football truth, predictive value, or production "
            "eligibility."
        ),
    }

def write_report(
    *,
    metadata: dict[str, Any],
    scope_summary: pd.DataFrame,
    relationship_deltas: pd.DataFrame,
    degree_comparison: pd.DataFrame,
    component_summary: pd.DataFrame,
) -> None:
    removed_summary = (
        relationship_deltas.loc[
            relationship_deltas[
                "removed_from_active"
            ]
        ]
        .groupby(
            [
                "rule_id",
                "hypothesis_status",
                "responsibility_type",
            ],
            as_index=False,
        )
        .agg(
            removed_record_count=(
                "nation",
                "count",
            ),
            team_count=(
                "nation",
                "nunique",
            ),
        )
        .sort_values(
            "removed_record_count",
            ascending=False,
        )
    )

    slot_degree_summary = (
        degree_comparison
        .groupby(
            "slot",
            as_index=False,
        )
        .agg(
            all_mean_out_degree=(
                "out_degree_all_hypotheses",
                "mean",
            ),
            active_mean_out_degree=(
                "out_degree_active_diagnostic_only",
                "mean",
            ),
            all_mean_in_degree=(
                "in_degree_all_hypotheses",
                "mean",
            ),
            active_mean_in_degree=(
                "in_degree_active_diagnostic_only",
                "mean",
            ),
            mean_total_degree_delta=(
                "total_degree_delta_active_minus_all",
                "mean",
            ),
        )
        .sort_values("slot")
    )

    connectivity_summary = (
        component_summary
        .groupby(
            "scope",
            as_index=False,
        )
        .agg(
            team_count=(
                "nation",
                "nunique",
            ),
            mean_component_count=(
                "weak_component_count",
                "mean",
            ),
            mean_largest_component_size=(
                "largest_component_size",
                "mean",
            ),
            mean_isolated_slot_count=(
                "isolated_slot_count",
                "mean",
            ),
            fully_connected_team_count=(
                "fully_connected_weakly",
                "sum",
            ),
        )
    )

    report = f"""# Study 103A3 — Structural Network Scope Comparison

## Status

**PASS**

## Purpose

Compare the full expert-authored structural hypothesis network against
the conservative active-diagnostic subset.

## Scope summary

{scope_summary.to_markdown(index=False)}

## Removed hypotheses and relationships

{removed_summary.to_markdown(index=False)}

## Slot-degree effects

{slot_degree_summary.to_markdown(index=False)}

## Weak-connectivity diagnostics

{connectivity_summary.to_markdown(index=False)}

## Key result

- All-hypothesis relationships per team:
  {metadata["all_hypothesis_relationships_per_team"]}
- Active-diagnostic relationships per team:
  {metadata["active_relationships_per_team"]}
- Relationships removed per team:
  {metadata["relationships_removed_per_team"]}
- Active relationship fraction:
  {metadata["active_relationship_fraction"]:.6f}
- Mean active isolated slots:
  {metadata["active_mean_isolated_slot_count"]:.6f}
- Mean active largest component:
  {metadata["active_mean_largest_component_size"]:.6f}

## Interpretation boundary

The active network is a conservative diagnostic subnetwork. Sparse or
disconnected topology does not invalidate the retained hypotheses; it
indicates that the current active hypothesis set describes only part of
the team structure.

No predictive graph, weights, strength adjustments, repositories, or
simulations were created.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 103A3 — STRUCTURAL NETWORK "
        "SCOPE COMPARISON"
    )
    print("=" * 88)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        scope_frames,
        lineup_slots_by_team_scope,
    ) = build_scope_populations()

    for scope_name, frame in (
        scope_frames.items()
    ):
        validate_scope_population(
            scope_name=scope_name,
            frame=frame,
        )

    all_frame = scope_frames[
        "all_hypotheses"
    ]

    active_frame = scope_frames[
        "active_diagnostic_only"
    ]

    relationship_deltas = (
        build_relationship_deltas(
            all_frame=all_frame,
            active_frame=active_frame,
        )
    )

    degree_comparison = (
        build_degree_comparison(
            scope_frames=scope_frames,
            lineup_slots_by_team_scope=(
                lineup_slots_by_team_scope
            ),
        )
    )

    component_summary = (
        build_component_summary(
            scope_frames=scope_frames,
            lineup_slots_by_team_scope=(
                lineup_slots_by_team_scope
            ),
        )
    )

    scope_summary = (
        build_scope_summary(
            scope_frames=scope_frames,
            component_summary=(
                component_summary
            ),
        )
    )

    validate_comparison(
        scope_frames=scope_frames,
        relationship_deltas=(
            relationship_deltas
        ),
        degree_comparison=(
            degree_comparison
        ),
        component_summary=(
            component_summary
        ),
        scope_summary=scope_summary,
    )

    all_frame.to_csv(
        ALL_RELATIONSHIPS_PATH,
        index=False,
    )

    active_frame.to_csv(
        ACTIVE_RELATIONSHIPS_PATH,
        index=False,
    )

    relationship_deltas.to_csv(
        RELATIONSHIP_DELTAS_PATH,
        index=False,
    )

    degree_comparison.to_csv(
        DEGREE_COMPARISON_PATH,
        index=False,
    )

    component_summary.to_csv(
        COMPONENT_SUMMARY_PATH,
        index=False,
    )

    scope_summary.to_csv(
        SCOPE_SUMMARY_PATH,
        index=False,
    )

    metadata = build_metadata(
        scope_summary=scope_summary,
        relationship_deltas=(
            relationship_deltas
        ),
        component_summary=(
            component_summary
        ),
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        metadata=metadata,
        scope_summary=scope_summary,
        relationship_deltas=(
            relationship_deltas
        ),
        degree_comparison=(
            degree_comparison
        ),
        component_summary=(
            component_summary
        ),
    )

    print()
    print("Scope comparison summary")
    print("-" * 88)
    print(
        "  All-hypothesis relationships per team: "
        f"{metadata['all_hypothesis_relationships_per_team']}"
    )
    print(
        "  Active relationships per team: "
        f"{metadata['active_relationships_per_team']}"
    )
    print(
        "  Relationships removed per team: "
        f"{metadata['relationships_removed_per_team']}"
    )
    print(
        "  Active relationship fraction: "
        f"{metadata['active_relationship_fraction']:.6f}"
    )
    print(
        "  Active mean isolated slots: "
        f"{metadata['active_mean_isolated_slot_count']:.6f}"
    )
    print(
        "  Active mean largest component: "
        f"{metadata['active_mean_largest_component_size']:.6f}"
    )
    print(
        "  Active relationships strict subset: PASS"
    )
    print(
        "  Lifecycle status reconciliation: PASS"
    )
    print(
        "  Degree reconciliation: PASS"
    )
    print(
        "  Connectivity diagnostics: PASS"
    )
    print(
        "  Weights created: NO"
    )
    print(
        "  Football graph created: NO"
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
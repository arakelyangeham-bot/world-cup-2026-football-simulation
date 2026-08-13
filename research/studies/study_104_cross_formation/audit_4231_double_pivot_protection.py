#audit_4231_double_pivot_protection

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research.player_intelligence.football_responsibility import (
    ResponsibilityType,
)
from research.player_intelligence.formation_geometry import (
    FormationGeometry,
    load_formation_geometry,
)
from research.player_intelligence.player_schema import (
    StartingXI,
)
from research.player_intelligence.positional_responsibility_generator import (
    PositionalResponsibilitySet,
    broad_corridor,
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
    / "study_104_cross_formation"
    / "study_104b"
)

POSITIONAL_PATH = (
    OUTPUT_DIRECTORY
    / "formation_4231_positional_relationships.csv"
)

CURRENT_STRUCTURAL_PATH = (
    OUTPUT_DIRECTORY
    / "formation_4231_structural_relationships_current.csv"
)

PROTECTION_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "formation_4231_shared_vs_side_specific_protection.csv"
)

TEAM_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "team_4231_double_pivot_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_104b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_104B_REPORT.md"
)


FORMATION = "4-2-3-1"
EXPECTED_TEAM_COUNT = 48
EXPECTED_SLOT_COUNT = 11

DM_SLOTS = {
    "DM1",
    "DM2",
}

CB_SLOTS = {
    "CB1",
    "CB2",
}

WORLD_CUP_TEAMS = {
    normalize_team_name(team)
    for group in GROUPS.values()
    for team in group
}

def positional_types_by_pair(
    positional_set: PositionalResponsibilitySet,
) -> dict[
    frozenset[str],
    set[ResponsibilityType],
]:
    mapping: dict[
        frozenset[str],
        set[ResponsibilityType],
    ] = {}

    for relationship in positional_set.responsibilities:
        key = frozenset(
            {
                relationship.source_slot,
                relationship.target_slot,
            }
        )

        mapping.setdefault(
            key,
            set(),
        ).add(
            relationship.responsibility_type
        )

    return mapping

def current_protection_rows(
    *,
    starting_xi: StartingXI,
    geometry: FormationGeometry,
    positional_set: PositionalResponsibilitySet,
    nation: str,
    am_fallback_applied: bool,
    am_fallback_source_role: str,
) -> list[dict[str, Any]]:
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

    rows: list[dict[str, Any]] = []

    for relationship in (
        structural_set.responsibilities
    ):
        if (
            relationship.responsibility_type
            != ResponsibilityType.PROTECTION
        ):
            continue

        rows.append(
            {
                "nation": nation,
                "formation": FORMATION,
                "am_fallback_applied":
                    am_fallback_applied,
                "am_fallback_source_role":
                    am_fallback_source_role,
                "source_slot":
                    relationship.source_slot,
                "target_slot":
                    relationship.target_slot,
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
                "interpretation":
                    "current_adjacent_line_rule",
                "candidate_only": False,
            }
        )

    return rows

def build_protection_candidates(
    *,
    nation: str,
    geometry: FormationGeometry,
    positional_set: PositionalResponsibilitySet,
    am_fallback_applied: bool,
    am_fallback_source_role: str,
) -> list[dict[str, Any]]:
    positions_by_slot = {
        position.slot: position
        for position in geometry.positions
    }

    evidence_by_pair = positional_types_by_pair(
        positional_set
    )

    rows: list[dict[str, Any]] = []

    for dm_slot in sorted(DM_SLOTS):
        for cb_slot in sorted(CB_SLOTS):
            pair_key = frozenset(
                {
                    dm_slot,
                    cb_slot,
                }
            )

            evidence = evidence_by_pair.get(
                pair_key,
                set(),
            )

            adjacent_line = (
                ResponsibilityType.ADJACENT_LINE
                in evidence
            )

            same_corridor = (
                ResponsibilityType.SAME_CORRIDOR
                in evidence
            )

            dm_position = positions_by_slot[
                dm_slot
            ]

            cb_position = positions_by_slot[
                cb_slot
            ]

            shared_protection = bool(
                adjacent_line
                and dm_position.y
                > cb_position.y
            )

            side_specific_protection = bool(
                shared_protection
                and same_corridor
                and broad_corridor(
                    dm_position
                )
                == broad_corridor(
                    cb_position
                )
            )

            rows.append(
                {
                    "nation": nation,
                    "formation": FORMATION,
                    "am_fallback_applied":
                        am_fallback_applied,
                    "am_fallback_source_role":
                        am_fallback_source_role,
                    "dm_slot": dm_slot,
                    "cb_slot": cb_slot,
                    "dm_corridor":
                        broad_corridor(
                            dm_position
                        ),
                    "cb_corridor":
                        broad_corridor(
                            cb_position
                        ),
                    "adjacent_line":
                        adjacent_line,
                    "same_corridor":
                        same_corridor,
                    "dm_more_advanced":
                        bool(
                            dm_position.y
                            > cb_position.y
                        ),
                    "shared_protection_candidate":
                        shared_protection,
                    "side_specific_protection_candidate":
                        side_specific_protection,
                }
            )

    return rows

def build_population() -> tuple[
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

    positional_rows: list[
        dict[str, Any]
    ] = []

    structural_rows: list[
        dict[str, Any]
    ] = []

    comparison_rows: list[
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

        starting_xi = build_4231_lineup_with_am_fallback(
            lineup_builder=lineup_builder,
            squad=squad,
            formation_df=formation_df,
        )

        team_geometry = geometry_for_lineup(
            base_geometry=geometry,
            starting_xi=starting_xi,
        )

        am_assignment = next(
            assignment
            for assignment in starting_xi.assignments
            if assignment.slot == "AM1"
        )

        am_fallback_applied = bool(
            am_assignment.tactical_role != "AM"
        )

        am_fallback_source_role = (
            am_assignment.tactical_role
        )

        if len(
            starting_xi.assignments
        ) != EXPECTED_SLOT_COUNT:
            raise AssertionError(
                f"{nation!r} has an invalid "
                "4-2-3-1 lineup population."
            )

        positional_set = (
            generate_positional_responsibilities(
                starting_xi=starting_xi,
                geometry=team_geometry,
            )
        )

        for relationship in (
            positional_set.responsibilities
        ):
            if not (
                {
                    relationship.source_slot,
                    relationship.target_slot,
                }
                & DM_SLOTS
            ):
                continue

            positional_rows.append(
                {
                    "nation": nation,
                    "formation": FORMATION,
                    "am_fallback_applied":
                        am_fallback_applied,
                    "am_fallback_source_role":
                        am_fallback_source_role,
                    "source_slot":
                        relationship.source_slot,
                    "target_slot":
                        relationship.target_slot,
                    "responsibility_type":
                        relationship
                        .responsibility_type.value,
                }
            )

        structural_rows.extend(
            current_protection_rows(
                starting_xi=starting_xi,
                geometry=team_geometry,
                positional_set=positional_set,
                nation=nation,
                am_fallback_applied=(
                    am_fallback_applied
                ),
                am_fallback_source_role=(
                    am_fallback_source_role
                ),
            )
        )

        comparison_rows.extend(
            build_protection_candidates(
                nation=nation,
                geometry=team_geometry,
                positional_set=positional_set,
                am_fallback_applied=(
                    am_fallback_applied
                ),
                am_fallback_source_role=(
                    am_fallback_source_role
                ),
            )
        )

    return (
        pd.DataFrame(
            positional_rows
        ),
        pd.DataFrame(
            structural_rows
        ),
        pd.DataFrame(
            comparison_rows
        ),
    )

def build_4231_lineup_with_am_fallback(
    *,
    lineup_builder: StartingXIBuilder,
    squad,
    formation_df: pd.DataFrame,
) -> StartingXI:
    try:
        return lineup_builder.build_for_squad(
            squad=squad,
            formation_df=formation_df,
        )

    except ValueError as exc:
        message = str(exc)

        if "AM1 (AM)" not in message:
            raise

        fallback_formation = (
            formation_df.copy()
        )

        fallback_formation.loc[
            fallback_formation["slot"].eq("AM1"),
            "role",
        ] = "CM"

        return lineup_builder.build_for_squad(
            squad=squad,
            formation_df=fallback_formation,
        )

def geometry_for_lineup(
    *,
    base_geometry: FormationGeometry,
    starting_xi: StartingXI,
) -> FormationGeometry:
    role_by_slot = {
        assignment.slot:
            assignment.tactical_role
        for assignment in starting_xi.assignments
    }

    positions = tuple(
        type(position)(
            formation=position.formation,
            slot=position.slot,
            role=role_by_slot[
                position.slot
            ],
            x=position.x,
            y=position.y,
            tactical_line=position.tactical_line,
            side=position.side,
        )
        for position in base_geometry.positions
    )

    return FormationGeometry(
        formation=base_geometry.formation,
        positions=positions,
    )

def validate_population(
    *,
    positional_frame: pd.DataFrame,
    structural_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
) -> None:
    if (
        positional_frame.empty
        or structural_frame.empty
        or comparison_frame.empty
    ):
        raise AssertionError(
            "At least one Study 104B output is empty."
        )

    observed_teams = set(
        comparison_frame[
            "nation"
        ].astype(str)
    )

    if observed_teams != WORLD_CUP_TEAMS:
        raise AssertionError(
            "Study 104B does not cover all "
            "World Cup teams."
        )

    if comparison_frame[
        [
            "nation",
            "dm_slot",
            "cb_slot",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Duplicate DM-CB candidate pairs found."
        )

    candidates_per_team = (
        comparison_frame
        .groupby(
            "nation"
        )
        .size()
    )

    if not candidates_per_team.eq(
        4
    ).all():
        raise AssertionError(
            "Each team must contain four possible "
            "DM-CB pairings."
        )

    current_edges_per_team = (
        structural_frame
        .groupby(
            "nation"
        )
        .size()
    )

    if not current_edges_per_team.eq(
        4
    ).all():
        raise AssertionError(
            "The current adjacent-line rule does not "
            "generate four protection edges per team."
        )

    shared_per_team = (
        comparison_frame
        .groupby(
            "nation"
        )[
            "shared_protection_candidate"
        ]
        .sum()
    )

    if not shared_per_team.eq(
        4
    ).all():
        raise AssertionError(
            "Shared-protection interpretation does not "
            "produce four candidate edges per team."
        )

    side_specific_per_team = (
        comparison_frame
        .groupby(
            "nation"
        )[
            "side_specific_protection_candidate"
        ]
        .sum()
    )

    if not side_specific_per_team.eq(
        2
    ).all():
        raise AssertionError(
            "Side-specific interpretation does not "
            "produce two candidate edges per team."
        )

    expected_side_specific = {
        ("DM1", "CB1"),
        ("DM2", "CB2"),
    }

    observed_side_specific = {
        (
            str(row.dm_slot),
            str(row.cb_slot),
        )
        for row in (
            comparison_frame.loc[
                comparison_frame[
                    "side_specific_protection_candidate"
                ]
            ]
            .drop_duplicates(
                subset=[
                    "dm_slot",
                    "cb_slot",
                ]
            )
            .itertuples(
                index=False
            )
        )
    }

    if (
        observed_side_specific
        != expected_side_specific
    ):
        raise AssertionError(
            "Unexpected side-specific protection "
            f"pairs: {sorted(observed_side_specific)}."
        )

    if not structural_frame[
        "rule_id"
    ].eq(
        "dm_protects_cb_v1"
    ).all():
        raise AssertionError(
            "Unexpected structural protection rule."
        )

    if not structural_frame[
        "hypothesis_status"
    ].eq(
        StructuralHypothesisStatus
        .ACTIVE_DIAGNOSTIC
        .value
    ).all():
        raise AssertionError(
            "Current protection edges are not "
            "active-diagnostic."
        )

    fallback_counts = (
        comparison_frame[
            [
                "nation",
                "am_fallback_applied",
                "am_fallback_source_role",
            ]
        ]
        .drop_duplicates()
    )

    if fallback_counts[
        "nation"
    ].duplicated().any():
        raise AssertionError(
            "A team has inconsistent AM fallback metadata."
        )

    invalid_fallback_roles = (
        fallback_counts.loc[
            fallback_counts[
                "am_fallback_applied"
            ]
            & ~fallback_counts[
                "am_fallback_source_role"
            ].eq("CM")
        ]
    )

    if not invalid_fallback_roles.empty:
        raise AssertionError(
            "AM fallback used an unexpected tactical role."
        )

    invalid_non_fallback_roles = (
        fallback_counts.loc[
            ~fallback_counts[
                "am_fallback_applied"
            ]
            & ~fallback_counts[
                "am_fallback_source_role"
            ].eq("AM")
        ]
    )

    if not invalid_non_fallback_roles.empty:
        raise AssertionError(
            "A non-fallback lineup has an unexpected "
            "AM1 tactical role."
        )

def build_team_summary(
    *,
    structural_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
) -> pd.DataFrame:
    current_counts = (
        structural_frame
        .groupby(
            "nation"
        )
        .size()
        .rename(
            "current_protection_edges"
        )
    )

    candidate_counts = (
        comparison_frame
        .groupby(
            "nation"
        )
        .agg(
            possible_dm_cb_pairs=(
                "cb_slot",
                "count",
            ),
            shared_protection_edges=(
                "shared_protection_candidate",
                "sum",
            ),
            side_specific_protection_edges=(
                "side_specific_protection_candidate",
                "sum",
            ),
        )
    )

    return (
        candidate_counts
        .join(
            current_counts
        )
        .reset_index()
        .sort_values(
            "nation"
        )
        .reset_index(drop=True)
    )

def write_report(
    *,
    team_summary: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    am_fallback_team_count: int,
) -> None:
    topology = (
        comparison_frame[
            [
                "dm_slot",
                "cb_slot",
                "dm_corridor",
                "cb_corridor",
                "adjacent_line",
                "same_corridor",
                "shared_protection_candidate",
                "side_specific_protection_candidate",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "dm_slot",
                "cb_slot",
            ]
        )
    )

    summary = (
        team_summary[
            [
                "possible_dm_cb_pairs",
                "current_protection_edges",
                "shared_protection_edges",
                "side_specific_protection_edges",
            ]
        ]
        .drop_duplicates()
    )

    report = f"""# Study 104B — Double-Pivot Protection Audit

## Status

**PASS**

## Purpose

Compare two diagnostic interpretations of center-back protection in
the registered 4-2-3-1 double-pivot geometry.

## Candidate topology

{topology.to_markdown(index=False)}

## Per-team comparison

{summary.to_markdown(index=False)}

## Interpretation 1 — Shared protection

Both defensive midfielders protect both center-backs.

Result:

- Four protection edges per team.
- Matches the current adjacent-line-only hypothesis.
- Treats the double pivot as a shared central screen.

## Lineup fallback boundary

{am_fallback_team_count} teams required a diagnostic CM-to-AM1
eligibility fallback because their squad could not populate the AM
role directly.

The fallback is local to Study 104B. It does not modify the canonical
formation manifest, player eligibility rules, production lineups, or
team strength.

## Interpretation 2 — Side-specific protection

Each defensive midfielder protects the center-back in the same broad
corridor.

Result:

- Two protection edges per team.
- DM1 protects CB1.
- DM2 protects CB2.
- Introduces a stronger geometric assumption.

## Permitted conclusion

The current single-pivot rule generalizes mechanically to 4-2-3-1,
but its four-edge output represents one particular shared-protection
interpretation rather than a formation-neutral truth.

## Prohibited conclusion

This study does not establish which protection interpretation better
describes real 4-2-3-1 teams.

No hypothesis was promoted, revised, weighted, or used for prediction.
"""
    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 104B — DOUBLE-PIVOT "
        "PROTECTION AUDIT"
    )
    print("=" * 88)

    (
        positional_frame,
        structural_frame,
        comparison_frame,
    ) = build_population()

    validate_population(
        positional_frame=positional_frame,
        structural_frame=structural_frame,
        comparison_frame=comparison_frame,
    )

    team_summary = build_team_summary(
        structural_frame=structural_frame,
        comparison_frame=comparison_frame,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    positional_frame.to_csv(
        POSITIONAL_PATH,
        index=False,
    )

    structural_frame.to_csv(
        CURRENT_STRUCTURAL_PATH,
        index=False,
    )

    comparison_frame.to_csv(
        PROTECTION_COMPARISON_PATH,
        index=False,
    )

    team_summary.to_csv(
        TEAM_SUMMARY_PATH,
        index=False,
    )

    fallback_team_frame = (
        comparison_frame[
            [
                "nation",
                "am_fallback_applied",
                "am_fallback_source_role",
            ]
        ]
        .drop_duplicates()
    )

    am_fallback_team_count = int(
        fallback_team_frame[
            "am_fallback_applied"
        ].sum()
    )

    metadata = {
        "study_id": "104B",
        "study_name": (
            "Double-Pivot Protection Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "formation": FORMATION,
        "team_count": int(
            comparison_frame[
                "nation"
            ].nunique()
        ),
        "am_fallback_team_count":
            am_fallback_team_count,
        "am_fallback_role": "CM",
        "canonical_formation_manifest_changed":
            False,
        "fallback_local_to_study": True,
        "defensive_midfielders_per_team": 2,
        "center_backs_per_team": 2,
        "possible_dm_cb_pairs_per_team": 4,
        "current_rule_edges_per_team": 4,
        "shared_protection_edges_per_team": 4,
        "side_specific_edges_per_team": 2,
        "current_rule_matches_shared_interpretation":
            True,
        "side_specific_pairs": [
            "DM1->CB1",
            "DM2->CB2",
        ],
        "hypothesis_promoted": False,
        "structural_rules_changed": False,
        "defensive_graph_created": False,
        "diagnostics_generated": False,
        "weights_created": False,
        "team_strength_changed": False,
        "repository_changed": False,
        "simulation_run": False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "The study compares expert candidate "
            "interpretations of double-pivot protection. "
            "It does not establish which interpretation "
            "best describes real football."
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
        team_summary=team_summary,
        comparison_frame=comparison_frame,
        am_fallback_team_count=(
            am_fallback_team_count
        ),
    )

    print()
    print("Double-pivot summary")
    print("-" * 88)
    print(
        f"  Teams audited: "
        f"{metadata['team_count']}"
    )
    print(
        "  Teams requiring CM-to-AM1 fallback: "
        f"{metadata['am_fallback_team_count']}"
    )
    print(
        "  Canonical formation manifest changed: NO"
    )
    print(
        "  Defensive midfielders per team: 2"
    )
    print(
        "  Center-backs per team: 2"
    )
    print(
        "  Possible DM-CB pairs per team: 4"
    )
    print(
        "  Current-rule protection edges: 4"
    )
    print(
        "  Shared-protection candidate edges: 4"
    )
    print(
        "  Side-specific candidate edges: 2"
    )
    print(
        "  Current rule matches shared interpretation: PASS"
    )
    print(
        "  Side-specific pairs: DM1->CB1 | DM2->CB2"
    )
    print(
        "  Candidate interpretations separated: PASS"
    )
    print(
        "  Hypothesis promoted: NO"
    )
    print(
        "  Structural rules changed: NO"
    )
    print(
        "  Defensive graph created: NO"
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
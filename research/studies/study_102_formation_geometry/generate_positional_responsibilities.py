#generate_positional_responsibilities

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.player_intelligence.football_responsibility import (
    ResponsibilityType,
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
    / "study_102_formation_geometry"
    / "study_102c"
)

RELATIONSHIPS_PATH = (
    OUTPUT_DIRECTORY
    / "world_cup_positional_responsibilities.csv"
)

TEAM_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "team_positional_responsibility_summary.csv"
)

PAIR_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "formation_pair_responsibility_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_102c_metadata.json"
)


FORMATION = "4-3-3"
EXPECTED_TEAM_COUNT = 48
EXPECTED_SLOT_COUNT = 11

WORLD_CUP_TEAMS = {
    normalize_team_name(team)
    for teams in GROUPS.values()
    for team in teams
}


def build_relationship_population() -> pd.DataFrame:
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

    rows: list[dict[str, object]] = []

    for team in roster_builder.list_teams():
        canonical_team = normalize_team_name(
            team
        )

        if canonical_team not in WORLD_CUP_TEAMS:
            continue

        squad = roster_builder.get_squad(
            team
        )

        starting_xi = (
            lineup_builder.build_for_squad(
                squad=squad,
                formation_df=formation_df,
            )
        )

        result = (
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

        for responsibility in (
            result.responsibilities
        ):
            source_assignment = (
                assignments_by_slot[
                    responsibility.source_slot
                ]
            )

            target_assignment = (
                assignments_by_slot[
                    responsibility.target_slot
                ]
            )

            rows.append(
                {
                    "nation":
                        canonical_team,
                    "formation":
                        FORMATION,
                    "source_slot":
                        responsibility.source_slot,
                    "source_role":
                        source_assignment.tactical_role,
                    "source_player_id":
                        source_assignment.player.identity.player_id,
                    "source_player":
                        source_assignment.player.identity.name,
                    "target_slot":
                        responsibility.target_slot,
                    "target_role":
                        target_assignment.tactical_role,
                    "target_player_id":
                        target_assignment.player.identity.player_id,
                    "target_player":
                        target_assignment.player.identity.name,
                    "responsibility_type":
                        responsibility.responsibility_type.value,
                    "directional":
                        False,
                    "weighted":
                        False,
                }
            )

    return pd.DataFrame(rows)


def validate_relationship_population(
    relationships: pd.DataFrame,
) -> None:
    if relationships.empty:
        raise AssertionError(
            "No positional responsibilities were generated."
        )

    observed_teams = set(
        relationships[
            "nation"
        ].astype(str)
    )

    if observed_teams != WORLD_CUP_TEAMS:
        missing = sorted(
            WORLD_CUP_TEAMS - observed_teams
        )

        extra = sorted(
            observed_teams - WORLD_CUP_TEAMS
        )

        raise AssertionError(
            "Responsibility population does not cover the "
            "World Cup field. "
            f"Missing={missing}, extra={extra}"
        )

    allowed_types = {
        ResponsibilityType.SAME_LINE.value,
        ResponsibilityType.ADJACENT_LINE.value,
        ResponsibilityType.SAME_CORRIDOR.value,
    }

    observed_types = set(
        relationships[
            "responsibility_type"
        ].astype(str)
    )

    if not observed_types.issubset(
        allowed_types
    ):
        raise AssertionError(
            "Unexpected responsibility types were generated: "
            f"{sorted(observed_types - allowed_types)}"
        )

    self_relationships = relationships.loc[
        relationships[
            "source_slot"
        ].eq(
            relationships[
                "target_slot"
            ]
        )
    ]

    if not self_relationships.empty:
        raise AssertionError(
            "Self-relationships were generated."
        )

    duplicate_columns = [
        "nation",
        "source_slot",
        "target_slot",
        "responsibility_type",
    ]

    if relationships.duplicated(
        subset=duplicate_columns
    ).any():
        raise AssertionError(
            "Duplicate positional responsibilities were "
            "generated."
        )

    if not relationships[
        "source_slot"
    ].lt(
        relationships[
            "target_slot"
        ]
    ).all():
        raise AssertionError(
            "At least one symmetric relationship is not in "
            "canonical slot order."
        )

    if relationships[
        "directional"
    ].any():
        raise AssertionError(
            "Study 102C generated a directional relationship."
        )

    if relationships[
        "weighted"
    ].any():
        raise AssertionError(
            "Study 102C generated a weighted relationship."
        )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 102C — POSITIONAL RESPONSIBILITY "
        "GENERATION"
    )
    print("=" * 88)

    relationships = (
        build_relationship_population()
    )

    validate_relationship_population(
        relationships
    )

    team_summary = (
        relationships
        .groupby(
            [
                "nation",
                "responsibility_type",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "relationship_count",
            }
        )
    )

    pair_summary = (
        relationships
        .groupby(
            [
                "formation",
                "source_slot",
                "source_role",
                "target_slot",
                "target_role",
                "responsibility_type",
            ],
            as_index=False,
        )
        .agg(
            team_count=(
                "nation",
                "nunique",
            )
        )
        .sort_values(
            [
                "responsibility_type",
                "source_slot",
                "target_slot",
            ]
        )
        .reset_index(drop=True)
    )

    per_team_total = (
        relationships
        .groupby(
            "nation"
        )
        .size()
    )

    if per_team_total.nunique() != 1:
        raise AssertionError(
            "Identical formations generated different positional "
            "relationship counts across teams."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    relationships.to_csv(
        RELATIONSHIPS_PATH,
        index=False,
    )

    team_summary.to_csv(
        TEAM_SUMMARY_PATH,
        index=False,
    )

    pair_summary.to_csv(
        PAIR_SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "study_id": "102C",
        "study_name": (
            "Positional Responsibility Generation"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "formation": FORMATION,
        "team_count": int(
            relationships[
                "nation"
            ].nunique()
        ),
        "slot_count": EXPECTED_SLOT_COUNT,
        "responsibility_type_count": int(
            relationships[
                "responsibility_type"
            ].nunique()
        ),
        "relationships_per_team": int(
            per_team_total.iloc[0]
        ),
        "total_relationship_count": len(
            relationships
        ),
        "relationship_types_generated": sorted(
            relationships[
                "responsibility_type"
            ].unique().tolist()
        ),
        "all_relationships_symmetric":
            True,
        "all_relationships_unweighted":
            True,
        "self_relationship_count": 0,
        "duplicate_relationship_count": 0,
        "football_graph_created": False,
        "structural_responsibilities_generated":
            False,
        "functional_responsibilities_generated":
            False,
        "emergent_responsibilities_generated":
            False,
        "team_strength_changed": False,
        "simulation_run": False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "Generated records are positional descriptions "
            "derived from formation geometry. They do not yet "
            "assert support, protection, coverage, connection, "
            "chemistry, or causal football influence."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Generation summary")
    print("-" * 88)
    print(
        f"  Formation: {FORMATION}"
    )
    print(
        f"  Teams: {metadata['team_count']}"
    )
    print(
        "  Responsibility types: "
        f"{metadata['responsibility_type_count']}"
    )
    print(
        "  Relationships per team: "
        f"{metadata['relationships_per_team']}"
    )
    print(
        "  Complete World Cup population: PASS"
    )
    print(
        "  Deterministic topology by formation: PASS"
    )
    print(
        "  Valid lineup-slot references: PASS"
    )
    print(
        "  Canonical symmetric ordering: PASS"
    )
    print(
        "  Self-relationships absent: PASS"
    )
    print(
        "  Duplicate relationships absent: PASS"
    )
    print(
        "  Relationship weights created: NO"
    )
    print(
        "  Structural semantics generated: NO"
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
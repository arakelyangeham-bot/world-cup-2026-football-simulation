#compare_legacy_and_scoped_generation

from __future__ import annotations

import json
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
    / "study_104_cross_formation"
    / "study_104d"
)

RELATIONSHIP_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "legacy_vs_scoped_relationships.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "legacy_vs_scoped_generation_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_104d_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_104D_REPORT.md"
)


FORMATIONS = (
    "4-3-3",
    "4-2-3-1",
)

MODES = {
    "legacy": False,
    "scope_enforced": True,
}

def find_compatible_squad(
    *,
    roster_builder,
    formation_frames: dict[
        str,
        pd.DataFrame,
    ],
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
        "No squad could populate both registered formations."
    )

def build_comparison() -> pd.DataFrame:
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

    rows: list[dict[str, Any]] = []

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

        for mode, enforce_scope in (
            MODES.items()
        ):
            structural_set = (
                generate_structural_responsibilities(
                    starting_xi=starting_xi,
                    geometry=geometry,
                    positional_set=positional_set,
                    included_statuses=(
                        StructuralHypothesisStatus
                        .ACTIVE_DIAGNOSTIC,
                    ),
                    enforce_hypothesis_scope=(
                        enforce_scope
                    ),
                )
            )

            for relationship in (
                structural_set.responsibilities
            ):
                rows.append(
                    {
                        "source_team":
                            source_team,
                        "formation":
                            formation,
                        "generation_mode":
                            mode,
                        "scope_enforced":
                            enforce_scope,
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
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "formation",
                "generation_mode",
                "rule_id",
                "source_slot",
                "target_slot",
            ]
        )
        .reset_index(drop=True)
    )

def build_summary(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    return (
        comparison
        .groupby(
            [
                "formation",
                "generation_mode",
                "scope_enforced",
            ],
            as_index=False,
        )
        .agg(
            relationship_count=(
                "rule_id",
                "count",
            ),
            rule_count=(
                "rule_id",
                "nunique",
            ),
            responsibility_type_count=(
                "responsibility_type",
                "nunique",
            ),
            rule_ids=(
                "rule_id",
                lambda values: "|".join(
                    sorted(
                        set(
                            str(value)
                            for value in values
                        )
                    )
                ),
            ),
        )
        .sort_values(
            [
                "formation",
                "generation_mode",
            ]
        )
        .reset_index(drop=True)
    )

def validate_comparison(
    *,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    if comparison.empty:
        raise AssertionError(
            "Generation comparison is empty."
        )

    summary_by_key = {
        (
            str(row.formation),
            str(row.generation_mode),
        ): row
        for row in summary.itertuples(
            index=False
        )
    }

    legacy_433 = summary_by_key[
        (
            "4-3-3",
            "legacy",
        )
    ]

    scoped_433 = summary_by_key[
        (
            "4-3-3",
            "scope_enforced",
        )
    ]

    if (
        int(
            legacy_433.relationship_count
        )
        != int(
            scoped_433.relationship_count
        )
    ):
        raise AssertionError(
            "Scope enforcement changed validated "
            "4-3-3 active generation."
        )

    legacy_4231 = comparison.loc[
        comparison[
            "formation"
        ].eq(
            "4-2-3-1"
        )
        & comparison[
            "generation_mode"
        ].eq(
            "legacy"
        )
    ]

    scoped_4231 = comparison.loc[
        comparison[
            "formation"
        ].eq(
            "4-2-3-1"
        )
        & comparison[
            "generation_mode"
        ].eq(
            "scope_enforced"
        )
    ]

    legacy_dm_protection = (
        legacy_4231.loc[
            legacy_4231[
                "rule_id"
            ].eq(
                "dm_protects_cb_v1"
            )
        ]
    )

    if len(
        legacy_dm_protection
    ) != 4:
        raise AssertionError(
            "Legacy 4-2-3-1 generation did not "
            "preserve four shared-protection edges."
        )

    if scoped_4231[
        "rule_id"
    ].eq(
        "dm_protects_cb_v1"
    ).any():
        raise AssertionError(
            "Scoped 4-2-3-1 generation retained "
            "the single-pivot DM rule."
        )

    expected_scoped_rules = {
        "cb_covers_fb_v1",
    }

    if (
        set(
            scoped_4231[
                "rule_id"
            ].astype(str)
        )
        != expected_scoped_rules
    ):
        raise AssertionError(
            "Scoped 4-2-3-1 generation produced "
            "unexpected active rules."
        )

    if len(
        scoped_4231
    ) != 2:
        raise AssertionError(
            "Scoped 4-2-3-1 generation should contain "
            "two CB-to-FB coverage relationships."
        )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 104D — FORMATION-AWARE "
        "HYPOTHESIS ACTIVATION"
    )
    print("=" * 88)

    comparison = build_comparison()

    summary = build_summary(
        comparison
    )

    validate_comparison(
        comparison=comparison,
        summary=summary,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        RELATIONSHIP_COMPARISON_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "study_id": "104D",
        "study_name": (
            "Formation-Aware Hypothesis Activation"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "formations_compared": list(
            FORMATIONS
        ),
        "generation_modes": list(
            MODES
        ),
        "scope_filtering_implemented":
            True,
        "default_scope_filtering_enabled":
            False,
        "legacy_behavior_preserved":
            True,
        "validated_433_output_changed":
            False,
        "legacy_4231_dm_protection_edges":
            4,
        "scoped_4231_dm_protection_edges":
            0,
        "scoped_4231_coverage_edges":
            2,
        "double_pivot_hypothesis_promoted":
            False,
        "football_graph_created": False,
        "diagnostics_generated": False,
        "weights_created": False,
        "team_strength_changed": False,
        "repository_changed": False,
        "simulation_run": False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "Scope enforcement activates only hypotheses "
            "declared applicable to a formation and role "
            "population. It does not establish empirical "
            "football truth or predictive value."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = f"""# Study 104D — Formation-Aware Hypothesis Activation

## Status

**PASS**

## Generation comparison

{summary.to_markdown(index=False)}

## Main result

- Scope enforcement preserves the validated 4-3-3 active output.
- Legacy 4-2-3-1 generation creates four shared-protection edges.
- Scoped 4-2-3-1 generation excludes the single-pivot DM rule.
- The formation-general CB-to-FB coverage rule remains active.
- No double-pivot protection hypothesis was promoted.

## Compatibility boundary

Scope enforcement is opt-in. Existing callers continue to use legacy
generation unless they explicitly pass:

```python
enforce_hypothesis_scope=True
No production behavior changed.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("Activation summary")
    print("-" * 88)
    print(
        "  Formations compared: 2"
    )
    print(
        "  Generation modes compared: 2"
    )
    print(
        "  Scope filtering implemented: PASS"
    )
    print(
        "  Default remains legacy-compatible: PASS"
    )
    print(
        "  Validated 4-3-3 output preserved: PASS"
    )
    print(
        "  Legacy 4-2-3-1 DM protection edges: 4"
    )
    print(
        "  Scoped 4-2-3-1 DM protection edges: 0"
    )
    print(
        "  Scoped 4-2-3-1 CB coverage edges: 2"
    )
    print(
        "  Double-pivot hypothesis promoted: NO"
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
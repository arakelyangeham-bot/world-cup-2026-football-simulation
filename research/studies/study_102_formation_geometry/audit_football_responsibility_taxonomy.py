#audit_football_responsibility_taxonomy

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.player_intelligence.football_responsibility import (
    RESPONSIBILITY_DEFINITIONS,
    ResponsibilityFamily,
    ResponsibilityType,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_102_formation_geometry"
    / "study_102b"
)

TAXONOMY_PATH = (
    OUTPUT_DIRECTORY
    / "football_responsibility_taxonomy.csv"
)

FAMILY_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "responsibility_family_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_102b_metadata.json"
)


def build_taxonomy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "responsibility_type":
                    definition.responsibility_type.value,
                "family":
                    definition.family.value,
                "definition":
                    definition.definition,
                "directional":
                    definition.directional,
                "symmetric":
                    definition.symmetric,
                "structural":
                    definition.structural,
                "style_dependent":
                    definition.style_dependent,
                "directly_observable":
                    definition.directly_observable,
                "stable_across_matches":
                    definition.stable_across_matches,
                "generation_enabled":
                    definition.generation_enabled,
            }
            for definition
            in RESPONSIBILITY_DEFINITIONS
        ]
    )


def validate_taxonomy(
    taxonomy: pd.DataFrame,
) -> None:
    if len(taxonomy) != len(
        ResponsibilityType
    ):
        raise AssertionError(
            "Taxonomy row count does not match the "
            "responsibility vocabulary."
        )

    if taxonomy[
        "responsibility_type"
    ].duplicated().any():
        raise AssertionError(
            "Taxonomy contains duplicate responsibility types."
        )

    invalid_directionality = taxonomy.loc[
        taxonomy["directional"]
        & taxonomy["symmetric"]
    ]

    if not invalid_directionality.empty:
        raise AssertionError(
            "Taxonomy contains relationships declared both "
            "directional and symmetric."
        )

    emergent_enabled = taxonomy.loc[
        taxonomy["family"].eq(
            ResponsibilityFamily.EMERGENT.value
        )
        & taxonomy[
            "generation_enabled"
        ]
    ]

    if not emergent_enabled.empty:
        raise AssertionError(
            "Emergent relationships are enabled without "
            "observable proxies."
        )

    if taxonomy[
        "definition"
    ].astype(str).str.strip().eq("").any():
        raise AssertionError(
            "Taxonomy contains an empty definition."
        )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 102B — FOOTBALL RESPONSIBILITY "
        "TAXONOMY"
    )
    print("=" * 88)

    taxonomy = build_taxonomy_frame()

    validate_taxonomy(
        taxonomy
    )

    family_summary = (
        taxonomy
        .groupby(
            "family",
            as_index=False,
        )
        .agg(
            responsibility_count=(
                "responsibility_type",
                "count",
            ),
            enabled_count=(
                "generation_enabled",
                "sum",
            ),
            directional_count=(
                "directional",
                "sum",
            ),
            symmetric_count=(
                "symmetric",
                "sum",
            ),
            directly_observable_count=(
                "directly_observable",
                "sum",
            ),
        )
        .sort_values("family")
        .reset_index(drop=True)
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    taxonomy.to_csv(
        TAXONOMY_PATH,
        index=False,
    )

    family_summary.to_csv(
        FAMILY_SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "study_id": "102B",
        "study_name": (
            "Football Responsibility Taxonomy"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "responsibility_type_count": len(
            taxonomy
        ),
        "family_count": int(
            taxonomy[
                "family"
            ].nunique()
        ),
        "generation_enabled_count": int(
            taxonomy[
                "generation_enabled"
            ].sum()
        ),
        "emergent_generation_enabled_count": int(
            (
                taxonomy[
                    "family"
                ].eq(
                    ResponsibilityFamily.EMERGENT.value
                )
                & taxonomy[
                    "generation_enabled"
                ]
            ).sum()
        ),
        "relationships_generated": False,
        "football_graph_created": False,
        "interaction_weights_created": False,
        "team_strength_changed": False,
        "simulation_run": False,
        "production_configuration_changed": False,
        "interpretation_boundary": (
            "The taxonomy defines football relationship "
            "semantics only. It does not assert that any "
            "relationship exists between actual players."
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
    print("Taxonomy summary")
    print("-" * 88)
    print(
        "  Responsibility types: "
        f"{metadata['responsibility_type_count']}"
    )
    print(
        "  Responsibility families: "
        f"{metadata['family_count']}"
    )
    print(
        "  Initially enabled types: "
        f"{metadata['generation_enabled_count']}"
    )
    print(
        "  Emergent relationships enabled: "
        f"{metadata['emergent_generation_enabled_count']}"
    )
    print(
        "  Complete vocabulary coverage: PASS"
    )
    print(
        "  Directionality validation: PASS"
    )
    print(
        "  Emergent relationships disabled: PASS"
    )
    print(
        "  Relationships generated: NO"
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
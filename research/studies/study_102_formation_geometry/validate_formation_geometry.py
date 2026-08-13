#validate_formation_geometry

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.player_intelligence.formation_geometry import (
    load_formation_geometry,
    validate_geometry_against_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

GEOMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "formation_geometry.csv"
)

FORMATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "formation_manifest.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_102_formation_geometry"
    / "study_102a"
)

GEOMETRY_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "formation_geometry_audit.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_102a_metadata.json"
)

FORMATION = "4-3-3"
EXPECTED_SLOT_COUNT = 11


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 102A — FORMATION GEOMETRY "
        "FOUNDATIONS"
    )
    print("=" * 88)

    geometry = load_formation_geometry(
        path=GEOMETRY_PATH,
        formation=FORMATION,
    )

    formation_manifest = pd.read_csv(
        FORMATION_MANIFEST_PATH,
        low_memory=False,
    )

    validate_geometry_against_manifest(
        geometry=geometry,
        formation_manifest=formation_manifest,
    )

    if len(
        geometry.positions
    ) != EXPECTED_SLOT_COUNT:
        raise AssertionError(
            "Unexpected geometry slot count: "
            f"{len(geometry.positions)} vs "
            f"{EXPECTED_SLOT_COUNT}."
        )

    audit = pd.DataFrame(
        [
            {
                "formation":
                    position.formation,
                "slot":
                    position.slot,
                "role":
                    position.role,
                "x":
                    position.x,
                "y":
                    position.y,
                "tactical_line":
                    position.tactical_line,
                "side":
                    position.side,
            }
            for position in geometry.positions
        ]
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        GEOMETRY_AUDIT_PATH,
        index=False,
    )

    metadata = {
        "study_id": "102A",
        "study_name": (
            "Formation Geometry Foundations"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "formation": FORMATION,
        "slot_count": len(
            geometry.positions
        ),
        "geometry_manifest_separate":
            True,
        "lineup_manifest_modified":
            False,
        "graph_edges_created":
            False,
        "interaction_weights_created":
            False,
        "team_strength_changed":
            False,
        "simulation_run":
            False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "Coordinates are normalized tactical geometry "
            "for future topology experiments. They are not "
            "empirical tracking coordinates."
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
    print("Geometry summary")
    print("-" * 88)
    print(
        f"  Formation: {FORMATION}"
    )
    print(
        f"  Slots: {len(geometry.positions)}"
    )
    print(
        "  Geometry matches lineup manifest: PASS"
    )
    print(
        "  Unique slots: PASS"
    )
    print(
        "  Finite coordinates: PASS"
    )
    print(
        "  Formation manifest modified: NO"
    )
    print(
        "  Graph edges created: NO"
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
#validate_4231_registration

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

FORMATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "formation_manifest.csv"
)

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
    / "study_104a"
)

FORMATION_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "formation_4231_registration_audit.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_104a_metadata.json"
)


FORMATION = "4-2-3-1"
EXPECTED_SLOT_COUNT = 11

EXPECTED_ROLE_COUNTS = {
    "GK": 1,
    "FB": 2,
    "CB": 2,
    "DM": 2,
    "W": 2,
    "AM": 1,
    "ST": 1,
}

EXPECTED_SLOTS = {
    "GK",
    "FB1",
    "CB1",
    "CB2",
    "FB2",
    "DM1",
    "DM2",
    "W1",
    "AM1",
    "W2",
    "ST1",
}


def validate_registration(
    *,
    manifest: pd.DataFrame,
    geometry_frame: pd.DataFrame,
) -> None:
    manifest_rows = manifest.loc[
        manifest["formation"]
        .astype(str)
        .eq(FORMATION)
    ].copy()

    geometry_rows = geometry_frame.loc[
        geometry_frame["formation"]
        .astype(str)
        .eq(FORMATION)
    ].copy()

    if len(manifest_rows) != EXPECTED_SLOT_COUNT:
        raise AssertionError(
            "Unexpected 4-2-3-1 manifest slot count: "
            f"{len(manifest_rows)}."
        )

    if len(geometry_rows) != EXPECTED_SLOT_COUNT:
        raise AssertionError(
            "Unexpected 4-2-3-1 geometry slot count: "
            f"{len(geometry_rows)}."
        )

    manifest_slots = set(
        manifest_rows["slot"].astype(str)
    )

    geometry_slots = set(
        geometry_rows["slot"].astype(str)
    )

    if manifest_slots != EXPECTED_SLOTS:
        raise AssertionError(
            "4-2-3-1 manifest slot population differs "
            f"from expectation: {sorted(manifest_slots)}."
        )

    if geometry_slots != EXPECTED_SLOTS:
        raise AssertionError(
            "4-2-3-1 geometry slot population differs "
            f"from expectation: {sorted(geometry_slots)}."
        )

    if manifest_rows["slot"].duplicated().any():
        raise AssertionError(
            "4-2-3-1 manifest contains duplicate slots."
        )

    if geometry_rows["slot"].duplicated().any():
        raise AssertionError(
            "4-2-3-1 geometry contains duplicate slots."
        )

    role_counts = (
        manifest_rows["role"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    if role_counts != EXPECTED_ROLE_COUNTS:
        raise AssertionError(
            "Unexpected 4-2-3-1 role population: "
            f"{role_counts}."
        )

    numeric_coordinates = geometry_rows[
        [
            "x",
            "y",
            "tactical_line",
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric_coordinates.isna().any().any():
        raise AssertionError(
            "4-2-3-1 geometry contains invalid coordinates."
        )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 104A — 4-2-3-1 FORMATION "
        "REGISTRATION"
    )
    print("=" * 88)

    manifest = pd.read_csv(
        FORMATION_MANIFEST_PATH,
        low_memory=False,
    )

    geometry_frame = pd.read_csv(
        GEOMETRY_PATH,
        low_memory=False,
    )

    validate_registration(
        manifest=manifest,
        geometry_frame=geometry_frame,
    )

    geometry = load_formation_geometry(
        path=GEOMETRY_PATH,
        formation=FORMATION,
    )

    validate_geometry_against_manifest(
        geometry=geometry,
        formation_manifest=manifest,
    )

    formation_rows = (
        geometry_frame.loc[
            geometry_frame[
                "formation"
            ].astype(str).eq(
                FORMATION
            )
        ]
        .sort_values(
            [
                "tactical_line",
                "x",
                "slot",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    formation_rows.to_csv(
        FORMATION_AUDIT_PATH,
        index=False,
    )

    metadata = {
        "study_id": "104A",
        "study_name": (
            "4-2-3-1 Formation Registration"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "formation": FORMATION,
        "slot_count": len(
            formation_rows
        ),
        "role_counts": EXPECTED_ROLE_COUNTS,
        "manifest_geometry_parity": True,
        "duplicate_manifest_slots": 0,
        "duplicate_geometry_slots": 0,
        "structural_responsibilities_generated":
            False,
        "defensive_graph_created": False,
        "diagnostics_generated": False,
        "team_strength_changed": False,
        "repository_changed": False,
        "simulation_run": False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "This study registers an abstract 4-2-3-1 "
            "formation and geometry only. It does not establish "
            "tactical superiority, structural responsibility, "
            "or predictive value."
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
    print("Registration summary")
    print("-" * 88)
    print(
        f"  Formation: {FORMATION}"
    )
    print(
        f"  Manifest slots: "
        f"{len(formation_rows)}"
    )
    print(
        "  Geometry slots: 11"
    )
    print(
        "  Manifest/geometry slot parity: PASS"
    )
    print(
        "  Manifest/geometry role parity: PASS"
    )
    print(
        "  Unique formation slots: PASS"
    )
    print(
        "  Finite geometry coordinates: PASS"
    )
    print(
        "  Existing 4-3-3 modified: NO"
    )
    print(
        "  Structural responsibilities generated: NO"
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
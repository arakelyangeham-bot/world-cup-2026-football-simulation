#formation_geometry

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd


GEOMETRY_COLUMNS = (
    "formation",
    "slot",
    "role",
    "x",
    "y",
    "tactical_line",
    "side",
)


@dataclass(frozen=True)
class FormationPosition:
    """
    One tactical slot inside an abstract formation geometry.

    Coordinates are normalized tactical coordinates rather than
    measured physical locations.
    """

    formation: str
    slot: str
    role: str

    x: float
    y: float

    tactical_line: int
    side: str

    def __post_init__(self) -> None:
        if not self.formation.strip():
            raise ValueError(
                "Formation must not be empty."
            )

        if not self.slot.strip():
            raise ValueError(
                "Formation slot must not be empty."
            )

        if not self.role.strip():
            raise ValueError(
                "Formation role must not be empty."
            )

        if not math.isfinite(self.x):
            raise ValueError(
                "Formation x-coordinate must be finite."
            )

        if not math.isfinite(self.y):
            raise ValueError(
                "Formation y-coordinate must be finite."
            )

        if self.tactical_line < 0:
            raise ValueError(
                "Tactical line cannot be negative."
            )

        if not self.side.strip():
            raise ValueError(
                "Formation side must not be empty."
            )


@dataclass(frozen=True)
class FormationGeometry:
    """
    Immutable geometry for one formation.

    This object contains no graph edges, interaction weights,
    football-strength adjustments, or simulation behavior.
    """

    formation: str
    positions: tuple[
        FormationPosition,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.formation.strip():
            raise ValueError(
                "Formation must not be empty."
            )

        if not self.positions:
            raise ValueError(
                "Formation geometry must contain positions."
            )

        mismatched = [
            position.slot
            for position in self.positions
            if position.formation != self.formation
        ]

        if mismatched:
            raise ValueError(
                "Formation geometry contains positions from "
                "another formation."
            )

        slots = tuple(
            position.slot
            for position in self.positions
        )

        if len(slots) != len(set(slots)):
            raise ValueError(
                "Formation geometry contains duplicate slots."
            )

    def position_by_slot(
        self,
        slot: str,
    ) -> FormationPosition:
        for position in self.positions:
            if position.slot == slot:
                return position

        raise KeyError(
            f"Unknown formation slot: {slot!r}"
        )

    def ordered_slots(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            position.slot
            for position in self.positions
        )


def load_formation_geometry(
    *,
    path: Path,
    formation: str,
) -> FormationGeometry:
    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    missing = (
        set(GEOMETRY_COLUMNS)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Formation geometry file is missing required "
            f"columns: {sorted(missing)}"
        )

    selected = frame.loc[
        frame["formation"]
        .astype(str)
        .eq(formation)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No geometry found for formation {formation!r}."
        )

    if selected["slot"].isna().any():
        raise ValueError(
            "Formation geometry contains missing slots."
        )

    if selected["slot"].duplicated().any():
        duplicates = (
            selected.loc[
                selected["slot"].duplicated(
                    keep=False
                ),
                "slot",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Formation geometry contains duplicate slots: "
            f"{duplicates}"
        )

    positions = tuple(
        FormationPosition(
            formation=str(row.formation),
            slot=str(row.slot),
            role=str(row.role),
            x=float(row.x),
            y=float(row.y),
            tactical_line=int(
                row.tactical_line
            ),
            side=str(row.side),
        )
        for row in selected.itertuples(
            index=False
        )
    )

    return FormationGeometry(
        formation=formation,
        positions=positions,
    )


def validate_geometry_against_manifest(
    *,
    geometry: FormationGeometry,
    formation_manifest: pd.DataFrame,
) -> None:
    """
    Prove that the geometry describes exactly the same formation
    slots and roles as the lineup-selection manifest.
    """

    required_columns = {
        "slot",
        "role",
    }

    missing = (
        required_columns
        - set(formation_manifest.columns)
    )

    if missing:
        raise ValueError(
            "Formation manifest is missing required "
            f"columns: {sorted(missing)}"
        )

    if "formation" in formation_manifest.columns:
        selected = formation_manifest.loc[
            formation_manifest[
                "formation"
            ]
            .astype(str)
            .eq(
                geometry.formation
            )
        ].copy()
    else:
        selected = (
            formation_manifest.copy()
        )

    if selected.empty:
        raise ValueError(
            "Formation manifest contains no matching rows "
            f"for {geometry.formation!r}."
        )

    manifest_mapping = {
        str(row.slot): str(row.role)
        for row in selected.itertuples(
            index=False
        )
    }

    geometry_mapping = {
        position.slot: position.role
        for position in geometry.positions
    }

    if set(manifest_mapping) != set(
        geometry_mapping
    ):
        missing_geometry = sorted(
            set(manifest_mapping)
            - set(geometry_mapping)
        )

        extra_geometry = sorted(
            set(geometry_mapping)
            - set(manifest_mapping)
        )

        raise AssertionError(
            "Formation geometry and lineup manifest have "
            "different slot populations. "
            f"Missing geometry={missing_geometry}, "
            f"extra geometry={extra_geometry}"
        )

    role_mismatches = [
        {
            "slot": slot,
            "manifest_role":
                manifest_mapping[slot],
            "geometry_role":
                geometry_mapping[slot],
        }
        for slot in manifest_mapping
        if (
            manifest_mapping[slot]
            != geometry_mapping[slot]
        )
    ]

    if role_mismatches:
        raise AssertionError(
            "Formation geometry role assignments differ "
            "from the lineup manifest: "
            f"{role_mismatches}"
        )
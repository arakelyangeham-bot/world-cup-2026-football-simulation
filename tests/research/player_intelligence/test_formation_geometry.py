#test_formation_geometry

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.player_intelligence.formation_geometry import (
    FormationGeometry,
    FormationPosition,
    load_formation_geometry,
    validate_geometry_against_manifest,
)


def make_position(
    *,
    slot: str,
    role: str,
    x: float,
    y: float,
) -> FormationPosition:
    return FormationPosition(
        formation="4-3-3",
        slot=slot,
        role=role,
        x=x,
        y=y,
        tactical_line=1,
        side="center",
    )


def test_geometry_rejects_duplicate_slots() -> None:
    position = make_position(
        slot="CM1",
        role="CM",
        x=0.0,
        y=3.0,
    )

    with pytest.raises(
        ValueError,
        match="duplicate slots",
    ):
        FormationGeometry(
            formation="4-3-3",
            positions=(
                position,
                position,
            ),
        )


def test_position_lookup_is_deterministic() -> None:
    left = make_position(
        slot="CM1",
        role="CM",
        x=-0.8,
        y=3.0,
    )

    right = make_position(
        slot="CM2",
        role="CM",
        x=0.8,
        y=3.0,
    )

    geometry = FormationGeometry(
        formation="4-3-3",
        positions=(
            left,
            right,
        ),
    )

    assert (
        geometry.position_by_slot(
            "CM1"
        )
        == left
    )

    assert geometry.ordered_slots() == (
        "CM1",
        "CM2",
    )


def test_geometry_matches_lineup_manifest() -> None:
    geometry = FormationGeometry(
        formation="4-3-3",
        positions=(
            make_position(
                slot="DM1",
                role="DM",
                x=0.0,
                y=2.0,
            ),
            make_position(
                slot="CM1",
                role="CM",
                x=-0.8,
                y=3.0,
            ),
        ),
    )

    manifest = pd.DataFrame(
        [
            {
                "formation": "4-3-3",
                "slot": "DM1",
                "role": "DM",
            },
            {
                "formation": "4-3-3",
                "slot": "CM1",
                "role": "CM",
            },
        ]
    )

    validate_geometry_against_manifest(
        geometry=geometry,
        formation_manifest=manifest,
    )


def test_role_mismatch_is_rejected() -> None:
    geometry = FormationGeometry(
        formation="4-3-3",
        positions=(
            make_position(
                slot="DM1",
                role="CM",
                x=0.0,
                y=2.0,
            ),
        ),
    )

    manifest = pd.DataFrame(
        [
            {
                "formation": "4-3-3",
                "slot": "DM1",
                "role": "DM",
            },
        ]
    )

    with pytest.raises(
        AssertionError,
        match="role assignments differ",
    ):
        validate_geometry_against_manifest(
            geometry=geometry,
            formation_manifest=manifest,
        )


def test_geometry_csv_loader(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "formation_geometry.csv"
    )

    pd.DataFrame(
        [
            {
                "formation": "4-3-3",
                "slot": "GK1",
                "role": "GK",
                "x": 0.0,
                "y": 0.0,
                "tactical_line": 0,
                "side": "center",
            },
            {
                "formation": "4-3-3",
                "slot": "ST1",
                "role": "ST",
                "x": 0.0,
                "y": 4.5,
                "tactical_line": 4,
                "side": "center",
            },
        ]
    ).to_csv(
        path,
        index=False,
    )

    geometry = load_formation_geometry(
        path=path,
        formation="4-3-3",
    )

    assert geometry.formation == "4-3-3"
    assert len(geometry.positions) == 2
    assert (
        geometry.position_by_slot(
            "ST1"
        ).y
        == pytest.approx(4.5)
    )
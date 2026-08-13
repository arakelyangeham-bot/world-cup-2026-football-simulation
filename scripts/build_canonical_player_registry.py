#build_canonical_player_registry

from __future__ import annotations

from pathlib import Path

import pandas as pd
import argparse
import ast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_REGISTRY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_profiles.csv"
)

DEFAULT_OUTPUT_REGISTRY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical_player_registry.csv"
)

ROLE_MAP = {
    "GK": "GK",
    "DC": "CB",
    "DL": "FB",
    "DR": "FB",
    "DM": "DM",
    "MC": "CM",
    "AM": "AM",
    "ML": "WM",
    "MR": "WM",
    "LW": "W",
    "RW": "W",
    "ST": "ST",
}

ROLE_PRIORITY = [
    "GK",
    "CB",
    "FB",
    "DM",
    "CM",
    "AM",
    "WM",
    "W",
    "ST",
]

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one authoritative canonical-player registry "
            "from the current player-profile population."
        )
    )

    parser.add_argument(
        "--source-registry",
        type=Path,
        default=DEFAULT_SOURCE_REGISTRY,
        help=(
            "Profile or registry input used to construct one "
            "authoritative row per canonical player."
        ),
    )

    parser.add_argument(
        "--output-registry",
        type=Path,
        default=DEFAULT_OUTPUT_REGISTRY,
        help=(
            "Destination canonical-player registry."
        ),
    )

    return parser.parse_args()

def parse_positions(value):
    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

    except Exception:
        pass

    return [str(value)]


def infer_eligible_roles(
    positions_detailed,
):
    roles = []

    for position in parse_positions(
        positions_detailed
    ):
        role = ROLE_MAP.get(position)

        if (
            role
            and role not in roles
        ):
            roles.append(role)

    return roles


def infer_primary_role(
    eligible_roles,
):
    if not eligible_roles:
        return pd.NA

    for role in ROLE_PRIORITY:
        if role in eligible_roles:
            return role

    return eligible_roles[0]

def main() -> None:

    arguments = parse_arguments()

    source_registry = arguments.source_registry
    output_registry = arguments.output_registry

    registry = pd.read_csv(
        source_registry,
        low_memory=False,
    )

    if "canonical_player_id" not in registry.columns:
        registry[
            "canonical_player_id"
        ] = pd.to_numeric(
            registry["player_id"],
            errors="raise",
        ).astype(int)

    if "positions_detailed" not in registry.columns:
        raise ValueError(
            "Profile registry is missing "
            "positions_detailed."
        )

    registry["eligible_roles"] = (
        registry[
            "positions_detailed"
        ].apply(
            infer_eligible_roles
        )
    )

    registry["position"] = (
        registry[
            "eligible_roles"
        ].apply(
            infer_primary_role
        )
    )
    
    required_columns = {
        "player_id",
        "canonical_player_id",
    }

    missing_columns = (
        required_columns
        - set(registry.columns)
    )

    if missing_columns:
        raise ValueError(
            "Source registry is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    registry["player_id"] = pd.to_numeric(
        registry["player_id"],
        errors="raise",
    ).astype(int)

    registry["canonical_player_id"] = pd.to_numeric(
        registry["canonical_player_id"],
        errors="raise",
    ).astype(int)

    source_rows = len(registry)

    source_players = int(
        registry["player_id"].nunique()
    )

    canonical_players = int(
        registry[
            "canonical_player_id"
        ].nunique()
    )

    canonical_family_sizes = (
        registry.groupby(
            "canonical_player_id"
        )["player_id"]
        .nunique()
    )

    multi_source_families = (
        canonical_family_sizes[
            canonical_family_sizes > 1
        ]
    )

    print(
        f"Source registry rows: "
        f"{source_rows:,}"
    )

    print(
        f"Unique source players: "
        f"{source_players:,}"
    )

    print(
        f"Unique canonical players: "
        f"{canonical_players:,}"
    )

    print(
        "Canonical families with >1 source ID: "
        f"{len(multi_source_families):,}"
    )

    multi_source_ids = set(
        multi_source_families.index
    )

    missing_native_rows = []

    for canonical_player_id in multi_source_ids:
        family = registry.loc[
            registry[
                "canonical_player_id"
            ].eq(canonical_player_id)
        ]

        has_native_row = (
            family["player_id"]
            .eq(canonical_player_id)
            .any()
        )

        if not has_native_row:
            missing_native_rows.append(
                canonical_player_id
            )

    if missing_native_rows:
        raise AssertionError(
            "Multi-source canonical families are missing "
            "their native canonical source row: "
            f"{sorted(missing_native_rows)}"
        )

    print(
        "Multi-source families with native "
        "canonical row: PASS"
    )

    registry["_native_canonical_source"] = (
        registry["player_id"]
        == registry["canonical_player_id"]
    )

    canonical_registry = (
        registry
        .sort_values(
            [
                "canonical_player_id",
                "_native_canonical_source",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["canonical_player_id"],
            keep="first",
        )
        .drop(
            columns=["_native_canonical_source"]
        )
        .reset_index(drop=True)
    )

    if canonical_registry[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Canonical registry still contains duplicate "
            "canonical player IDs."
        )

    if len(canonical_registry) != canonical_players:
        raise AssertionError(
            "Canonical registry row count does not match "
            "the expected canonical player population: "
            f"{len(canonical_registry)} vs "
            f"{canonical_players}."
        )

    if set(
        canonical_registry["canonical_player_id"]
    ) != set(
        registry["canonical_player_id"]
    ):
        raise AssertionError(
            "Canonical player population changed during "
            "registry resolution."
        )

    removed_rows = (
        len(registry)
        - len(canonical_registry)
    )

    expected_removed_rows = (
        source_players
        - canonical_players
    )

    if removed_rows != expected_removed_rows:
        raise AssertionError(
            "Unexpected number of source registry rows "
            "removed during canonical resolution: "
            f"{removed_rows} vs expected "
            f"{expected_removed_rows}."
        )

    output_registry.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_registry.to_csv(
        output_registry,
        index=False,
    )

    print()
    print("Canonical player registry")
    print("-" * 88)
    print(
        f"Output rows: "
        f"{len(canonical_registry):,}"
    )
    print(
        "Unique canonical players: "
        f"{canonical_registry['canonical_player_id'].nunique():,}"
    )
    print(
        f"Source rows removed: "
        f"{removed_rows:,}"
    )
    print(
        "Duplicate canonical IDs: "
        f"{canonical_registry['canonical_player_id'].duplicated().sum()}"
    )
    print(
        f"Wrote: {output_registry}"
    )

if __name__ == "__main__":
    main()
#build_101f_canonical_player_registry

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_REGISTRY = (
    PROJECT_ROOT
    / "outputs"
    / "study_101f_source_player_registry.csv"
)

OUTPUT_REGISTRY = (
    PROJECT_ROOT
    / "outputs"
    / "study_101f_canonical_player_registry.csv"
)


def main() -> None:
    registry = pd.read_csv(
        SOURCE_REGISTRY,
        low_memory=False,
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

    OUTPUT_REGISTRY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_registry.to_csv(
        OUTPUT_REGISTRY,
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
        f"Wrote: {OUTPUT_REGISTRY}"
    )

if __name__ == "__main__":
    main()
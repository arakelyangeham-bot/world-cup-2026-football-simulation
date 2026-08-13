#resolve_player_evidence

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_stats.csv"
)

DEFAULT_ALIAS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_id_aliases.csv"
)

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical_player_evidence.csv"
)

CANONICAL_TASK_KEY = [
    "competition_id",
    "season_id",
    "canonical_player_id",
]

SOURCE_IDENTITY_COLUMNS = {
    "id",
    "player_id",
    "player",
    "player_slug",
    "canonical_player_id",
    "_evidence_source",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve source player identities in "
            "competition-season evidence to canonical "
            "player identities."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
    )

    parser.add_argument(
        "--alias-file",
        type=Path,
        default=DEFAULT_ALIAS_FILE,
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )

    return parser.parse_args()


def load_reviewed_aliases(
    alias_file: Path,
) -> pd.DataFrame:
    required_columns = {
        "source_player_id",
        "canonical_player_id",
        "review_status",
    }

    if (
        not alias_file.exists()
        or alias_file.stat().st_size == 0
    ):
        return pd.DataFrame(
            columns=sorted(required_columns)
        )

    aliases = pd.read_csv(
        alias_file,
        low_memory=False,
    )

    missing_columns = (
        required_columns
        - set(aliases.columns)
    )

    if missing_columns:
        raise ValueError(
            "Player ID alias file is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    reviewed = aliases.loc[
        aliases["review_status"].eq("reviewed")
    ].copy()

    reviewed["source_player_id"] = pd.to_numeric(
        reviewed["source_player_id"],
        errors="raise",
    ).astype(int)

    reviewed["canonical_player_id"] = pd.to_numeric(
        reviewed["canonical_player_id"],
        errors="raise",
    ).astype(int)

    if reviewed[
        "source_player_id"
    ].duplicated().any():
        duplicate_ids = sorted(
            reviewed.loc[
                reviewed[
                    "source_player_id"
                ].duplicated(keep=False),
                "source_player_id",
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate reviewed source player IDs "
            f"in alias file: {duplicate_ids}"
        )

    return reviewed


def main() -> None:
    arguments = parse_arguments()

    evidence = pd.read_csv(
        arguments.input_file,
        dtype={"season_year": str},
        low_memory=False,
    )

    required_evidence_columns = {
        "player_id",
        "competition_id",
        "season_id",
    }

    missing_evidence_columns = (
        required_evidence_columns
        - set(evidence.columns)
    )

    if missing_evidence_columns:
        raise ValueError(
            "Player evidence is missing required "
            f"columns: {sorted(missing_evidence_columns)}"
        )

    evidence["player_id"] = pd.to_numeric(
        evidence["player_id"],
        errors="raise",
    ).astype(int)

    reviewed_aliases = load_reviewed_aliases(
        arguments.alias_file
    )

    source_population = set(
        evidence["player_id"]
    )

    applicable_aliases = (
        reviewed_aliases.loc[
            reviewed_aliases[
                "source_player_id"
            ].isin(source_population)
        ]
        .copy()
    )

    alias_map = dict(
        zip(
            applicable_aliases[
                "source_player_id"
            ],
            applicable_aliases[
                "canonical_player_id"
            ],
        )
    )

    evidence["canonical_player_id"] = (
        evidence["player_id"]
        .map(alias_map)
        .fillna(evidence["player_id"])
        .astype(int)
    )

    collision_mask = evidence.duplicated(
        CANONICAL_TASK_KEY,
        keep=False,
    )

    collisions = (
        evidence.loc[
            collision_mask
        ]
        .sort_values(
            CANONICAL_TASK_KEY
            + ["player_id"]
        )
        .copy()
    )

    collision_groups = (
        collisions[
            CANONICAL_TASK_KEY
        ]
        .drop_duplicates()
    )

    print(
        f"Source evidence rows: "
        f"{len(evidence):,}"
    )

    print(
        "Unique source players: "
        f"{evidence['player_id'].nunique():,}"
    )

    print(
        "Unique canonical players: "
        f"{evidence['canonical_player_id'].nunique():,}"
    )

    print(
        "Reviewed aliases available: "
        f"{len(reviewed_aliases):,}"
    )

    print(
        "Reviewed aliases applicable to evidence: "
        f"{len(applicable_aliases):,}"
    )

    print(
        "Canonical collision groups: "
        f"{len(collision_groups):,}"
    )

    print(
        "Rows in canonical collisions: "
        f"{len(collisions):,}"
    )

    conflicting_groups = []

    for task_key, group in collisions.groupby(
        CANONICAL_TASK_KEY,
        sort=True,
    ):
        comparison_columns = [
            column
            for column in group.columns
            if column not in SOURCE_IDENTITY_COLUMNS
        ]

        reference = group.iloc[0]

        group_conflicts = []

        for row_index in range(
            1,
            len(group),
        ):
            candidate = group.iloc[
                row_index
            ]

            for column in comparison_columns:
                left = reference[column]
                right = candidate[column]

                if (
                    pd.isna(left)
                    and pd.isna(right)
                ):
                    continue

                if (
                    pd.api.types.is_number(left)
                    and pd.api.types.is_number(right)
                ):
                    if (
                        pd.isna(left)
                        or pd.isna(right)
                    ):
                        group_conflicts.append(
                            column
                        )
                        continue

                    equal_numeric = (
                        pd.Series(
                            [
                                float(left),
                                float(right),
                            ]
                        )
                        .round(12)
                        .nunique()
                        == 1
                    )

                    if not equal_numeric:
                        group_conflicts.append(
                            column
                        )

                    continue

                if str(left) != str(right):
                    group_conflicts.append(
                        column
                    )

        group_conflicts = sorted(
            set(group_conflicts)
        )

        if group_conflicts:
            conflicting_groups.append(
                {
                    "task_key": task_key,
                    "columns":
                        group_conflicts,
                }
            )

    if conflicting_groups:
        print()
        print(
            "Conflicting canonical collisions"
        )
        print("-" * 88)

        for conflict in conflicting_groups:
            print(
                f"{conflict['task_key']}: "
                f"{conflict['columns']}"
            )

        raise AssertionError(
            "Canonical collision evidence "
            "is not equivalent."
        )

    print(
        "Canonical collision equivalence: PASS"
    )

    collision_groups_without_native = []

    for task_key, group in collisions.groupby(
        CANONICAL_TASK_KEY,
        sort=True,
    ):
        canonical_player_id = (
            task_key[2]
        )

        has_native_source = (
            group["player_id"]
            .eq(canonical_player_id)
            .any()
        )

        if not has_native_source:
            collision_groups_without_native.append(
                task_key
            )

    if collision_groups_without_native:
        raise AssertionError(
            "Equivalent canonical collisions exist "
            "without a native canonical source row: "
            f"{collision_groups_without_native}"
        )

    print(
        "Canonical collision native-source "
        "availability: PASS"
    )

    evidence["_native_canonical_source"] = (
        evidence["player_id"]
        == evidence["canonical_player_id"]
    )

    canonical_evidence = (
        evidence
        .sort_values(
            CANONICAL_TASK_KEY
            + ["_native_canonical_source"],
            ascending=[
                True,
                True,
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=CANONICAL_TASK_KEY,
            keep="first",
        )
        .drop(
            columns=["_native_canonical_source"]
        )
        .reset_index(drop=True)
    )

    remaining_collisions = int(
        canonical_evidence.duplicated(
            CANONICAL_TASK_KEY
        ).sum()
    )

    if remaining_collisions:
        raise AssertionError(
            "Canonical evidence still contains duplicate "
            "canonical task keys after resolution."
        )

    removed_rows = (
        len(evidence)
        - len(canonical_evidence)
    )

    expected_removed_rows = sum(
        len(group) - 1
        for _, group in collisions.groupby(
            CANONICAL_TASK_KEY,
            sort=False,
        )
    )

    if removed_rows != expected_removed_rows:
        raise AssertionError(
            "Unexpected number of rows removed during "
            "canonical evidence resolution: "
            f"{removed_rows} vs expected "
            f"{expected_removed_rows}."
        )

    if (
        canonical_evidence[
            "canonical_player_id"
        ].nunique()
        != evidence[
            "canonical_player_id"
        ].nunique()
    ):
        raise AssertionError(
            "Canonical player population changed during "
            "evidence resolution."
        )

    arguments.output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_evidence.to_csv(
        arguments.output_file,
        index=False,
    )

    print()
    print("Canonical evidence resolution")
    print("-" * 88)
    print(
        f"Input rows: "
        f"{len(evidence):,}"
    )
    print(
        f"Output rows: "
        f"{len(canonical_evidence):,}"
    )
    print(
        f"Rows removed: "
        f"{removed_rows:,}"
    )
    print(
        "Canonical players preserved: "
        f"{canonical_evidence['canonical_player_id'].nunique():,}"
    )
    print(
        "Remaining canonical collisions: "
        f"{remaining_collisions}"
    )
    print(
        f"Wrote: {arguments.output_file}"
    )

if __name__ == "__main__":
    main()
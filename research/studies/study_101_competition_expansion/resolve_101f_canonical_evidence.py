#resolve_101f_canonical_evidence

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STUDY_101D_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "study_101d_expanded_player_intelligence"
)

INPUT_EVIDENCE = (
    STUDY_101D_ROOT
    / "candidate_player_stats.csv"
)

PLAYER_ID_ALIASES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_id_aliases.csv"
)

OUTPUT_EVIDENCE = (
    PROJECT_ROOT
    / "outputs"
    / "study_101f_canonical_evidence.csv"
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

def main() -> None:
    evidence = pd.read_csv(
        INPUT_EVIDENCE,
        dtype={"season_year": str},
        low_memory=False,
    )

    aliases = pd.read_csv(
        PLAYER_ID_ALIASES,
        low_memory=False,
    )

    required_alias_columns = {
        "source_player_id",
        "canonical_player_id",
        "review_status",
    }

    missing_alias_columns = (
        required_alias_columns
        - set(aliases.columns)
    )

    if missing_alias_columns:
        raise ValueError(
            "Player ID alias file is missing required columns: "
            f"{sorted(missing_alias_columns)}"
        )

    reviewed_aliases = (
        aliases.loc[
            aliases["review_status"].eq("reviewed")
        ]
        .copy()
    )

    if reviewed_aliases[
        "source_player_id"
    ].duplicated().any():
        duplicate_source_ids = (
            reviewed_aliases.loc[
                reviewed_aliases[
                    "source_player_id"
                ].duplicated(keep=False),
                "source_player_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate reviewed source player IDs: "
            f"{duplicate_source_ids}"
        )

    evidence["player_id"] = pd.to_numeric(
        evidence["player_id"],
        errors="raise",
    ).astype(int)

    reviewed_aliases[
        "source_player_id"
    ] = pd.to_numeric(
        reviewed_aliases[
            "source_player_id"
        ],
        errors="raise",
    ).astype(int)

    reviewed_aliases[
        "canonical_player_id"
    ] = pd.to_numeric(
        reviewed_aliases[
            "canonical_player_id"
        ],
        errors="raise",
    ).astype(int)

    source_population = set(
        evidence["player_id"]
    )

    missing_alias_sources = sorted(
        set(
            reviewed_aliases[
                "source_player_id"
            ]
        )
        - source_population
    )

    if missing_alias_sources:
        raise ValueError(
            "Reviewed alias source IDs are absent "
            "from candidate evidence: "
            f"{missing_alias_sources}"
        )

    missing_alias_targets = sorted(
        set(
            reviewed_aliases[
                "canonical_player_id"
            ]
        )
        - source_population
    )

    if missing_alias_targets:
        raise ValueError(
            "Reviewed canonical player IDs are absent "
            "from candidate evidence: "
            f"{missing_alias_targets}"
        )

    alias_map = dict(
        zip(
            reviewed_aliases[
                "source_player_id"
            ],
            reviewed_aliases[
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
        f"Reviewed alias rows: "
        f"{len(reviewed_aliases):,}"
    )

    print(
        "Canonical collision groups: "
        f"{len(collision_groups):,}"
    )

    print(
        "Rows in canonical collisions: "
        f"{len(collisions):,}"
    )

    if not collisions.empty:
        print()
        print("Canonical collisions")
        print("-" * 88)

        display_columns = [
            column
            for column in [
                "canonical_player_id",
                "player_id",
                "player",
                "competition",
                "competition_id",
                "season_id",
                "season_year",
                "team",
                "minutesPlayed",
            ]
            if column in collisions.columns
        ]

        print(
            collisions[
                display_columns
            ].to_string(
                index=False
            )
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

        for row_index in range(1, len(group)):
            candidate = group.iloc[row_index]

            for column in comparison_columns:
                left = reference[column]
                right = candidate[column]

                if pd.isna(left) and pd.isna(right):
                    continue

                if pd.api.types.is_number(left) and pd.api.types.is_number(right):
                    if pd.isna(left) or pd.isna(right):
                        group_conflicts.append(column)
                        continue

                    if not pd.Series(
                        [float(left), float(right)]
                    ).round(12).nunique() == 1:
                        group_conflicts.append(column)

                    continue

                if str(left) != str(right):
                    group_conflicts.append(column)

        group_conflicts = sorted(
            set(group_conflicts)
        )

        if group_conflicts:
            conflicting_groups.append(
                {
                    "task_key": task_key,
                    "columns": group_conflicts,
                }
            )

    if conflicting_groups:
        print()
        print("Conflicting canonical collisions")
        print("-" * 88)

        for conflict in conflicting_groups:
            print(
                f"{conflict['task_key']}: "
                f"{conflict['columns']}"
            )

        raise AssertionError(
            "Canonical collision evidence is not equivalent."
        )

    print()
    print(
        "Canonical collision equivalence: PASS"
    )

    # Prefer native canonical source observations when equivalent
    # canonical task-key collisions occur.
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

    expected_removed_rows = (
        len(evidence)
        - len(collision_groups)
        - (
            len(evidence)
            - len(collisions)
        )
    )

    removed_rows = (
        len(evidence)
        - len(canonical_evidence)
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
            "evidence deduplication."
        )

    OUTPUT_EVIDENCE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_evidence.to_csv(
        OUTPUT_EVIDENCE,
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
        f"Wrote: {OUTPUT_EVIDENCE}"
    )

if __name__ == "__main__":
    main()
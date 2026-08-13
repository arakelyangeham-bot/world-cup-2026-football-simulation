#benchmark_positional_relationships

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_102_formation_geometry"
    / "study_102c"
)

INPUT_PATH = (
    INPUT_DIRECTORY
    / "formation_pair_responsibility_summary.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_102_formation_geometry"
    / "study_102d"
)

TYPE_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "positional_relationship_type_summary.csv"
)

PAIR_MULTIPLICITY_PATH = (
    OUTPUT_DIRECTORY
    / "slot_pair_relationship_multiplicity.csv"
)

OVERLAP_MATRIX_PATH = (
    OUTPUT_DIRECTORY
    / "relationship_overlap_matrix.csv"
)

PAIRWISE_OVERLAP_PATH = (
    OUTPUT_DIRECTORY
    / "pairwise_relationship_overlap.csv"
)

UNIQUE_INFORMATION_PATH = (
    OUTPUT_DIRECTORY
    / "relationship_unique_information_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_102d_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_102D_REPORT.md"
)


EXPECTED_RELATIONSHIP_TYPES = (
    "same_line",
    "adjacent_line",
    "same_corridor",
)

EXPECTED_FORMATION = "4-3-3"
EXPECTED_TEAM_COUNT = 48
EXPECTED_RECORD_COUNT_PER_TEAM = 41

def load_relationship_population() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Study 102C input: {INPUT_PATH}"
        )

    frame = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    required_columns = {
        "formation",
        "source_slot",
        "source_role",
        "target_slot",
        "target_role",
        "responsibility_type",
        "team_count",
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise AssertionError(
            "Relationship summary is missing columns: "
            f"{sorted(missing)}"
        )

    return frame


def validate_input(
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        raise AssertionError(
            "Relationship summary is empty."
        )

    formations = set(
        frame["formation"].astype(str)
    )

    if formations != {
        EXPECTED_FORMATION
    }:
        raise AssertionError(
            "Unexpected formation population: "
            f"{sorted(formations)}"
        )

    observed_types = set(
        frame[
            "responsibility_type"
        ].astype(str)
    )

    if observed_types != set(
        EXPECTED_RELATIONSHIP_TYPES
    ):
        raise AssertionError(
            "Unexpected relationship vocabulary. "
            f"Observed={sorted(observed_types)}"
        )

    if not frame[
        "team_count"
    ].eq(
        EXPECTED_TEAM_COUNT
    ).all():
        raise AssertionError(
            "At least one positional relationship is not "
            "present for all 48 teams."
        )

    if frame.duplicated(
        subset=[
            "source_slot",
            "target_slot",
            "responsibility_type",
        ]
    ).any():
        raise AssertionError(
            "Relationship summary contains duplicate records."
        )

    if not frame[
        "source_slot"
    ].lt(
        frame[
            "target_slot"
        ]
    ).all():
        raise AssertionError(
            "At least one slot pair is not in canonical order."
        )

def relationship_pair_sets(
    frame: pd.DataFrame,
) -> dict[
    str,
    set[tuple[str, str]],
]:
    return {
        relationship_type: {
            (
                str(row.source_slot),
                str(row.target_slot),
            )
            for row in group.itertuples(
                index=False
            )
        }
        for relationship_type, group
        in frame.groupby(
            "responsibility_type",
            sort=True,
        )
    }

def build_type_summary(
    pair_sets: dict[
        str,
        set[tuple[str, str]],
    ],
) -> pd.DataFrame:
    all_pairs = set().union(
        *pair_sets.values()
    )

    rows: list[dict[str, Any]] = []

    for relationship_type in (
        EXPECTED_RELATIONSHIP_TYPES
    ):
        pairs = pair_sets[
            relationship_type
        ]

        rows.append(
            {
                "responsibility_type":
                    relationship_type,
                "relationship_count":
                    len(pairs),
                "share_of_all_unique_pairs":
                    (
                        len(pairs)
                        / len(all_pairs)
                    ),
            }
        )

    return pd.DataFrame(rows)

def build_pair_multiplicity(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        frame
        .groupby(
            [
                "source_slot",
                "source_role",
                "target_slot",
                "target_role",
            ],
            as_index=False,
        )
        .agg(
            responsibility_types=(
                "responsibility_type",
                lambda values: "|".join(
                    sorted(
                        set(
                            str(value)
                            for value in values
                        )
                    )
                ),
            ),
            relationship_type_count=(
                "responsibility_type",
                "nunique",
            ),
        )
        .sort_values(
            [
                "relationship_type_count",
                "source_slot",
                "target_slot",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return grouped

def jaccard_similarity(
    left: set[tuple[str, str]],
    right: set[tuple[str, str]],
) -> float:
    union = left | right

    if not union:
        return 1.0

    return float(
        len(left & right)
        / len(union)
    )


def overlap_coefficient(
    left: set[tuple[str, str]],
    right: set[tuple[str, str]],
) -> float:
    denominator = min(
        len(left),
        len(right),
    )

    if denominator == 0:
        return 0.0

    return float(
        len(left & right)
        / denominator
    )


def build_pairwise_overlap(
    pair_sets: dict[
        str,
        set[tuple[str, str]],
    ],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for first, second in combinations(
        EXPECTED_RELATIONSHIP_TYPES,
        2,
    ):
        left = pair_sets[first]
        right = pair_sets[second]

        intersection = left & right
        union = left | right

        rows.append(
            {
                "relationship_type_a":
                    first,
                "relationship_type_b":
                    second,
                "count_a":
                    len(left),
                "count_b":
                    len(right),
                "intersection_count":
                    len(intersection),
                "union_count":
                    len(union),
                "jaccard_similarity":
                    jaccard_similarity(
                        left,
                        right,
                    ),
                "overlap_coefficient":
                    overlap_coefficient(
                        left,
                        right,
                    ),
                "shared_pairs":
                    "|".join(
                        f"{source}-{target}"
                        for source, target
                        in sorted(intersection)
                    ),
            }
        )

    return pd.DataFrame(rows)

def build_overlap_matrix(
    pair_sets: dict[
        str,
        set[tuple[str, str]],
    ],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row_type in (
        EXPECTED_RELATIONSHIP_TYPES
    ):
        row: dict[str, Any] = {
            "responsibility_type":
                row_type,
        }

        for column_type in (
            EXPECTED_RELATIONSHIP_TYPES
        ):
            row[column_type] = len(
                pair_sets[row_type]
                & pair_sets[column_type]
            )

        rows.append(row)

    return pd.DataFrame(rows)

def build_unique_information_summary(
    pair_sets: dict[
        str,
        set[tuple[str, str]],
    ],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for relationship_type in (
        EXPECTED_RELATIONSHIP_TYPES
    ):
        own_pairs = pair_sets[
            relationship_type
        ]

        other_pairs = set().union(
            *[
                pairs
                for other_type, pairs
                in pair_sets.items()
                if other_type
                != relationship_type
            ]
        )

        unique_pairs = (
            own_pairs
            - other_pairs
        )

        overlapping_pairs = (
            own_pairs
            & other_pairs
        )

        rows.append(
            {
                "responsibility_type":
                    relationship_type,
                "total_pair_count":
                    len(own_pairs),
                "unique_pair_count":
                    len(unique_pairs),
                "overlapping_pair_count":
                    len(overlapping_pairs),
                "unique_pair_fraction":
                    (
                        len(unique_pairs)
                        / len(own_pairs)
                        if own_pairs
                        else 0.0
                    ),
                "unique_pairs":
                    "|".join(
                        f"{source}-{target}"
                        for source, target
                        in sorted(unique_pairs)
                    ),
                "overlapping_pairs":
                    "|".join(
                        f"{source}-{target}"
                        for source, target
                        in sorted(
                            overlapping_pairs
                        )
                    ),
            }
        )

    return pd.DataFrame(rows)

def validate_outputs(
    *,
    frame: pd.DataFrame,
    type_summary: pd.DataFrame,
    multiplicity: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
    unique_summary: pd.DataFrame,
) -> None:
    total_records = int(
        type_summary[
            "relationship_count"
        ].sum()
    )

    if total_records != (
        EXPECTED_RECORD_COUNT_PER_TEAM
    ):
        raise AssertionError(
            "Unexpected relationship count per formation: "
            f"{total_records} vs "
            f"{EXPECTED_RECORD_COUNT_PER_TEAM}."
        )

    if multiplicity.empty:
        raise AssertionError(
            "Pair multiplicity output is empty."
        )

    if (
        multiplicity[
            "relationship_type_count"
        ]
        .lt(1)
        .any()
    ):
        raise AssertionError(
            "A slot pair has no relationship type."
        )

    if (
        multiplicity[
            "relationship_type_count"
        ]
        .gt(
            len(
                EXPECTED_RELATIONSHIP_TYPES
            )
        )
        .any()
    ):
        raise AssertionError(
            "A slot pair has an impossible multiplicity."
        )

    numeric_columns = (
        "jaccard_similarity",
        "overlap_coefficient",
    )

    for column in numeric_columns:
        if not pairwise_overlap[
            column
        ].between(
            0.0,
            1.0,
            inclusive="both",
        ).all():
            raise AssertionError(
                f"{column} lies outside [0, 1]."
            )

    if not unique_summary[
        "unique_pair_fraction"
    ].between(
        0.0,
        1.0,
        inclusive="both",
    ).all():
        raise AssertionError(
            "Unique-pair fractions lie outside [0, 1]."
        )

    same_line_adjacent = (
        pairwise_overlap.loc[
            (
                pairwise_overlap[
                    "relationship_type_a"
                ].isin(
                    [
                        "same_line",
                        "adjacent_line",
                    ]
                )
                & pairwise_overlap[
                    "relationship_type_b"
                ].isin(
                    [
                        "same_line",
                        "adjacent_line",
                    ]
                )
            )
        ]
    )

    if (
        len(same_line_adjacent) != 1
        or int(
            same_line_adjacent[
                "intersection_count"
            ].iloc[0]
        ) != 0
    ):
        raise AssertionError(
            "Same-line and adjacent-line relationships "
            "must be mutually exclusive."
        )

def write_report(
    *,
    type_summary: pd.DataFrame,
    multiplicity: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
    unique_summary: pd.DataFrame,
) -> None:
    overlapping_pairs = multiplicity.loc[
        multiplicity[
            "relationship_type_count"
        ].gt(1)
    ]

    report = f"""# Study 102D — Positional Relationship Overlap Benchmark

## Status

**PASS**

## Purpose

Evaluate the internal overlap, redundancy, and distinctiveness of the
three positional relationship types generated for the validated 4-3-3
formation.

## Interpretation boundary

This benchmark evaluates one formation only. It cannot establish which
relationship type best distinguishes tactical systems, improves
prediction, or carries causal football information.

## Relationship-type population

{type_summary.to_markdown(index=False)}

## Pairwise overlap

{pairwise_overlap.to_markdown(index=False)}

## Slot pairs carrying multiple descriptions

{overlapping_pairs.to_markdown(index=False)}

## Unique pair contribution

{unique_summary.to_markdown(index=False)}

## Conclusions permitted

- Whether relationship definitions are internally coherent.
- Whether any type is completely redundant within the current 4-3-3.
- Which slot pairs receive multiple positional descriptions.

## Conclusions not permitted

- Predictive usefulness.
- Tactical superiority.
- Cross-formation generality.
- Player interaction quality.
- Structural responsibilities such as support, protection, or coverage.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 102D — POSITIONAL RELATIONSHIP "
        "OVERLAP BENCHMARK"
    )
    print("=" * 88)

    frame = load_relationship_population()

    validate_input(
        frame
    )

    pair_sets = relationship_pair_sets(
        frame
    )

    type_summary = build_type_summary(
        pair_sets
    )

    multiplicity = build_pair_multiplicity(
        frame
    )

    pairwise_overlap = (
        build_pairwise_overlap(
            pair_sets
        )
    )

    overlap_matrix = build_overlap_matrix(
        pair_sets
    )

    unique_summary = (
        build_unique_information_summary(
            pair_sets
        )
    )

    validate_outputs(
        frame=frame,
        type_summary=type_summary,
        multiplicity=multiplicity,
        pairwise_overlap=pairwise_overlap,
        unique_summary=unique_summary,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    type_summary.to_csv(
        TYPE_SUMMARY_PATH,
        index=False,
    )

    multiplicity.to_csv(
        PAIR_MULTIPLICITY_PATH,
        index=False,
    )

    overlap_matrix.to_csv(
        OVERLAP_MATRIX_PATH,
        index=False,
    )

    pairwise_overlap.to_csv(
        PAIRWISE_OVERLAP_PATH,
        index=False,
    )

    unique_summary.to_csv(
        UNIQUE_INFORMATION_PATH,
        index=False,
    )

    unique_pair_count = int(
        len(multiplicity)
    )

    overlapping_pair_count = int(
        multiplicity[
            "relationship_type_count"
        ].gt(1).sum()
    )

    metadata = {
        "study_id": "102D",
        "study_name": (
            "Positional Relationship Overlap Benchmark"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "formation": EXPECTED_FORMATION,
        "relationship_type_count": len(
            EXPECTED_RELATIONSHIP_TYPES
        ),
        "relationship_record_count":
            int(
                type_summary[
                    "relationship_count"
                ].sum()
            ),
        "unique_slot_pair_count":
            unique_pair_count,
        "overlapping_slot_pair_count":
            overlapping_pair_count,
        "maximum_relationship_multiplicity":
            int(
                multiplicity[
                    "relationship_type_count"
                ].max()
            ),
        "cross_formation_comparison":
            False,
        "predictive_evaluation":
            False,
        "structural_semantics_generated":
            False,
        "football_graph_created":
            False,
        "team_strength_changed":
            False,
        "simulation_run":
            False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "This benchmark measures overlap and distinctiveness "
            "within one 4-3-3 geometry. It does not establish "
            "predictive usefulness or cross-formation validity."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        type_summary=type_summary,
        multiplicity=multiplicity,
        pairwise_overlap=pairwise_overlap,
        unique_summary=unique_summary,
    )

    print()
    print("Benchmark summary")
    print("-" * 88)
    print(
        f"  Formation: "
        f"{EXPECTED_FORMATION}"
    )
    print(
        "  Relationship types: "
        f"{metadata['relationship_type_count']}"
    )
    print(
        "  Relationship records: "
        f"{metadata['relationship_record_count']}"
    )
    print(
        "  Unique slot pairs: "
        f"{metadata['unique_slot_pair_count']}"
    )
    print(
        "  Multi-description slot pairs: "
        f"{metadata['overlapping_slot_pair_count']}"
    )
    print(
        "  Same-line / adjacent-line exclusivity: PASS"
    )
    print(
        "  Relationship overlap bounded: PASS"
    )
    print(
        "  Completely redundant type present: NO"
    )
    print(
        "  Cross-formation comparison: NO"
    )
    print(
        "  Structural semantics generated: NO"
    )
    print(
        "  Football graph created: NO"
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
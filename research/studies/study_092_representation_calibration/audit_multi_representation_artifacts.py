#audit_multi_representation_artifacts

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "study_092c1"
)

OUTPUT_DIRECTORY = INPUT_DIRECTORY

TRANSFORMATIONS = (
    "global_zscore",
    "percentile_normal",
    "robust_zscore",
)

ATTRIBUTE_PATHS = {
    transformation: (
        INPUT_DIRECTORY
        / f"player_attribute_scores_{transformation}.csv"
    )
    for transformation in TRANSFORMATIONS
}

RATING_PATHS = {
    transformation: (
        INPUT_DIRECTORY
        / f"player_ratings_{transformation}.csv"
    )
    for transformation in TRANSFORMATIONS
}

AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "multi_representation_artifact_audit.csv"
)

ATTRIBUTE_DIFFERENCE_PATH = (
    OUTPUT_DIRECTORY
    / "attribute_branch_differences.csv"
)

RATING_DIFFERENCE_PATH = (
    OUTPUT_DIRECTORY
    / "rating_branch_differences.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_092c1a_metadata.json"
)


IDENTITY_COLUMNS = (
    "canonical_player_id",
    "player_id",
    "player",
)

ATTRIBUTE_PROVENANCE_COLUMNS = (
    "minutesPlayed",
    "total_weighted_evidence",
    "evidence_confidence",
    "source_competitions",
    "competition_count",
    "season_count",
)

RATING_PROVENANCE_COLUMNS = (
    "country",
    "current_team",
    "position",
    "positions_detailed",
    "eligible_roles",
    "minutesPlayed",
    "total_weighted_evidence",
    "evidence_confidence",
    "source_competitions",
    "competition_count",
    "season_count",
)


def load_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"{label} is empty."
        )

    if "canonical_player_id" not in dataframe.columns:
        raise ValueError(
            f"{label} lacks canonical_player_id."
        )

    if dataframe["canonical_player_id"].duplicated().any():
        raise ValueError(
            f"{label} contains duplicate canonical player IDs."
        )

    return dataframe


def assert_same_schema(
    frames: dict[str, pd.DataFrame],
    *,
    label: str,
) -> None:
    baseline_name = TRANSFORMATIONS[0]
    baseline_columns = list(
        frames[baseline_name].columns
    )

    for transformation, dataframe in frames.items():
        if list(dataframe.columns) != baseline_columns:
            raise AssertionError(
                f"{label} schema differs for "
                f"{transformation!r}."
            )


def assert_same_population(
    frames: dict[str, pd.DataFrame],
    *,
    label: str,
) -> None:
    baseline_name = TRANSFORMATIONS[0]

    baseline = (
        frames[baseline_name]
        .sort_values("canonical_player_id")
        .reset_index(drop=True)
    )

    for transformation, dataframe in frames.items():
        candidate = (
            dataframe
            .sort_values("canonical_player_id")
            .reset_index(drop=True)
        )

        if len(candidate) != len(baseline):
            raise AssertionError(
                f"{label} row count differs for "
                f"{transformation!r}."
            )

        if not baseline[
            "canonical_player_id"
        ].equals(
            candidate[
                "canonical_player_id"
            ]
        ):
            raise AssertionError(
                f"{label} player population differs for "
                f"{transformation!r}."
            )


def assert_equal_columns(
    frames: dict[str, pd.DataFrame],
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    baseline_name = TRANSFORMATIONS[0]

    baseline = (
        frames[baseline_name]
        .sort_values("canonical_player_id")
        .reset_index(drop=True)
    )

    for transformation, dataframe in frames.items():
        candidate = (
            dataframe
            .sort_values("canonical_player_id")
            .reset_index(drop=True)
        )

        for column in columns:
            if column not in baseline.columns:
                continue

            left = baseline[column]
            right = candidate[column]

            if pd.api.types.is_numeric_dtype(left):
                equal = np.allclose(
                    pd.to_numeric(
                        left,
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    pd.to_numeric(
                        right,
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    equal_nan=True,
                    atol=0.0,
                    rtol=0.0,
                )
            else:
                equal = (
                    left.fillna("<missing>")
                    .astype(str)
                    .equals(
                        right.fillna("<missing>")
                        .astype(str)
                    )
                )

            if not equal:
                raise AssertionError(
                    f"{label} differs unexpectedly in "
                    f"{column!r} for {transformation!r}."
                )


def build_artifact_audit(
    *,
    attributes: dict[str, pd.DataFrame],
    ratings: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for transformation in TRANSFORMATIONS:
        for artifact_type, dataframe in (
            ("attributes", attributes[transformation]),
            ("ratings", ratings[transformation]),
        ):
            numeric = dataframe.select_dtypes(
                include="number"
            )

            finite = numeric.to_numpy(
                dtype=float
            )

            records.append(
                {
                    "transformation":
                        transformation,
                    "artifact_type":
                        artifact_type,
                    "row_count":
                        len(dataframe),
                    "column_count":
                        len(dataframe.columns),
                    "unique_player_count":
                        dataframe[
                            "canonical_player_id"
                        ].nunique(),
                    "duplicate_player_count":
                        int(
                            dataframe[
                                "canonical_player_id"
                            ].duplicated().sum()
                        ),
                    "missing_value_count":
                        int(
                            dataframe.isna().sum().sum()
                        ),
                    "non_finite_numeric_count":
                        int(
                            (
                                ~np.isfinite(finite)
                                & ~np.isnan(finite)
                            ).sum()
                        ),
                }
            )

    return pd.DataFrame(records)


def build_difference_summary(
    frames: dict[str, pd.DataFrame],
    *,
    prefix: str,
) -> pd.DataFrame:
    baseline = (
        frames["global_zscore"]
        .sort_values("canonical_player_id")
        .reset_index(drop=True)
    )

    numeric_columns = [
        column
        for column in baseline.columns
        if (
            pd.api.types.is_numeric_dtype(
                baseline[column]
            )
            and (
                column.startswith(prefix)
                or column.startswith("raw_rating_")
                or column.startswith("rating_")
                or column == "best_rating"
            )
        )
    ]

    records: list[dict[str, object]] = []

    for transformation in (
        "percentile_normal",
        "robust_zscore",
    ):
        candidate = (
            frames[transformation]
            .sort_values("canonical_player_id")
            .reset_index(drop=True)
        )

        for column in numeric_columns:
            left = pd.to_numeric(
                baseline[column],
                errors="coerce",
            ).to_numpy(dtype=float)

            right = pd.to_numeric(
                candidate[column],
                errors="coerce",
            ).to_numpy(dtype=float)

            valid = (
                np.isfinite(left)
                & np.isfinite(right)
            )

            differences = (
                right[valid]
                - left[valid]
            )

            records.append(
                {
                    "candidate_transformation":
                        transformation,
                    "column":
                        column,
                    "matched_value_count":
                        int(valid.sum()),
                    "mean_difference":
                        (
                            float(differences.mean())
                            if differences.size
                            else None
                        ),
                    "mean_absolute_difference":
                        (
                            float(
                                np.abs(differences).mean()
                            )
                            if differences.size
                            else None
                        ),
                    "maximum_absolute_difference":
                        (
                            float(
                                np.abs(differences).max()
                            )
                            if differences.size
                            else None
                        ),
                    "changed_value_count":
                        int(
                            (
                                np.abs(differences)
                                > 1e-12
                            ).sum()
                        ),
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    attributes = {
        transformation: load_csv(
            ATTRIBUTE_PATHS[transformation],
            label=(
                f"{transformation} attribute artifact"
            ),
        )
        for transformation in TRANSFORMATIONS
    }

    ratings = {
        transformation: load_csv(
            RATING_PATHS[transformation],
            label=(
                f"{transformation} rating artifact"
            ),
        )
        for transformation in TRANSFORMATIONS
    }

    assert_same_schema(
        attributes,
        label="Attribute artifacts",
    )

    assert_same_schema(
        ratings,
        label="Rating artifacts",
    )

    assert_same_population(
        attributes,
        label="Attribute artifacts",
    )

    assert_same_population(
        ratings,
        label="Rating artifacts",
    )

    assert_equal_columns(
        attributes,
        (
            *IDENTITY_COLUMNS,
            *ATTRIBUTE_PROVENANCE_COLUMNS,
        ),
        label="Attribute provenance",
    )

    assert_equal_columns(
        ratings,
        (
            *IDENTITY_COLUMNS,
            *RATING_PROVENANCE_COLUMNS,
        ),
        label="Rating provenance",
    )

    audit = build_artifact_audit(
        attributes=attributes,
        ratings=ratings,
    )

    attribute_differences = (
        build_difference_summary(
            attributes,
            prefix="attribute_",
        )
    )

    rating_differences = (
        build_difference_summary(
            ratings,
            prefix="rating_",
        )
    )

    if attribute_differences[
        "changed_value_count"
    ].sum() == 0:
        raise AssertionError(
            "Alternative attribute branches do not differ "
            "from the global-z-score control."
        )

    if rating_differences[
        "changed_value_count"
    ].sum() == 0:
        raise AssertionError(
            "Alternative rating branches do not differ "
            "from the global-z-score control."
        )

    metadata = {
        "study_id": "092C1A",
        "study_name": (
            "Multi-Representation Artifact Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "transformations": list(
            TRANSFORMATIONS
        ),
        "attribute_player_count": int(
            audit.loc[
                audit["artifact_type"].eq(
                    "attributes"
                ),
                "unique_player_count",
            ].iloc[0]
        ),
        "rating_player_count": int(
            audit.loc[
                audit["artifact_type"].eq(
                    "ratings"
                ),
                "unique_player_count",
            ].iloc[0]
        ),
        "attribute_schema_match": True,
        "rating_schema_match": True,
        "player_population_match": True,
        "identity_columns_match": True,
        "evidence_provenance_match": True,
        "alternative_attribute_values_detected": True,
        "alternative_rating_values_detected": True,
        "canonical_files_overwritten": False,
        "club_repositories_generated": False,
        "goal_models_fitted": False,
    }

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    attribute_differences.to_csv(
        ATTRIBUTE_DIFFERENCE_PATH,
        index=False,
    )

    rating_differences.to_csv(
        RATING_DIFFERENCE_PATH,
        index=False,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print(
        "STUDY 092C1A — MULTI-REPRESENTATION ARTIFACT AUDIT"
    )
    print("=" * 88)
    print()
    print(audit.to_string(index=False))
    print()
    print("Validation summary")
    print("  Attribute schemas matched: PASS")
    print("  Rating schemas matched: PASS")
    print("  Player populations matched: PASS")
    print("  Identity fields matched: PASS")
    print("  Evidence provenance matched: PASS")
    print("  Alternative attribute values detected: PASS")
    print("  Alternative rating values detected: PASS")
    print("  Canonical artifacts overwritten: NO")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
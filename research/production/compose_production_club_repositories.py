#compose_production_club_repositories

from __future__ import annotations

from pathlib import Path

import pandas as pd


COMPATIBILITY_COLUMNS = (
    "repository_version",
    "representation_type",
    "aggregation_profile",
)


def _require_single_value(
    dataframe: pd.DataFrame,
    column: str,
    source_path: Path,
) -> str:
    values = (
        dataframe[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(values) != 1:
        raise ValueError(
            f"{source_path} must contain exactly one "
            f"{column!r} value; found {values}."
        )

    return values[0]


def compose_production_club_repositories(
    *,
    source_paths: list[Path],
    repository_scope: str,
    output_path: Path,
) -> pd.DataFrame:
    if not source_paths:
        raise ValueError(
            "At least one source repository is required."
        )

    if not repository_scope.strip():
        raise ValueError(
            "repository_scope must not be empty."
        )

    frames: list[pd.DataFrame] = []

    expected_columns: list[str] | None = None
    expected_metadata: dict[str, str] | None = None

    for source_path in source_paths:
        source_path = Path(source_path)

        dataframe = pd.read_csv(
            source_path,
            low_memory=False,
        )

        if dataframe.empty:
            raise ValueError(
                f"Source repository is empty: {source_path}"
            )

        columns = dataframe.columns.tolist()

        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(
                "Source repository schemas do not match."
            )

        metadata = {
            column: _require_single_value(
                dataframe,
                column,
                source_path,
            )
            for column in COMPATIBILITY_COLUMNS
        }

        if expected_metadata is None:
            expected_metadata = metadata
        else:
            for column, expected_value in (
                expected_metadata.items()
            ):
                actual_value = metadata[column]

                if actual_value != expected_value:
                    raise ValueError(
                        f"Incompatible {column}: "
                        f"{expected_value!r} != "
                        f"{actual_value!r}."
                    )

        frames.append(dataframe)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    duplicate_mask = combined["club"].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicates = sorted(
            combined.loc[
                duplicate_mask,
                "club",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Combined production repository contains "
            f"duplicate clubs: {duplicates}"
        )

    combined["repository_scope"] = (
        repository_scope.strip()
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    return combined
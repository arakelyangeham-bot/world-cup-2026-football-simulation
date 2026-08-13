#pca_geometry

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectorySummary:
    """
    Geometric summary of one ordered trajectory in PCA space.
    """

    group_key: str
    observation_count: int
    transition_count: int
    total_length: float
    direct_displacement: float
    mean_step_distance: float
    maximum_step_distance: float


def validate_component_columns(
    dataframe: pd.DataFrame,
    component_columns: Sequence[str],
) -> tuple[str, ...]:
    """
    Validate and normalize the PCA component columns used for
    geometric calculations.
    """

    normalized_columns = tuple(
        str(column).strip()
        for column in component_columns
    )

    if not normalized_columns:
        raise ValueError(
            "At least one PCA component column is required."
        )

    if len(set(normalized_columns)) != len(
        normalized_columns
    ):
        raise ValueError(
            "PCA component columns must be unique."
        )

    missing_columns = (
        set(normalized_columns)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "PCA coordinate dataframe is missing component "
            f"columns: {sorted(missing_columns)}"
        )

    return normalized_columns


def extract_coordinate_matrix(
    dataframe: pd.DataFrame,
    component_columns: Sequence[str],
) -> np.ndarray:
    """
    Return a validated finite numeric coordinate matrix.
    """

    validated_columns = validate_component_columns(
        dataframe=dataframe,
        component_columns=component_columns,
    )

    coordinates = dataframe[
        list(validated_columns)
    ].copy()

    for column in validated_columns:
        coordinates[column] = pd.to_numeric(
            coordinates[column],
            errors="raise",
        )

    values = coordinates.to_numpy(
        dtype=float,
        copy=True,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "PCA coordinates contain missing or non-finite values."
        )

    return values


def euclidean_distance(
    point_a: Sequence[float] | np.ndarray,
    point_b: Sequence[float] | np.ndarray,
) -> float:
    """
    Calculate Euclidean distance between two points.
    """

    array_a = np.asarray(
        point_a,
        dtype=float,
    )

    array_b = np.asarray(
        point_b,
        dtype=float,
    )

    if array_a.ndim != 1 or array_b.ndim != 1:
        raise ValueError(
            "Euclidean distance requires two one-dimensional "
            "coordinate vectors."
        )

    if array_a.shape != array_b.shape:
        raise ValueError(
            "Coordinate vectors must have matching dimensions. "
            f"Observed shapes: {array_a.shape} and {array_b.shape}."
        )

    if not (
        np.isfinite(array_a).all()
        and np.isfinite(array_b).all()
    ):
        raise ValueError(
            "Coordinate vectors must contain only finite values."
        )

    return float(
        np.linalg.norm(
            array_a - array_b
        )
    )


def build_pairwise_distance_table(
    dataframe: pd.DataFrame,
    id_column: str,
    component_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Calculate every unique pairwise distance between observations.
    """

    if id_column not in dataframe.columns:
        raise ValueError(
            f"Missing observation ID column {id_column!r}."
        )

    observation_ids = (
        dataframe[id_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if observation_ids.eq("").any():
        raise ValueError(
            f"Observation ID column {id_column!r} contains "
            "empty values."
        )

    if observation_ids.duplicated().any():
        duplicates = sorted(
            observation_ids[
                observation_ids.duplicated(
                    keep=False
                )
            ].unique()
        )

        raise ValueError(
            "Observation IDs must be unique for pairwise "
            f"distance analysis. Duplicates: {duplicates}"
        )

    values = extract_coordinate_matrix(
        dataframe=dataframe,
        component_columns=component_columns,
    )

    rows: list[dict[str, object]] = []

    for first_index, second_index in combinations(
        range(len(dataframe)),
        2,
    ):
        rows.append(
            {
                "observation_id_a": (
                    observation_ids.iloc[
                        first_index
                    ]
                ),
                "observation_id_b": (
                    observation_ids.iloc[
                        second_index
                    ]
                ),
                "distance": euclidean_distance(
                    values[first_index],
                    values[second_index],
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "observation_id_a",
            "observation_id_b",
            "distance",
        ],
    )


def build_pairwise_distance_matrix(
    dataframe: pd.DataFrame,
    id_column: str,
    component_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Build a square symmetric observation-distance matrix.
    """

    if id_column not in dataframe.columns:
        raise ValueError(
            f"Missing observation ID column {id_column!r}."
        )

    observation_ids = (
        dataframe[id_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if observation_ids.eq("").any():
        raise ValueError(
            f"Observation ID column {id_column!r} contains "
            "empty values."
        )

    if observation_ids.duplicated().any():
        raise ValueError(
            "Observation IDs must be unique when building a "
            "distance matrix."
        )

    values = extract_coordinate_matrix(
        dataframe=dataframe,
        component_columns=component_columns,
    )

    differences = (
        values[:, np.newaxis, :]
        - values[np.newaxis, :, :]
    )

    distance_values = np.linalg.norm(
        differences,
        axis=2,
    )

    return pd.DataFrame(
        distance_values,
        index=observation_ids,
        columns=observation_ids,
    )


def build_consecutive_trajectory_distances(
    dataframe: pd.DataFrame,
    group_column: str,
    order_column: str,
    id_column: str,
    component_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Calculate distances between consecutive ordered observations
    within each group.
    """

    required_columns = {
        group_column,
        order_column,
        id_column,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Trajectory analysis is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    validate_component_columns(
        dataframe=dataframe,
        component_columns=component_columns,
    )

    rows: list[dict[str, object]] = []

    for group_key, group in dataframe.groupby(
        group_column,
        sort=True,
    ):
        ordered = (
            group.sort_values(
                order_column
            )
            .reset_index(drop=True)
        )

        if ordered[order_column].duplicated().any():
            duplicate_values = sorted(
                ordered.loc[
                    ordered[
                        order_column
                    ].duplicated(
                        keep=False
                    ),
                    order_column,
                ]
                .tolist()
            )

            raise ValueError(
                f"Group {group_key!r} contains duplicate "
                f"{order_column!r} values: {duplicate_values}"
            )

        values = extract_coordinate_matrix(
            dataframe=ordered,
            component_columns=component_columns,
        )

        for index in range(
            1,
            len(ordered),
        ):
            previous_row = ordered.iloc[
                index - 1
            ]

            current_row = ordered.iloc[
                index
            ]

            rows.append(
                {
                    group_column: group_key,
                    "from_observation_id": (
                        previous_row[id_column]
                    ),
                    "to_observation_id": (
                        current_row[id_column]
                    ),
                    "from_order": (
                        previous_row[order_column]
                    ),
                    "to_order": (
                        current_row[order_column]
                    ),
                    "distance": euclidean_distance(
                        values[index - 1],
                        values[index],
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize_trajectories(
    dataframe: pd.DataFrame,
    group_column: str,
    order_column: str,
    id_column: str,
    component_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Summarize trajectory length and displacement for every group.
    """

    transition_table = (
        build_consecutive_trajectory_distances(
            dataframe=dataframe,
            group_column=group_column,
            order_column=order_column,
            id_column=id_column,
            component_columns=component_columns,
        )
    )

    rows: list[dict[str, object]] = []

    for group_key, group in dataframe.groupby(
        group_column,
        sort=True,
    ):
        ordered = (
            group.sort_values(
                order_column
            )
            .reset_index(drop=True)
        )

        values = extract_coordinate_matrix(
            dataframe=ordered,
            component_columns=component_columns,
        )

        group_transitions = (
            transition_table.loc[
                transition_table[
                    group_column
                ].eq(group_key)
            ]
        )

        if len(ordered) == 1:
            direct_displacement = 0.0
        else:
            direct_displacement = (
                euclidean_distance(
                    values[0],
                    values[-1],
                )
            )

        if group_transitions.empty:
            total_length = 0.0
            mean_step_distance = 0.0
            maximum_step_distance = 0.0
        else:
            total_length = float(
                group_transitions[
                    "distance"
                ].sum()
            )

            mean_step_distance = float(
                group_transitions[
                    "distance"
                ].mean()
            )

            maximum_step_distance = float(
                group_transitions[
                    "distance"
                ].max()
            )

        rows.append(
            {
                group_column: group_key,
                "observation_count": len(
                    ordered
                ),
                "transition_count": max(
                    len(ordered) - 1,
                    0,
                ),
                "total_trajectory_length": (
                    total_length
                ),
                "direct_displacement": (
                    direct_displacement
                ),
                "mean_step_distance": (
                    mean_step_distance
                ),
                "maximum_step_distance": (
                    maximum_step_distance
                ),
            }
        )

    return pd.DataFrame(rows)


def build_group_centroids(
    dataframe: pd.DataFrame,
    group_column: str,
    component_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Calculate the centroid of each group in PCA space.
    """

    if group_column not in dataframe.columns:
        raise ValueError(
            f"Missing grouping column {group_column!r}."
        )

    validated_columns = validate_component_columns(
        dataframe=dataframe,
        component_columns=component_columns,
    )

    working = dataframe.copy()

    for column in validated_columns:
        working[column] = pd.to_numeric(
            working[column],
            errors="raise",
        )

    centroids = (
        working.groupby(
            group_column,
            as_index=False,
            sort=True,
        )[
            list(validated_columns)
        ]
        .mean()
    )

    return centroids


def add_distance_to_group_centroid(
    dataframe: pd.DataFrame,
    group_column: str,
    component_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Add each observation's Euclidean distance from its group
    centroid.
    """

    validated_columns = validate_component_columns(
        dataframe=dataframe,
        component_columns=component_columns,
    )

    centroids = build_group_centroids(
        dataframe=dataframe,
        group_column=group_column,
        component_columns=validated_columns,
    )

    centroid_columns = {
        column: f"{column}_centroid"
        for column in validated_columns
    }

    centroids = centroids.rename(
        columns=centroid_columns
    )

    result = dataframe.merge(
        centroids,
        on=group_column,
        how="left",
        validate="many_to_one",
    )

    coordinate_values = (
        result[
            list(validated_columns)
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .to_numpy(dtype=float)
    )

    centroid_values = (
        result[
            [
                centroid_columns[column]
                for column in validated_columns
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .to_numpy(dtype=float)
    )

    result["distance_to_group_centroid"] = (
        np.linalg.norm(
            coordinate_values
            - centroid_values,
            axis=1,
        )
    )

    return result


def compare_within_and_between_group_distances(
    dataframe: pd.DataFrame,
    group_column: str,
    id_column: str,
    component_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify every pairwise distance as within-group or
    between-group and return both the pair table and summary.
    """

    required_columns = {
        group_column,
        id_column,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Group-distance analysis is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    pairwise = build_pairwise_distance_table(
        dataframe=dataframe,
        id_column=id_column,
        component_columns=component_columns,
    )

    identity_lookup = (
        dataframe[
            [
                id_column,
                group_column,
            ]
        ]
        .copy()
    )

    identity_lookup[id_column] = (
        identity_lookup[id_column]
        .astype(str)
        .str.strip()
    )

    group_lookup = dict(
        zip(
            identity_lookup[id_column],
            identity_lookup[group_column],
        )
    )

    pairwise["group_a"] = (
        pairwise["observation_id_a"]
        .map(group_lookup)
    )

    pairwise["group_b"] = (
        pairwise["observation_id_b"]
        .map(group_lookup)
    )

    pairwise["comparison_type"] = np.where(
        pairwise["group_a"]
        .eq(pairwise["group_b"]),
        "within_group",
        "between_group",
    )

    summary = (
        pairwise.groupby(
            "comparison_type",
            as_index=False,
        )
        .agg(
            pair_count=(
                "distance",
                "size",
            ),
            mean_distance=(
                "distance",
                "mean",
            ),
            median_distance=(
                "distance",
                "median",
            ),
            minimum_distance=(
                "distance",
                "min",
            ),
            maximum_distance=(
                "distance",
                "max",
            ),
            standard_deviation=(
                "distance",
                "std",
            ),
        )
    )

    return pairwise, summary
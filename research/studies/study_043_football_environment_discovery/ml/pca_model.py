#pca_model

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from research.studies.study_043_football_environment_discovery.ml.pca_loader import (
    PCAFeatureMatrix,
)


@dataclass(frozen=True)
class PCAResult:
    """
    Mathematical outputs from one fitted PCA model.
    """

    feature_set: str
    feature_names: tuple[str, ...]
    component_names: tuple[str, ...]
    model: PCA
    coordinates: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    component_correlations: pd.DataFrame
    reconstruction_errors: pd.DataFrame

    @property
    def observation_count(self) -> int:
        return len(self.coordinates)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def component_count(self) -> int:
        return len(self.component_names)


def validate_component_count(
    feature_matrix: PCAFeatureMatrix,
    n_components: int | None,
) -> int:
    """
    Resolve and validate the requested number of PCA components.
    """

    maximum_components = min(
        feature_matrix.observation_count,
        feature_matrix.feature_count,
    )

    if n_components is None:
        return maximum_components

    if not isinstance(n_components, int):
        raise TypeError(
            "n_components must be an integer or None."
        )

    if n_components <= 0:
        raise ValueError(
            "n_components must be greater than zero."
        )

    if n_components > maximum_components:
        raise ValueError(
            f"Requested {n_components} PCA components, "
            f"but the maximum available is "
            f"{maximum_components}."
        )

    return n_components


def build_component_names(
    component_count: int,
) -> tuple[str, ...]:
    return tuple(
        f"PC{index}"
        for index in range(
            1,
            component_count + 1,
        )
    )


def build_coordinate_dataframe(
    transformed: np.ndarray,
    component_names: tuple[str, ...],
    feature_matrix: PCAFeatureMatrix,
) -> pd.DataFrame:
    coordinates = feature_matrix.metadata.copy()

    component_dataframe = pd.DataFrame(
        transformed,
        columns=component_names,
        index=coordinates.index,
    )

    return pd.concat(
        [
            coordinates,
            component_dataframe,
        ],
        axis=1,
    )


def build_loading_dataframe(
    model: PCA,
    component_names: tuple[str, ...],
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """
    Return PCA eigenvector coefficients.

    Rows are original features and columns are components.
    """

    loadings = pd.DataFrame(
        model.components_.T,
        index=feature_names,
        columns=component_names,
    )

    loadings.index.name = "feature_key"

    return loadings.reset_index()


def build_explained_variance_dataframe(
    model: PCA,
    component_names: tuple[str, ...],
) -> pd.DataFrame:
    explained_variance_ratio = (
        model.explained_variance_ratio_
    )

    cumulative_explained_variance = (
        np.cumsum(
            explained_variance_ratio
        )
    )

    return pd.DataFrame(
        {
            "component": component_names,
            "explained_variance": (
                model.explained_variance_
            ),
            "explained_variance_ratio": (
                explained_variance_ratio
            ),
            "cumulative_explained_variance_ratio": (
                cumulative_explained_variance
            ),
            "singular_value": (
                model.singular_values_
            ),
        }
    )


def build_component_correlation_dataframe(
    standardized_values: np.ndarray,
    transformed: np.ndarray,
    feature_names: tuple[str, ...],
    component_names: tuple[str, ...],
) -> pd.DataFrame:
    """
    Calculate correlations between original standardized features
    and PCA component scores.
    """

    combined = np.column_stack(
        [
            standardized_values,
            transformed,
        ]
    )

    correlation_matrix = np.corrcoef(
        combined,
        rowvar=False,
    )

    feature_count = len(
        feature_names
    )

    feature_component_correlations = (
        correlation_matrix[
            :feature_count,
            feature_count:,
        ]
    )

    correlations = pd.DataFrame(
        feature_component_correlations,
        index=feature_names,
        columns=component_names,
    )

    correlations.index.name = "feature_key"

    return correlations.reset_index()


def calculate_reconstruction_errors(
    standardized_values: np.ndarray,
    maximum_components: int,
) -> pd.DataFrame:
    """
    Fit PCA models using 1 through maximum_components and measure
    reconstruction error in standardized feature space.

    Mean squared error is averaged over all observations and features.
    """

    rows: list[dict[str, float | int]] = []

    total_variance = float(
        np.mean(
            standardized_values ** 2
        )
    )

    for component_count in range(
        1,
        maximum_components + 1,
    ):
        model = PCA(
            n_components=component_count,
            svd_solver="full",
        )

        transformed = model.fit_transform(
            standardized_values
        )

        reconstructed = model.inverse_transform(
            transformed
        )

        residuals = (
            standardized_values
            - reconstructed
        )

        mean_squared_error = float(
            np.mean(
                residuals ** 2
            )
        )

        root_mean_squared_error = float(
            np.sqrt(
                mean_squared_error
            )
        )

        if np.isclose(
            total_variance,
            0.0,
        ):
            variance_reconstruction_ratio = (
                np.nan
            )
        else:
            variance_reconstruction_ratio = (
                1.0
                - mean_squared_error
                / total_variance
            )

        rows.append(
            {
                "component_count": (
                    component_count
                ),
                "mean_squared_error": (
                    mean_squared_error
                ),
                "root_mean_squared_error": (
                    root_mean_squared_error
                ),
                "variance_reconstruction_ratio": (
                    variance_reconstruction_ratio
                ),
            }
        )

    return pd.DataFrame(rows)


def fit_pca(
    feature_matrix: PCAFeatureMatrix,
    n_components: int | None = None,
) -> PCAResult:
    """
    Fit classical PCA to a validated standardized feature matrix.
    """

    resolved_component_count = (
        validate_component_count(
            feature_matrix=feature_matrix,
            n_components=n_components,
        )
    )

    values = feature_matrix.to_numpy()

    model = PCA(
        n_components=resolved_component_count,
        svd_solver="full",
    )

    transformed = model.fit_transform(
        values
    )

    component_names = (
        build_component_names(
            resolved_component_count
        )
    )

    coordinates = (
        build_coordinate_dataframe(
            transformed=transformed,
            component_names=component_names,
            feature_matrix=feature_matrix,
        )
    )

    loadings = build_loading_dataframe(
        model=model,
        component_names=component_names,
        feature_names=(
            feature_matrix.feature_names
        ),
    )

    explained_variance = (
        build_explained_variance_dataframe(
            model=model,
            component_names=component_names,
        )
    )

    component_correlations = (
        build_component_correlation_dataframe(
            standardized_values=values,
            transformed=transformed,
            feature_names=(
                feature_matrix.feature_names
            ),
            component_names=component_names,
        )
    )

    reconstruction_errors = (
        calculate_reconstruction_errors(
            standardized_values=values,
            maximum_components=(
                resolved_component_count
            ),
        )
    )

    return PCAResult(
        feature_set=(
            feature_matrix.feature_set
        ),
        feature_names=(
            feature_matrix.feature_names
        ),
        component_names=component_names,
        model=model,
        coordinates=coordinates,
        loadings=loadings,
        explained_variance=(
            explained_variance
        ),
        component_correlations=(
            component_correlations
        ),
        reconstruction_errors=(
            reconstruction_errors
        ),
    )
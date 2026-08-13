#pca_loader

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_STANDARDIZED_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_043_football_environment_discovery"
    / "outputs"
)

STANDARDIZED_FEATURE_FILES = {
    "reduced": (
        "standardized_environment_features_reduced.csv"
    ),
    "full": (
        "standardized_environment_features_full.csv"
    ),
}

IDENTITY_COLUMNS = [
    "observation_id",
    "feature_set",
    "competition_key",
    "competition_name",
    "season_start_year",
]


@dataclass(frozen=True)
class PCAFeatureMatrix:
    """
    Validated input data for a PCA experiment.

    The metadata dataframe preserves observation identities while
    the numeric matrix contains only standardized model features.
    """

    feature_set: str
    source_path: Path
    metadata: pd.DataFrame
    features: pd.DataFrame
    feature_names: tuple[str, ...]

    @property
    def observation_count(self) -> int:
        return len(self.features)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def to_numpy(self) -> np.ndarray:
        """
        Return the numeric feature matrix in the validated
        feature-column order.
        """

        return self.features.to_numpy(
            dtype=float,
            copy=True,
        )


def resolve_feature_matrix_path(
    feature_set: str,
    input_directory: Path = (
        DEFAULT_STANDARDIZED_DIRECTORY
    ),
) -> Path:
    """
    Resolve the standardized matrix path for one feature set.
    """

    normalized_feature_set = (
        feature_set.strip().lower()
    )

    try:
        filename = STANDARDIZED_FEATURE_FILES[
            normalized_feature_set
        ]

    except KeyError as exc:
        raise ValueError(
            f"Unknown PCA feature set "
            f"{normalized_feature_set!r}. "
            "Available feature sets: "
            f"{sorted(STANDARDIZED_FEATURE_FILES)}"
        ) from exc

    return input_directory / filename


def validate_identity_columns(
    dataframe: pd.DataFrame,
    source_path: Path,
) -> None:
    missing_columns = (
        set(IDENTITY_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{source_path.name} is missing required "
            f"identity columns: "
            f"{sorted(missing_columns)}"
        )

    for column in [
        "observation_id",
        "feature_set",
        "competition_key",
        "competition_name",
    ]:
        empty_mask = (
            dataframe[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        if empty_mask.any():
            row_numbers = (
                dataframe.index[
                    empty_mask
                ]
                .add(2)
                .tolist()
            )

            raise ValueError(
                f"{source_path.name} contains empty "
                f"values in {column!r} at CSV rows "
                f"{row_numbers[:20]}."
            )

    duplicate_observations = dataframe[
        dataframe["observation_id"].duplicated(
            keep=False
        )
    ]

    if not duplicate_observations.empty:
        duplicate_ids = sorted(
            duplicate_observations[
                "observation_id"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{source_path.name} contains duplicate "
            f"observation IDs: {duplicate_ids}"
        )

    dataframe["season_start_year"] = (
        pd.to_numeric(
            dataframe["season_start_year"],
            errors="raise",
        )
        .astype(int)
    )


def identify_feature_columns(
    dataframe: pd.DataFrame,
    source_path: Path,
) -> list[str]:
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in IDENTITY_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            f"{source_path.name} contains no PCA "
            "feature columns."
        )

    return feature_columns


def validate_feature_set_label(
    dataframe: pd.DataFrame,
    expected_feature_set: str,
    source_path: Path,
) -> None:
    observed_feature_sets = {
        value.strip().lower()
        for value in dataframe[
            "feature_set"
        ].astype(str)
    }

    if observed_feature_sets != {
        expected_feature_set
    }:
        raise ValueError(
            f"{source_path.name} contains unexpected "
            "feature-set labels. "
            f"Expected: {expected_feature_set!r}. "
            f"Observed: "
            f"{sorted(observed_feature_sets)}"
        )


def convert_and_validate_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    source_path: Path,
) -> pd.DataFrame:
    features = dataframe[
        feature_columns
    ].copy()

    for feature in feature_columns:
        features[feature] = pd.to_numeric(
            features[feature],
            errors="raise",
        )

        missing_count = int(
            features[feature].isna().sum()
        )

        if missing_count:
            raise ValueError(
                f"{source_path.name} contains "
                f"{missing_count} missing value(s) "
                f"in feature {feature!r}."
            )

        values = features[
            feature
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{source_path.name} contains "
                f"non-finite values in feature "
                f"{feature!r}."
            )

        standard_deviation = float(
            np.std(
                values,
                ddof=0,
            )
        )

        if np.isclose(
            standard_deviation,
            0.0,
        ):
            raise ValueError(
                f"Feature {feature!r} has zero "
                "variance and cannot be used in PCA."
            )

    return features


def validate_standardization(
    features: pd.DataFrame,
    source_path: Path,
    mean_tolerance: float = 1e-8,
    standard_deviation_tolerance: float = 1e-8,
) -> None:
    """
    Confirm that each feature is already standardized.

    PCA experiments should consume the Phase 2 standardized
    matrices without silently applying new preprocessing.
    """

    for feature in features.columns:
        values = features[
            feature
        ].to_numpy(dtype=float)

        observed_mean = float(
            np.mean(values)
        )

        observed_standard_deviation = float(
            np.std(
                values,
                ddof=0,
            )
        )

        if abs(
            observed_mean
        ) > mean_tolerance:
            raise ValueError(
                f"Feature {feature!r} in "
                f"{source_path.name} has mean "
                f"{observed_mean:.12g}; expected "
                "approximately 0."
            )

        if abs(
            observed_standard_deviation
            - 1.0
        ) > standard_deviation_tolerance:
            raise ValueError(
                f"Feature {feature!r} in "
                f"{source_path.name} has population "
                "standard deviation "
                f"{observed_standard_deviation:.12g}; "
                "expected approximately 1."
            )


def load_pca_feature_matrix(
    feature_set: str = "reduced",
    input_directory: Path = (
        DEFAULT_STANDARDIZED_DIRECTORY
    ),
    validate_standardized: bool = True,
) -> PCAFeatureMatrix:
    """
    Load and validate one standardized Study 043 feature matrix.
    """

    normalized_feature_set = (
        feature_set.strip().lower()
    )

    source_path = resolve_feature_matrix_path(
        feature_set=normalized_feature_set,
        input_directory=input_directory,
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "Standardized PCA feature matrix "
            f"was not found:\n{source_path}"
        )

    dataframe = pd.read_csv(
        source_path
    )

    if dataframe.empty:
        raise ValueError(
            f"PCA feature matrix is empty: "
            f"{source_path}"
        )

    dataframe = dataframe.copy()

    validate_identity_columns(
        dataframe=dataframe,
        source_path=source_path,
    )

    validate_feature_set_label(
        dataframe=dataframe,
        expected_feature_set=(
            normalized_feature_set
        ),
        source_path=source_path,
    )

    feature_columns = (
        identify_feature_columns(
            dataframe=dataframe,
            source_path=source_path,
        )
    )

    features = (
        convert_and_validate_features(
            dataframe=dataframe,
            feature_columns=feature_columns,
            source_path=source_path,
        )
    )

    if validate_standardized:
        validate_standardization(
            features=features,
            source_path=source_path,
        )

    metadata = dataframe[
        IDENTITY_COLUMNS
    ].copy()

    return PCAFeatureMatrix(
        feature_set=normalized_feature_set,
        source_path=source_path,
        metadata=metadata,
        features=features,
        feature_names=tuple(
            feature_columns
        ),
    )
# feature_transformations.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


DEFAULT_WINSOR_LOWER_QUANTILE = 0.01
DEFAULT_WINSOR_UPPER_QUANTILE = 0.99
DEFAULT_ROBUST_SCALE_CONSTANT = 1.4826


def _coerce_numeric(
    values: pd.Series,
    *,
    label: str,
) -> pd.Series:
    """
    Convert a feature series to floating-point values while preserving
    missing observations.

    Non-numeric values become missing. Infinite values are rejected.
    """

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    finite_values = numeric.dropna().to_numpy(
        dtype=float
    )

    if not np.isfinite(
        finite_values
    ).all():
        raise ValueError(
            f"{label} contains non-finite values."
        )

    return numeric


@dataclass(frozen=True)
class FeatureTransformationMetadata:
    """
    Stable descriptive metadata for one transformation strategy.
    """

    transformation_id: str
    display_name: str
    description: str

    def validate(self) -> None:
        if not self.transformation_id.strip():
            raise ValueError(
                "transformation_id must not be empty."
            )

        if not self.display_name.strip():
            raise ValueError(
                "display_name must not be empty."
            )

        if not self.description.strip():
            raise ValueError(
                "description must not be empty."
            )


class FeatureTransformationStrategy(ABC):
    """
    Abstract interface for transforming one raw football feature.

    Implementations must:

    - preserve the original index;
    - preserve missing-value locations;
    - return finite values for every non-missing input;
    - avoid mutating the input Series.
    """

    metadata: FeatureTransformationMetadata

    @abstractmethod
    def transform(
        self,
        values: pd.Series,
    ) -> pd.Series:
        """
        Transform one raw feature series.
        """

    def validate_output(
        self,
        *,
        original: pd.Series,
        transformed: pd.Series,
    ) -> None:
        self.metadata.validate()

        if not transformed.index.equals(
            original.index
        ):
            raise AssertionError(
                "Feature transformation changed the input index."
            )

        original_numeric = _coerce_numeric(
            original,
            label=(
                f"{self.metadata.transformation_id} input"
            ),
        )

        transformed_numeric = _coerce_numeric(
            transformed,
            label=(
                f"{self.metadata.transformation_id} output"
            ),
        )

        original_missing = (
            original_numeric.isna()
        )

        transformed_missing = (
            transformed_numeric.isna()
        )

        if not original_missing.equals(
            transformed_missing
        ):
            raise AssertionError(
                "Feature transformation changed the "
                "missing-value pattern."
            )

        finite_output = (
            transformed_numeric
            .dropna()
            .to_numpy(dtype=float)
        )

        if not np.isfinite(
            finite_output
        ).all():
            raise AssertionError(
                "Feature transformation produced "
                "non-finite values."
            )

    def __call__(
        self,
        values: pd.Series,
    ) -> pd.Series:
        original = values.copy(
            deep=True
        )

        transformed = self.transform(
            values
        )

        self.validate_output(
            original=original,
            transformed=transformed,
        )

        if not values.equals(
            original
        ):
            raise AssertionError(
                "Feature transformation mutated its input."
            )

        return transformed


@dataclass(frozen=True)
class GlobalZScoreTransformation(
    FeatureTransformationStrategy
):
    """
    Existing production transformation.

    Uses pandas' sample standard deviation, matching the historical
    ``Series.std()`` implementation in score_player_attributes.py.
    """

    metadata: FeatureTransformationMetadata = (
        FeatureTransformationMetadata(
            transformation_id="global_zscore",
            display_name="Global z-score",
            description=(
                "Centers each feature on its global mean and divides "
                "by the global sample standard deviation."
            ),
        )
    )

    def transform(
        self,
        values: pd.Series,
    ) -> pd.Series:
        numeric = _coerce_numeric(
            values,
            label="global z-score input",
        )

        standard_deviation = numeric.std(
            ddof=1
        )

        if (
            pd.isna(
                standard_deviation
            )
            or math.isclose(
                float(
                    standard_deviation
                ),
                0.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            return pd.Series(
                np.where(
                    numeric.isna(),
                    np.nan,
                    0.0,
                ),
                index=numeric.index,
                dtype=float,
            )

        return (
            numeric
            - numeric.mean()
        ) / standard_deviation


@dataclass(frozen=True)
class WinsorizedZScoreTransformation(
    FeatureTransformationStrategy
):
    lower_quantile: float = (
        DEFAULT_WINSOR_LOWER_QUANTILE
    )

    upper_quantile: float = (
        DEFAULT_WINSOR_UPPER_QUANTILE
    )

    metadata: FeatureTransformationMetadata = (
        FeatureTransformationMetadata(
            transformation_id="winsorized_zscore",
            display_name="Winsorized z-score",
            description=(
                "Clips each feature at configured lower and upper "
                "quantiles before global z-score standardization."
            ),
        )
    )

    def __post_init__(
        self,
    ) -> None:
        if not (
            0.0
            <= self.lower_quantile
            < self.upper_quantile
            <= 1.0
        ):
            raise ValueError(
                "Winsorization quantiles must satisfy "
                "0 <= lower < upper <= 1."
            )

    def transform(
        self,
        values: pd.Series,
    ) -> pd.Series:
        numeric = _coerce_numeric(
            values,
            label="winsorized z-score input",
        )

        valid = numeric.dropna()

        if valid.empty:
            return numeric

        lower = float(
            valid.quantile(
                self.lower_quantile
            )
        )

        upper = float(
            valid.quantile(
                self.upper_quantile
            )
        )

        clipped = numeric.clip(
            lower=lower,
            upper=upper,
        )

        return GlobalZScoreTransformation()(
            clipped
        )


@dataclass(frozen=True)
class RobustZScoreTransformation(
    FeatureTransformationStrategy
):
    scale_constant: float = (
        DEFAULT_ROBUST_SCALE_CONSTANT
    )

    metadata: FeatureTransformationMetadata = (
        FeatureTransformationMetadata(
            transformation_id="robust_zscore",
            display_name="Robust z-score",
            description=(
                "Centers each feature on its median and divides by "
                "a median-absolute-deviation estimate of scale."
            ),
        )
    )

    def __post_init__(
        self,
    ) -> None:
        if self.scale_constant <= 0.0:
            raise ValueError(
                "Robust scale constant must be positive."
            )

    def transform(
        self,
        values: pd.Series,
    ) -> pd.Series:
        numeric = _coerce_numeric(
            values,
            label="robust z-score input",
        )

        valid = numeric.dropna()

        if valid.empty:
            return numeric

        median = float(
            valid.median()
        )

        median_absolute_deviation = float(
            (
                valid
                - median
            )
            .abs()
            .median()
        )

        robust_scale = (
            self.scale_constant
            * median_absolute_deviation
        )

        if math.isclose(
            robust_scale,
            0.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return pd.Series(
                np.where(
                    numeric.isna(),
                    np.nan,
                    0.0,
                ),
                index=numeric.index,
                dtype=float,
            )

        return (
            numeric
            - median
        ) / robust_scale


@dataclass(frozen=True)
class PercentileNormalTransformation(
    FeatureTransformationStrategy
):
    metadata: FeatureTransformationMetadata = (
        FeatureTransformationMetadata(
            transformation_id="percentile_normal",
            display_name="Percentile-normal",
            description=(
                "Maps empirical feature ranks to standard-normal "
                "quantiles while preventing infinite tail values."
            ),
        )
    )

    def transform(
        self,
        values: pd.Series,
    ) -> pd.Series:
        numeric = _coerce_numeric(
            values,
            label="percentile-normal input",
        )

        valid_mask = numeric.notna()

        output = pd.Series(
            np.nan,
            index=numeric.index,
            dtype=float,
        )

        valid = numeric.loc[
            valid_mask
        ]

        if valid.empty:
            return output

        if valid.nunique(
            dropna=True
        ) <= 1:
            output.loc[
                valid_mask
            ] = 0.0

            return output

        percentiles = valid.rank(
            method="average",
            pct=True,
        )

        sample_size = len(valid)

        lower_bound = (
            0.5
            / max(
                sample_size,
                1,
            )
        )

        upper_bound = (
            1.0
            - lower_bound
        )

        clipped = percentiles.clip(
            lower=lower_bound,
            upper=upper_bound,
        )

        try:
            from scipy.stats import norm
        except ImportError as error:
            raise ImportError(
                "PercentileNormalTransformation "
                "requires scipy."
            ) from error

        output.loc[
            valid_mask
        ] = norm.ppf(
            clipped.to_numpy(
                dtype=float
            )
        )

        return output


def build_feature_transformation_registry(
) -> dict[
    str,
    FeatureTransformationStrategy,
]:
    strategies: tuple[
        FeatureTransformationStrategy,
        ...,
    ] = (
        GlobalZScoreTransformation(),
        WinsorizedZScoreTransformation(),
        RobustZScoreTransformation(),
        PercentileNormalTransformation(),
    )

    registry = {
        strategy.metadata.transformation_id:
            strategy
        for strategy in strategies
    }

    if len(registry) != len(
        strategies
    ):
        raise AssertionError(
            "Feature transformation IDs must be unique."
        )

    return registry


def get_feature_transformation(
    transformation_id: str,
) -> FeatureTransformationStrategy:
    normalized = str(
        transformation_id
    ).strip()

    if not normalized:
        raise ValueError(
            "Transformation ID must not be empty."
        )

    registry = (
        build_feature_transformation_registry()
    )

    try:
        return registry[
            normalized
        ]

    except KeyError as error:
        raise KeyError(
            "Unknown feature transformation "
            f"{normalized!r}. Available transformations: "
            f"{sorted(registry)}"
        ) from error


def list_feature_transformations(
) -> tuple[
    FeatureTransformationMetadata,
    ...,
]:
    registry = (
        build_feature_transformation_registry()
    )

    return tuple(
        registry[
            transformation_id
        ].metadata
        for transformation_id in sorted(
            registry
        )
    )
# test_feature_transformations.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.player_intelligence.feature_transformations import (
    GlobalZScoreTransformation,
    PercentileNormalTransformation,
    RobustZScoreTransformation,
    WinsorizedZScoreTransformation,
    build_feature_transformation_registry,
    get_feature_transformation,
    list_feature_transformations,
)


@pytest.fixture
def values() -> pd.Series:
    return pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            100.0,
            np.nan,
        ],
        index=[
            "a",
            "b",
            "c",
            "d",
            "e",
            "missing",
        ],
        dtype=float,
    )


def test_global_zscore_matches_historical_formula(
    values: pd.Series,
) -> None:
    strategy = (
        GlobalZScoreTransformation()
    )

    result = strategy(
        values
    )

    expected = (
        values
        - values.mean()
    ) / values.std()

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_global_zscore_has_zero_mean(
    values: pd.Series,
) -> None:
    result = (
        GlobalZScoreTransformation()(
            values
        )
    )

    assert result.dropna().mean() == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_global_zscore_has_sample_standard_deviation_one(
    values: pd.Series,
) -> None:
    result = (
        GlobalZScoreTransformation()(
            values
        )
    )

    assert result.dropna().std(
        ddof=1
    ) == pytest.approx(
        1.0
    )


def test_winsorized_zscore_reduces_extreme_value(
    values: pd.Series,
) -> None:
    global_result = (
        GlobalZScoreTransformation()(
            values
        )
    )

    winsorized_result = (
        WinsorizedZScoreTransformation()(
            values
        )
    )

    assert abs(
        winsorized_result.loc["e"]
    ) < abs(
        global_result.loc["e"]
    )


def test_robust_zscore_centers_on_median(
    values: pd.Series,
) -> None:
    result = (
        RobustZScoreTransformation()(
            values
        )
    )

    median_index = "c"

    assert result.loc[
        median_index
    ] == pytest.approx(
        0.0
    )


def test_percentile_normal_is_monotonic(
    values: pd.Series,
) -> None:
    result = (
        PercentileNormalTransformation()(
            values
        )
    )

    observed = result.dropna()

    assert observed.is_monotonic_increasing


@pytest.mark.parametrize(
    "strategy",
    [
        GlobalZScoreTransformation(),
        WinsorizedZScoreTransformation(),
        RobustZScoreTransformation(),
        PercentileNormalTransformation(),
    ],
)
def test_transformations_preserve_index_and_missing_values(
    strategy,
    values: pd.Series,
) -> None:
    result = strategy(
        values
    )

    assert result.index.equals(
        values.index
    )

    assert result.isna().equals(
        values.isna()
    )


@pytest.mark.parametrize(
    "strategy",
    [
        GlobalZScoreTransformation(),
        WinsorizedZScoreTransformation(),
        RobustZScoreTransformation(),
        PercentileNormalTransformation(),
    ],
)
def test_transformations_do_not_mutate_input(
    strategy,
    values: pd.Series,
) -> None:
    original = values.copy(
        deep=True
    )

    strategy(
        values
    )

    pd.testing.assert_series_equal(
        values,
        original,
    )


@pytest.mark.parametrize(
    "strategy",
    [
        GlobalZScoreTransformation(),
        WinsorizedZScoreTransformation(),
        RobustZScoreTransformation(),
        PercentileNormalTransformation(),
    ],
)
def test_constant_feature_returns_zero_for_observed_values(
    strategy,
) -> None:
    values = pd.Series(
        [
            4.0,
            4.0,
            np.nan,
            4.0,
        ],
        dtype=float,
    )

    result = strategy(
        values
    )

    assert result.iloc[0] == pytest.approx(
        0.0
    )

    assert result.iloc[1] == pytest.approx(
        0.0
    )

    assert pd.isna(
        result.iloc[2]
    )

    assert result.iloc[3] == pytest.approx(
        0.0
    )


def test_winsorized_quantiles_are_validated() -> None:
    with pytest.raises(
        ValueError,
        match="0 <= lower < upper <= 1",
    ):
        WinsorizedZScoreTransformation(
            lower_quantile=0.99,
            upper_quantile=0.01,
        )


def test_robust_scale_constant_is_validated() -> None:
    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        RobustZScoreTransformation(
            scale_constant=0.0
        )


def test_registry_contains_expected_transformations() -> None:
    registry = (
        build_feature_transformation_registry()
    )

    assert set(
        registry
    ) == {
        "global_zscore",
        "winsorized_zscore",
        "robust_zscore",
        "percentile_normal",
    }


def test_get_feature_transformation_returns_requested_strategy() -> None:
    strategy = get_feature_transformation(
        "robust_zscore"
    )

    assert isinstance(
        strategy,
        RobustZScoreTransformation,
    )


def test_unknown_transformation_is_rejected() -> None:
    with pytest.raises(
        KeyError,
        match="Unknown feature transformation",
    ):
        get_feature_transformation(
            "unknown"
        )


def test_metadata_listing_is_stable() -> None:
    metadata = (
        list_feature_transformations()
    )

    identifiers = tuple(
        item.transformation_id
        for item in metadata
    )

    assert identifiers == tuple(
        sorted(
            identifiers
        )
    )

    assert set(
        identifiers
    ) == {
        "global_zscore",
        "winsorized_zscore",
        "robust_zscore",
        "percentile_normal",
    }
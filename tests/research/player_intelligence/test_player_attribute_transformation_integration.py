#test_player_attribute_transformation_integration

from __future__ import annotations

import numpy as np
import pandas as pd

from research.player_intelligence.feature_transformations import (
    GlobalZScoreTransformation,
)


def _historical_zscore(
    values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    standard_deviation = numeric.std()

    if (
        pd.isna(standard_deviation)
        or standard_deviation == 0
    ):
        return pd.Series(
            0,
            index=numeric.index,
            dtype=float,
        )

    return (
        numeric
        - numeric.mean()
    ) / standard_deviation


def test_global_strategy_reproduces_historical_nonconstant_feature() -> None:
    values = pd.Series(
        [
            1.0,
            3.0,
            np.nan,
            7.0,
            10.0,
        ],
        index=[
            "a",
            "b",
            "missing",
            "c",
            "d",
        ],
        dtype=float,
    )

    historical = _historical_zscore(
        values
    )

    strategy = (
        GlobalZScoreTransformation()(
            values
        )
    )

    pd.testing.assert_series_equal(
        strategy,
        historical,
        check_exact=True,
    )


def test_global_strategy_matches_pandas_sample_standard_deviation() -> None:
    values = pd.Series(
        [
            -2.5,
            0.0,
            1.5,
            4.0,
            9.0,
        ],
        dtype=float,
    )

    expected = (
        values
        - values.mean()
    ) / values.std(
        ddof=1
    )

    observed = (
        GlobalZScoreTransformation()(
            values
        )
    )

    pd.testing.assert_series_equal(
        observed,
        expected,
        check_exact=True,
    )


def test_global_strategy_preserves_missing_values() -> None:
    values = pd.Series(
        [
            1.0,
            np.nan,
            2.0,
        ],
        dtype=float,
    )

    result = (
        GlobalZScoreTransformation()(
            values
        )
    )

    assert result.isna().equals(
        values.isna()
    )
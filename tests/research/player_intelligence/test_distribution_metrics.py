#test_distribution_metrics

from __future__ import annotations

import math

import pytest

from research.player_intelligence.aggregation_functions import (
    DistributionShape,
    distribution_shape,
)


def test_distribution_shape_returns_dataclass() -> None:
    result = distribution_shape(
        [0.95, 0.90, 0.85, 0.80, 0.75],
        k=5,
    )

    assert isinstance(result, DistributionShape)


def test_distribution_shape_returns_expected_core_values() -> None:
    result = distribution_shape(
        [0.95, 0.90, 0.85, 0.80, 0.75],
        k=5,
    )

    assert result.mean == pytest.approx(0.85)
    assert result.maximum == pytest.approx(0.95)
    assert result.minimum == pytest.approx(0.75)
    assert result.value_range == pytest.approx(0.20)
    assert result.player_count == 5


def test_distribution_shape_returns_expected_standard_deviation() -> None:
    values = [0.90, 0.80, 0.70]

    result = distribution_shape(
        values,
        k=3,
    )

    mean_value = 0.80
    expected_variance = (
        (0.90 - mean_value) ** 2
        + (0.80 - mean_value) ** 2
        + (0.70 - mean_value) ** 2
    ) / 3

    assert result.standard_deviation == pytest.approx(
        math.sqrt(expected_variance)
    )


def test_distribution_shape_returns_expected_star_gap() -> None:
    result = distribution_shape(
        [0.95, 0.90, 0.85, 0.80, 0.75],
        k=5,
    )

    supporting_mean = (
        0.90
        + 0.85
        + 0.80
        + 0.75
    ) / 4

    assert result.star_gap == pytest.approx(
        0.95 - supporting_mean
    )


def test_distribution_shape_returns_expected_concentration() -> None:
    values = [0.95, 0.90, 0.85, 0.80, 0.75]

    result = distribution_shape(
        values,
        k=5,
    )

    assert result.concentration == pytest.approx(
        0.95 / sum(values)
    )


def test_balanced_population_has_zero_spread() -> None:
    result = distribution_shape(
        [0.85, 0.85, 0.85, 0.85, 0.85],
        k=5,
    )

    assert result.value_range == pytest.approx(0.0)
    assert result.standard_deviation == pytest.approx(0.0)
    assert result.star_gap == pytest.approx(0.0)
    assert result.concentration == pytest.approx(0.20)


def test_top_heavy_population_has_positive_spread() -> None:
    result = distribution_shape(
        [0.95, 0.90, 0.85, 0.80, 0.75],
        k=5,
    )

    assert result.value_range > 0.0
    assert result.standard_deviation > 0.0
    assert result.star_gap > 0.0
    assert result.concentration > 0.20


def test_extreme_superstar_increases_star_gap() -> None:
    balanced = distribution_shape(
        [0.758, 0.758, 0.758, 0.758, 0.758],
        k=5,
    )
    superstar = distribution_shape(
        [0.99, 0.70, 0.70, 0.70, 0.70],
        k=5,
    )

    assert superstar.star_gap > balanced.star_gap


def test_extreme_superstar_increases_concentration() -> None:
    balanced = distribution_shape(
        [0.758, 0.758, 0.758, 0.758, 0.758],
        k=5,
    )
    superstar = distribution_shape(
        [0.99, 0.70, 0.70, 0.70, 0.70],
        k=5,
    )

    assert superstar.concentration > balanced.concentration


def test_distribution_shape_uses_only_top_k() -> None:
    baseline = distribution_shape(
        [0.95, 0.90, 0.85, 0.80, 0.75],
        k=5,
    )
    expanded = distribution_shape(
        [
            0.95,
            0.90,
            0.85,
            0.80,
            0.75,
            0.20,
            0.10,
        ],
        k=5,
    )

    assert expanded == baseline


def test_distribution_shape_is_permutation_invariant() -> None:
    population_a = [0.95, 0.90, 0.85, 0.80, 0.75]
    population_b = [0.80, 0.95, 0.75, 0.85, 0.90]

    assert distribution_shape(
        population_a,
        k=5,
    ) == distribution_shape(
        population_b,
        k=5,
    )


def test_distribution_shape_k_one_has_expected_values() -> None:
    result = distribution_shape(
        [0.90, 0.80],
        k=1,
    )

    assert result.mean == pytest.approx(0.90)
    assert result.maximum == pytest.approx(0.90)
    assert result.minimum == pytest.approx(0.90)
    assert result.value_range == pytest.approx(0.0)
    assert result.standard_deviation == pytest.approx(0.0)
    assert result.star_gap == pytest.approx(0.0)
    assert result.concentration == pytest.approx(1.0)
    assert result.player_count == 1


def test_zero_population_has_zero_concentration() -> None:
    result = distribution_shape(
        [0.0, 0.0, 0.0],
        k=3,
    )

    assert result.concentration == pytest.approx(0.0)


def test_distribution_shape_is_deterministic() -> None:
    values = [0.95, 0.90, 0.85, 0.80, 0.75]

    results = [
        distribution_shape(
            values,
            k=5,
        )
        for _ in range(10)
    ]

    assert all(
        result == results[0]
        for result in results
    )
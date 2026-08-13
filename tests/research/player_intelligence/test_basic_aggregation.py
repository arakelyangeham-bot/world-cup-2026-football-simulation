#test_basic_aggregation

from __future__ import annotations

import pytest

from research.player_intelligence.aggregation_functions import (
    arithmetic_mean,
    maximum_value,
    top_k_mean,
)


# ---------------------------------------------------------------------
# arithmetic_mean
# ---------------------------------------------------------------------


def test_arithmetic_mean_returns_expected_value() -> None:
    result = arithmetic_mean([1.0, 2.0, 3.0, 4.0])

    assert result == pytest.approx(2.5)


def test_arithmetic_mean_preserves_constant_population() -> None:
    result = arithmetic_mean([0.75, 0.75, 0.75, 0.75])

    assert result == pytest.approx(0.75)


def test_arithmetic_mean_is_permutation_invariant() -> None:
    population_a = [0.90, 0.80, 0.70, 0.60]
    population_b = [0.60, 0.90, 0.70, 0.80]

    assert arithmetic_mean(population_a) == pytest.approx(
        arithmetic_mean(population_b)
    )


def test_arithmetic_mean_is_monotonic() -> None:
    baseline = arithmetic_mean([0.90, 0.80, 0.70])
    improved = arithmetic_mean([0.91, 0.80, 0.70])

    assert improved > baseline


def test_arithmetic_mean_is_bounded() -> None:
    values = [0.95, 0.85, 0.75, 0.65]

    result = arithmetic_mean(values)

    assert min(values) <= result <= max(values)


# ---------------------------------------------------------------------
# maximum_value
# ---------------------------------------------------------------------


def test_maximum_value_returns_expected_value() -> None:
    result = maximum_value([0.50, 0.90, 0.70])

    assert result == pytest.approx(0.90)


def test_maximum_value_preserves_constant_population() -> None:
    result = maximum_value([0.75, 0.75, 0.75])

    assert result == pytest.approx(0.75)


def test_maximum_value_is_permutation_invariant() -> None:
    population_a = [0.50, 0.90, 0.70]
    population_b = [0.70, 0.50, 0.90]

    assert maximum_value(population_a) == pytest.approx(
        maximum_value(population_b)
    )


def test_maximum_value_is_monotonic() -> None:
    baseline = maximum_value([0.90, 0.80, 0.70])
    improved = maximum_value([0.91, 0.80, 0.70])

    assert improved > baseline


def test_maximum_value_ignores_non_maximum_improvement() -> None:
    baseline = maximum_value([0.90, 0.80, 0.70])
    modified = maximum_value([0.90, 0.81, 0.70])

    assert modified == pytest.approx(baseline)


# ---------------------------------------------------------------------
# top_k_mean
# ---------------------------------------------------------------------


def test_top_k_mean_returns_expected_value() -> None:
    values = [
        0.70,
        0.90,
        0.50,
        0.80,
        0.60,
        0.75,
        0.55,
        0.85,
        0.45,
        0.65,
    ]

    result = top_k_mean(
        values,
        k=5,
    )

    expected = (
        0.90
        + 0.85
        + 0.80
        + 0.75
        + 0.70
    ) / 5

    assert result == pytest.approx(expected)


def test_top_k_mean_preserves_constant_population() -> None:
    result = top_k_mean(
        [0.80, 0.80, 0.80, 0.80, 0.80],
        k=5,
    )

    assert result == pytest.approx(0.80)


def test_top_k_mean_is_permutation_invariant() -> None:
    population_a = [
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
    ]
    population_b = [
        0.65,
        0.80,
        0.90,
        0.70,
        0.85,
        0.75,
    ]

    assert top_k_mean(
        population_a,
        k=5,
    ) == pytest.approx(
        top_k_mean(
            population_b,
            k=5,
        )
    )


def test_top_k_mean_is_monotonic_inside_top_k() -> None:
    baseline = top_k_mean(
        [0.90, 0.85, 0.80, 0.75, 0.70],
        k=5,
    )
    improved = top_k_mean(
        [0.91, 0.85, 0.80, 0.75, 0.70],
        k=5,
    )

    assert improved > baseline


def test_top_k_mean_is_monotonic_outside_top_k() -> None:
    baseline = top_k_mean(
        [0.90, 0.85, 0.80, 0.75, 0.70, 0.60],
        k=5,
    )
    improved = top_k_mean(
        [0.90, 0.85, 0.80, 0.75, 0.70, 0.61],
        k=5,
    )

    assert improved >= baseline


def test_top_k_mean_ignores_weak_fringe_players() -> None:
    baseline = top_k_mean(
        [0.90, 0.85, 0.80, 0.75, 0.70],
        k=5,
    )
    expanded = top_k_mean(
        [
            0.90,
            0.85,
            0.80,
            0.75,
            0.70,
            0.20,
            0.10,
        ],
        k=5,
    )

    assert expanded == pytest.approx(baseline)


def test_top_k_mean_is_bounded_by_selected_values() -> None:
    values = [
        0.95,
        0.90,
        0.80,
        0.70,
        0.60,
        0.10,
    ]

    result = top_k_mean(
        values,
        k=5,
    )

    selected_values = [
        0.95,
        0.90,
        0.80,
        0.70,
        0.60,
    ]

    assert min(selected_values) <= result <= max(selected_values)


def test_top_k_mean_threshold_crossing_is_small() -> None:
    baseline = top_k_mean(
        [
            0.90,
            0.85,
            0.80,
            0.75,
            0.7000,
            0.6999,
        ],
        k=5,
    )
    modified = top_k_mean(
        [
            0.90,
            0.85,
            0.80,
            0.75,
            0.7000,
            0.7001,
        ],
        k=5,
    )

    assert modified > baseline
    assert modified - baseline == pytest.approx(
        0.0001 / 5
    )


# ---------------------------------------------------------------------
# Cross-function control behavior
# ---------------------------------------------------------------------


def test_maximum_is_at_least_top_k_mean() -> None:
    values = [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]

    assert maximum_value(values) >= top_k_mean(
        values,
        k=5,
    )


def test_top_k_mean_is_at_least_whole_population_mean() -> None:
    values = [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.40,
        0.30,
    ]

    assert top_k_mean(
        values,
        k=5,
    ) >= arithmetic_mean(values)


def test_all_basic_aggregators_agree_for_constant_population() -> None:
    values = [
        0.75,
        0.75,
        0.75,
        0.75,
        0.75,
    ]

    expected = 0.75

    assert arithmetic_mean(values) == pytest.approx(expected)
    assert maximum_value(values) == pytest.approx(expected)
    assert top_k_mean(
        values,
        k=5,
    ) == pytest.approx(expected)
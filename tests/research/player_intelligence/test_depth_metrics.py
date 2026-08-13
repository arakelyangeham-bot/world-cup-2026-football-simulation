#test_depth_metrics

from __future__ import annotations

import pytest

from research.player_intelligence.aggregation_functions import (
    replacement_dropoff,
    replacement_group_mean,
    top_k_mean,
)


def test_replacement_group_mean_uses_expected_ranks() -> None:
    values = [
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
        0.50,
        0.45,
    ]

    result = replacement_group_mean(
        values,
        primary_k=5,
        replacement_k=5,
    )

    expected = (
        0.65
        + 0.60
        + 0.55
        + 0.50
        + 0.45
    ) / 5

    assert result == pytest.approx(expected)


def test_replacement_group_mean_sorts_before_selecting() -> None:
    values = [
        0.55,
        0.90,
        0.45,
        0.70,
        0.60,
        0.85,
        0.50,
        0.75,
        0.65,
        0.80,
    ]

    result = replacement_group_mean(
        values,
        primary_k=5,
        replacement_k=5,
    )

    expected = (
        0.65
        + 0.60
        + 0.55
        + 0.50
        + 0.45
    ) / 5

    assert result == pytest.approx(expected)


def test_replacement_group_mean_is_permutation_invariant() -> None:
    population_a = [
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
        0.50,
        0.45,
    ]
    population_b = [
        0.55,
        0.90,
        0.45,
        0.70,
        0.60,
        0.85,
        0.50,
        0.75,
        0.65,
        0.80,
    ]

    assert replacement_group_mean(
        population_a,
        primary_k=5,
        replacement_k=5,
    ) == pytest.approx(
        replacement_group_mean(
            population_b,
            primary_k=5,
            replacement_k=5,
        )
    )


def test_replacement_group_mean_rejects_insufficient_population() -> None:
    with pytest.raises(
        ValueError,
        match="required 10, received 9",
    ):
        replacement_group_mean(
            [
                0.90,
                0.85,
                0.80,
                0.75,
                0.70,
                0.65,
                0.60,
                0.55,
                0.50,
            ],
            primary_k=5,
            replacement_k=5,
        )


def test_replacement_group_mean_rejects_invalid_primary_k() -> None:
    with pytest.raises(
        ValueError,
        match="primary_k",
    ):
        replacement_group_mean(
            [0.90, 0.80],
            primary_k=0,
            replacement_k=1,
        )


def test_replacement_group_mean_rejects_invalid_replacement_k() -> None:
    with pytest.raises(
        ValueError,
        match="replacement_k",
    ):
        replacement_group_mean(
            [0.90, 0.80],
            primary_k=1,
            replacement_k=0,
        )


def test_replacement_group_mean_ignores_lower_fringe_players() -> None:
    baseline = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.78,
        0.76,
        0.74,
        0.72,
        0.70,
    ]
    expanded = baseline + [
        0.30,
        0.20,
        0.10,
    ]

    assert replacement_group_mean(
        baseline,
        primary_k=5,
        replacement_k=5,
    ) == pytest.approx(
        replacement_group_mean(
            expanded,
            primary_k=5,
            replacement_k=5,
        )
    )


def test_replacement_group_mean_improves_when_replacements_improve() -> None:
    baseline = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.65,
        0.63,
        0.61,
        0.59,
        0.57,
    ]
    improved = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.75,
        0.73,
        0.71,
        0.69,
        0.67,
    ]

    assert replacement_group_mean(
        improved,
        primary_k=5,
        replacement_k=5,
    ) > replacement_group_mean(
        baseline,
        primary_k=5,
        replacement_k=5,
    )


def test_replacement_dropoff_returns_expected_value() -> None:
    values = [
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
        0.50,
        0.45,
    ]

    result = replacement_dropoff(
        values,
        primary_k=5,
        replacement_k=5,
    )

    primary_mean = (
        0.90
        + 0.85
        + 0.80
        + 0.75
        + 0.70
    ) / 5
    replacement_mean = (
        0.65
        + 0.60
        + 0.55
        + 0.50
        + 0.45
    ) / 5

    assert result == pytest.approx(
        primary_mean - replacement_mean
    )


def test_replacement_dropoff_matches_component_difference() -> None:
    values = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.78,
        0.76,
        0.74,
        0.72,
        0.70,
    ]

    primary = top_k_mean(
        values,
        k=5,
    )
    replacement = replacement_group_mean(
        values,
        primary_k=5,
        replacement_k=5,
    )

    assert replacement_dropoff(
        values,
        primary_k=5,
        replacement_k=5,
    ) == pytest.approx(
        primary - replacement
    )


def test_replacement_improvement_reduces_dropoff() -> None:
    baseline = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.65,
        0.63,
        0.61,
        0.59,
        0.57,
    ]
    improved = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.75,
        0.73,
        0.71,
        0.69,
        0.67,
    ]

    baseline_dropoff = replacement_dropoff(
        baseline,
        primary_k=5,
        replacement_k=5,
    )
    improved_dropoff = replacement_dropoff(
        improved,
        primary_k=5,
        replacement_k=5,
    )

    assert improved_dropoff < baseline_dropoff


def test_primary_improvement_increases_dropoff_when_bench_is_fixed() -> None:
    baseline = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.78,
        0.76,
        0.74,
        0.72,
        0.70,
    ]
    improved = [
        0.92,
        0.90,
        0.88,
        0.86,
        0.84,
        0.78,
        0.76,
        0.74,
        0.72,
        0.70,
    ]

    assert replacement_dropoff(
        improved,
        primary_k=5,
        replacement_k=5,
    ) > replacement_dropoff(
        baseline,
        primary_k=5,
        replacement_k=5,
    )


def test_uniform_improvement_preserves_dropoff() -> None:
    baseline = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.78,
        0.76,
        0.74,
        0.72,
        0.70,
    ]
    improved = [
        value + 0.02
        for value in baseline
    ]

    assert replacement_dropoff(
        improved,
        primary_k=5,
        replacement_k=5,
    ) == pytest.approx(
        replacement_dropoff(
            baseline,
            primary_k=5,
            replacement_k=5,
        )
    )


def test_weak_fringe_addition_does_not_change_dropoff() -> None:
    baseline = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.78,
        0.76,
        0.74,
        0.72,
        0.70,
    ]
    expanded = baseline + [
        0.30,
        0.20,
        0.10,
    ]

    assert replacement_dropoff(
        expanded,
        primary_k=5,
        replacement_k=5,
    ) == pytest.approx(
        replacement_dropoff(
            baseline,
            primary_k=5,
            replacement_k=5,
        )
    )


def test_balanced_primary_and_replacement_groups_have_zero_dropoff() -> None:
    values = [
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
    ]

    assert replacement_dropoff(
        values,
        primary_k=5,
        replacement_k=5,
    ) == pytest.approx(0.0)


def test_replacement_metrics_are_deterministic() -> None:
    values = [
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.78,
        0.76,
        0.74,
        0.72,
        0.70,
    ]

    replacement_results = [
        replacement_group_mean(
            values,
            primary_k=5,
            replacement_k=5,
        )
        for _ in range(10)
    ]
    dropoff_results = [
        replacement_dropoff(
            values,
            primary_k=5,
            replacement_k=5,
        )
        for _ in range(10)
    ]

    assert all(
        result == replacement_results[0]
        for result in replacement_results
    )
    assert all(
        result == dropoff_results[0]
        for result in dropoff_results
    )
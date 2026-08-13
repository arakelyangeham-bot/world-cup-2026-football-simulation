#test_weighted_aggregation

from __future__ import annotations

import math

import pytest

from research.player_intelligence.aggregation_functions import (
    ability_power_weighted_mean,
    power_mean_top_k,
    rank_weighted_top_k,
    softmax_weighted_top_k,
    star_influence_top_k,
    top_k_mean,
)


MILD_WEIGHTS = [
    0.24,
    0.22,
    0.20,
    0.18,
    0.16,
]

MODERATE_WEIGHTS = [
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
]

STRONG_WEIGHTS = [
    0.40,
    0.25,
    0.15,
    0.12,
    0.08,
]


# ---------------------------------------------------------------------
# rank_weighted_top_k
# ---------------------------------------------------------------------


def test_rank_weighted_top_k_returns_expected_value() -> None:
    values = [
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
    ]

    result = rank_weighted_top_k(
        values,
        weights=MODERATE_WEIGHTS,
    )

    expected = (
        0.90 * 0.30
        + 0.80 * 0.25
        + 0.70 * 0.20
        + 0.60 * 0.15
        + 0.50 * 0.10
    )

    assert result == pytest.approx(expected)


def test_rank_weighted_top_k_sorts_before_weighting() -> None:
    values = [
        0.50,
        0.90,
        0.60,
        0.80,
        0.70,
    ]

    result = rank_weighted_top_k(
        values,
        weights=MODERATE_WEIGHTS,
    )

    expected = rank_weighted_top_k(
        sorted(values, reverse=True),
        weights=MODERATE_WEIGHTS,
    )

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "weights",
    [
        MILD_WEIGHTS,
        MODERATE_WEIGHTS,
        STRONG_WEIGHTS,
    ],
)
def test_rank_weighted_top_k_preserves_constant_population(
    weights: list[float],
) -> None:
    result = rank_weighted_top_k(
        [0.80, 0.80, 0.80, 0.80, 0.80],
        weights=weights,
    )

    assert result == pytest.approx(0.80)


def test_rank_weighted_top_k_rewards_top_heavy_population() -> None:
    top_heavy = [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]
    balanced = [
        0.85,
        0.85,
        0.85,
        0.85,
        0.85,
    ]

    assert rank_weighted_top_k(
        top_heavy,
        weights=MODERATE_WEIGHTS,
    ) > rank_weighted_top_k(
        balanced,
        weights=MODERATE_WEIGHTS,
    )


def test_stronger_rank_weighting_increases_top_heavy_score() -> None:
    values = [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]

    mild = rank_weighted_top_k(
        values,
        weights=MILD_WEIGHTS,
    )
    moderate = rank_weighted_top_k(
        values,
        weights=MODERATE_WEIGHTS,
    )
    strong = rank_weighted_top_k(
        values,
        weights=STRONG_WEIGHTS,
    )

    assert mild < moderate < strong


def test_rank_weighted_top_k_is_permutation_invariant() -> None:
    population_a = [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]
    population_b = [
        0.80,
        0.95,
        0.75,
        0.85,
        0.90,
    ]

    assert rank_weighted_top_k(
        population_a,
        weights=MODERATE_WEIGHTS,
    ) == pytest.approx(
        rank_weighted_top_k(
            population_b,
            weights=MODERATE_WEIGHTS,
        )
    )


def test_rank_weighted_top_k_rejects_negative_weight() -> None:
    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        rank_weighted_top_k(
            [0.90, 0.80],
            weights=[1.10, -0.10],
        )


def test_rank_weighted_top_k_rejects_invalid_weight_sum() -> None:
    with pytest.raises(
        ValueError,
        match="must sum to one",
    ):
        rank_weighted_top_k(
            [0.90, 0.80],
            weights=[0.40, 0.40],
        )


def test_rank_weighted_top_k_rejects_all_zero_weights() -> None:
    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        rank_weighted_top_k(
            [0.90, 0.80],
            weights=[0.0, 0.0],
        )


# ---------------------------------------------------------------------
# star_influence_top_k
# ---------------------------------------------------------------------


def test_star_influence_returns_expected_value() -> None:
    values = [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]

    result = star_influence_top_k(
        values,
        k=5,
        alpha=0.20,
    )

    expected = (
        0.80 * 0.85
        + 0.20 * 0.95
    )

    assert result == pytest.approx(expected)


def test_star_influence_alpha_zero_matches_top_k_mean() -> None:
    values = [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]

    assert star_influence_top_k(
        values,
        k=5,
        alpha=0.0,
    ) == pytest.approx(
        top_k_mean(
            values,
            k=5,
        )
    )


def test_star_influence_alpha_one_matches_maximum() -> None:
    values = [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]

    assert star_influence_top_k(
        values,
        k=5,
        alpha=1.0,
    ) == pytest.approx(0.95)


def test_star_influence_increases_with_alpha() -> None:
    values = [
        0.99,
        0.70,
        0.70,
        0.70,
        0.70,
    ]

    alpha_10 = star_influence_top_k(
        values,
        k=5,
        alpha=0.10,
    )
    alpha_20 = star_influence_top_k(
        values,
        k=5,
        alpha=0.20,
    )
    alpha_30 = star_influence_top_k(
        values,
        k=5,
        alpha=0.30,
    )

    assert alpha_10 < alpha_20 < alpha_30


@pytest.mark.parametrize(
    "alpha",
    [
        -0.01,
        1.01,
    ],
)
def test_star_influence_rejects_alpha_outside_unit_interval(
    alpha: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"within \[0, 1\]",
    ):
        star_influence_top_k(
            [0.90, 0.80],
            k=2,
            alpha=alpha,
        )


# ---------------------------------------------------------------------
# power_mean_top_k
# ---------------------------------------------------------------------


def test_power_mean_power_one_matches_top_k_mean() -> None:
    values = [
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
    ]

    assert power_mean_top_k(
        values,
        k=5,
        power=1.0,
    ) == pytest.approx(
        top_k_mean(
            values,
            k=5,
        )
    )


def test_power_mean_power_two_matches_hand_calculation() -> None:
    values = [
        1.0,
        2.0,
        3.0,
    ]

    result = power_mean_top_k(
        values,
        k=3,
        power=2.0,
    )

    expected = math.sqrt(
        (
            1.0**2
            + 2.0**2
            + 3.0**2
        )
        / 3
    )

    assert result == pytest.approx(expected)


def test_power_mean_increases_with_power_for_unequal_values() -> None:
    values = [
        0.99,
        0.70,
        0.70,
        0.70,
        0.70,
    ]

    power_1 = power_mean_top_k(
        values,
        k=5,
        power=1.0,
    )
    power_15 = power_mean_top_k(
        values,
        k=5,
        power=1.5,
    )
    power_2 = power_mean_top_k(
        values,
        k=5,
        power=2.0,
    )

    assert power_1 < power_15 < power_2


def test_power_mean_preserves_constant_population() -> None:
    result = power_mean_top_k(
        [0.75, 0.75, 0.75, 0.75, 0.75],
        k=5,
        power=2.0,
    )

    assert result == pytest.approx(0.75)


def test_power_mean_rejects_non_positive_power() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        power_mean_top_k(
            [0.90, 0.80],
            k=2,
            power=0.0,
        )


def test_power_mean_rejects_negative_input_values() -> None:
    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        power_mean_top_k(
            [0.90, -0.10],
            k=2,
            power=2.0,
        )


# ---------------------------------------------------------------------
# softmax_weighted_top_k
# ---------------------------------------------------------------------


def test_softmax_beta_zero_matches_top_k_mean() -> None:
    values = [
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
    ]

    assert softmax_weighted_top_k(
        values,
        k=5,
        beta=0.0,
    ) == pytest.approx(
        top_k_mean(
            values,
            k=5,
        )
    )


def test_softmax_score_increases_with_beta() -> None:
    values = [
        0.99,
        0.70,
        0.70,
        0.70,
        0.70,
    ]

    beta_0 = softmax_weighted_top_k(
        values,
        k=5,
        beta=0.0,
    )
    beta_1 = softmax_weighted_top_k(
        values,
        k=5,
        beta=1.0,
    )
    beta_3 = softmax_weighted_top_k(
        values,
        k=5,
        beta=3.0,
    )
    beta_5 = softmax_weighted_top_k(
        values,
        k=5,
        beta=5.0,
    )

    assert beta_0 < beta_1 < beta_3 < beta_5


def test_softmax_output_is_bounded() -> None:
    values = [
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
    ]

    result = softmax_weighted_top_k(
        values,
        k=5,
        beta=3.0,
    )

    assert min(values) <= result <= max(values)


def test_softmax_is_numerically_stable() -> None:
    result = softmax_weighted_top_k(
        [
            1_000_000.0,
            999_999.0,
            999_998.0,
        ],
        k=3,
        beta=10.0,
    )

    assert math.isfinite(result)
    assert 999_998.0 <= result <= 1_000_000.0


def test_softmax_rejects_negative_beta() -> None:
    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        softmax_weighted_top_k(
            [0.90, 0.80],
            k=2,
            beta=-1.0,
        )


# ---------------------------------------------------------------------
# historical Study 011 ability-power weighting
# ---------------------------------------------------------------------


def test_ability_power_weighted_mean_matches_historical_formula() -> None:
    values = [
        1.0,
        2.0,
        3.0,
    ]

    result = ability_power_weighted_mean(
        values,
        gamma=2.0,
    )

    expected = (
        1.0 * 1.0**2
        + 2.0 * 2.0**2
        + 3.0 * 3.0**2
    ) / (
        1.0**2
        + 2.0**2
        + 3.0**2
    )

    assert result == pytest.approx(expected)


def test_ability_power_weighted_gamma_zero_matches_mean() -> None:
    values = [
        0.90,
        0.80,
        0.70,
    ]

    result = ability_power_weighted_mean(
        values,
        gamma=0.0,
    )

    assert result == pytest.approx(
        sum(values) / len(values)
    )


def test_ability_power_weighted_clamps_negative_values() -> None:
    result = ability_power_weighted_mean(
        [-1.0, 0.0, 1.0],
        gamma=2.0,
    )

    assert result == pytest.approx(1.0)


def test_ability_power_weighted_all_zero_values_returns_zero() -> None:
    result = ability_power_weighted_mean(
        [0.0, 0.0, 0.0],
        gamma=2.0,
    )

    assert result == pytest.approx(0.0)


def test_ability_power_weighted_rejects_negative_gamma() -> None:
    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        ability_power_weighted_mean(
            [0.90, 0.80],
            gamma=-1.0,
        )


# ---------------------------------------------------------------------
# Comparative behavior
# ---------------------------------------------------------------------


def test_all_weighted_methods_preserve_constant_population() -> None:
    values = [
        0.80,
        0.80,
        0.80,
        0.80,
        0.80,
    ]

    expected = 0.80

    assert rank_weighted_top_k(
        values,
        weights=MODERATE_WEIGHTS,
    ) == pytest.approx(expected)

    assert star_influence_top_k(
        values,
        k=5,
        alpha=0.20,
    ) == pytest.approx(expected)

    assert power_mean_top_k(
        values,
        k=5,
        power=1.50,
    ) == pytest.approx(expected)

    assert softmax_weighted_top_k(
        values,
        k=5,
        beta=3.0,
    ) == pytest.approx(expected)

    assert ability_power_weighted_mean(
        values,
        gamma=2.0,
    ) == pytest.approx(expected)


def test_all_weighted_methods_are_permutation_invariant() -> None:
    population_a = [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    ]
    population_b = [
        0.80,
        0.95,
        0.75,
        0.85,
        0.90,
    ]

    assert rank_weighted_top_k(
        population_a,
        weights=MODERATE_WEIGHTS,
    ) == pytest.approx(
        rank_weighted_top_k(
            population_b,
            weights=MODERATE_WEIGHTS,
        )
    )

    assert star_influence_top_k(
        population_a,
        k=5,
        alpha=0.20,
    ) == pytest.approx(
        star_influence_top_k(
            population_b,
            k=5,
            alpha=0.20,
        )
    )

    assert power_mean_top_k(
        population_a,
        k=5,
        power=1.50,
    ) == pytest.approx(
        power_mean_top_k(
            population_b,
            k=5,
            power=1.50,
        )
    )

    assert softmax_weighted_top_k(
        population_a,
        k=5,
        beta=3.0,
    ) == pytest.approx(
        softmax_weighted_top_k(
            population_b,
            k=5,
            beta=3.0,
        )
    )

    assert ability_power_weighted_mean(
        population_a,
        gamma=2.0,
    ) == pytest.approx(
        ability_power_weighted_mean(
            population_b,
            gamma=2.0,
        )
    )
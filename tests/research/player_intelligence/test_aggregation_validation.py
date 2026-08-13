# test_aggregation_validation.py

from __future__ import annotations

import math

import pytest

from research.player_intelligence.aggregation_validation import (
    sorted_descending,
    top_k_values,
    validated_positive_integer,
    validated_values,
)


# ---------------------------------------------------------------------
# validated_values
# ---------------------------------------------------------------------


def test_validated_values_returns_tuple_of_floats() -> None:
    result = validated_values(
        [1, 2.5, "3.0"],  # type: ignore[list-item]
    )

    assert result == (1.0, 2.5, 3.0)
    assert isinstance(result, tuple)
    assert all(
        isinstance(value, float)
        for value in result
    )


def test_validated_values_does_not_mutate_input() -> None:
    values = [
        0.70,
        0.90,
        0.80,
    ]
    original = list(values)

    validated_values(values)

    assert values == original


def test_validated_values_rejects_empty_population() -> None:
    with pytest.raises(
        ValueError,
        match="requires at least one value",
    ):
        validated_values([])


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validated_values_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        validated_values(
            [0.80, invalid_value]
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        object(),
        "not-a-number",
    ],
)
def test_validated_values_rejects_non_numeric_values(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be real numbers",
    ):
        validated_values(
            [0.80, invalid_value],  # type: ignore[list-item]
        )


def test_validated_values_accepts_numeric_strings() -> None:
    result = validated_values(
        ["0.90", "0.80"],  # type: ignore[list-item]
    )

    assert result == pytest.approx(
        (0.90, 0.80)
    )


# ---------------------------------------------------------------------
# validated_positive_integer
# ---------------------------------------------------------------------


def test_validated_positive_integer_returns_value() -> None:
    result = validated_positive_integer(
        5,
        parameter_name="k",
    )

    assert result == 5


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        -1,
        -10,
    ],
)
def test_validated_positive_integer_rejects_non_positive_values(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        validated_positive_integer(
            invalid_value,
            parameter_name="k",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.5,
        "5",
        None,
        True,
        False,
    ],
)
def test_validated_positive_integer_rejects_non_integer_values(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        validated_positive_integer(
            invalid_value,  # type: ignore[arg-type]
            parameter_name="k",
        )


def test_validated_positive_integer_uses_parameter_name() -> None:
    with pytest.raises(
        ValueError,
        match="replacement_k",
    ):
        validated_positive_integer(
            0,
            parameter_name="replacement_k",
        )


# ---------------------------------------------------------------------
# sorted_descending
# ---------------------------------------------------------------------


def test_sorted_descending_orders_values() -> None:
    result = sorted_descending(
        [
            0.70,
            0.90,
            0.80,
            0.60,
        ]
    )

    assert result == pytest.approx(
        (
            0.90,
            0.80,
            0.70,
            0.60,
        )
    )


def test_sorted_descending_preserves_duplicate_values() -> None:
    result = sorted_descending(
        [
            0.80,
            0.90,
            0.80,
            0.70,
        ]
    )

    assert result == pytest.approx(
        (
            0.90,
            0.80,
            0.80,
            0.70,
        )
    )


def test_sorted_descending_does_not_mutate_input() -> None:
    values = [
        0.70,
        0.90,
        0.80,
    ]
    original = list(values)

    sorted_descending(values)

    assert values == original


def test_sorted_descending_rejects_non_finite_values() -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        sorted_descending(
            [
                0.90,
                math.inf,
            ]
        )


# ---------------------------------------------------------------------
# top_k_values
# ---------------------------------------------------------------------


def test_top_k_values_returns_strongest_values() -> None:
    result = top_k_values(
        [
            0.70,
            0.90,
            0.60,
            0.80,
            0.75,
        ],
        k=3,
    )

    assert result == pytest.approx(
        (
            0.90,
            0.80,
            0.75,
        )
    )


def test_top_k_values_is_permutation_invariant() -> None:
    population_a = [
        0.90,
        0.80,
        0.70,
        0.60,
    ]
    population_b = [
        0.60,
        0.90,
        0.70,
        0.80,
    ]

    assert top_k_values(
        population_a,
        k=3,
    ) == pytest.approx(
        top_k_values(
            population_b,
            k=3,
        )
    )


def test_top_k_values_allows_exact_population_size() -> None:
    result = top_k_values(
        [
            0.90,
            0.80,
            0.70,
        ],
        k=3,
    )

    assert result == pytest.approx(
        (
            0.90,
            0.80,
            0.70,
        )
    )


def test_top_k_values_rejects_insufficient_population() -> None:
    with pytest.raises(
        ValueError,
        match="required 5, received 3",
    ):
        top_k_values(
            [
                0.90,
                0.80,
                0.70,
            ],
            k=5,
        )


def test_top_k_values_rejects_invalid_k() -> None:
    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        top_k_values(
            [
                0.90,
                0.80,
            ],
            k=0,
        )
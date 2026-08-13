# aggregation_functions.py

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from research.player_intelligence.aggregation_validation import (
    sorted_descending,
    top_k_values,
    validated_positive_integer,
    validated_values,
)


@dataclass(frozen=True)
class DistributionShape:
    """
    Distribution diagnostics for the strongest k values.

    These fields supplement a primary strength feature rather than
    replacing it.
    """

    mean: float
    maximum: float
    minimum: float
    value_range: float
    standard_deviation: float
    star_gap: float
    concentration: float
    player_count: int



def arithmetic_mean(
    values: Sequence[float],
) -> float:
    """
    Return the arithmetic mean of all values.
    """

    validated = validated_values(values)

    return float(
        math.fsum(validated) / len(validated)
    )


def maximum_value(
    values: Sequence[float],
) -> float:
    """
    Return the maximum value in the population.
    """

    validated = validated_values(values)

    return float(max(validated))


def top_k_mean(
    values: Sequence[float],
    *,
    k: int,
) -> float:
    """
    Return the arithmetic mean of the strongest k values.
    """

    top_values = top_k_values(
        values,
        k=k,
    )

    return float(
        math.fsum(top_values) / len(top_values)
    )


def rank_weighted_top_k(
    values: Sequence[float],
    *,
    weights: Sequence[float],
) -> float:
    """
    Return a rank-weighted mean of the strongest values.

    The number of weights determines k.

    The weights must:

    - be finite;
    - be non-negative;
    - contain at least one positive value;
    - sum to one within floating-point tolerance.

    Values are sorted from strongest to weakest before weights are
    applied.
    """

    validated_weights = validated_values(weights)

    if any(weight < 0.0 for weight in validated_weights):
        raise ValueError(
            "Rank weights must be non-negative."
        )

    total_weight = math.fsum(validated_weights)

    if total_weight <= 0.0:
        raise ValueError(
            "At least one rank weight must be positive."
        )

    if not math.isclose(
        total_weight,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Rank weights must sum to one. "
            f"Observed sum: {total_weight!r}."
        )

    top_values = top_k_values(
        values,
        k=len(validated_weights),
    )

    return float(
        math.fsum(
            value * weight
            for value, weight in zip(
                top_values,
                validated_weights,
                strict=True,
            )
        )
    )


def star_influence_top_k(
    values: Sequence[float],
    *,
    k: int,
    alpha: float,
) -> float:
    """
    Blend the top-k arithmetic mean with the strongest value.

    Formula:

        (1 - alpha) * mean(top k)
        + alpha * max(top k)

    alpha must lie in [0, 1].
    """

    try:
        validated_alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "alpha must be a real number."
        ) from error

    if not math.isfinite(validated_alpha):
        raise ValueError(
            "alpha must be finite."
        )

    if not 0.0 <= validated_alpha <= 1.0:
        raise ValueError(
            "alpha must lie within [0, 1]."
        )

    top_values = top_k_values(
        values,
        k=k,
    )
    top_mean = math.fsum(top_values) / len(top_values)
    strongest = top_values[0]

    return float(
        (1.0 - validated_alpha) * top_mean
        + validated_alpha * strongest
    )


def power_mean_top_k(
    values: Sequence[float],
    *,
    k: int,
    power: float,
) -> float:
    """
    Return the generalized power mean of the strongest k values.

    Study 089 initially restricts power to positive values because the
    player-projection scale may contain zero.

    power = 1 reproduces the arithmetic top-k mean.
    """

    try:
        validated_power = float(power)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "power must be a real number."
        ) from error

    if not math.isfinite(validated_power):
        raise ValueError(
            "power must be finite."
        )

    if validated_power <= 0.0:
        raise ValueError(
            "power must be greater than zero."
        )

    top_values = top_k_values(
        values,
        k=k,
    )

    if any(value < 0.0 for value in top_values):
        raise ValueError(
            "Power-mean inputs must be non-negative."
        )

    powered_mean = (
        math.fsum(
            value ** validated_power
            for value in top_values
        )
        / len(top_values)
    )

    return float(
        powered_mean ** (1.0 / validated_power)
    )


def softmax_weighted_top_k(
    values: Sequence[float],
    *,
    k: int,
    beta: float,
) -> float:
    """
    Return a softmax-weighted mean of the strongest k values.

    beta controls concentration.

    beta = 0 reproduces the arithmetic top-k mean.

    A numerically stable exponent calculation is used by subtracting the
    largest exponent input before exponentiation.
    """

    try:
        validated_beta = float(beta)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "beta must be a real number."
        ) from error

    if not math.isfinite(validated_beta):
        raise ValueError(
            "beta must be finite."
        )

    if validated_beta < 0.0:
        raise ValueError(
            "beta must be non-negative."
        )

    top_values = top_k_values(
        values,
        k=k,
    )

    scaled_values = tuple(
        validated_beta * value
        for value in top_values
    )
    maximum_scaled_value = max(scaled_values)

    exponentials = tuple(
        math.exp(value - maximum_scaled_value)
        for value in scaled_values
    )
    denominator = math.fsum(exponentials)

    if denominator <= 0.0:
        raise RuntimeError(
            "Softmax denominator must be positive."
        )

    weights = tuple(
        exponential / denominator
        for exponential in exponentials
    )

    return float(
        math.fsum(
            value * weight
            for value, weight in zip(
                top_values,
                weights,
                strict=True,
            )
        )
    )


def ability_power_weighted_mean(
    values: Sequence[float],
    *,
    gamma: float = 2.0,
) -> float:
    """
    Reproduce the historical Study 011 star-weighted formula.

    Formula:

        weight_i = max(value_i, 0) ** gamma

        result =
            sum(value_i * weight_i)
            /
            sum(weight_i)

    This is not the same as a mathematical power mean and is not the same
    as the Study 089 star-influence blend.

    It is retained as an explicitly named historical control.
    """

    try:
        validated_gamma = float(gamma)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "gamma must be a real number."
        ) from error

    if not math.isfinite(validated_gamma):
        raise ValueError(
            "gamma must be finite."
        )

    if validated_gamma < 0.0:
        raise ValueError(
            "gamma must be non-negative."
        )

    validated = validated_values(values)

    non_negative_values = tuple(
        max(value, 0.0)
        for value in validated
    )

    if validated_gamma == 0.0:
        weights = tuple(
            1.0
            for _ in non_negative_values
        )
    else:
        weights = tuple(
            value ** validated_gamma
            for value in non_negative_values
        )

    total_weight = math.fsum(weights)

    if total_weight == 0.0:
        return 0.0

    return float(
        math.fsum(
            value * weight
            for value, weight in zip(
                non_negative_values,
                weights,
                strict=True,
            )
        )
        / total_weight
    )


def replacement_group_mean(
    values: Sequence[float],
    *,
    primary_k: int,
    replacement_k: int,
) -> float:
    """
    Return the mean of the replacement group immediately below the
    primary group.

    Example:

        primary_k = 5
        replacement_k = 5

    uses ranks 6 through 10.
    """

    validated_primary_k = validated_positive_integer(
        primary_k,
        parameter_name="primary_k",
    )
    validated_replacement_k = validated_positive_integer(
        replacement_k,
        parameter_name="replacement_k",
    )

    ordered = sorted_descending(values)
    required_count = (
        validated_primary_k
        + validated_replacement_k
    )

    if len(ordered) < required_count:
        raise ValueError(
            "Insufficient values for replacement-group aggregation: "
            f"required {required_count}, received {len(ordered)}."
        )

    replacement_values = ordered[
        validated_primary_k:required_count
    ]

    return float(
        math.fsum(replacement_values)
        / len(replacement_values)
    )


def replacement_dropoff(
    values: Sequence[float],
    *,
    primary_k: int,
    replacement_k: int,
) -> float:
    """
    Return the difference between primary and replacement quality.

    A larger value indicates a steeper reduction from the primary group
    to the replacement group.
    """

    primary_mean = top_k_mean(
        values,
        k=primary_k,
    )
    replacement_mean = replacement_group_mean(
        values,
        primary_k=primary_k,
        replacement_k=replacement_k,
    )

    return float(
        primary_mean - replacement_mean
    )


def distribution_shape(
    values: Sequence[float],
    *,
    k: int,
) -> DistributionShape:
    """
    Return interpretable distribution diagnostics for the strongest k
    values.

    star_gap is defined as:

        strongest player
        minus
        mean of the remaining top-k players

    For k = 1, star_gap is defined as zero.

    concentration is defined as:

        strongest value
        /
        sum of top-k values

    If the top-k sum is zero, concentration is defined as zero.
    """

    top_values = top_k_values(
        values,
        k=k,
    )

    count = len(top_values)
    mean_value = (
        math.fsum(top_values)
        / count
    )
    maximum = top_values[0]
    minimum = top_values[-1]
    value_range = maximum - minimum

    variance = (
        math.fsum(
            (value - mean_value) ** 2
            for value in top_values
        )
        / count
    )
    standard_deviation = math.sqrt(variance)

    if count == 1:
        star_gap = 0.0
    else:
        supporting_mean = (
            math.fsum(top_values[1:])
            / (count - 1)
        )
        star_gap = maximum - supporting_mean

    total = math.fsum(top_values)

    if total == 0.0:
        concentration = 0.0
    else:
        concentration = maximum / total

    return DistributionShape(
        mean=float(mean_value),
        maximum=float(maximum),
        minimum=float(minimum),
        value_range=float(value_range),
        standard_deviation=float(standard_deviation),
        star_gap=float(star_gap),
        concentration=float(concentration),
        player_count=count,
    )
#aggregation_validation

# aggregation_validation.py

from __future__ import annotations

from collections.abc import Sequence
import math


def validated_values(
    values: Sequence[float],
) -> tuple[float, ...]:
    """
    Convert an input sequence to a non-empty tuple of finite floats.

    This low-level research layer is intentionally strict. Production
    adapters may implement their own explicit empty-population behavior
    before calling aggregation functions.
    """

    if not values:
        raise ValueError(
            "Aggregation requires at least one value."
        )

    validated: list[float] = []

    for index, value in enumerate(values):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Aggregation values must be real numbers. "
                f"Invalid value at index {index}: {value!r}."
            ) from error

        if not math.isfinite(numeric_value):
            raise ValueError(
                "Aggregation values must be finite. "
                f"Invalid value at index {index}: {value!r}."
            )

        validated.append(numeric_value)

    return tuple(validated)


def validated_positive_integer(
    value: int,
    *,
    parameter_name: str,
) -> int:
    """
    Validate a strictly positive integer parameter.

    Boolean values are rejected because bool is a subclass of int.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{parameter_name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{parameter_name} must be greater than zero."
        )

    return value


def sorted_descending(
    values: Sequence[float],
) -> tuple[float, ...]:
    """
    Validate and return values in descending order.

    The original input is not mutated.
    """

    return tuple(
        sorted(
            validated_values(values),
            reverse=True,
        )
    )


def top_k_values(
    values: Sequence[float],
    *,
    k: int,
) -> tuple[float, ...]:
    """
    Return the strongest k validated values.
    """

    validated_k = validated_positive_integer(
        k,
        parameter_name="k",
    )
    ordered = sorted_descending(values)

    if len(ordered) < validated_k:
        raise ValueError(
            "Insufficient values for top-k aggregation: "
            f"required {validated_k}, received {len(ordered)}."
        )

    return ordered[:validated_k]
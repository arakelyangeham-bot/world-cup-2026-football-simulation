# aggregation_adapter.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from research.player_intelligence.aggregation_functions import (
    ability_power_weighted_mean,
    arithmetic_mean,
    power_mean_top_k,
    rank_weighted_top_k,
    replacement_dropoff,
    replacement_group_mean,
    softmax_weighted_top_k,
    star_influence_top_k,
    top_k_mean,
)


SCALAR_AGGREGATION_FAMILIES = {
    "arithmetic_mean",
    "top_k_mean",
    "rank_weighted_top_k",
    "star_influence_top_k",
    "power_mean_top_k",
    "softmax_weighted_top_k",
    "ability_power_weighted_mean",
    "replacement_group_mean",
    "replacement_dropoff",
}


@runtime_checkable
class AggregationSpecificationLike(Protocol):
    """
    Structural interface required by the player-intelligence adapter.

    Study-specific specification classes may satisfy this interface
    without the reusable player-intelligence layer importing them.
    """

    aggregation_family: str
    output_type: str
    parameters: Mapping[str, object]

    def validate(self) -> None:
        ...


def minimum_required_population(
    specification: AggregationSpecificationLike,
) -> int:
    """
    Return the minimum number of player values required.

    This adapter is intentionally strict. An undersized real-team
    population is a coverage problem and must not silently fall back
    to another aggregation method.
    """

    specification.validate()

    family = specification.aggregation_family
    parameters = specification.parameters

    if family in {
        "arithmetic_mean",
        "ability_power_weighted_mean",
    }:
        return 1

    if family in {
        "top_k_mean",
        "star_influence_top_k",
        "power_mean_top_k",
        "softmax_weighted_top_k",
    }:
        return int(parameters["k"])

    if family == "rank_weighted_top_k":
        return len(tuple(parameters["weights"]))  # type: ignore[arg-type]

    if family in {
        "replacement_group_mean",
        "replacement_dropoff",
    }:
        return (
            int(parameters["primary_k"])
            + int(parameters["replacement_k"])
        )

    if family == "distribution_shape":
        raise TypeError(
            "Distribution diagnostics do not produce one scalar "
            "team-strength component."
        )

    raise KeyError(
        "No population-size rule exists for aggregation family "
        f"{family!r}."
    )


def aggregate_dimension_values(
    values: Sequence[float],
    *,
    specification: AggregationSpecificationLike,
) -> float:
    """
    Aggregate one real-team dimension using a validated scalar
    aggregation specification.

    Parameters
    ----------
    values:
        Player-level values for one dimension, such as attack,
        midfield, defense, goalkeeper, or overall ability.

    specification:
        Any validated object exposing ``aggregation_family``,
        ``output_type``, ``parameters``, and ``validate()``.

    Returns
    -------
    float
        Scalar aggregate for the supplied dimension.

    Raises
    ------
    ValueError
        If the population is too small.

    TypeError
        If the specification produces distribution diagnostics rather
        than a scalar representation.
    """

    specification.validate()

    family = specification.aggregation_family

    if family not in SCALAR_AGGREGATION_FAMILIES:
        if family == "distribution_shape":
            raise TypeError(
                "distribution_shape cannot be used as a scalar "
                "team-representation aggregation."
            )

        raise KeyError(
            "Unsupported scalar aggregation family "
            f"{family!r}."
        )

    required_size = minimum_required_population(
        specification
    )
    observed_size = len(values)

    if observed_size < required_size:
        raise ValueError(
            "Insufficient player population for aggregation "
            f"family {family!r}: required {required_size}, "
            f"received {observed_size}."
        )

    parameters = specification.parameters

    if family == "arithmetic_mean":
        return arithmetic_mean(values)

    if family == "top_k_mean":
        return top_k_mean(
            values,
            k=int(parameters["k"]),
        )

    if family == "rank_weighted_top_k":
        return rank_weighted_top_k(
            values,
            weights=tuple(
                float(weight)
                for weight in parameters["weights"]  # type: ignore[union-attr]
            ),
        )

    if family == "star_influence_top_k":
        return star_influence_top_k(
            values,
            k=int(parameters["k"]),
            alpha=float(parameters["alpha"]),
        )

    if family == "power_mean_top_k":
        return power_mean_top_k(
            values,
            k=int(parameters["k"]),
            power=float(parameters["power"]),
        )

    if family == "softmax_weighted_top_k":
        return softmax_weighted_top_k(
            values,
            k=int(parameters["k"]),
            beta=float(parameters["beta"]),
        )

    if family == "ability_power_weighted_mean":
        return ability_power_weighted_mean(
            values,
            gamma=float(parameters["gamma"]),
        )

    if family == "replacement_group_mean":
        return replacement_group_mean(
            values,
            primary_k=int(parameters["primary_k"]),
            replacement_k=int(parameters["replacement_k"]),
        )

    if family == "replacement_dropoff":
        return replacement_dropoff(
            values,
            primary_k=int(parameters["primary_k"]),
            replacement_k=int(parameters["replacement_k"]),
        )

    raise AssertionError(
        "A supported scalar aggregation family reached no "
        "dispatch branch."
    )


def aggregation_profile_name(
    specification: AggregationSpecificationLike,
) -> str:
    """
    Return a stable human-readable profile identifier.

    The Study 089 specification currently exposes ``specification_id``,
    but the adapter remains compatible with other specification
    implementations.
    """

    specification.validate()

    specification_id = getattr(
        specification,
        "specification_id",
        None,
    )

    if specification_id is not None:
        normalized = str(specification_id).strip()

        if normalized:
            return normalized

    return specification.aggregation_family
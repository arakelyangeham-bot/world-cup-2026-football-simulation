# aggregation_specifications.py

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping


VALID_AGGREGATION_FAMILIES = {
    "arithmetic_mean",
    "top_k_mean",
    "rank_weighted_top_k",
    "star_influence_top_k",
    "power_mean_top_k",
    "softmax_weighted_top_k",
    "ability_power_weighted_mean",
    "replacement_group_mean",
    "replacement_dropoff",
    "distribution_shape",
}


VALID_OUTPUT_TYPES = {
    "primary_strength",
    "depth_strength",
    "depth_dropoff",
    "distribution_diagnostics",
}


@dataclass(frozen=True)
class AggregationSpecification:
    """
    Immutable Study 089B aggregation specification.

    The specification describes how an aggregation function should be
    invoked. It does not contain player populations or benchmark logic.
    """

    specification_id: str
    aggregation_family: str
    display_name: str
    output_type: str
    parameters: Mapping[str, object]
    description: str
    historical_control: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(
                dict(self.parameters)
            ),
        )

    def validate(self) -> None:
        if not self.specification_id.strip():
            raise ValueError(
                "Aggregation specification ID must not be empty."
            )

        if (
            self.aggregation_family
            not in VALID_AGGREGATION_FAMILIES
        ):
            raise ValueError(
                "Unknown aggregation family "
                f"{self.aggregation_family!r} for "
                f"{self.specification_id!r}."
            )

        if not self.display_name.strip():
            raise ValueError(
                f"Aggregation specification "
                f"{self.specification_id!r} must have a "
                "display name."
            )

        if self.output_type not in VALID_OUTPUT_TYPES:
            raise ValueError(
                "Unknown aggregation output type "
                f"{self.output_type!r} for "
                f"{self.specification_id!r}."
            )

        if not self.description.strip():
            raise ValueError(
                f"Aggregation specification "
                f"{self.specification_id!r} must have a "
                "description."
            )

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        family = self.aggregation_family
        parameters = dict(self.parameters)

        expected_parameter_names = {
            "arithmetic_mean": set(),
            "top_k_mean": {"k"},
            "rank_weighted_top_k": {"weights"},
            "star_influence_top_k": {"k", "alpha"},
            "power_mean_top_k": {"k", "power"},
            "softmax_weighted_top_k": {"k", "beta"},
            "ability_power_weighted_mean": {"gamma"},
            "replacement_group_mean": {
                "primary_k",
                "replacement_k",
            },
            "replacement_dropoff": {
                "primary_k",
                "replacement_k",
            },
            "distribution_shape": {"k"},
        }[family]

        observed_parameter_names = set(parameters)

        if observed_parameter_names != expected_parameter_names:
            raise ValueError(
                "Aggregation specification "
                f"{self.specification_id!r} has parameter names "
                f"{sorted(observed_parameter_names)} but family "
                f"{family!r} requires "
                f"{sorted(expected_parameter_names)}."
            )

        if "k" in parameters:
            _validate_positive_integer(
                parameters["k"],
                parameter_name="k",
            )

        if "primary_k" in parameters:
            _validate_positive_integer(
                parameters["primary_k"],
                parameter_name="primary_k",
            )

        if "replacement_k" in parameters:
            _validate_positive_integer(
                parameters["replacement_k"],
                parameter_name="replacement_k",
            )

        if "weights" in parameters:
            _validate_rank_weights(
                parameters["weights"]
            )

        if "alpha" in parameters:
            alpha = _validated_finite_float(
                parameters["alpha"],
                parameter_name="alpha",
            )

            if not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    "alpha must lie within [0, 1]."
                )

        if "power" in parameters:
            power = _validated_finite_float(
                parameters["power"],
                parameter_name="power",
            )

            if power <= 0.0:
                raise ValueError(
                    "power must be greater than zero."
                )

        if "beta" in parameters:
            beta = _validated_finite_float(
                parameters["beta"],
                parameter_name="beta",
            )

            if beta < 0.0:
                raise ValueError(
                    "beta must be non-negative."
                )

        if "gamma" in parameters:
            gamma = _validated_finite_float(
                parameters["gamma"],
                parameter_name="gamma",
            )

            if gamma < 0.0:
                raise ValueError(
                    "gamma must be non-negative."
                )

    def to_record(self) -> dict[str, object]:
        self.validate()

        parameters = dict(self.parameters)

        return {
            "specification_id": self.specification_id,
            "aggregation_family": self.aggregation_family,
            "display_name": self.display_name,
            "output_type": self.output_type,
            "parameterization": _format_parameterization(
                parameters
            ),
            "description": self.description,
            "historical_control": self.historical_control,
        }


def _validate_positive_integer(
    value: object,
    *,
    parameter_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{parameter_name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{parameter_name} must be greater than zero."
        )

    return value


def _validated_finite_float(
    value: object,
    *,
    parameter_name: str,
) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{parameter_name} must be a real number."
        ) from error

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{parameter_name} must be finite."
        )

    return numeric_value


def _validate_rank_weights(
    weights: object,
) -> tuple[float, ...]:
    if isinstance(
        weights,
        (str, bytes),
    ):
        raise TypeError(
            "weights must be a numeric sequence."
        )

    try:
        values = tuple(
            float(value)
            for value in weights  # type: ignore[union-attr]
        )
    except TypeError as error:
        raise TypeError(
            "weights must be a numeric sequence."
        ) from error
    except ValueError as error:
        raise TypeError(
            "weights must contain only real numbers."
        ) from error

    if not values:
        raise ValueError(
            "weights must not be empty."
        )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "weights must contain only finite values."
        )

    if any(
        value < 0.0
        for value in values
    ):
        raise ValueError(
            "weights must be non-negative."
        )

    total = math.fsum(values)

    if total <= 0.0:
        raise ValueError(
            "At least one rank weight must be positive."
        )

    if not math.isclose(
        total,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "weights must sum to one. "
            f"Observed sum: {total!r}."
        )

    return values


def _format_parameterization(
    parameters: Mapping[str, object],
) -> str:
    if not parameters:
        return "none"

    parts: list[str] = []

    for name in sorted(parameters):
        value = parameters[name]

        if name == "weights":
            weights = tuple(
                float(weight)
                for weight in value  # type: ignore[union-attr]
            )

            formatted_value = (
                "["
                + ", ".join(
                    f"{weight:.12g}"
                    for weight in weights
                )
                + "]"
            )
        elif isinstance(value, float):
            formatted_value = f"{value:.12g}"
        else:
            formatted_value = str(value)

        parts.append(
            f"{name}={formatted_value}"
        )

    return "; ".join(parts)


def build_aggregation_specifications(
) -> tuple[AggregationSpecification, ...]:
    specifications = (
        AggregationSpecification(
            specification_id="arithmetic_all",
            aggregation_family="arithmetic_mean",
            display_name="Whole-population arithmetic mean",
            output_type="depth_strength",
            parameters={},
            description=(
                "Arithmetic mean of the complete player "
                "population. Retained as the current depth-style "
                "control."
            ),
        ),
        AggregationSpecification(
            specification_id="top5_arithmetic",
            aggregation_family="top_k_mean",
            display_name="Arithmetic top-five mean",
            output_type="primary_strength",
            parameters={
                "k": 5,
            },
            description=(
                "Arithmetic mean of the five strongest player "
                "projections. Current primary-strength control."
            ),
        ),
        AggregationSpecification(
            specification_id="top5_rank_mild",
            aggregation_family="rank_weighted_top_k",
            display_name="Mild rank-weighted top five",
            output_type="primary_strength",
            parameters={
                "weights": (
                    0.24,
                    0.22,
                    0.20,
                    0.18,
                    0.16,
                ),
            },
            description=(
                "Top-five rank weighting with a mild preference "
                "for stronger contributors."
            ),
        ),
        AggregationSpecification(
            specification_id="top5_rank_moderate",
            aggregation_family="rank_weighted_top_k",
            display_name="Moderate rank-weighted top five",
            output_type="primary_strength",
            parameters={
                "weights": (
                    0.30,
                    0.25,
                    0.20,
                    0.15,
                    0.10,
                ),
            },
            description=(
                "Top-five rank weighting with a moderate "
                "preference for stronger contributors."
            ),
        ),
        AggregationSpecification(
            specification_id="top5_rank_strong",
            aggregation_family="rank_weighted_top_k",
            display_name="Strong rank-weighted top five",
            output_type="primary_strength",
            parameters={
                "weights": (
                    0.40,
                    0.25,
                    0.15,
                    0.12,
                    0.08,
                ),
            },
            description=(
                "Top-five rank weighting with a strong preference "
                "for the highest-rated players."
            ),
        ),
        *(
            AggregationSpecification(
                specification_id=(
                    f"top5_star_alpha_{alpha:.2f}"
                    .replace(".", "_")
                ),
                aggregation_family="star_influence_top_k",
                display_name=(
                    f"Top-five star influence, alpha={alpha:.2f}"
                ),
                output_type="primary_strength",
                parameters={
                    "k": 5,
                    "alpha": alpha,
                },
                description=(
                    "Blend the arithmetic top-five mean with the "
                    "strongest player."
                ),
            )
            for alpha in (
                0.10,
                0.20,
                0.30,
            )
        ),
        *(
            AggregationSpecification(
                specification_id=(
                    f"top5_power_{power:.2f}"
                    .replace(".", "_")
                ),
                aggregation_family="power_mean_top_k",
                display_name=(
                    f"Top-five power mean, p={power:.2f}"
                ),
                output_type="primary_strength",
                parameters={
                    "k": 5,
                    "power": power,
                },
                description=(
                    "Generalized power mean of the five strongest "
                    "player projections."
                ),
            )
            for power in (
                1.25,
                1.50,
                2.00,
            )
        ),
        *(
            AggregationSpecification(
                specification_id=(
                    f"top5_softmax_beta_{beta}"
                ),
                aggregation_family="softmax_weighted_top_k",
                display_name=(
                    f"Top-five softmax weighting, beta={beta}"
                ),
                output_type="primary_strength",
                parameters={
                    "k": 5,
                    "beta": float(beta),
                },
                description=(
                    "Softmax-weighted top-five mean on the "
                    "normalized [0, 1] player scale."
                ),
            )
            for beta in (
                1,
                3,
                5,
            )
        ),
        AggregationSpecification(
            specification_id="ability_power_gamma_2",
            aggregation_family="ability_power_weighted_mean",
            display_name="Historical ability-power weighting",
            output_type="primary_strength",
            parameters={
                "gamma": 2.0,
            },
            description=(
                "Historical Study 011 star-weighted formula, "
                "retained as a compatibility control."
            ),
            historical_control=True,
        ),
        AggregationSpecification(
            specification_id="replacement_mean_5_5",
            aggregation_family="replacement_group_mean",
            display_name="Replacement-group mean, ranks 6-10",
            output_type="depth_strength",
            parameters={
                "primary_k": 5,
                "replacement_k": 5,
            },
            description=(
                "Arithmetic mean of the five players immediately "
                "below the primary top-five group."
            ),
        ),
        AggregationSpecification(
            specification_id="replacement_dropoff_5_5",
            aggregation_family="replacement_dropoff",
            display_name="Primary-to-replacement drop-off",
            output_type="depth_dropoff",
            parameters={
                "primary_k": 5,
                "replacement_k": 5,
            },
            description=(
                "Difference between the top-five mean and the "
                "mean of ranks six through ten."
            ),
        ),
        AggregationSpecification(
            specification_id="distribution_shape_top5",
            aggregation_family="distribution_shape",
            display_name="Top-five distribution diagnostics",
            output_type="distribution_diagnostics",
            parameters={
                "k": 5,
            },
            description=(
                "Top-five mean, extrema, range, standard "
                "deviation, star gap, and concentration."
            ),
        ),
    )

    validate_aggregation_specifications(
        specifications
    )

    return specifications


def validate_aggregation_specifications(
    specifications: tuple[
        AggregationSpecification,
        ...,
    ],
) -> None:
    if not specifications:
        raise ValueError(
            "Aggregation specification registry must not be empty."
        )

    specification_ids: set[str] = set()

    for specification in specifications:
        specification.validate()

        if (
            specification.specification_id
            in specification_ids
        ):
            raise ValueError(
                "Aggregation specification registry contains "
                "duplicate specification ID "
                f"{specification.specification_id!r}."
            )

        specification_ids.add(
            specification.specification_id
        )


def aggregation_specification_records(
    specifications: tuple[
        AggregationSpecification,
        ...,
    ] | None = None,
) -> list[dict[str, object]]:
    selected = (
        specifications
        if specifications is not None
        else build_aggregation_specifications()
    )

    validate_aggregation_specifications(
        selected
    )

    return [
        specification.to_record()
        for specification in selected
    ]
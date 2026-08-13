# aggregation_evaluator.py

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections.abc import Sequence
from typing import Literal

from research.player_intelligence.aggregation_functions import (
    ability_power_weighted_mean,
    arithmetic_mean,
    distribution_shape,
    power_mean_top_k,
    rank_weighted_top_k,
    replacement_dropoff,
    replacement_group_mean,
    softmax_weighted_top_k,
    star_influence_top_k,
    top_k_mean,
)

from research.studies.study_089_aggregation_mathematics.aggregation_specifications import (
    AggregationSpecification,
)


EvaluationStatus = Literal[
    "evaluated",
    "not_applicable",
    "failed",
]


@dataclass(frozen=True)
class ScalarAggregationEvaluation:
    """
    Result of applying one scalar aggregation specification to one
    player population.
    """

    specification_id: str
    aggregation_family: str
    output_type: str

    status: EvaluationStatus
    value: float | None

    population_size: int
    minimum_population_size: int

    error_type: str | None = None
    error_message: str | None = None

    def validate(self) -> None:
        if not self.specification_id.strip():
            raise ValueError(
                "Scalar evaluation specification ID must not be empty."
            )

        if not self.aggregation_family.strip():
            raise ValueError(
                "Scalar evaluation aggregation family must not be empty."
            )

        if not self.output_type.strip():
            raise ValueError(
                "Scalar evaluation output type must not be empty."
            )

        if self.status not in {
            "evaluated",
            "not_applicable",
            "failed",
        }:
            raise ValueError(
                f"Unknown scalar evaluation status {self.status!r}."
            )

        if self.population_size < 0:
            raise ValueError(
                "population_size must not be negative."
            )

        if self.minimum_population_size <= 0:
            raise ValueError(
                "minimum_population_size must be positive."
            )

        if self.status == "evaluated":
            if self.value is None:
                raise ValueError(
                    "Evaluated scalar result must contain a value."
                )

            if not math.isfinite(self.value):
                raise ValueError(
                    "Evaluated scalar result must be finite."
                )

            if (
                self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError(
                    "Evaluated scalar result must not contain "
                    "error information."
                )

        else:
            if self.value is not None:
                raise ValueError(
                    "Unevaluated scalar result must not contain "
                    "a numeric value."
                )

            if not self.error_type:
                raise ValueError(
                    "Unevaluated scalar result must identify "
                    "an error type."
                )

            if not self.error_message:
                raise ValueError(
                    "Unevaluated scalar result must identify "
                    "an error message."
                )

    def to_record(self) -> dict[str, object]:
        self.validate()

        return asdict(self)


@dataclass(frozen=True)
class DistributionAggregationEvaluation:
    """
    Result of applying one distribution-diagnostic specification to one
    player population.
    """

    specification_id: str
    aggregation_family: str
    output_type: str

    status: EvaluationStatus

    population_size: int
    minimum_population_size: int

    mean: float | None = None
    maximum: float | None = None
    minimum: float | None = None
    value_range: float | None = None
    standard_deviation: float | None = None
    star_gap: float | None = None
    concentration: float | None = None
    player_count: int | None = None

    error_type: str | None = None
    error_message: str | None = None

    def validate(self) -> None:
        if not self.specification_id.strip():
            raise ValueError(
                "Distribution evaluation specification ID "
                "must not be empty."
            )

        if self.status not in {
            "evaluated",
            "not_applicable",
            "failed",
        }:
            raise ValueError(
                "Unknown distribution evaluation status "
                f"{self.status!r}."
            )

        diagnostic_values = (
            self.mean,
            self.maximum,
            self.minimum,
            self.value_range,
            self.standard_deviation,
            self.star_gap,
            self.concentration,
        )

        if self.status == "evaluated":
            if any(
                value is None
                for value in diagnostic_values
            ):
                raise ValueError(
                    "Evaluated distribution result must contain "
                    "all diagnostic values."
                )

            if not all(
                math.isfinite(float(value))
                for value in diagnostic_values
            ):
                raise ValueError(
                    "Distribution diagnostics must be finite."
                )

            if self.player_count is None:
                raise ValueError(
                    "Evaluated distribution result must contain "
                    "player_count."
                )

            if self.player_count <= 0:
                raise ValueError(
                    "Distribution player_count must be positive."
                )

            if (
                self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError(
                    "Evaluated distribution result must not "
                    "contain error information."
                )

        else:
            if any(
                value is not None
                for value in diagnostic_values
            ):
                raise ValueError(
                    "Unevaluated distribution result must not "
                    "contain diagnostic values."
                )

            if self.player_count is not None:
                raise ValueError(
                    "Unevaluated distribution result must not "
                    "contain player_count."
                )

            if not self.error_type:
                raise ValueError(
                    "Unevaluated distribution result must "
                    "identify an error type."
                )

            if not self.error_message:
                raise ValueError(
                    "Unevaluated distribution result must "
                    "identify an error message."
                )

    def to_record(self) -> dict[str, object]:
        self.validate()

        return asdict(self)


AggregationEvaluation = (
    ScalarAggregationEvaluation
    | DistributionAggregationEvaluation
)


def minimum_population_size(
    specification: AggregationSpecification,
) -> int:
    """
    Return the minimum population size required by one specification.
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
        "distribution_shape",
    }:
        return int(parameters["k"])

    if family == "rank_weighted_top_k":
        return len(
            tuple(parameters["weights"])
        )

    if family in {
        "replacement_group_mean",
        "replacement_dropoff",
    }:
        return (
            int(parameters["primary_k"])
            + int(parameters["replacement_k"])
        )

    raise KeyError(
        "No minimum-population rule exists for aggregation "
        f"family {family!r}."
    )


def is_distribution_specification(
    specification: AggregationSpecification,
) -> bool:
    specification.validate()

    return (
        specification.aggregation_family
        == "distribution_shape"
    )


def _evaluate_scalar(
    values: Sequence[float],
    *,
    specification: AggregationSpecification,
) -> float:
    family = specification.aggregation_family
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
                for weight in parameters["weights"]
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

    if family == "distribution_shape":
        raise TypeError(
            "distribution_shape is not a scalar aggregation family."
        )

    raise KeyError(
        "No scalar evaluator exists for aggregation family "
        f"{family!r}."
    )


def evaluate_aggregation(
    values: Sequence[float],
    *,
    specification: AggregationSpecification,
) -> AggregationEvaluation:
    """
    Apply one aggregation specification to one player population.

    Population-size incompatibility is returned as ``not_applicable``.

    Unexpected runtime errors are returned as ``failed`` so the study
    runner can preserve diagnostics rather than terminating the entire
    benchmark.
    """

    specification.validate()

    required_size = minimum_population_size(
        specification
    )
    observed_size = len(values)

    if observed_size < required_size:
        message = (
            "Population is too small for aggregation "
            f"specification {specification.specification_id!r}: "
            f"required {required_size}, received {observed_size}."
        )

        if is_distribution_specification(
            specification
        ):
            result = DistributionAggregationEvaluation(
                specification_id=(
                    specification.specification_id
                ),
                aggregation_family=(
                    specification.aggregation_family
                ),
                output_type=specification.output_type,
                status="not_applicable",
                population_size=observed_size,
                minimum_population_size=required_size,
                error_type="InsufficientPopulation",
                error_message=message,
            )
        else:
            result = ScalarAggregationEvaluation(
                specification_id=(
                    specification.specification_id
                ),
                aggregation_family=(
                    specification.aggregation_family
                ),
                output_type=specification.output_type,
                status="not_applicable",
                value=None,
                population_size=observed_size,
                minimum_population_size=required_size,
                error_type="InsufficientPopulation",
                error_message=message,
            )

        result.validate()

        return result

    try:
        if is_distribution_specification(
            specification
        ):
            shape = distribution_shape(
                values,
                k=int(
                    specification.parameters["k"]
                ),
            )

            result = DistributionAggregationEvaluation(
                specification_id=(
                    specification.specification_id
                ),
                aggregation_family=(
                    specification.aggregation_family
                ),
                output_type=specification.output_type,
                status="evaluated",
                population_size=observed_size,
                minimum_population_size=required_size,
                mean=shape.mean,
                maximum=shape.maximum,
                minimum=shape.minimum,
                value_range=shape.value_range,
                standard_deviation=shape.standard_deviation,
                star_gap=shape.star_gap,
                concentration=shape.concentration,
                player_count=shape.player_count,
            )

            result.validate()

            return result

        value = _evaluate_scalar(
            values,
            specification=specification,
        )

        result = ScalarAggregationEvaluation(
            specification_id=(
                specification.specification_id
            ),
            aggregation_family=(
                specification.aggregation_family
            ),
            output_type=specification.output_type,
            status="evaluated",
            value=float(value),
            population_size=observed_size,
            minimum_population_size=required_size,
        )

        result.validate()

        return result

    except Exception as error:
        if is_distribution_specification(
            specification
        ):
            result = DistributionAggregationEvaluation(
                specification_id=(
                    specification.specification_id
                ),
                aggregation_family=(
                    specification.aggregation_family
                ),
                output_type=specification.output_type,
                status="failed",
                population_size=observed_size,
                minimum_population_size=required_size,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        else:
            result = ScalarAggregationEvaluation(
                specification_id=(
                    specification.specification_id
                ),
                aggregation_family=(
                    specification.aggregation_family
                ),
                output_type=specification.output_type,
                status="failed",
                value=None,
                population_size=observed_size,
                minimum_population_size=required_size,
                error_type=type(error).__name__,
                error_message=str(error),
            )

        result.validate()

        return result


def evaluation_to_record(
    evaluation: AggregationEvaluation,
) -> dict[str, object]:
    return evaluation.to_record()
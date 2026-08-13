# axiom_evaluator.py

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

from research.studies.study_089_aggregation_mathematics.aggregation_evaluator import (
    ScalarAggregationEvaluation,
)

from research.studies.study_089_aggregation_mathematics.synthetic_scenario_registry import (
    ABSOLUTE_TOLERANCE,
    RELATIVE_TOLERANCE,
    SyntheticScenario,
)


AxiomStatus = Literal[
    "pass",
    "fail",
    "not_applicable",
]


SUPPORTED_AXIOM_PROPERTIES = {
    "determinism",
    "permutation_invariance",
    "monotonicity",
    "continuity",
    "identity",
    "boundedness",
}


@dataclass(frozen=True)
class AxiomEvaluation:
    """
    Mathematical verdict for one scalar aggregation specification
    under one synthetic axiom scenario.
    """

    scenario_id: str
    evaluated_property: str
    specification_id: str
    aggregation_family: str
    output_type: str

    status: AxiomStatus
    axiom_pass: bool | None

    criterion: str

    value_a: float | None
    value_b: float | None

    observed_metric: float | None
    allowed_threshold: float | None

    error_type: str | None = None
    error_message: str | None = None
    notes: str = ""

    def validate(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError(
                "Axiom evaluation scenario ID must not be empty."
            )

        if (
            self.evaluated_property
            not in SUPPORTED_AXIOM_PROPERTIES
        ):
            raise ValueError(
                "Unsupported axiom property "
                f"{self.evaluated_property!r}."
            )

        if not self.specification_id.strip():
            raise ValueError(
                "Axiom evaluation specification ID must not be empty."
            )

        if self.status not in {
            "pass",
            "fail",
            "not_applicable",
        }:
            raise ValueError(
                f"Unknown axiom status {self.status!r}."
            )

        if not self.criterion.strip():
            raise ValueError(
                "Axiom evaluation criterion must not be empty."
            )

        if self.status == "pass":
            if self.axiom_pass is not True:
                raise ValueError(
                    "Passing axiom evaluation must set "
                    "axiom_pass=True."
                )

            if (
                self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError(
                    "Passing axiom evaluation must not contain "
                    "error information."
                )

        elif self.status == "fail":
            if self.axiom_pass is not False:
                raise ValueError(
                    "Failing axiom evaluation must set "
                    "axiom_pass=False."
                )

            if not self.error_type:
                raise ValueError(
                    "Failing axiom evaluation must identify "
                    "an error type."
                )

            if not self.error_message:
                raise ValueError(
                    "Failing axiom evaluation must identify "
                    "an error message."
                )

        else:
            if self.axiom_pass is not None:
                raise ValueError(
                    "Not-applicable axiom evaluation must set "
                    "axiom_pass=None."
                )

            if not self.error_type:
                raise ValueError(
                    "Not-applicable axiom evaluation must identify "
                    "an error type."
                )

            if not self.error_message:
                raise ValueError(
                    "Not-applicable axiom evaluation must identify "
                    "an error message."
                )

        for field_name, value in {
            "value_a": self.value_a,
            "value_b": self.value_b,
            "observed_metric": self.observed_metric,
            "allowed_threshold": self.allowed_threshold,
        }.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    f"{field_name} must be finite when present."
                )

    def to_record(self) -> dict[str, object]:
        self.validate()

        return asdict(self)


def _close(
    value_a: float,
    value_b: float,
) -> bool:
    return math.isclose(
        value_a,
        value_b,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=ABSOLUTE_TOLERANCE,
    )


def _not_applicable_result(
    *,
    scenario: SyntheticScenario,
    evaluation_a: ScalarAggregationEvaluation,
    evaluation_b: ScalarAggregationEvaluation | None,
) -> AxiomEvaluation:
    error_messages = [
        message
        for message in (
            evaluation_a.error_message,
            (
                evaluation_b.error_message
                if evaluation_b is not None
                else None
            ),
        )
        if message
    ]

    result = AxiomEvaluation(
        scenario_id=scenario.scenario_id,
        evaluated_property=scenario.evaluated_property,
        specification_id=evaluation_a.specification_id,
        aggregation_family=evaluation_a.aggregation_family,
        output_type=evaluation_a.output_type,
        status="not_applicable",
        axiom_pass=None,
        criterion=(
            "The aggregation specification must be applicable "
            "to every population required by this axiom scenario."
        ),
        value_a=evaluation_a.value,
        value_b=(
            evaluation_b.value
            if evaluation_b is not None
            else None
        ),
        observed_metric=None,
        allowed_threshold=None,
        error_type="InsufficientPopulation",
        error_message=(
            " | ".join(error_messages)
            if error_messages
            else (
                "One or more scenario populations do not satisfy "
                "the aggregation minimum-size requirement."
            )
        ),
        notes=(
            "Not-applicable rows are excluded from mathematical "
            "failure counts."
        ),
    )

    result.validate()

    return result


def _failed_runtime_result(
    *,
    scenario: SyntheticScenario,
    evaluation_a: ScalarAggregationEvaluation,
    evaluation_b: ScalarAggregationEvaluation | None,
) -> AxiomEvaluation:
    error_types = [
        error_type
        for error_type in (
            evaluation_a.error_type,
            (
                evaluation_b.error_type
                if evaluation_b is not None
                else None
            ),
        )
        if error_type
    ]

    error_messages = [
        message
        for message in (
            evaluation_a.error_message,
            (
                evaluation_b.error_message
                if evaluation_b is not None
                else None
            ),
        )
        if message
    ]

    result = AxiomEvaluation(
        scenario_id=scenario.scenario_id,
        evaluated_property=scenario.evaluated_property,
        specification_id=evaluation_a.specification_id,
        aggregation_family=evaluation_a.aggregation_family,
        output_type=evaluation_a.output_type,
        status="fail",
        axiom_pass=False,
        criterion=(
            "A mandatory axiom evaluation must complete without "
            "a runtime aggregation failure."
        ),
        value_a=evaluation_a.value,
        value_b=(
            evaluation_b.value
            if evaluation_b is not None
            else None
        ),
        observed_metric=None,
        allowed_threshold=None,
        error_type=(
            " | ".join(error_types)
            if error_types
            else "AggregationRuntimeFailure"
        ),
        error_message=(
            " | ".join(error_messages)
            if error_messages
            else "Aggregation evaluation failed."
        ),
    )

    result.validate()

    return result


def _extract_note_float(
    notes: str,
    *,
    key: str,
) -> float:
    """
    Extract a semicolon-delimited numeric note.

    Example:

        input_delta=-0.0001; zero_based_rank_index=4
    """

    entries = [
        entry.strip()
        for entry in notes.split(";")
        if entry.strip()
    ]

    for entry in entries:
        name, separator, raw_value = entry.partition("=")

        if separator and name.strip() == key:
            try:
                value = float(raw_value.strip())
            except ValueError as error:
                raise ValueError(
                    f"Scenario note {key!r} must be numeric."
                ) from error

            if not math.isfinite(value):
                raise ValueError(
                    f"Scenario note {key!r} must be finite."
                )

            return value

    raise KeyError(
        f"Scenario notes do not define required key {key!r}."
    )


def _pass_result(
    *,
    scenario: SyntheticScenario,
    evaluation_a: ScalarAggregationEvaluation,
    evaluation_b: ScalarAggregationEvaluation | None,
    criterion: str,
    observed_metric: float | None,
    allowed_threshold: float | None,
    notes: str = "",
) -> AxiomEvaluation:
    result = AxiomEvaluation(
        scenario_id=scenario.scenario_id,
        evaluated_property=scenario.evaluated_property,
        specification_id=evaluation_a.specification_id,
        aggregation_family=evaluation_a.aggregation_family,
        output_type=evaluation_a.output_type,
        status="pass",
        axiom_pass=True,
        criterion=criterion,
        value_a=evaluation_a.value,
        value_b=(
            evaluation_b.value
            if evaluation_b is not None
            else None
        ),
        observed_metric=observed_metric,
        allowed_threshold=allowed_threshold,
        notes=notes,
    )

    result.validate()

    return result


def _fail_result(
    *,
    scenario: SyntheticScenario,
    evaluation_a: ScalarAggregationEvaluation,
    evaluation_b: ScalarAggregationEvaluation | None,
    criterion: str,
    observed_metric: float | None,
    allowed_threshold: float | None,
    error_message: str,
    notes: str = "",
) -> AxiomEvaluation:
    result = AxiomEvaluation(
        scenario_id=scenario.scenario_id,
        evaluated_property=scenario.evaluated_property,
        specification_id=evaluation_a.specification_id,
        aggregation_family=evaluation_a.aggregation_family,
        output_type=evaluation_a.output_type,
        status="fail",
        axiom_pass=False,
        criterion=criterion,
        value_a=evaluation_a.value,
        value_b=(
            evaluation_b.value
            if evaluation_b is not None
            else None
        ),
        observed_metric=observed_metric,
        allowed_threshold=allowed_threshold,
        error_type="AxiomViolation",
        error_message=error_message,
        notes=notes,
    )

    result.validate()

    return result


def evaluate_scalar_axiom(
    *,
    scenario: SyntheticScenario,
    evaluation_a: ScalarAggregationEvaluation,
    evaluation_b: ScalarAggregationEvaluation | None,
) -> AxiomEvaluation:
    """
    Evaluate one mandatory scalar axiom according to its own
    mathematical contract.
    """

    scenario.validate()
    evaluation_a.validate()

    if evaluation_b is not None:
        evaluation_b.validate()

    if scenario.scenario_family != "axiom":
        raise ValueError(
            "Axiom evaluator received a non-axiom scenario "
            f"{scenario.scenario_id!r}."
        )

    if (
        scenario.evaluated_property
        not in SUPPORTED_AXIOM_PROPERTIES
    ):
        raise ValueError(
            "No scalar axiom rule exists for property "
            f"{scenario.evaluated_property!r}."
        )

    statuses = {
        evaluation_a.status,
        *(
            {evaluation_b.status}
            if evaluation_b is not None
            else set()
        ),
    }

    if "failed" in statuses:
        return _failed_runtime_result(
            scenario=scenario,
            evaluation_a=evaluation_a,
            evaluation_b=evaluation_b,
        )

    if "not_applicable" in statuses:
        return _not_applicable_result(
            scenario=scenario,
            evaluation_a=evaluation_a,
            evaluation_b=evaluation_b,
        )

    if evaluation_a.value is None:
        raise AssertionError(
            "Evaluated population A is missing a scalar value."
        )

    value_a = float(evaluation_a.value)

    value_b = (
        float(evaluation_b.value)
        if (
            evaluation_b is not None
            and evaluation_b.value is not None
        )
        else None
    )

    property_name = scenario.evaluated_property

    # -------------------------------------------------------------
    # Determinism and permutation invariance
    # -------------------------------------------------------------

    if property_name in {
        "determinism",
        "permutation_invariance",
    }:
        if value_b is None:
            raise ValueError(
                f"{property_name} requires population B."
            )

        difference = abs(value_b - value_a)

        criterion = (
            "|value_b - value_a| must be within the frozen "
            "floating-point tolerance."
        )

        if _close(value_a, value_b):
            return _pass_result(
                scenario=scenario,
                evaluation_a=evaluation_a,
                evaluation_b=evaluation_b,
                criterion=criterion,
                observed_metric=difference,
                allowed_threshold=ABSOLUTE_TOLERANCE,
            )

        return _fail_result(
            scenario=scenario,
            evaluation_a=evaluation_a,
            evaluation_b=evaluation_b,
            criterion=criterion,
            observed_metric=difference,
            allowed_threshold=ABSOLUTE_TOLERANCE,
            error_message=(
                f"{property_name} failed: values differ by "
                f"{difference!r}."
            ),
        )

    # -------------------------------------------------------------
    # Monotonicity
    # -------------------------------------------------------------

    if property_name == "monotonicity":
        if value_b is None:
            raise ValueError(
                "Monotonicity requires population B."
            )

        output_delta = value_b - value_a

        criterion = (
            "Improving one player must not reduce the aggregated "
            "team value."
        )

        passes = (
            output_delta >= -ABSOLUTE_TOLERANCE
            or _close(value_a, value_b)
        )

        if passes:
            return _pass_result(
                scenario=scenario,
                evaluation_a=evaluation_a,
                evaluation_b=evaluation_b,
                criterion=criterion,
                observed_metric=output_delta,
                allowed_threshold=-ABSOLUTE_TOLERANCE,
            )

        return _fail_result(
            scenario=scenario,
            evaluation_a=evaluation_a,
            evaluation_b=evaluation_b,
            criterion=criterion,
            observed_metric=output_delta,
            allowed_threshold=-ABSOLUTE_TOLERANCE,
            error_message=(
                "Monotonicity failed: player improvement reduced "
                f"the aggregated value by {abs(output_delta)!r}."
            ),
        )

    # -------------------------------------------------------------
    # Continuity
    # -------------------------------------------------------------

    if property_name == "continuity":
        if value_b is None:
            raise ValueError(
                "Continuity requires population B."
            )

        input_delta = _extract_note_float(
            scenario.notes,
            key="input_delta",
        )

        output_delta = value_b - value_a
        absolute_input_delta = abs(input_delta)
        absolute_output_delta = abs(output_delta)

        criterion = (
            "The absolute output change must not exceed the "
            "absolute one-player input perturbation, apart from "
            "floating-point tolerance."
        )

        allowed = (
            absolute_input_delta
            + ABSOLUTE_TOLERANCE
        )

        if absolute_output_delta <= allowed:
            local_sensitivity = (
                absolute_output_delta
                / absolute_input_delta
                if absolute_input_delta > 0.0
                else 0.0
            )

            return _pass_result(
                scenario=scenario,
                evaluation_a=evaluation_a,
                evaluation_b=evaluation_b,
                criterion=criterion,
                observed_metric=absolute_output_delta,
                allowed_threshold=allowed,
                notes=(
                    f"input_delta={input_delta}; "
                    f"local_sensitivity={local_sensitivity}"
                ),
            )

        return _fail_result(
            scenario=scenario,
            evaluation_a=evaluation_a,
            evaluation_b=evaluation_b,
            criterion=criterion,
            observed_metric=absolute_output_delta,
            allowed_threshold=allowed,
            error_message=(
                "Continuity failed: output changed by "
                f"{absolute_output_delta!r} after an input "
                f"perturbation of {absolute_input_delta!r}."
            ),
            notes=f"input_delta={input_delta}",
        )

    # -------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------

    if property_name == "identity":
        common_values = set(
            scenario.population_a.values
        )

        if len(common_values) != 1:
            raise ValueError(
                "Identity scenario population must contain one "
                "common player value."
            )

        expected_value = float(
            next(iter(common_values))
        )

        difference = abs(
            value_a - expected_value
        )

        criterion = (
            "An aggregation of a constant population must equal "
            "the common player value."
        )

        if _close(value_a, expected_value):
            return _pass_result(
                scenario=scenario,
                evaluation_a=evaluation_a,
                evaluation_b=evaluation_b,
                criterion=criterion,
                observed_metric=difference,
                allowed_threshold=ABSOLUTE_TOLERANCE,
                notes=(
                    f"expected_common_value={expected_value}"
                ),
            )

        return _fail_result(
            scenario=scenario,
            evaluation_a=evaluation_a,
            evaluation_b=evaluation_b,
            criterion=criterion,
            observed_metric=difference,
            allowed_threshold=ABSOLUTE_TOLERANCE,
            error_message=(
                "Identity failed: aggregation returned "
                f"{value_a!r}, expected {expected_value!r}."
            ),
        )

    # -------------------------------------------------------------
    # Boundedness
    # -------------------------------------------------------------

    if property_name == "boundedness":
        population_minimum = min(
            scenario.population_a.values
        )
        population_maximum = max(
            scenario.population_a.values
        )

        lower_violation = max(
            0.0,
            population_minimum - value_a,
        )

        upper_violation = max(
            0.0,
            value_a - population_maximum,
        )

        maximum_violation = max(
            lower_violation,
            upper_violation,
        )

        criterion = (
            "Scalar strength must remain within the minimum and "
            "maximum player values."
        )

        passes = (
            value_a
            >= population_minimum
            - ABSOLUTE_TOLERANCE
            and value_a
            <= population_maximum
            + ABSOLUTE_TOLERANCE
        )

        if passes:
            return _pass_result(
                scenario=scenario,
                evaluation_a=evaluation_a,
                evaluation_b=evaluation_b,
                criterion=criterion,
                observed_metric=maximum_violation,
                allowed_threshold=ABSOLUTE_TOLERANCE,
                notes=(
                    f"population_minimum={population_minimum}; "
                    f"population_maximum={population_maximum}"
                ),
            )

        return _fail_result(
            scenario=scenario,
            evaluation_a=evaluation_a,
            evaluation_b=evaluation_b,
            criterion=criterion,
            observed_metric=maximum_violation,
            allowed_threshold=ABSOLUTE_TOLERANCE,
            error_message=(
                "Boundedness failed: aggregation value "
                f"{value_a!r} lies outside "
                f"[{population_minimum!r}, "
                f"{population_maximum!r}]."
            ),
        )

    raise AssertionError(
        "A supported axiom property reached no evaluation rule."
    )


def axiom_evaluation_to_record(
    evaluation: AxiomEvaluation,
) -> dict[str, object]:
    return evaluation.to_record()
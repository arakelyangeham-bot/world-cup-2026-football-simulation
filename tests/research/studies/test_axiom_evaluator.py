# test_axiom_evaluator.py

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research.studies.study_089_aggregation_mathematics.aggregation_evaluator import (
    ScalarAggregationEvaluation,
    evaluate_aggregation,
)

from research.studies.study_089_aggregation_mathematics.aggregation_specifications import (
    AggregationSpecification,
    build_aggregation_specifications,
)

from research.studies.study_089_aggregation_mathematics.axiom_evaluator import (
    AxiomEvaluation,
    axiom_evaluation_to_record,
    evaluate_scalar_axiom,
)

from research.studies.study_089_aggregation_mathematics.synthetic_scenario_registry import (
    SyntheticPopulation,
    SyntheticScenario,
    build_scenario_registry,
)


def _specification_by_id(
    specification_id: str,
) -> AggregationSpecification:
    return next(
        specification
        for specification in build_aggregation_specifications()
        if specification.specification_id == specification_id
    )


def _scenario_by_id(
    scenario_id: str,
) -> SyntheticScenario:
    return next(
        scenario
        for scenario in build_scenario_registry()
        if scenario.scenario_id == scenario_id
    )


def _scalar_evaluation(
    *,
    specification_id: str = "test_specification",
    aggregation_family: str = "top_k_mean",
    output_type: str = "primary_strength",
    status: str = "evaluated",
    value: float | None = 0.80,
    population_size: int = 5,
    minimum_population_size: int = 5,
    error_type: str | None = None,
    error_message: str | None = None,
) -> ScalarAggregationEvaluation:
    return ScalarAggregationEvaluation(
        specification_id=specification_id,
        aggregation_family=aggregation_family,
        output_type=output_type,
        status=status,  # type: ignore[arg-type]
        value=value,
        population_size=population_size,
        minimum_population_size=minimum_population_size,
        error_type=error_type,
        error_message=error_message,
    )


def _simple_axiom_scenario(
    *,
    scenario_id: str = "AX-TEST",
    evaluated_property: str = "determinism",
    values_a: tuple[float, ...] = (0.90, 0.80, 0.70),
    values_b: tuple[float, ...] | None = (0.90, 0.80, 0.70),
    notes: str = "",
) -> SyntheticScenario:
    population_a = SyntheticPopulation(
        population_id=f"{scenario_id}-a",
        values=values_a,
        description="Population A.",
    )

    population_b = (
        SyntheticPopulation(
            population_id=f"{scenario_id}-b",
            values=values_b,
            description="Population B.",
        )
        if values_b is not None
        else None
    )

    return SyntheticScenario(
        scenario_id=scenario_id,
        scenario_family="axiom",
        name="Test axiom",
        description="Test axiom scenario.",
        population_a=population_a,
        population_b=population_b,
        evaluated_property=evaluated_property,
        expected_direction="descriptive",
        binary_pass_expected=True,
        notes=notes,
    )


# ---------------------------------------------------------------------
# AxiomEvaluation validation
# ---------------------------------------------------------------------


def test_axiom_evaluation_is_immutable() -> None:
    evaluation = AxiomEvaluation(
        scenario_id="AX-001",
        evaluated_property="determinism",
        specification_id="top5_arithmetic",
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        status="pass",
        axiom_pass=True,
        criterion="Values must match.",
        value_a=0.80,
        value_b=0.80,
        observed_metric=0.0,
        allowed_threshold=1e-12,
    )

    with pytest.raises(FrozenInstanceError):
        evaluation.status = "fail"  # type: ignore[misc]


def test_valid_pass_evaluation_validates() -> None:
    evaluation = AxiomEvaluation(
        scenario_id="AX-001",
        evaluated_property="determinism",
        specification_id="top5_arithmetic",
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        status="pass",
        axiom_pass=True,
        criterion="Values must match.",
        value_a=0.80,
        value_b=0.80,
        observed_metric=0.0,
        allowed_threshold=1e-12,
    )

    evaluation.validate()


def test_pass_requires_axiom_pass_true() -> None:
    evaluation = AxiomEvaluation(
        scenario_id="AX-001",
        evaluated_property="determinism",
        specification_id="top5_arithmetic",
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        status="pass",
        axiom_pass=False,
        criterion="Values must match.",
        value_a=0.80,
        value_b=0.80,
        observed_metric=0.0,
        allowed_threshold=1e-12,
    )

    with pytest.raises(
        ValueError,
        match="axiom_pass=True",
    ):
        evaluation.validate()


def test_fail_requires_axiom_pass_false() -> None:
    evaluation = AxiomEvaluation(
        scenario_id="AX-001",
        evaluated_property="determinism",
        specification_id="top5_arithmetic",
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        status="fail",
        axiom_pass=True,
        criterion="Values must match.",
        value_a=0.80,
        value_b=0.79,
        observed_metric=0.01,
        allowed_threshold=1e-12,
        error_type="AxiomViolation",
        error_message="Values differ.",
    )

    with pytest.raises(
        ValueError,
        match="axiom_pass=False",
    ):
        evaluation.validate()


def test_fail_requires_error_information() -> None:
    evaluation = AxiomEvaluation(
        scenario_id="AX-001",
        evaluated_property="determinism",
        specification_id="top5_arithmetic",
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        status="fail",
        axiom_pass=False,
        criterion="Values must match.",
        value_a=0.80,
        value_b=0.79,
        observed_metric=0.01,
        allowed_threshold=1e-12,
    )

    with pytest.raises(
        ValueError,
        match="error type",
    ):
        evaluation.validate()


def test_not_applicable_requires_axiom_pass_none() -> None:
    evaluation = AxiomEvaluation(
        scenario_id="AX-001",
        evaluated_property="determinism",
        specification_id="replacement_mean_5_5",
        aggregation_family="replacement_group_mean",
        output_type="depth_strength",
        status="not_applicable",
        axiom_pass=False,
        criterion="Population must be large enough.",
        value_a=None,
        value_b=None,
        observed_metric=None,
        allowed_threshold=None,
        error_type="InsufficientPopulation",
        error_message="Too few players.",
    )

    with pytest.raises(
        ValueError,
        match="axiom_pass=None",
    ):
        evaluation.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "value_a",
        "value_b",
        "observed_metric",
        "allowed_threshold",
    ],
)
def test_axiom_evaluation_rejects_non_finite_metrics(
    field_name: str,
) -> None:
    kwargs = {
        "scenario_id": "AX-001",
        "evaluated_property": "determinism",
        "specification_id": "top5_arithmetic",
        "aggregation_family": "top_k_mean",
        "output_type": "primary_strength",
        "status": "pass",
        "axiom_pass": True,
        "criterion": "Values must match.",
        "value_a": 0.80,
        "value_b": 0.80,
        "observed_metric": 0.0,
        "allowed_threshold": 1e-12,
    }

    kwargs[field_name] = float("nan")

    evaluation = AxiomEvaluation(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        evaluation.validate()


def test_axiom_to_record_is_deterministic() -> None:
    evaluation = AxiomEvaluation(
        scenario_id="AX-001",
        evaluated_property="determinism",
        specification_id="top5_arithmetic",
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        status="pass",
        axiom_pass=True,
        criterion="Values must match.",
        value_a=0.80,
        value_b=0.80,
        observed_metric=0.0,
        allowed_threshold=1e-12,
    )

    assert evaluation.to_record() == evaluation.to_record()
    assert axiom_evaluation_to_record(
        evaluation
    ) == evaluation.to_record()


# ---------------------------------------------------------------------
# Determinism and permutation invariance
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_id",
    [
        "AX-001",
        "AX-002",
    ],
)
def test_equality_axioms_pass_for_canonical_top5(
    scenario_id: str,
) -> None:
    scenario = _scenario_by_id(
        scenario_id
    )

    specification = _specification_by_id(
        "top5_arithmetic"
    )

    evaluation_a = evaluate_aggregation(
        scenario.population_a.values,
        specification=specification,
    )

    assert isinstance(
        evaluation_a,
        ScalarAggregationEvaluation,
    )

    assert scenario.population_b is not None

    evaluation_b = evaluate_aggregation(
        scenario.population_b.values,
        specification=specification,
    )

    assert isinstance(
        evaluation_b,
        ScalarAggregationEvaluation,
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=evaluation_a,
        evaluation_b=evaluation_b,
    )

    assert result.status == "pass"
    assert result.axiom_pass is True
    assert result.observed_metric == pytest.approx(0.0)


def test_determinism_fails_when_outputs_differ() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="determinism",
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=_scalar_evaluation(
            value=0.80,
        ),
        evaluation_b=_scalar_evaluation(
            value=0.79,
        ),
    )

    assert result.status == "fail"
    assert result.axiom_pass is False
    assert result.error_type == "AxiomViolation"


# ---------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_id",
    [
        "AX-003-R1",
        "AX-003-R2",
        "AX-003-R3",
        "AX-003-R4",
        "AX-003-R5",
    ],
)
def test_canonical_monotonicity_scenarios_pass(
    scenario_id: str,
) -> None:
    scenario = _scenario_by_id(
        scenario_id
    )

    specification = _specification_by_id(
        "top5_arithmetic"
    )

    evaluation_a = evaluate_aggregation(
        scenario.population_a.values,
        specification=specification,
    )

    assert scenario.population_b is not None

    evaluation_b = evaluate_aggregation(
        scenario.population_b.values,
        specification=specification,
    )

    assert isinstance(
        evaluation_a,
        ScalarAggregationEvaluation,
    )

    assert isinstance(
        evaluation_b,
        ScalarAggregationEvaluation,
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=evaluation_a,
        evaluation_b=evaluation_b,
    )

    assert result.status == "pass"
    assert result.axiom_pass is True


def test_monotonicity_allows_equal_output() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="monotonicity",
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=_scalar_evaluation(
            value=0.80,
        ),
        evaluation_b=_scalar_evaluation(
            value=0.80,
        ),
    )

    assert result.status == "pass"


def test_monotonicity_fails_when_output_decreases() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="monotonicity",
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=_scalar_evaluation(
            value=0.80,
        ),
        evaluation_b=_scalar_evaluation(
            value=0.79,
        ),
    )

    assert result.status == "fail"
    assert result.axiom_pass is False


# ---------------------------------------------------------------------
# Continuity
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_id",
    [
        "AX-004-01",
        "AX-004-06",
        "AX-004-07",
        "AX-004-12",
        "AX-004-13",
        "AX-004-18",
    ],
)
def test_selected_canonical_continuity_scenarios_pass(
    scenario_id: str,
) -> None:
    scenario = _scenario_by_id(
        scenario_id
    )

    specification = _specification_by_id(
        "top5_arithmetic"
    )

    evaluation_a = evaluate_aggregation(
        scenario.population_a.values,
        specification=specification,
    )

    assert scenario.population_b is not None

    evaluation_b = evaluate_aggregation(
        scenario.population_b.values,
        specification=specification,
    )

    assert isinstance(
        evaluation_a,
        ScalarAggregationEvaluation,
    )

    assert isinstance(
        evaluation_b,
        ScalarAggregationEvaluation,
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=evaluation_a,
        evaluation_b=evaluation_b,
    )

    assert result.status == "pass"
    assert result.axiom_pass is True
    assert result.allowed_threshold is not None


def test_continuity_fails_when_output_jump_exceeds_input_change() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="continuity",
        notes="input_delta=0.01; zero_based_rank_index=0",
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=_scalar_evaluation(
            value=0.80,
        ),
        evaluation_b=_scalar_evaluation(
            value=0.90,
        ),
    )

    assert result.status == "fail"
    assert result.axiom_pass is False


def test_continuity_requires_input_delta_note() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="continuity",
        notes="",
    )

    with pytest.raises(
        KeyError,
        match="input_delta",
    ):
        evaluate_scalar_axiom(
            scenario=scenario,
            evaluation_a=_scalar_evaluation(
                value=0.80,
            ),
            evaluation_b=_scalar_evaluation(
                value=0.81,
            ),
        )


# ---------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_id",
    [
        "AX-005-01",
        "AX-005-02",
        "AX-005-03",
        "AX-005-04",
        "AX-005-05",
    ],
)
def test_canonical_identity_scenarios_pass(
    scenario_id: str,
) -> None:
    scenario = _scenario_by_id(
        scenario_id
    )

    specification = _specification_by_id(
        "top5_arithmetic"
    )

    evaluation_a = evaluate_aggregation(
        scenario.population_a.values,
        specification=specification,
    )

    assert scenario.population_b is not None

    evaluation_b = evaluate_aggregation(
        scenario.population_b.values,
        specification=specification,
    )

    assert isinstance(
        evaluation_a,
        ScalarAggregationEvaluation,
    )

    assert isinstance(
        evaluation_b,
        ScalarAggregationEvaluation,
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=evaluation_a,
        evaluation_b=evaluation_b,
    )

    assert result.status == "pass"
    assert result.axiom_pass is True


def test_identity_fails_for_wrong_output() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="identity",
        values_a=(0.50, 0.50, 0.50),
        values_b=(0.50, 0.50, 0.50),
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=_scalar_evaluation(
            value=0.60,
            population_size=3,
            minimum_population_size=1,
        ),
        evaluation_b=_scalar_evaluation(
            value=0.60,
            population_size=3,
            minimum_population_size=1,
        ),
    )

    assert result.status == "fail"
    assert result.axiom_pass is False


def test_identity_requires_uniform_population() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="identity",
        values_a=(0.50, 0.60, 0.50),
        values_b=(0.50, 0.60, 0.50),
    )

    with pytest.raises(
        ValueError,
        match="one common player value",
    ):
        evaluate_scalar_axiom(
            scenario=scenario,
            evaluation_a=_scalar_evaluation(
                value=0.53,
                population_size=3,
                minimum_population_size=1,
            ),
            evaluation_b=_scalar_evaluation(
                value=0.53,
                population_size=3,
                minimum_population_size=1,
            ),
        )


# ---------------------------------------------------------------------
# Boundedness
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_id",
    [
        "AX-006-01",
        "AX-006-02",
        "AX-006-03",
        "AX-006-04",
    ],
)
def test_canonical_boundedness_scenarios_pass(
    scenario_id: str,
) -> None:
    scenario = _scenario_by_id(
        scenario_id
    )

    specification = _specification_by_id(
        "top5_arithmetic"
    )

    evaluation_a = evaluate_aggregation(
        scenario.population_a.values,
        specification=specification,
    )

    assert scenario.population_b is not None

    evaluation_b = evaluate_aggregation(
        scenario.population_b.values,
        specification=specification,
    )

    assert isinstance(
        evaluation_a,
        ScalarAggregationEvaluation,
    )

    assert isinstance(
        evaluation_b,
        ScalarAggregationEvaluation,
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=evaluation_a,
        evaluation_b=evaluation_b,
    )

    assert result.status == "pass"
    assert result.axiom_pass is True
    assert result.observed_metric == pytest.approx(0.0)


def test_boundedness_fails_above_population_maximum() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="boundedness",
        values_a=(0.90, 0.80, 0.70),
        values_b=(0.90, 0.80, 0.70),
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=_scalar_evaluation(
            value=0.95,
            population_size=3,
            minimum_population_size=1,
        ),
        evaluation_b=_scalar_evaluation(
            value=0.95,
            population_size=3,
            minimum_population_size=1,
        ),
    )

    assert result.status == "fail"
    assert result.axiom_pass is False
    assert result.observed_metric == pytest.approx(0.05)


def test_boundedness_fails_below_population_minimum() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="boundedness",
        values_a=(0.90, 0.80, 0.70),
        values_b=(0.90, 0.80, 0.70),
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=_scalar_evaluation(
            value=0.65,
            population_size=3,
            minimum_population_size=1,
        ),
        evaluation_b=_scalar_evaluation(
            value=0.65,
            population_size=3,
            minimum_population_size=1,
        ),
    )

    assert result.status == "fail"
    assert result.axiom_pass is False
    assert result.observed_metric == pytest.approx(0.05)


# ---------------------------------------------------------------------
# Applicability and runtime failure handling
# ---------------------------------------------------------------------


def test_not_applicable_aggregation_is_not_counted_as_failure() -> None:
    scenario = _scenario_by_id(
        "AX-005-03"
    )

    evaluation_a = _scalar_evaluation(
        specification_id="replacement_mean_5_5",
        aggregation_family="replacement_group_mean",
        output_type="depth_strength",
        status="not_applicable",
        value=None,
        population_size=5,
        minimum_population_size=10,
        error_type="InsufficientPopulation",
        error_message="Required 10, received 5.",
    )

    evaluation_b = _scalar_evaluation(
        specification_id="replacement_mean_5_5",
        aggregation_family="replacement_group_mean",
        output_type="depth_strength",
        status="not_applicable",
        value=None,
        population_size=5,
        minimum_population_size=10,
        error_type="InsufficientPopulation",
        error_message="Required 10, received 5.",
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=evaluation_a,
        evaluation_b=evaluation_b,
    )

    assert result.status == "not_applicable"
    assert result.axiom_pass is None
    assert result.error_type == "InsufficientPopulation"


def test_runtime_failure_becomes_axiom_failure() -> None:
    scenario = _scenario_by_id(
        "AX-001"
    )

    evaluation_a = _scalar_evaluation(
        status="failed",
        value=None,
        population_size=7,
        minimum_population_size=5,
        error_type="ValueError",
        error_message="Runtime failure.",
    )

    evaluation_b = _scalar_evaluation(
        status="evaluated",
        value=0.80,
        population_size=7,
        minimum_population_size=5,
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=evaluation_a,
        evaluation_b=evaluation_b,
    )

    assert result.status == "fail"
    assert result.axiom_pass is False
    assert result.error_type == "ValueError"


# ---------------------------------------------------------------------
# Input and contract validation
# ---------------------------------------------------------------------


def test_non_axiom_scenario_is_rejected() -> None:
    scenario = SyntheticScenario(
        scenario_id="ST-TEST",
        scenario_family="stability",
        name="Stability",
        description="Non-axiom scenario.",
        population_a=SyntheticPopulation(
            population_id="a",
            values=(0.90, 0.80),
            description="A.",
        ),
        population_b=SyntheticPopulation(
            population_id="b",
            values=(0.90, 0.79),
            description="B.",
        ),
        evaluated_property="determinism",
        expected_direction="equal",
    )

    with pytest.raises(
        ValueError,
        match="non-axiom scenario",
    ):
        evaluate_scalar_axiom(
            scenario=scenario,
            evaluation_a=_scalar_evaluation(
                value=0.85,
                population_size=2,
                minimum_population_size=1,
            ),
            evaluation_b=_scalar_evaluation(
                value=0.845,
                population_size=2,
                minimum_population_size=1,
            ),
        )


def test_unsupported_axiom_property_is_rejected() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="unsupported_property",
    )

    with pytest.raises(
        ValueError,
        match="No scalar axiom rule exists",
    ):
        evaluate_scalar_axiom(
            scenario=scenario,
            evaluation_a=_scalar_evaluation(),
            evaluation_b=_scalar_evaluation(),
        )


def test_missing_population_b_is_rejected_for_determinism() -> None:
    scenario = _simple_axiom_scenario(
        evaluated_property="determinism",
        values_b=None,
    )

    with pytest.raises(
        ValueError,
        match="requires population B",
    ):
        evaluate_scalar_axiom(
            scenario=scenario,
            evaluation_a=_scalar_evaluation(),
            evaluation_b=None,
        )


# ---------------------------------------------------------------------
# Canonical registry coverage
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "specification_id",
    [
        "arithmetic_all",
        "top5_arithmetic",
        "top5_rank_mild",
        "top5_rank_moderate",
        "top5_rank_strong",
        "top5_star_alpha_0_10",
        "top5_star_alpha_0_20",
        "top5_star_alpha_0_30",
        "top5_power_1_25",
        "top5_power_1_50",
        "top5_power_2_00",
        "top5_softmax_beta_1",
        "top5_softmax_beta_3",
        "top5_softmax_beta_5",
        "ability_power_gamma_2",
    ],
)
def test_primary_scalar_specifications_pass_ax001(
    specification_id: str,
) -> None:
    scenario = _scenario_by_id(
        "AX-001"
    )

    specification = _specification_by_id(
        specification_id
    )

    evaluation_a = evaluate_aggregation(
        scenario.population_a.values,
        specification=specification,
    )

    assert scenario.population_b is not None

    evaluation_b = evaluate_aggregation(
        scenario.population_b.values,
        specification=specification,
    )

    assert isinstance(
        evaluation_a,
        ScalarAggregationEvaluation,
    )

    assert isinstance(
        evaluation_b,
        ScalarAggregationEvaluation,
    )

    result = evaluate_scalar_axiom(
        scenario=scenario,
        evaluation_a=evaluation_a,
        evaluation_b=evaluation_b,
    )

    assert result.status == "pass"
    assert result.axiom_pass is True


def test_replacement_specs_are_not_applicable_on_short_axiom_population() -> None:
    scenario = _scenario_by_id(
        "AX-001"
    )

    for specification_id in (
        "replacement_mean_5_5",
        "replacement_dropoff_5_5",
    ):
        specification = _specification_by_id(
            specification_id
        )

        evaluation_a = evaluate_aggregation(
            scenario.population_a.values,
            specification=specification,
        )

        assert scenario.population_b is not None

        evaluation_b = evaluate_aggregation(
            scenario.population_b.values,
            specification=specification,
        )

        assert isinstance(
            evaluation_a,
            ScalarAggregationEvaluation,
        )

        assert isinstance(
            evaluation_b,
            ScalarAggregationEvaluation,
        )

        result = evaluate_scalar_axiom(
            scenario=scenario,
            evaluation_a=evaluation_a,
            evaluation_b=evaluation_b,
        )

        assert result.status == "not_applicable"
        assert result.axiom_pass is None
# test_aggregation_evaluator.py

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

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

from research.studies.study_089_aggregation_mathematics.aggregation_evaluator import (
    DistributionAggregationEvaluation,
    ScalarAggregationEvaluation,
    evaluate_aggregation,
    evaluation_to_record,
    is_distribution_specification,
    minimum_population_size,
)

from research.studies.study_089_aggregation_mathematics.aggregation_specifications import (
    AggregationSpecification,
    build_aggregation_specifications,
)


FIVE_PLAYER_POPULATION = (
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
)

TEN_PLAYER_POPULATION = (
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
    0.50,
)


def _specification_by_id(
    specification_id: str,
) -> AggregationSpecification:
    return next(
        specification
        for specification
        in build_aggregation_specifications()
        if specification.specification_id
        == specification_id
    )


def _scalar_evaluation(
    *,
    status: str = "evaluated",
    value: float | None = 0.80,
    error_type: str | None = None,
    error_message: str | None = None,
) -> ScalarAggregationEvaluation:
    return ScalarAggregationEvaluation(
        specification_id="test_scalar",
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        status=status,  # type: ignore[arg-type]
        value=value,
        population_size=5,
        minimum_population_size=5,
        error_type=error_type,
        error_message=error_message,
    )


def _distribution_evaluation(
    *,
    status: str = "evaluated",
    error_type: str | None = None,
    error_message: str | None = None,
) -> DistributionAggregationEvaluation:
    evaluated = status == "evaluated"

    return DistributionAggregationEvaluation(
        specification_id="test_distribution",
        aggregation_family="distribution_shape",
        output_type="distribution_diagnostics",
        status=status,  # type: ignore[arg-type]
        population_size=5,
        minimum_population_size=5,
        mean=0.85 if evaluated else None,
        maximum=0.95 if evaluated else None,
        minimum=0.75 if evaluated else None,
        value_range=0.20 if evaluated else None,
        standard_deviation=0.07 if evaluated else None,
        star_gap=0.125 if evaluated else None,
        concentration=0.2235 if evaluated else None,
        player_count=5 if evaluated else None,
        error_type=error_type,
        error_message=error_message,
    )


# ---------------------------------------------------------------------
# Scalar result validation
# ---------------------------------------------------------------------


def test_scalar_evaluation_is_immutable() -> None:
    evaluation = _scalar_evaluation()

    with pytest.raises(FrozenInstanceError):
        evaluation.value = 0.90  # type: ignore[misc]


def test_valid_scalar_evaluation_passes_validation() -> None:
    evaluation = _scalar_evaluation()

    evaluation.validate()


@pytest.mark.parametrize(
    "invalid_status",
    [
        "",
        "unknown",
        "success",
    ],
)
def test_scalar_evaluation_rejects_unknown_status(
    invalid_status: str,
) -> None:
    evaluation = _scalar_evaluation(
        status=invalid_status,
    )

    with pytest.raises(
        ValueError,
        match="Unknown scalar evaluation status",
    ):
        evaluation.validate()


def test_evaluated_scalar_requires_value() -> None:
    evaluation = _scalar_evaluation(
        value=None,
    )

    with pytest.raises(
        ValueError,
        match="must contain a value",
    ):
        evaluation.validate()


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_evaluated_scalar_requires_finite_value(
    invalid_value: float,
) -> None:
    evaluation = _scalar_evaluation(
        value=invalid_value,
    )

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        evaluation.validate()


def test_evaluated_scalar_rejects_error_information() -> None:
    evaluation = _scalar_evaluation(
        error_type="ValueError",
        error_message="Unexpected error.",
    )

    with pytest.raises(
        ValueError,
        match="must not contain error information",
    ):
        evaluation.validate()


@pytest.mark.parametrize(
    "status",
    [
        "not_applicable",
        "failed",
    ],
)
def test_unevaluated_scalar_requires_no_value(
    status: str,
) -> None:
    evaluation = _scalar_evaluation(
        status=status,
        value=0.80,
        error_type="TestError",
        error_message="Test message.",
    )

    with pytest.raises(
        ValueError,
        match="must not contain a numeric value",
    ):
        evaluation.validate()


@pytest.mark.parametrize(
    "status",
    [
        "not_applicable",
        "failed",
    ],
)
def test_unevaluated_scalar_requires_error_type(
    status: str,
) -> None:
    evaluation = _scalar_evaluation(
        status=status,
        value=None,
        error_type=None,
        error_message="Test message.",
    )

    with pytest.raises(
        ValueError,
        match="must identify an error type",
    ):
        evaluation.validate()


@pytest.mark.parametrize(
    "status",
    [
        "not_applicable",
        "failed",
    ],
)
def test_unevaluated_scalar_requires_error_message(
    status: str,
) -> None:
    evaluation = _scalar_evaluation(
        status=status,
        value=None,
        error_type="TestError",
        error_message=None,
    )

    with pytest.raises(
        ValueError,
        match="must identify an error message",
    ):
        evaluation.validate()


def test_scalar_to_record_is_deterministic() -> None:
    evaluation = _scalar_evaluation()

    assert evaluation.to_record() == evaluation.to_record()


# ---------------------------------------------------------------------
# Distribution result validation
# ---------------------------------------------------------------------


def test_distribution_evaluation_is_immutable() -> None:
    evaluation = _distribution_evaluation()

    with pytest.raises(FrozenInstanceError):
        evaluation.mean = 0.90  # type: ignore[misc]


def test_valid_distribution_evaluation_passes_validation() -> None:
    evaluation = _distribution_evaluation()

    evaluation.validate()


def test_evaluated_distribution_requires_all_diagnostics() -> None:
    evaluation = DistributionAggregationEvaluation(
        specification_id="distribution",
        aggregation_family="distribution_shape",
        output_type="distribution_diagnostics",
        status="evaluated",
        population_size=5,
        minimum_population_size=5,
        mean=0.85,
        maximum=0.95,
        minimum=0.75,
        value_range=0.20,
        standard_deviation=None,
        star_gap=0.125,
        concentration=0.2235,
        player_count=5,
    )

    with pytest.raises(
        ValueError,
        match="all diagnostic values",
    ):
        evaluation.validate()


def test_evaluated_distribution_requires_player_count() -> None:
    evaluation = DistributionAggregationEvaluation(
        specification_id="distribution",
        aggregation_family="distribution_shape",
        output_type="distribution_diagnostics",
        status="evaluated",
        population_size=5,
        minimum_population_size=5,
        mean=0.85,
        maximum=0.95,
        minimum=0.75,
        value_range=0.20,
        standard_deviation=0.07,
        star_gap=0.125,
        concentration=0.2235,
        player_count=None,
    )

    with pytest.raises(
        ValueError,
        match="must contain player_count",
    ):
        evaluation.validate()


def test_unevaluated_distribution_rejects_diagnostics() -> None:
    evaluation = DistributionAggregationEvaluation(
        specification_id="distribution",
        aggregation_family="distribution_shape",
        output_type="distribution_diagnostics",
        status="failed",
        population_size=5,
        minimum_population_size=5,
        mean=0.85,
        error_type="ValueError",
        error_message="Failure.",
    )

    with pytest.raises(
        ValueError,
        match="must not contain diagnostic values",
    ):
        evaluation.validate()


def test_distribution_to_record_is_deterministic() -> None:
    evaluation = _distribution_evaluation()

    assert evaluation.to_record() == evaluation.to_record()


# ---------------------------------------------------------------------
# Minimum population requirements
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "specification_id",
        "expected_size",
    ),
    [
        ("arithmetic_all", 1),
        ("top5_arithmetic", 5),
        ("top5_rank_mild", 5),
        ("top5_rank_moderate", 5),
        ("top5_rank_strong", 5),
        ("top5_star_alpha_0_10", 5),
        ("top5_star_alpha_0_20", 5),
        ("top5_star_alpha_0_30", 5),
        ("top5_power_1_25", 5),
        ("top5_power_1_50", 5),
        ("top5_power_2_00", 5),
        ("top5_softmax_beta_1", 5),
        ("top5_softmax_beta_3", 5),
        ("top5_softmax_beta_5", 5),
        ("ability_power_gamma_2", 1),
        ("replacement_mean_5_5", 10),
        ("replacement_dropoff_5_5", 10),
        ("distribution_shape_top5", 5),
    ],
)
def test_minimum_population_size_for_canonical_specifications(
    specification_id: str,
    expected_size: int,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    assert minimum_population_size(
        specification
    ) == expected_size


# ---------------------------------------------------------------------
# Distribution identification
# ---------------------------------------------------------------------


def test_only_distribution_shape_is_distribution_specification() -> None:
    specifications = build_aggregation_specifications()

    observed = {
        specification.specification_id:
            is_distribution_specification(
                specification
            )
        for specification in specifications
    }

    assert observed[
        "distribution_shape_top5"
    ] is True

    assert all(
        value is False
        for specification_id, value in observed.items()
        if specification_id
        != "distribution_shape_top5"
    )


# ---------------------------------------------------------------------
# Scalar dispatch correctness
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "specification_id",
        "expected_value",
    ),
    [
        (
            "arithmetic_all",
            arithmetic_mean(
                TEN_PLAYER_POPULATION
            ),
        ),
        (
            "top5_arithmetic",
            top_k_mean(
                TEN_PLAYER_POPULATION,
                k=5,
            ),
        ),
        (
            "top5_rank_mild",
            rank_weighted_top_k(
                TEN_PLAYER_POPULATION,
                weights=(
                    0.24,
                    0.22,
                    0.20,
                    0.18,
                    0.16,
                ),
            ),
        ),
        (
            "top5_rank_moderate",
            rank_weighted_top_k(
                TEN_PLAYER_POPULATION,
                weights=(
                    0.30,
                    0.25,
                    0.20,
                    0.15,
                    0.10,
                ),
            ),
        ),
        (
            "top5_rank_strong",
            rank_weighted_top_k(
                TEN_PLAYER_POPULATION,
                weights=(
                    0.40,
                    0.25,
                    0.15,
                    0.12,
                    0.08,
                ),
            ),
        ),
        (
            "top5_star_alpha_0_20",
            star_influence_top_k(
                TEN_PLAYER_POPULATION,
                k=5,
                alpha=0.20,
            ),
        ),
        (
            "top5_power_1_50",
            power_mean_top_k(
                TEN_PLAYER_POPULATION,
                k=5,
                power=1.50,
            ),
        ),
        (
            "top5_softmax_beta_3",
            softmax_weighted_top_k(
                TEN_PLAYER_POPULATION,
                k=5,
                beta=3.0,
            ),
        ),
        (
            "ability_power_gamma_2",
            ability_power_weighted_mean(
                TEN_PLAYER_POPULATION,
                gamma=2.0,
            ),
        ),
        (
            "replacement_mean_5_5",
            replacement_group_mean(
                TEN_PLAYER_POPULATION,
                primary_k=5,
                replacement_k=5,
            ),
        ),
        (
            "replacement_dropoff_5_5",
            replacement_dropoff(
                TEN_PLAYER_POPULATION,
                primary_k=5,
                replacement_k=5,
            ),
        ),
    ],
)
def test_scalar_dispatch_matches_direct_function_call(
    specification_id: str,
    expected_value: float,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    evaluation = evaluate_aggregation(
        TEN_PLAYER_POPULATION,
        specification=specification,
    )

    assert isinstance(
        evaluation,
        ScalarAggregationEvaluation,
    )

    assert evaluation.status == "evaluated"
    assert evaluation.value == pytest.approx(
        expected_value
    )


@pytest.mark.parametrize(
    "specification_id",
    [
        "top5_star_alpha_0_10",
        "top5_star_alpha_0_20",
        "top5_star_alpha_0_30",
        "top5_power_1_25",
        "top5_power_1_50",
        "top5_power_2_00",
        "top5_softmax_beta_1",
        "top5_softmax_beta_3",
        "top5_softmax_beta_5",
    ],
)
def test_parameterized_primary_specifications_evaluate(
    specification_id: str,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    evaluation = evaluate_aggregation(
        TEN_PLAYER_POPULATION,
        specification=specification,
    )

    assert isinstance(
        evaluation,
        ScalarAggregationEvaluation,
    )

    assert evaluation.status == "evaluated"
    assert evaluation.value is not None
    assert math.isfinite(evaluation.value)


# ---------------------------------------------------------------------
# Distribution dispatch
# ---------------------------------------------------------------------


def test_distribution_dispatch_matches_direct_function_call() -> None:
    specification = _specification_by_id(
        "distribution_shape_top5"
    )

    expected = distribution_shape(
        TEN_PLAYER_POPULATION,
        k=5,
    )

    evaluation = evaluate_aggregation(
        TEN_PLAYER_POPULATION,
        specification=specification,
    )

    assert isinstance(
        evaluation,
        DistributionAggregationEvaluation,
    )

    assert evaluation.status == "evaluated"
    assert evaluation.mean == pytest.approx(
        expected.mean
    )
    assert evaluation.maximum == pytest.approx(
        expected.maximum
    )
    assert evaluation.minimum == pytest.approx(
        expected.minimum
    )
    assert evaluation.value_range == pytest.approx(
        expected.value_range
    )
    assert (
        evaluation.standard_deviation
        == pytest.approx(
            expected.standard_deviation
        )
    )
    assert evaluation.star_gap == pytest.approx(
        expected.star_gap
    )
    assert evaluation.concentration == pytest.approx(
        expected.concentration
    )
    assert evaluation.player_count == (
        expected.player_count
    )


# ---------------------------------------------------------------------
# Applicability handling
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "specification_id",
    [
        "replacement_mean_5_5",
        "replacement_dropoff_5_5",
    ],
)
def test_replacement_methods_are_not_applicable_to_five_players(
    specification_id: str,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    evaluation = evaluate_aggregation(
        FIVE_PLAYER_POPULATION,
        specification=specification,
    )

    assert isinstance(
        evaluation,
        ScalarAggregationEvaluation,
    )

    assert evaluation.status == "not_applicable"
    assert evaluation.value is None
    assert evaluation.error_type == (
        "InsufficientPopulation"
    )
    assert evaluation.error_message is not None
    assert evaluation.population_size == 5
    assert evaluation.minimum_population_size == 10


@pytest.mark.parametrize(
    "specification_id",
    [
        "replacement_mean_5_5",
        "replacement_dropoff_5_5",
    ],
)
def test_replacement_methods_evaluate_with_ten_players(
    specification_id: str,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    evaluation = evaluate_aggregation(
        TEN_PLAYER_POPULATION,
        specification=specification,
    )

    assert evaluation.status == "evaluated"


def test_top_five_method_is_not_applicable_to_four_players() -> None:
    specification = _specification_by_id(
        "top5_arithmetic"
    )

    evaluation = evaluate_aggregation(
        (0.90, 0.80, 0.70, 0.60),
        specification=specification,
    )

    assert evaluation.status == "not_applicable"
    assert evaluation.error_type == (
        "InsufficientPopulation"
    )


def test_distribution_method_is_not_applicable_to_four_players() -> None:
    specification = _specification_by_id(
        "distribution_shape_top5"
    )

    evaluation = evaluate_aggregation(
        (0.90, 0.80, 0.70, 0.60),
        specification=specification,
    )

    assert isinstance(
        evaluation,
        DistributionAggregationEvaluation,
    )

    assert evaluation.status == "not_applicable"
    assert evaluation.mean is None
    assert evaluation.player_count is None
    assert evaluation.error_type == (
        "InsufficientPopulation"
    )


# ---------------------------------------------------------------------
# Runtime failure preservation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_numeric_population_returns_failed_scalar_result(
    invalid_value: float,
) -> None:
    specification = _specification_by_id(
        "top5_arithmetic"
    )

    values = (
        0.90,
        0.80,
        0.70,
        0.60,
        invalid_value,
    )

    evaluation = evaluate_aggregation(
        values,
        specification=specification,
    )

    assert isinstance(
        evaluation,
        ScalarAggregationEvaluation,
    )

    assert evaluation.status == "failed"
    assert evaluation.value is None
    assert evaluation.error_type == "ValueError"
    assert evaluation.error_message


def test_non_numeric_population_returns_failed_result() -> None:
    specification = _specification_by_id(
        "arithmetic_all"
    )

    values = (
        0.90,
        "invalid",
        0.70,
    )

    evaluation = evaluate_aggregation(
        values,  # type: ignore[arg-type]
        specification=specification,
    )

    assert evaluation.status == "failed"
    assert evaluation.error_type == "TypeError"
    assert evaluation.error_message


def test_invalid_distribution_population_returns_failed_result() -> None:
    specification = _specification_by_id(
        "distribution_shape_top5"
    )

    values = (
        0.90,
        0.80,
        0.70,
        0.60,
        float("nan"),
    )

    evaluation = evaluate_aggregation(
        values,
        specification=specification,
    )

    assert isinstance(
        evaluation,
        DistributionAggregationEvaluation,
    )

    assert evaluation.status == "failed"
    assert evaluation.mean is None
    assert evaluation.player_count is None
    assert evaluation.error_type == "ValueError"


# ---------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "specification_id",
    [
        specification.specification_id
        for specification
        in build_aggregation_specifications()
    ],
)
def test_evaluator_does_not_mutate_input(
    specification_id: str,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    values = list(
        TEN_PLAYER_POPULATION
    )

    original = list(values)

    evaluate_aggregation(
        values,
        specification=specification,
    )

    assert values == original


# ---------------------------------------------------------------------
# Canonical registry execution
# ---------------------------------------------------------------------


def test_every_canonical_specification_executes_on_ten_players() -> None:
    specifications = build_aggregation_specifications()

    for specification in specifications:
        evaluation = evaluate_aggregation(
            TEN_PLAYER_POPULATION,
            specification=specification,
        )

        assert evaluation.status == "evaluated", (
            specification.specification_id,
            evaluation.error_type,
            evaluation.error_message,
        )


def test_every_canonical_evaluation_has_correct_metadata() -> None:
    specifications = build_aggregation_specifications()

    for specification in specifications:
        evaluation = evaluate_aggregation(
            TEN_PLAYER_POPULATION,
            specification=specification,
        )

        assert (
            evaluation.specification_id
            == specification.specification_id
        )

        assert (
            evaluation.aggregation_family
            == specification.aggregation_family
        )

        assert (
            evaluation.output_type
            == specification.output_type
        )

        assert evaluation.population_size == 10


# ---------------------------------------------------------------------
# Record serialization
# ---------------------------------------------------------------------


def test_evaluation_to_record_matches_result_method() -> None:
    specification = _specification_by_id(
        "top5_arithmetic"
    )

    evaluation = evaluate_aggregation(
        TEN_PLAYER_POPULATION,
        specification=specification,
    )

    assert evaluation_to_record(
        evaluation
    ) == evaluation.to_record()


def test_scalar_record_contains_expected_fields() -> None:
    specification = _specification_by_id(
        "top5_arithmetic"
    )

    evaluation = evaluate_aggregation(
        TEN_PLAYER_POPULATION,
        specification=specification,
    )

    record = evaluation_to_record(
        evaluation
    )

    required_fields = {
        "specification_id",
        "aggregation_family",
        "output_type",
        "status",
        "value",
        "population_size",
        "minimum_population_size",
        "error_type",
        "error_message",
    }

    assert required_fields.issubset(record)


def test_distribution_record_contains_expected_fields() -> None:
    specification = _specification_by_id(
        "distribution_shape_top5"
    )

    evaluation = evaluate_aggregation(
        TEN_PLAYER_POPULATION,
        specification=specification,
    )

    record = evaluation_to_record(
        evaluation
    )

    required_fields = {
        "specification_id",
        "aggregation_family",
        "output_type",
        "status",
        "population_size",
        "minimum_population_size",
        "mean",
        "maximum",
        "minimum",
        "value_range",
        "standard_deviation",
        "star_gap",
        "concentration",
        "player_count",
        "error_type",
        "error_message",
    }

    assert required_fields.issubset(record)
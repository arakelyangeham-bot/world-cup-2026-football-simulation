# test_aggregation_specifications.py

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from research.studies.study_089_aggregation_mathematics.aggregation_specifications import (
    AggregationSpecification,
    aggregation_specification_records,
    build_aggregation_specifications,
    validate_aggregation_specifications,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _specification(
    *,
    specification_id: str = "test_specification",
    aggregation_family: str = "top_k_mean",
    display_name: str = "Test specification",
    output_type: str = "primary_strength",
    parameters: dict[str, object] | None = None,
    description: str = "Test aggregation specification.",
    historical_control: bool = False,
) -> AggregationSpecification:
    return AggregationSpecification(
        specification_id=specification_id,
        aggregation_family=aggregation_family,
        display_name=display_name,
        output_type=output_type,
        parameters=(
            {"k": 5}
            if parameters is None
            else parameters
        ),
        description=description,
        historical_control=historical_control,
    )


# ---------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------


def test_aggregation_specification_is_immutable() -> None:
    specification = _specification()

    with pytest.raises(FrozenInstanceError):
        specification.display_name = "Changed"  # type: ignore[misc]


def test_parameter_mapping_is_immutable() -> None:
    specification = _specification()

    assert isinstance(
        specification.parameters,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        specification.parameters["k"] = 10  # type: ignore[index]


def test_input_parameter_dictionary_is_copied() -> None:
    parameters = {
        "k": 5,
    }

    specification = _specification(
        parameters=parameters,
    )

    parameters["k"] = 10

    assert specification.parameters["k"] == 5


def test_specification_validation_passes() -> None:
    specification = _specification()

    specification.validate()


# ---------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------


def test_specification_rejects_empty_id() -> None:
    specification = _specification(
        specification_id="",
    )

    with pytest.raises(
        ValueError,
        match="ID must not be empty",
    ):
        specification.validate()


def test_specification_rejects_empty_display_name() -> None:
    specification = _specification(
        display_name="",
    )

    with pytest.raises(
        ValueError,
        match="display name",
    ):
        specification.validate()


def test_specification_rejects_empty_description() -> None:
    specification = _specification(
        description="",
    )

    with pytest.raises(
        ValueError,
        match="description",
    ):
        specification.validate()


def test_specification_rejects_unknown_family() -> None:
    specification = _specification(
        aggregation_family="unknown_family",
    )

    with pytest.raises(
        ValueError,
        match="Unknown aggregation family",
    ):
        specification.validate()


def test_specification_rejects_unknown_output_type() -> None:
    specification = _specification(
        output_type="unknown_output",
    )

    with pytest.raises(
        ValueError,
        match="Unknown aggregation output type",
    ):
        specification.validate()


# ---------------------------------------------------------------------
# Parameter-name contracts
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "family",
        "valid_parameters",
    ),
    [
        (
            "arithmetic_mean",
            {},
        ),
        (
            "top_k_mean",
            {"k": 5},
        ),
        (
            "rank_weighted_top_k",
            {
                "weights": (
                    0.30,
                    0.25,
                    0.20,
                    0.15,
                    0.10,
                ),
            },
        ),
        (
            "star_influence_top_k",
            {
                "k": 5,
                "alpha": 0.20,
            },
        ),
        (
            "power_mean_top_k",
            {
                "k": 5,
                "power": 1.50,
            },
        ),
        (
            "softmax_weighted_top_k",
            {
                "k": 5,
                "beta": 3.0,
            },
        ),
        (
            "ability_power_weighted_mean",
            {
                "gamma": 2.0,
            },
        ),
        (
            "replacement_group_mean",
            {
                "primary_k": 5,
                "replacement_k": 5,
            },
        ),
        (
            "replacement_dropoff",
            {
                "primary_k": 5,
                "replacement_k": 5,
            },
        ),
        (
            "distribution_shape",
            {
                "k": 5,
            },
        ),
    ],
)
def test_family_specific_parameter_contracts_pass(
    family: str,
    valid_parameters: dict[str, object],
) -> None:
    specification = _specification(
        aggregation_family=family,
        parameters=valid_parameters,
    )

    specification.validate()


@pytest.mark.parametrize(
    (
        "family",
        "invalid_parameters",
    ),
    [
        (
            "arithmetic_mean",
            {"k": 5},
        ),
        (
            "top_k_mean",
            {},
        ),
        (
            "rank_weighted_top_k",
            {"k": 5},
        ),
        (
            "star_influence_top_k",
            {"alpha": 0.20},
        ),
        (
            "power_mean_top_k",
            {
                "k": 5,
                "alpha": 0.20,
            },
        ),
        (
            "softmax_weighted_top_k",
            {
                "k": 5,
                "beta": 3.0,
                "extra": 1,
            },
        ),
        (
            "replacement_dropoff",
            {
                "primary_k": 5,
            },
        ),
    ],
)
def test_family_specific_parameter_contracts_reject_invalid_names(
    family: str,
    invalid_parameters: dict[str, object],
) -> None:
    specification = _specification(
        aggregation_family=family,
        parameters=invalid_parameters,
    )

    with pytest.raises(
        ValueError,
        match="parameter names",
    ):
        specification.validate()


# ---------------------------------------------------------------------
# Integer parameter validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_k",
    [
        0,
        -1,
    ],
)
def test_k_must_be_positive(
    invalid_k: int,
) -> None:
    specification = _specification(
        parameters={
            "k": invalid_k,
        },
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        specification.validate()


@pytest.mark.parametrize(
    "invalid_k",
    [
        5.0,
        "5",
        True,
    ],
)
def test_k_must_be_integer(
    invalid_k: object,
) -> None:
    specification = _specification(
        parameters={
            "k": invalid_k,
        },
    )

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        specification.validate()


@pytest.mark.parametrize(
    (
        "parameter_name",
        "invalid_value",
    ),
    [
        (
            "primary_k",
            0,
        ),
        (
            "replacement_k",
            -1,
        ),
    ],
)
def test_replacement_parameters_must_be_positive(
    parameter_name: str,
    invalid_value: int,
) -> None:
    parameters = {
        "primary_k": 5,
        "replacement_k": 5,
    }

    parameters[parameter_name] = invalid_value

    specification = _specification(
        aggregation_family="replacement_dropoff",
        output_type="depth_dropoff",
        parameters=parameters,
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        specification.validate()


# ---------------------------------------------------------------------
# Rank weights
# ---------------------------------------------------------------------


def test_rank_weights_validate() -> None:
    specification = _specification(
        aggregation_family="rank_weighted_top_k",
        parameters={
            "weights": (
                0.30,
                0.25,
                0.20,
                0.15,
                0.10,
            ),
        },
    )

    specification.validate()


@pytest.mark.parametrize(
    "weights",
    [
        (),
        [],
    ],
)
def test_rank_weights_reject_empty_sequence(
    weights: object,
) -> None:
    specification = _specification(
        aggregation_family="rank_weighted_top_k",
        parameters={
            "weights": weights,
        },
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        specification.validate()


@pytest.mark.parametrize(
    "weights",
    [
        "0.5,0.5",
        b"weights",
    ],
)
def test_rank_weights_reject_string_like_values(
    weights: object,
) -> None:
    specification = _specification(
        aggregation_family="rank_weighted_top_k",
        parameters={
            "weights": weights,
        },
    )

    with pytest.raises(
        TypeError,
        match="numeric sequence",
    ):
        specification.validate()


def test_rank_weights_reject_negative_value() -> None:
    specification = _specification(
        aggregation_family="rank_weighted_top_k",
        parameters={
            "weights": (
                0.50,
                0.30,
                0.30,
                -0.10,
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        specification.validate()


def test_rank_weights_reject_all_zero_values() -> None:
    specification = _specification(
        aggregation_family="rank_weighted_top_k",
        parameters={
            "weights": (
                0.0,
                0.0,
                0.0,
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="At least one rank weight",
    ):
        specification.validate()


def test_rank_weights_must_sum_to_one() -> None:
    specification = _specification(
        aggregation_family="rank_weighted_top_k",
        parameters={
            "weights": (
                0.40,
                0.30,
                0.20,
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        specification.validate()


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rank_weights_reject_non_finite_values(
    invalid_value: float,
) -> None:
    specification = _specification(
        aggregation_family="rank_weighted_top_k",
        parameters={
            "weights": (
                0.50,
                0.50,
                invalid_value,
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        specification.validate()


# ---------------------------------------------------------------------
# Scalar parameter bounds
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha",
    [
        0.0,
        0.5,
        1.0,
    ],
)
def test_alpha_accepts_unit_interval(
    alpha: float,
) -> None:
    specification = _specification(
        aggregation_family="star_influence_top_k",
        parameters={
            "k": 5,
            "alpha": alpha,
        },
    )

    specification.validate()


@pytest.mark.parametrize(
    "alpha",
    [
        -0.01,
        1.01,
    ],
)
def test_alpha_rejects_values_outside_unit_interval(
    alpha: float,
) -> None:
    specification = _specification(
        aggregation_family="star_influence_top_k",
        parameters={
            "k": 5,
            "alpha": alpha,
        },
    )

    with pytest.raises(
        ValueError,
        match=r"within \[0, 1\]",
    ):
        specification.validate()


@pytest.mark.parametrize(
    "power",
    [
        0.0,
        -1.0,
    ],
)
def test_power_must_be_positive(
    power: float,
) -> None:
    specification = _specification(
        aggregation_family="power_mean_top_k",
        parameters={
            "k": 5,
            "power": power,
        },
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        specification.validate()


@pytest.mark.parametrize(
    "beta",
    [
        0.0,
        1.0,
        5.0,
    ],
)
def test_beta_accepts_non_negative_values(
    beta: float,
) -> None:
    specification = _specification(
        aggregation_family="softmax_weighted_top_k",
        parameters={
            "k": 5,
            "beta": beta,
        },
    )

    specification.validate()


def test_beta_rejects_negative_value() -> None:
    specification = _specification(
        aggregation_family="softmax_weighted_top_k",
        parameters={
            "k": 5,
            "beta": -0.01,
        },
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        specification.validate()


@pytest.mark.parametrize(
    "gamma",
    [
        0.0,
        1.0,
        2.0,
    ],
)
def test_gamma_accepts_non_negative_values(
    gamma: float,
) -> None:
    specification = _specification(
        aggregation_family="ability_power_weighted_mean",
        parameters={
            "gamma": gamma,
        },
    )

    specification.validate()


def test_gamma_rejects_negative_value() -> None:
    specification = _specification(
        aggregation_family="ability_power_weighted_mean",
        parameters={
            "gamma": -0.01,
        },
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        specification.validate()


@pytest.mark.parametrize(
    (
        "family",
        "parameters",
        "expected_parameter_name",
    ),
    [
        (
            "star_influence_top_k",
            {
                "k": 5,
                "alpha": float("nan"),
            },
            "alpha",
        ),
        (
            "power_mean_top_k",
            {
                "k": 5,
                "power": float("inf"),
            },
            "power",
        ),
        (
            "softmax_weighted_top_k",
            {
                "k": 5,
                "beta": float("-inf"),
            },
            "beta",
        ),
        (
            "ability_power_weighted_mean",
            {
                "gamma": float("nan"),
            },
            "gamma",
        ),
    ],
)
def test_scalar_parameters_must_be_finite(
    family: str,
    parameters: dict[str, object],
    expected_parameter_name: str,
) -> None:
    specification = _specification(
        aggregation_family=family,
        parameters=parameters,
    )

    with pytest.raises(
        ValueError,
        match=f"{expected_parameter_name} must be finite",
    ):
        specification.validate()


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_to_record_without_parameters() -> None:
    specification = _specification(
        aggregation_family="arithmetic_mean",
        output_type="depth_strength",
        parameters={},
    )

    record = specification.to_record()

    assert record["parameterization"] == "none"


def test_to_record_with_scalar_parameters() -> None:
    specification = _specification(
        aggregation_family="star_influence_top_k",
        parameters={
            "k": 5,
            "alpha": 0.20,
        },
    )

    record = specification.to_record()

    assert record["parameterization"] == (
        "alpha=0.2; k=5"
    )


def test_to_record_with_weights() -> None:
    specification = _specification(
        aggregation_family="rank_weighted_top_k",
        parameters={
            "weights": (
                0.30,
                0.25,
                0.20,
                0.15,
                0.10,
            ),
        },
    )

    record = specification.to_record()

    assert record["parameterization"] == (
        "weights=[0.3, 0.25, 0.2, 0.15, 0.1]"
    )


def test_to_record_preserves_historical_control_flag() -> None:
    specification = _specification(
        historical_control=True,
    )

    record = specification.to_record()

    assert record["historical_control"] is True


# ---------------------------------------------------------------------
# Canonical specification registry
# ---------------------------------------------------------------------


def test_specification_registry_is_deterministic() -> None:
    specifications_a = build_aggregation_specifications()
    specifications_b = build_aggregation_specifications()

    assert specifications_a == specifications_b


def test_specification_registry_is_not_empty() -> None:
    specifications = build_aggregation_specifications()

    assert specifications


def test_specification_registry_contains_18_specifications() -> None:
    specifications = build_aggregation_specifications()

    assert len(specifications) == 18


def test_specification_ids_are_unique() -> None:
    specifications = build_aggregation_specifications()

    specification_ids = [
        specification.specification_id
        for specification in specifications
    ]

    assert len(specification_ids) == len(
        set(specification_ids)
    )


def test_registry_contains_expected_specification_ids() -> None:
    specifications = build_aggregation_specifications()

    specification_ids = {
        specification.specification_id
        for specification in specifications
    }

    assert specification_ids == {
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
        "replacement_mean_5_5",
        "replacement_dropoff_5_5",
        "distribution_shape_top5",
    }


def test_registry_output_type_counts() -> None:
    specifications = build_aggregation_specifications()

    counts: dict[str, int] = {}

    for specification in specifications:
        counts[specification.output_type] = (
            counts.get(
                specification.output_type,
                0,
            )
            + 1
        )

    assert counts == {
        "primary_strength": 14,
        "depth_strength": 2,
        "depth_dropoff": 1,
        "distribution_diagnostics": 1,
    }


def test_rank_weight_parameter_grid_is_frozen() -> None:
    specifications = build_aggregation_specifications()

    observed = {
        specification.specification_id: tuple(
            specification.parameters["weights"]
        )
        for specification in specifications
        if specification.aggregation_family
        == "rank_weighted_top_k"
    }

    assert observed == {
        "top5_rank_mild": (
            0.24,
            0.22,
            0.20,
            0.18,
            0.16,
        ),
        "top5_rank_moderate": (
            0.30,
            0.25,
            0.20,
            0.15,
            0.10,
        ),
        "top5_rank_strong": (
            0.40,
            0.25,
            0.15,
            0.12,
            0.08,
        ),
    }


def test_star_alpha_grid_is_frozen() -> None:
    specifications = build_aggregation_specifications()

    observed = sorted(
        float(
            specification.parameters["alpha"]
        )
        for specification in specifications
        if specification.aggregation_family
        == "star_influence_top_k"
    )

    assert observed == [
        0.10,
        0.20,
        0.30,
    ]


def test_power_grid_is_frozen() -> None:
    specifications = build_aggregation_specifications()

    observed = sorted(
        float(
            specification.parameters["power"]
        )
        for specification in specifications
        if specification.aggregation_family
        == "power_mean_top_k"
    )

    assert observed == [
        1.25,
        1.50,
        2.00,
    ]


def test_softmax_beta_grid_is_frozen() -> None:
    specifications = build_aggregation_specifications()

    observed = sorted(
        float(
            specification.parameters["beta"]
        )
        for specification in specifications
        if specification.aggregation_family
        == "softmax_weighted_top_k"
    )

    assert observed == [
        1.0,
        3.0,
        5.0,
    ]


def test_only_historical_control_is_ability_power() -> None:
    specifications = build_aggregation_specifications()

    historical_controls = [
        specification
        for specification in specifications
        if specification.historical_control
    ]

    assert len(historical_controls) == 1

    assert (
        historical_controls[0].specification_id
        == "ability_power_gamma_2"
    )


def test_distribution_shape_is_not_scalar_strength() -> None:
    specifications = build_aggregation_specifications()

    specification = next(
        specification
        for specification in specifications
        if specification.specification_id
        == "distribution_shape_top5"
    )

    assert (
        specification.output_type
        == "distribution_diagnostics"
    )


def test_all_canonical_specifications_validate() -> None:
    specifications = build_aggregation_specifications()

    for specification in specifications:
        specification.validate()


def test_validate_registry_rejects_empty_registry() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        validate_aggregation_specifications(())


def test_validate_registry_rejects_duplicate_ids() -> None:
    specification_a = _specification(
        specification_id="duplicate",
    )

    specification_b = _specification(
        specification_id="duplicate",
        display_name="Second specification",
    )

    with pytest.raises(
        ValueError,
        match="duplicate specification ID",
    ):
        validate_aggregation_specifications(
            (
                specification_a,
                specification_b,
            )
        )


# ---------------------------------------------------------------------
# Registry serialization
# ---------------------------------------------------------------------


def test_specification_records_match_registry_size() -> None:
    specifications = build_aggregation_specifications()
    records = aggregation_specification_records(
        specifications
    )

    assert len(records) == len(specifications)


def test_specification_records_are_deterministic() -> None:
    records_a = aggregation_specification_records()
    records_b = aggregation_specification_records()

    assert records_a == records_b


def test_specification_records_have_required_fields() -> None:
    records = aggregation_specification_records()

    required_fields = {
        "specification_id",
        "aggregation_family",
        "display_name",
        "output_type",
        "parameterization",
        "description",
        "historical_control",
    }

    assert required_fields.issubset(records[0])
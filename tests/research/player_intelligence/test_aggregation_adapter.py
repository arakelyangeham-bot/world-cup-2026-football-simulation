# test_aggregation_adapter.py

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import pytest

from research.player_intelligence.aggregation_adapter import (
    AggregationSpecificationLike,
    aggregate_dimension_values,
    aggregation_profile_name,
    minimum_required_population,
)

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

from research.studies.study_089_aggregation_mathematics.aggregation_specifications import (
    AggregationSpecification,
    build_aggregation_specifications,
)


FIVE_PLAYER_VALUES = (
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
)

TEN_PLAYER_VALUES = (
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
        for specification in build_aggregation_specifications()
        if specification.specification_id == specification_id
    )


@dataclass(frozen=True)
class MinimalSpecification:
    """
    Non-study specification used to verify protocol-based integration.
    """

    aggregation_family: str
    output_type: str
    parameters: Mapping[str, object]
    specification_id: str | None = None

    def validate(self) -> None:
        if not self.aggregation_family:
            raise ValueError(
                "aggregation_family must not be empty."
            )

        if not self.output_type:
            raise ValueError(
                "output_type must not be empty."
            )


# ---------------------------------------------------------------------
# Protocol boundary
# ---------------------------------------------------------------------


def test_study_specification_satisfies_adapter_protocol() -> None:
    specification = _specification_by_id(
        "top5_arithmetic"
    )

    assert isinstance(
        specification,
        AggregationSpecificationLike,
    )


def test_minimal_external_specification_satisfies_protocol() -> None:
    specification = MinimalSpecification(
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        parameters=MappingProxyType(
            {
                "k": 5,
            }
        ),
        specification_id="external_top5",
    )

    assert isinstance(
        specification,
        AggregationSpecificationLike,
    )


def test_adapter_accepts_non_study_specification() -> None:
    specification = MinimalSpecification(
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        parameters={
            "k": 5,
        },
        specification_id="external_top5",
    )

    result = aggregate_dimension_values(
        TEN_PLAYER_VALUES,
        specification=specification,
    )

    assert result == pytest.approx(
        top_k_mean(
            TEN_PLAYER_VALUES,
            k=5,
        )
    )


# ---------------------------------------------------------------------
# Minimum population rules
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
    ],
)
def test_minimum_required_population_for_scalar_specifications(
    specification_id: str,
    expected_size: int,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    assert minimum_required_population(
        specification
    ) == expected_size


def test_rank_weight_minimum_matches_weight_count() -> None:
    specification = MinimalSpecification(
        aggregation_family="rank_weighted_top_k",
        output_type="primary_strength",
        parameters={
            "weights": (
                0.50,
                0.30,
                0.20,
            ),
        },
    )

    assert minimum_required_population(
        specification
    ) == 3


def test_distribution_specification_has_no_scalar_population_rule() -> None:
    specification = _specification_by_id(
        "distribution_shape_top5"
    )

    with pytest.raises(
        TypeError,
        match="do not produce one scalar",
    ):
        minimum_required_population(
            specification
        )


def test_unknown_family_has_no_population_rule() -> None:
    specification = MinimalSpecification(
        aggregation_family="unknown_family",
        output_type="primary_strength",
        parameters={},
    )

    with pytest.raises(
        KeyError,
        match="No population-size rule exists",
    ):
        minimum_required_population(
            specification
        )


# ---------------------------------------------------------------------
# Exact scalar dispatch
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
                TEN_PLAYER_VALUES
            ),
        ),
        (
            "top5_arithmetic",
            top_k_mean(
                TEN_PLAYER_VALUES,
                k=5,
            ),
        ),
        (
            "top5_rank_mild",
            rank_weighted_top_k(
                TEN_PLAYER_VALUES,
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
                TEN_PLAYER_VALUES,
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
                TEN_PLAYER_VALUES,
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
            "top5_star_alpha_0_10",
            star_influence_top_k(
                TEN_PLAYER_VALUES,
                k=5,
                alpha=0.10,
            ),
        ),
        (
            "top5_star_alpha_0_20",
            star_influence_top_k(
                TEN_PLAYER_VALUES,
                k=5,
                alpha=0.20,
            ),
        ),
        (
            "top5_star_alpha_0_30",
            star_influence_top_k(
                TEN_PLAYER_VALUES,
                k=5,
                alpha=0.30,
            ),
        ),
        (
            "top5_power_1_25",
            power_mean_top_k(
                TEN_PLAYER_VALUES,
                k=5,
                power=1.25,
            ),
        ),
        (
            "top5_power_1_50",
            power_mean_top_k(
                TEN_PLAYER_VALUES,
                k=5,
                power=1.50,
            ),
        ),
        (
            "top5_power_2_00",
            power_mean_top_k(
                TEN_PLAYER_VALUES,
                k=5,
                power=2.00,
            ),
        ),
        (
            "top5_softmax_beta_1",
            softmax_weighted_top_k(
                TEN_PLAYER_VALUES,
                k=5,
                beta=1.0,
            ),
        ),
        (
            "top5_softmax_beta_3",
            softmax_weighted_top_k(
                TEN_PLAYER_VALUES,
                k=5,
                beta=3.0,
            ),
        ),
        (
            "top5_softmax_beta_5",
            softmax_weighted_top_k(
                TEN_PLAYER_VALUES,
                k=5,
                beta=5.0,
            ),
        ),
        (
            "ability_power_gamma_2",
            ability_power_weighted_mean(
                TEN_PLAYER_VALUES,
                gamma=2.0,
            ),
        ),
        (
            "replacement_mean_5_5",
            replacement_group_mean(
                TEN_PLAYER_VALUES,
                primary_k=5,
                replacement_k=5,
            ),
        ),
        (
            "replacement_dropoff_5_5",
            replacement_dropoff(
                TEN_PLAYER_VALUES,
                primary_k=5,
                replacement_k=5,
            ),
        ),
    ],
)
def test_adapter_matches_frozen_aggregation_mathematics(
    specification_id: str,
    expected_value: float,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    result = aggregate_dimension_values(
        TEN_PLAYER_VALUES,
        specification=specification,
    )

    assert result == pytest.approx(
        expected_value
    )


def test_every_canonical_scalar_specification_dispatches() -> None:
    specifications = [
        specification
        for specification in build_aggregation_specifications()
        if specification.output_type
        != "distribution_diagnostics"
    ]

    for specification in specifications:
        result = aggregate_dimension_values(
            TEN_PLAYER_VALUES,
            specification=specification,
        )

        assert isinstance(
            result,
            float,
        )


# ---------------------------------------------------------------------
# Strict population handling
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "specification_id",
    [
        "top5_arithmetic",
        "top5_rank_moderate",
        "top5_star_alpha_0_20",
        "top5_power_1_50",
        "top5_softmax_beta_3",
    ],
)
def test_top_five_methods_reject_four_players(
    specification_id: str,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    with pytest.raises(
        ValueError,
        match="required 5, received 4",
    ):
        aggregate_dimension_values(
            (
                0.90,
                0.80,
                0.70,
                0.60,
            ),
            specification=specification,
        )


@pytest.mark.parametrize(
    "specification_id",
    [
        "replacement_mean_5_5",
        "replacement_dropoff_5_5",
    ],
)
def test_replacement_methods_reject_five_players(
    specification_id: str,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    with pytest.raises(
        ValueError,
        match="required 10, received 5",
    ):
        aggregate_dimension_values(
            FIVE_PLAYER_VALUES,
            specification=specification,
        )


def test_replacement_methods_accept_exactly_ten_players() -> None:
    for specification_id in (
        "replacement_mean_5_5",
        "replacement_dropoff_5_5",
    ):
        specification = _specification_by_id(
            specification_id
        )

        result = aggregate_dimension_values(
            TEN_PLAYER_VALUES,
            specification=specification,
        )

        assert isinstance(
            result,
            float,
        )


def test_arithmetic_mean_accepts_single_player() -> None:
    specification = _specification_by_id(
        "arithmetic_all"
    )

    result = aggregate_dimension_values(
        (0.82,),
        specification=specification,
    )

    assert result == pytest.approx(
        0.82
    )


def test_empty_population_is_rejected() -> None:
    specification = _specification_by_id(
        "arithmetic_all"
    )

    with pytest.raises(
        ValueError,
        match="required 1, received 0",
    ):
        aggregate_dimension_values(
            (),
            specification=specification,
        )


# ---------------------------------------------------------------------
# Unsupported output contracts
# ---------------------------------------------------------------------


def test_distribution_specification_is_rejected_as_scalar() -> None:
    specification = _specification_by_id(
        "distribution_shape_top5"
    )

    with pytest.raises(
        TypeError,
        match="cannot be used as a scalar",
    ):
        aggregate_dimension_values(
            FIVE_PLAYER_VALUES,
            specification=specification,
        )


def test_unknown_scalar_family_is_rejected() -> None:
    specification = MinimalSpecification(
        aggregation_family="unknown_family",
        output_type="primary_strength",
        parameters={},
    )

    with pytest.raises(
        KeyError,
        match="Unsupported scalar aggregation family",
    ):
        aggregate_dimension_values(
            FIVE_PLAYER_VALUES,
            specification=specification,
        )


# ---------------------------------------------------------------------
# Validation delegation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class InvalidSpecification:
    aggregation_family: str = "top_k_mean"
    output_type: str = "primary_strength"
    parameters: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType(
            {
                "k": 5,
            }
        )
    )
    def validate(self) -> None:
        raise ValueError(
            "Intentional specification failure."
        )


def test_minimum_population_delegates_validation() -> None:
    specification = InvalidSpecification()

    with pytest.raises(
        ValueError,
        match="Intentional specification failure",
    ):
        minimum_required_population(
            specification
        )


def test_aggregate_dimension_values_delegates_validation() -> None:
    specification = InvalidSpecification()

    with pytest.raises(
        ValueError,
        match="Intentional specification failure",
    ):
        aggregate_dimension_values(
            FIVE_PLAYER_VALUES,
            specification=specification,
        )


# ---------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "specification_id",
    [
        specification.specification_id
        for specification in build_aggregation_specifications()
        if specification.output_type
        != "distribution_diagnostics"
    ],
)
def test_adapter_does_not_mutate_input_values(
    specification_id: str,
) -> None:
    specification = _specification_by_id(
        specification_id
    )

    values = list(
        TEN_PLAYER_VALUES
    )
    original = list(values)

    aggregate_dimension_values(
        values,
        specification=specification,
    )

    assert values == original


# ---------------------------------------------------------------------
# Aggregation profile naming
# ---------------------------------------------------------------------


def test_profile_name_uses_specification_id() -> None:
    specification = _specification_by_id(
        "top5_rank_moderate"
    )

    assert aggregation_profile_name(
        specification
    ) == "top5_rank_moderate"


def test_profile_name_falls_back_to_family() -> None:
    specification = MinimalSpecification(
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        parameters={
            "k": 5,
        },
        specification_id=None,
    )

    assert aggregation_profile_name(
        specification
    ) == "top_k_mean"


def test_profile_name_falls_back_when_id_is_blank() -> None:
    specification = MinimalSpecification(
        aggregation_family="top_k_mean",
        output_type="primary_strength",
        parameters={
            "k": 5,
        },
        specification_id="   ",
    )

    assert aggregation_profile_name(
        specification
    ) == "top_k_mean"
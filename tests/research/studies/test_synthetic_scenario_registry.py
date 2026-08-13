# test_synthetic_scenario_registry.py

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from research.studies.study_089_aggregation_mathematics.synthetic_scenario_registry import (
    ABSOLUTE_TOLERANCE,
    RELATIVE_TOLERANCE,
    SyntheticPopulation,
    SyntheticScenario,
    append_values,
    build_scenario_registry,
    multiply_values,
    population_registry_records,
    remove_value_once,
    scenario_registry_records,
    shift_values,
    uniform_population,
    validate_scenario_registry,
)


# ---------------------------------------------------------------------
# SyntheticPopulation
# ---------------------------------------------------------------------


def test_synthetic_population_is_immutable() -> None:
    population = SyntheticPopulation(
        population_id="test-population",
        values=(0.90, 0.80, 0.70),
        description="Test population.",
    )

    with pytest.raises(FrozenInstanceError):
        population.population_id = "changed"  # type: ignore[misc]


def test_synthetic_population_validation_passes() -> None:
    population = SyntheticPopulation(
        population_id="valid-population",
        values=(0.90, 0.80, 0.70),
        description="Valid population.",
    )

    population.validate()


def test_synthetic_population_rejects_empty_id() -> None:
    population = SyntheticPopulation(
        population_id="",
        values=(0.90,),
        description="Description.",
    )

    with pytest.raises(
        ValueError,
        match="ID must not be empty",
    ):
        population.validate()


def test_synthetic_population_rejects_empty_description() -> None:
    population = SyntheticPopulation(
        population_id="population",
        values=(0.90,),
        description="",
    )

    with pytest.raises(
        ValueError,
        match="description must not be empty",
    ):
        population.validate()


def test_synthetic_population_rejects_empty_values() -> None:
    population = SyntheticPopulation(
        population_id="population",
        values=(),
        description="Empty population.",
    )

    with pytest.raises(
        ValueError,
        match="at least one value",
    ):
        population.validate()


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01,
    ],
)
def test_synthetic_population_rejects_values_outside_unit_interval(
    invalid_value: float,
) -> None:
    population = SyntheticPopulation(
        population_id="population",
        values=(0.90, invalid_value),
        description="Invalid population.",
    )

    with pytest.raises(
        ValueError,
        match=r"within \[0, 1\]",
    ):
        population.validate()


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_synthetic_population_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    population = SyntheticPopulation(
        population_id="population",
        values=(0.90, invalid_value),
        description="Invalid population.",
    )

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        population.validate()


def test_synthetic_population_to_record() -> None:
    population = SyntheticPopulation(
        population_id="population",
        values=(0.90, 0.80, 0.70),
        description="Example population.",
    )

    record = population.to_record()

    assert record == {
        "population_id": "population",
        "description": "Example population.",
        "player_count": 3,
        "values": (
            "0.900000000000, "
            "0.800000000000, "
            "0.700000000000"
        ),
    }


# ---------------------------------------------------------------------
# SyntheticScenario
# ---------------------------------------------------------------------


def _example_population(
    population_id: str,
) -> SyntheticPopulation:
    return SyntheticPopulation(
        population_id=population_id,
        values=(0.90, 0.80, 0.70),
        description="Example population.",
    )


def test_synthetic_scenario_is_immutable() -> None:
    scenario = SyntheticScenario(
        scenario_id="ST-TEST",
        scenario_family="stability",
        name="Test scenario",
        description="Test description.",
        population_a=_example_population("population-a"),
        population_b=_example_population("population-b"),
        evaluated_property="stability",
        expected_direction="equal",
    )

    with pytest.raises(FrozenInstanceError):
        scenario.name = "changed"  # type: ignore[misc]


def test_synthetic_scenario_validation_passes() -> None:
    scenario = SyntheticScenario(
        scenario_id="ST-TEST",
        scenario_family="stability",
        name="Test scenario",
        description="Test description.",
        population_a=_example_population("population-a"),
        population_b=_example_population("population-b"),
        evaluated_property="stability",
        expected_direction="equal",
    )

    scenario.validate()


def test_synthetic_scenario_rejects_unknown_family() -> None:
    scenario = SyntheticScenario(
        scenario_id="TEST-001",
        scenario_family="unknown",
        name="Test",
        description="Test.",
        population_a=_example_population("population-a"),
        population_b=_example_population("population-b"),
        evaluated_property="test",
        expected_direction="equal",
    )

    with pytest.raises(
        ValueError,
        match="Unknown scenario family",
    ):
        scenario.validate()


def test_synthetic_scenario_rejects_unknown_direction() -> None:
    scenario = SyntheticScenario(
        scenario_id="ST-TEST",
        scenario_family="stability",
        name="Test",
        description="Test.",
        population_a=_example_population("population-a"),
        population_b=_example_population("population-b"),
        evaluated_property="test",
        expected_direction="unknown",
    )

    with pytest.raises(
        ValueError,
        match="Unknown expected direction",
    ):
        scenario.validate()


def test_directional_scenario_requires_population_b() -> None:
    scenario = SyntheticScenario(
        scenario_id="ST-TEST",
        scenario_family="stability",
        name="Test",
        description="Test.",
        population_a=_example_population("population-a"),
        population_b=None,
        evaluated_property="test",
        expected_direction="decrease",
    )

    with pytest.raises(
        ValueError,
        match="does not define population_b",
    ):
        scenario.validate()


def test_descriptive_scenario_allows_no_population_b() -> None:
    scenario = SyntheticScenario(
        scenario_id="TEST-001",
        scenario_family="regression",
        name="Descriptive test",
        description="Descriptive scenario.",
        population_a=_example_population("population-a"),
        population_b=None,
        evaluated_property="description",
        expected_direction="descriptive",
    )

    scenario.validate()


def test_synthetic_scenario_to_record() -> None:
    scenario = SyntheticScenario(
        scenario_id="ST-TEST",
        scenario_family="stability",
        name="Test scenario",
        description="Test description.",
        population_a=_example_population("population-a"),
        population_b=_example_population("population-b"),
        evaluated_property="stability",
        expected_direction="equal",
        comparison_label_a="before",
        comparison_label_b="after",
        binary_pass_expected=True,
        notes="Test notes.",
    )

    record = scenario.to_record()

    assert record["scenario_id"] == "ST-TEST"
    assert record["scenario_family"] == "stability"
    assert record["population_a_id"] == "population-a"
    assert record["population_b_id"] == "population-b"
    assert record["comparison_label_a"] == "before"
    assert record["comparison_label_b"] == "after"
    assert record["binary_pass_expected"] is True


# ---------------------------------------------------------------------
# Population helper functions
# ---------------------------------------------------------------------


def test_uniform_population_returns_expected_values() -> None:
    population = uniform_population(
        population_id="uniform",
        value=0.85,
        size=5,
        description="Balanced population.",
    )

    assert population.values == (
        0.85,
        0.85,
        0.85,
        0.85,
        0.85,
    )


@pytest.mark.parametrize(
    "invalid_size",
    [
        0,
        -1,
    ],
)
def test_uniform_population_rejects_non_positive_size(
    invalid_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        uniform_population(
            population_id="uniform",
            value=0.85,
            size=invalid_size,
            description="Population.",
        )


@pytest.mark.parametrize(
    "invalid_size",
    [
        1.5,
        "5",
        True,
    ],
)
def test_uniform_population_rejects_non_integer_size(
    invalid_size: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        uniform_population(
            population_id="uniform",
            value=0.85,
            size=invalid_size,  # type: ignore[arg-type]
            description="Population.",
        )


def test_remove_value_once_removes_only_one_matching_value() -> None:
    result = remove_value_once(
        [0.90, 0.80, 0.80, 0.70],
        value_to_remove=0.80,
    )

    assert result == (
        0.90,
        0.80,
        0.70,
    )


def test_remove_value_once_rejects_missing_value() -> None:
    with pytest.raises(
        ValueError,
        match="was not found",
    ):
        remove_value_once(
            [0.90, 0.80],
            value_to_remove=0.70,
        )


def test_append_values_returns_combined_tuple() -> None:
    result = append_values(
        [0.90, 0.80],
        [0.30, 0.20],
    )

    assert result == (
        0.90,
        0.80,
        0.30,
        0.20,
    )


def test_shift_values_returns_expected_values() -> None:
    result = shift_values(
        [0.90, 0.50, 0.10],
        amount=0.02,
    )

    assert result == pytest.approx(
        (
            0.92,
            0.52,
            0.12,
        )
    )


def test_shift_values_applies_bounds() -> None:
    result = shift_values(
        [0.99, 0.01],
        amount=0.05,
    )

    assert result == pytest.approx(
        (
            1.00,
            0.06,
        )
    )


def test_shift_values_rejects_invalid_bounds() -> None:
    with pytest.raises(
        ValueError,
        match="lower_bound",
    ):
        shift_values(
            [0.50],
            amount=0.01,
            lower_bound=1.0,
            upper_bound=0.0,
        )


def test_multiply_values_returns_expected_values() -> None:
    result = multiply_values(
        [0.80, 0.60, 0.40],
        factor=1.05,
    )

    assert result == pytest.approx(
        (
            0.84,
            0.63,
            0.42,
        )
    )


def test_multiply_values_applies_upper_bound() -> None:
    result = multiply_values(
        [0.99],
        factor=1.05,
    )

    assert result == pytest.approx((1.0,))


def test_multiply_values_rejects_negative_factor() -> None:
    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        multiply_values(
            [0.50],
            factor=-1.0,
        )


# ---------------------------------------------------------------------
# Canonical registry
# ---------------------------------------------------------------------


def test_scenario_registry_is_deterministic() -> None:
    registry_a = build_scenario_registry()
    registry_b = build_scenario_registry()

    assert registry_a == registry_b


def test_scenario_registry_is_not_empty() -> None:
    registry = build_scenario_registry()

    assert registry


def test_scenario_registry_has_unique_scenario_ids() -> None:
    registry = build_scenario_registry()

    scenario_ids = [
        scenario.scenario_id
        for scenario in registry
    ]

    assert len(scenario_ids) == len(set(scenario_ids))


def test_population_ids_are_consistently_defined() -> None:
    registry = build_scenario_registry()

    populations: dict[
        str,
        SyntheticPopulation,
    ] = {}

    for scenario in registry:
        for population in (
            scenario.population_a,
            scenario.population_b,
        ):
            if population is None:
                continue

            existing = populations.get(
                population.population_id
            )

            if existing is None:
                populations[
                    population.population_id
                ] = population
            else:
                assert existing == population


def test_registry_covers_expected_scalar_families() -> None:
    registry = build_scenario_registry()

    families = {
        scenario.scenario_family
        for scenario in registry
    }

    assert {
        "axiom",
        "stability",
        "elite_responsiveness",
        "distribution_shape",
        "depth",
        "rank_boundary",
        "scale_consistency",
    }.issubset(families)


def test_structural_scenarios_are_deferred() -> None:
    registry = build_scenario_registry()

    scenario_ids = {
        scenario.scenario_id
        for scenario in registry
    }

    assert not any(
        scenario_id.startswith("SR-")
        for scenario_id in scenario_ids
    )


def test_registry_contains_canonical_named_scenarios() -> None:
    registry = build_scenario_registry()

    scenario_ids = {
        scenario.scenario_id
        for scenario in registry
    }

    required_ids = {
        "AX-001",
        "AX-002",
        "ST-001",
        "ST-002",
        "ST-003",
        "ER-001",
        "ER-002",
        "ER-003",
        "ER-004",
        "DS-001",
        "DS-002",
        "DS-003",
        "DS-004",
        "DP-001",
        "DP-002",
        "DP-003",
        "RB-001",
        "RB-002",
        "RB-003",
        "SC-001",
        "SC-002",
        "SC-003",
    }

    assert required_ids.issubset(scenario_ids)


def test_ds001_populations_are_frozen() -> None:
    registry = build_scenario_registry()

    scenario = next(
        scenario
        for scenario in registry
        if scenario.scenario_id == "DS-001"
    )

    assert scenario.population_a.values == (
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
    )

    assert scenario.population_b is not None

    assert scenario.population_b.values == (
        0.85,
        0.85,
        0.85,
        0.85,
        0.85,
    )


def test_er001_removes_elite_player() -> None:
    registry = build_scenario_registry()

    scenario = next(
        scenario
        for scenario in registry
        if scenario.scenario_id == "ER-001"
    )

    assert scenario.population_a.values == (
        0.98,
        0.86,
        0.84,
        0.82,
        0.80,
        0.76,
    )

    assert scenario.population_b is not None
    assert 0.98 not in scenario.population_b.values
    assert len(scenario.population_b.values) == 5


def test_dp001_adds_only_fringe_players() -> None:
    registry = build_scenario_registry()

    scenario = next(
        scenario
        for scenario in registry
        if scenario.scenario_id == "DP-001"
    )

    assert scenario.population_b is not None

    assert scenario.population_b.values[:10] == (
        scenario.population_a.values
    )

    assert scenario.population_b.values[10:] == (
        0.30,
        0.25,
        0.20,
        0.15,
        0.10,
    )


def test_dp003_uniform_shift_is_exact() -> None:
    registry = build_scenario_registry()

    scenario = next(
        scenario
        for scenario in registry
        if scenario.scenario_id == "DP-003"
    )

    assert scenario.population_b is not None

    for baseline, improved in zip(
        scenario.population_a.values,
        scenario.population_b.values,
        strict=True,
    ):
        assert improved - baseline == pytest.approx(0.02)


def test_ax003_expands_to_one_scenario_per_rank() -> None:
    registry = build_scenario_registry()

    scenarios = [
        scenario
        for scenario in registry
        if scenario.scenario_id.startswith("AX-003-R")
    ]

    assert len(scenarios) == 5


def test_ax004_has_expected_number_of_perturbations() -> None:
    registry = build_scenario_registry()

    scenarios = [
        scenario
        for scenario in registry
        if scenario.scenario_id.startswith("AX-004-")
    ]

    assert len(scenarios) == 18


def test_all_scenarios_validate() -> None:
    registry = build_scenario_registry()

    for scenario in registry:
        scenario.validate()


def test_validate_registry_rejects_conflicting_population_definitions() -> None:
    population_a = SyntheticPopulation(
        population_id="shared",
        values=(0.90, 0.80),
        description="Population.",
    )

    conflicting_population = SyntheticPopulation(
        population_id="shared",
        values=(0.95, 0.75),
        description="Different football world.",
    )

    scenario_a = SyntheticScenario(
        scenario_id="A",
        scenario_family="stability",
        name="A",
        description="A",
        population_a=population_a,
        population_b=None,
        evaluated_property="test",
        expected_direction="descriptive",
    )

    scenario_b = SyntheticScenario(
        scenario_id="B",
        scenario_family="stability",
        name="B",
        description="B",
        population_a=conflicting_population,
        population_b=None,
        evaluated_property="test",
        expected_direction="descriptive",
    )

    with pytest.raises(
        ValueError,
        match="multiple different population definitions",
    ):
        validate_scenario_registry(
            (
                scenario_a,
                scenario_b,
            )
        )

def test_validate_registry_allows_reused_identical_population() -> None:
    shared_population = _example_population(
        "shared-population"
    )

    scenario_a = SyntheticScenario(
        scenario_id="ST-A",
        scenario_family="stability",
        name="Scenario A",
        description="Scenario A.",
        population_a=shared_population,
        population_b=None,
        evaluated_property="stability",
        expected_direction="descriptive",
    )

    scenario_b = SyntheticScenario(
        scenario_id="ST-B",
        scenario_family="stability",
        name="Scenario B",
        description="Scenario B.",
        population_a=shared_population,
        population_b=None,
        evaluated_property="stability",
        expected_direction="descriptive",
    )

    validate_scenario_registry(
        (
            scenario_a,
            scenario_b,
        )
    )


def test_validate_registry_rejects_conflicting_population_definitions() -> None:
    population_a = SyntheticPopulation(
        population_id="shared-population",
        values=(0.90, 0.80, 0.70),
        description="Original population.",
    )

    conflicting_population = SyntheticPopulation(
        population_id="shared-population",
        values=(0.95, 0.80, 0.70),
        description="Conflicting population.",
    )

    scenario_a = SyntheticScenario(
        scenario_id="ST-A",
        scenario_family="stability",
        name="Scenario A",
        description="Scenario A.",
        population_a=population_a,
        population_b=None,
        evaluated_property="stability",
        expected_direction="descriptive",
    )

    scenario_b = SyntheticScenario(
        scenario_id="ST-B",
        scenario_family="stability",
        name="Scenario B",
        description="Scenario B.",
        population_a=conflicting_population,
        population_b=None,
        evaluated_property="stability",
        expected_direction="descriptive",
    )

    with pytest.raises(
        ValueError,
        match=(
            "multiple different population definitions"
        ),
    ):
        validate_scenario_registry(
            (
                scenario_a,
                scenario_b,
            )
        )


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_scenario_registry_records_match_registry_size() -> None:
    registry = build_scenario_registry()
    records = scenario_registry_records(registry)

    assert len(records) == len(registry)


def test_scenario_registry_records_have_required_fields() -> None:
    records = scenario_registry_records()

    required_fields = {
        "scenario_id",
        "scenario_family",
        "name",
        "description",
        "population_a_id",
        "population_b_id",
        "evaluated_property",
        "expected_direction",
        "binary_pass_expected",
    }

    assert required_fields.issubset(records[0])


def test_population_registry_records_have_one_row_per_value() -> None:
    registry = build_scenario_registry()
    records = population_registry_records(registry)

    expected_count = 0

    for scenario in registry:
        expected_count += len(
            scenario.population_a.values
        )

        if scenario.population_b is not None:
            expected_count += len(
                scenario.population_b.values
            )

    assert len(records) == expected_count


def test_population_registry_records_have_required_fields() -> None:
    records = population_registry_records()

    required_fields = {
        "scenario_id",
        "scenario_family",
        "population_label",
        "population_id",
        "population_description",
        "source_order",
        "player_value",
    }

    assert required_fields.issubset(records[0])


# ---------------------------------------------------------------------
# Frozen numerical constants
# ---------------------------------------------------------------------


def test_registry_tolerances_are_frozen() -> None:
    assert ABSOLUTE_TOLERANCE == pytest.approx(1e-12)
    assert RELATIVE_TOLERANCE == pytest.approx(1e-12)


def test_tolerances_are_finite_and_positive() -> None:
    assert math.isfinite(ABSOLUTE_TOLERANCE)
    assert math.isfinite(RELATIVE_TOLERANCE)

    assert ABSOLUTE_TOLERANCE > 0.0
    assert RELATIVE_TOLERANCE > 0.0
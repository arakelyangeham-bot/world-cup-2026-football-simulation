# run_synthetic_aggregation_benchmark.py

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research.studies.study_089_aggregation_mathematics.aggregation_evaluator import (
    DistributionAggregationEvaluation,
    ScalarAggregationEvaluation,
    evaluate_aggregation,
)

from research.studies.study_089_aggregation_mathematics.aggregation_specifications import (
    AggregationSpecification,
    aggregation_specification_records,
    build_aggregation_specifications,
)

from research.studies.study_089_aggregation_mathematics.axiom_evaluator import (
    axiom_evaluation_to_record,
    evaluate_scalar_axiom,
)

from research.studies.study_089_aggregation_mathematics.synthetic_scenario_registry import (
    ABSOLUTE_TOLERANCE,
    DEFAULT_RANDOM_SEED,
    RELATIVE_TOLERANCE,
    SyntheticScenario,
    build_scenario_registry,
    population_registry_records,
    scenario_registry_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_089_aggregation_mathematics"
)

POPULATION_PATH = (
    OUTPUT_DIRECTORY
    / "synthetic_populations.csv"
)

SCENARIO_REGISTRY_PATH = (
    OUTPUT_DIRECTORY
    / "synthetic_scenario_registry.csv"
)

PARAMETER_REGISTRY_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_parameter_registry.csv"
)

SCALAR_RESULT_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_scenario_results.csv"
)

DISTRIBUTION_RESULT_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_distribution_results.csv"
)

AXIOM_RESULT_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_axiom_results.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_089b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_089b_report.md"
)


EVALUATION_STATUSES = {
    "evaluated",
    "not_applicable",
    "failed",
}

OBSERVED_DIRECTIONS = {
    "equal",
    "increase",
    "decrease",
    "not_available",
}


def classify_direction(
    value_a: float,
    value_b: float,
) -> str:
    """
    Classify the direction from population A to population B.
    """

    if math.isclose(
        value_a,
        value_b,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=ABSOLUTE_TOLERANCE,
    ):
        return "equal"

    if value_b > value_a:
        return "increase"

    return "decrease"


def direction_passes(
    *,
    expected_direction: str,
    observed_direction: str,
) -> bool | None:
    """
    Evaluate directional agreement.

    Descriptive scenarios intentionally receive no binary judgment.
    """

    if expected_direction == "descriptive":
        return None

    if observed_direction == "not_available":
        return False

    if expected_direction == "equal":
        return observed_direction == "equal"

    if expected_direction == "increase":
        return observed_direction == "increase"

    if expected_direction == "decrease":
        return observed_direction == "decrease"

    if expected_direction == "non_decrease":
        return observed_direction in {
            "equal",
            "increase",
        }

    if expected_direction == "non_increase":
        return observed_direction in {
            "equal",
            "decrease",
        }

    raise KeyError(
        "No directional pass rule exists for "
        f"{expected_direction!r}."
    )


def relative_delta(
    *,
    value_a: float,
    value_b: float,
) -> float | None:
    """
    Return relative change from A to B.

    Undefined when the baseline is numerically zero.
    """

    if math.isclose(
        value_a,
        0.0,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=ABSOLUTE_TOLERANCE,
    ):
        return None

    return float(
        (value_b - value_a) / abs(value_a)
    )


def combine_statuses(
    status_a: str,
    status_b: str,
) -> str:
    if status_a not in EVALUATION_STATUSES:
        raise ValueError(
            f"Unknown evaluation status {status_a!r}."
        )

    if status_b not in EVALUATION_STATUSES:
        raise ValueError(
            f"Unknown evaluation status {status_b!r}."
        )

    if "failed" in {
        status_a,
        status_b,
    }:
        return "failed"

    if "not_applicable" in {
        status_a,
        status_b,
    }:
        return "not_applicable"

    return "evaluated"


def build_scalar_result_record(
    *,
    scenario: SyntheticScenario,
    specification: AggregationSpecification,
) -> dict[str, object]:
    evaluation_a = evaluate_aggregation(
        scenario.population_a.values,
        specification=specification,
    )

    if not isinstance(
        evaluation_a,
        ScalarAggregationEvaluation,
    ):
        raise TypeError(
            "Scalar-result builder received a non-scalar "
            f"evaluation for {specification.specification_id!r}."
        )

    if scenario.population_b is None:
        evaluation_b = None
    else:
        raw_evaluation_b = evaluate_aggregation(
            scenario.population_b.values,
            specification=specification,
        )

        if not isinstance(
            raw_evaluation_b,
            ScalarAggregationEvaluation,
        ):
            raise TypeError(
                "Scalar-result builder received a non-scalar "
                "population-B evaluation."
            )

        evaluation_b = raw_evaluation_b

    if evaluation_b is None:
        combined_status = evaluation_a.status
    else:
        combined_status = combine_statuses(
            evaluation_a.status,
            evaluation_b.status,
        )

    value_a = evaluation_a.value
    value_b = (
        evaluation_b.value
        if evaluation_b is not None
        else None
    )

    if (
        combined_status == "evaluated"
        and value_a is not None
        and value_b is not None
    ):
        delta = float(
            value_b - value_a
        )

        absolute_delta = abs(delta)

        relative_change = relative_delta(
            value_a=value_a,
            value_b=value_b,
        )

        observed_direction = classify_direction(
            value_a,
            value_b,
        )
    else:
        delta = None
        absolute_delta = None
        relative_change = None
        observed_direction = "not_available"

    binary_pass = direction_passes(
        expected_direction=(
            scenario.expected_direction
        ),
        observed_direction=observed_direction,
    )

    return {
        "scenario_id": scenario.scenario_id,
        "scenario_family": scenario.scenario_family,
        "scenario_name": scenario.name,
        "evaluated_property":
            scenario.evaluated_property,
        "expected_direction":
            scenario.expected_direction,
        "binary_pass_expected":
            scenario.binary_pass_expected,
        "specification_id":
            specification.specification_id,
        "aggregation_family":
            specification.aggregation_family,
        "display_name":
            specification.display_name,
        "output_type":
            specification.output_type,
        "parameterization":
            specification.to_record()[
                "parameterization"
            ],
        "historical_control":
            specification.historical_control,
        "population_a_id":
            scenario.population_a.population_id,
        "population_b_id": (
            scenario.population_b.population_id
            if scenario.population_b is not None
            else None
        ),
        "population_a_label":
            scenario.comparison_label_a,
        "population_b_label": (
            scenario.comparison_label_b
            if scenario.population_b is not None
            else None
        ),
        "population_a_size":
            evaluation_a.population_size,
        "population_b_size": (
            evaluation_b.population_size
            if evaluation_b is not None
            else None
        ),
        "minimum_population_size":
            evaluation_a.minimum_population_size,
        "status_a": evaluation_a.status,
        "status_b": (
            evaluation_b.status
            if evaluation_b is not None
            else None
        ),
        "comparison_status":
            combined_status,
        "value_a": value_a,
        "value_b": value_b,
        "delta_b_minus_a": delta,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_change,
        "observed_direction":
            observed_direction,
        "direction_pass": binary_pass,
        "error_type_a":
            evaluation_a.error_type,
        "error_message_a":
            evaluation_a.error_message,
        "error_type_b": (
            evaluation_b.error_type
            if evaluation_b is not None
            else None
        ),
        "error_message_b": (
            evaluation_b.error_message
            if evaluation_b is not None
            else None
        ),
        "scenario_notes": scenario.notes,
    }


DISTRIBUTION_FIELDS = (
    "mean",
    "maximum",
    "minimum",
    "value_range",
    "standard_deviation",
    "star_gap",
    "concentration",
)


def build_distribution_result_records(
    *,
    scenario: SyntheticScenario,
    specification: AggregationSpecification,
) -> list[dict[str, object]]:
    evaluation_a = evaluate_aggregation(
        scenario.population_a.values,
        specification=specification,
    )

    if not isinstance(
        evaluation_a,
        DistributionAggregationEvaluation,
    ):
        raise TypeError(
            "Distribution-result builder received a scalar "
            "evaluation."
        )

    if scenario.population_b is None:
        evaluation_b = None
    else:
        raw_evaluation_b = evaluate_aggregation(
            scenario.population_b.values,
            specification=specification,
        )

        if not isinstance(
            raw_evaluation_b,
            DistributionAggregationEvaluation,
        ):
            raise TypeError(
                "Distribution-result builder received a scalar "
                "population-B evaluation."
            )

        evaluation_b = raw_evaluation_b

    if evaluation_b is None:
        combined_status = evaluation_a.status
    else:
        combined_status = combine_statuses(
            evaluation_a.status,
            evaluation_b.status,
        )

    records: list[dict[str, object]] = []

    for diagnostic_name in DISTRIBUTION_FIELDS:
        value_a = getattr(
            evaluation_a,
            diagnostic_name,
        )

        value_b = (
            getattr(
                evaluation_b,
                diagnostic_name,
            )
            if evaluation_b is not None
            else None
        )

        if (
            combined_status == "evaluated"
            and value_a is not None
            and value_b is not None
        ):
            delta = float(
                value_b - value_a
            )

            absolute_delta = abs(delta)

            relative_change = relative_delta(
                value_a=float(value_a),
                value_b=float(value_b),
            )

            observed_direction = classify_direction(
                float(value_a),
                float(value_b),
            )
        else:
            delta = None
            absolute_delta = None
            relative_change = None
            observed_direction = "not_available"

        records.append(
            {
                "scenario_id":
                    scenario.scenario_id,
                "scenario_family":
                    scenario.scenario_family,
                "scenario_name":
                    scenario.name,
                "evaluated_property":
                    scenario.evaluated_property,
                "specification_id":
                    specification.specification_id,
                "aggregation_family":
                    specification.aggregation_family,
                "output_type":
                    specification.output_type,
                "diagnostic_name":
                    diagnostic_name,
                "population_a_id":
                    scenario.population_a.population_id,
                "population_b_id": (
                    scenario.population_b.population_id
                    if scenario.population_b is not None
                    else None
                ),
                "population_a_size":
                    evaluation_a.population_size,
                "population_b_size": (
                    evaluation_b.population_size
                    if evaluation_b is not None
                    else None
                ),
                "minimum_population_size":
                    evaluation_a.minimum_population_size,
                "status_a":
                    evaluation_a.status,
                "status_b": (
                    evaluation_b.status
                    if evaluation_b is not None
                    else None
                ),
                "comparison_status":
                    combined_status,
                "value_a": value_a,
                "value_b": value_b,
                "delta_b_minus_a":
                    delta,
                "absolute_delta":
                    absolute_delta,
                "relative_delta":
                    relative_change,
                "observed_direction":
                    observed_direction,
                "error_type_a":
                    evaluation_a.error_type,
                "error_message_a":
                    evaluation_a.error_message,
                "error_type_b": (
                    evaluation_b.error_type
                    if evaluation_b is not None
                    else None
                ),
                "error_message_b": (
                    evaluation_b.error_message
                    if evaluation_b is not None
                    else None
                ),
                "scenario_notes":
                    scenario.notes,
            }
        )

    return records


def build_benchmark_results(
    *,
    scenarios: tuple[
        SyntheticScenario,
        ...,
    ],
    specifications: tuple[
        AggregationSpecification,
        ...,
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scalar_records: list[
        dict[str, object]
    ] = []

    distribution_records: list[
        dict[str, object]
    ] = []

    for scenario in scenarios:
        for specification in specifications:
            if (
                specification.output_type
                == "distribution_diagnostics"
            ):
                distribution_records.extend(
                    build_distribution_result_records(
                        scenario=scenario,
                        specification=specification,
                    )
                )
            else:
                scalar_records.append(
                    build_scalar_result_record(
                        scenario=scenario,
                        specification=specification,
                    )
                )

    scalar_results = pd.DataFrame(
        scalar_records
    )

    distribution_results = pd.DataFrame(
        distribution_records
    )

    return (
        scalar_results,
        distribution_results,
    )


def build_axiom_results(
    *,
    scenarios: tuple[
        SyntheticScenario,
        ...,
    ],
    specifications: tuple[
        AggregationSpecification,
        ...,
    ],
) -> pd.DataFrame:
    axiom_scenarios = [
        scenario
        for scenario in scenarios
        if scenario.scenario_family == "axiom"
    ]

    scalar_specifications = [
        specification
        for specification in specifications
        if specification.output_type
        != "distribution_diagnostics"
    ]

    records: list[
        dict[str, object]
    ] = []

    for scenario in axiom_scenarios:
        for specification in scalar_specifications:
            evaluation_a = evaluate_aggregation(
                scenario.population_a.values,
                specification=specification,
            )

            if not isinstance(
                evaluation_a,
                ScalarAggregationEvaluation,
            ):
                raise TypeError(
                    "Scalar axiom evaluation received a "
                    "distribution result."
                )

            if scenario.population_b is None:
                evaluation_b = None
            else:
                raw_evaluation_b = evaluate_aggregation(
                    scenario.population_b.values,
                    specification=specification,
                )

                if not isinstance(
                    raw_evaluation_b,
                    ScalarAggregationEvaluation,
                ):
                    raise TypeError(
                        "Scalar axiom population-B evaluation "
                        "received a distribution result."
                    )

                evaluation_b = raw_evaluation_b

            axiom_evaluation = evaluate_scalar_axiom(
                scenario=scenario,
                evaluation_a=evaluation_a,
                evaluation_b=evaluation_b,
            )

            records.append(
                axiom_evaluation_to_record(
                    axiom_evaluation
                )
            )

    output = pd.DataFrame(records)

    return (
        output
        .sort_values(
            [
                "scenario_id",
                "specification_id",
            ]
        )
        .reset_index(drop=True)
    )


def validate_benchmark_outputs(
    *,
    scenarios: tuple[
        SyntheticScenario,
        ...,
    ],
    specifications: tuple[
        AggregationSpecification,
        ...,
    ],
    scalar_results: pd.DataFrame,
    distribution_results: pd.DataFrame,
) -> None:
    scalar_specifications = [
        specification
        for specification in specifications
        if specification.output_type
        != "distribution_diagnostics"
    ]

    distribution_specifications = [
        specification
        for specification in specifications
        if specification.output_type
        == "distribution_diagnostics"
    ]

    expected_scalar_rows = (
        len(scenarios)
        * len(scalar_specifications)
    )

    expected_distribution_rows = (
        len(scenarios)
        * len(distribution_specifications)
        * len(DISTRIBUTION_FIELDS)
    )

    if len(scalar_results) != expected_scalar_rows:
        raise AssertionError(
            "Unexpected scalar benchmark row count. "
            f"Expected {expected_scalar_rows}, "
            f"received {len(scalar_results)}."
        )

    if (
        len(distribution_results)
        != expected_distribution_rows
    ):
        raise AssertionError(
            "Unexpected distribution benchmark row count. "
            f"Expected {expected_distribution_rows}, "
            f"received {len(distribution_results)}."
        )

    if scalar_results.empty:
        raise AssertionError(
            "Scalar benchmark results are empty."
        )

    if distribution_results.empty:
        raise AssertionError(
            "Distribution benchmark results are empty."
        )

    if scalar_results[
        [
            "scenario_id",
            "specification_id",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Scalar results contain duplicate "
            "scenario-specification rows."
        )

    if distribution_results[
        [
            "scenario_id",
            "specification_id",
            "diagnostic_name",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Distribution results contain duplicate "
            "scenario-specification-diagnostic rows."
        )

    invalid_scalar_statuses = (
        set(
            scalar_results[
                "comparison_status"
            ].dropna()
        )
        - EVALUATION_STATUSES
    )

    if invalid_scalar_statuses:
        raise AssertionError(
            "Scalar results contain invalid statuses: "
            f"{sorted(invalid_scalar_statuses)}"
        )

    invalid_distribution_statuses = (
        set(
            distribution_results[
                "comparison_status"
            ].dropna()
        )
        - EVALUATION_STATUSES
    )

    if invalid_distribution_statuses:
        raise AssertionError(
            "Distribution results contain invalid statuses: "
            f"{sorted(invalid_distribution_statuses)}"
        )

    failed_scalar_count = int(
        scalar_results[
            "comparison_status"
        ].eq("failed").sum()
    )

    failed_distribution_count = int(
        distribution_results[
            "comparison_status"
        ].eq("failed").sum()
    )

    if (
        failed_scalar_count > 0
        or failed_distribution_count > 0
    ):
        raise AssertionError(
            "Benchmark contains runtime evaluation failures. "
            f"Scalar failures={failed_scalar_count}, "
            "distribution failures="
            f"{failed_distribution_count}."
        )


def build_metadata(
    *,
    scenarios: tuple[
        SyntheticScenario,
        ...,
    ],
    specifications: tuple[
        AggregationSpecification,
        ...,
    ],
    scalar_results: pd.DataFrame,
    distribution_results: pd.DataFrame,
    axiom_results: pd.DataFrame,
) -> dict[str, Any]:\

    axiom_pass_count = int(
        axiom_results[
            "status"
        ].eq("pass").sum()
    )

    axiom_failure_count = int(
        axiom_results[
            "status"
        ].eq("fail").sum()
    )

    axiom_not_applicable_count = int(
        axiom_results[
            "status"
        ].eq("not_applicable").sum()
    )

    return {
        "study_id": "089B",
        "study_name": (
            "Synthetic Aggregation Benchmark"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "PASS"
            if axiom_failure_count == 0
            else "FAIL"
        ),
        "scenario_count": len(
            scenarios
        ),
        "aggregation_specification_count": len(
            specifications
        ),
        "scalar_result_count": len(
            scalar_results
        ),
        "distribution_result_count": len(
            distribution_results
        ),
        "axiom_result_count": len(
            axiom_results
        ),
        "axiom_pass_count":
            axiom_pass_count,
        "axiom_failure_count":
            axiom_failure_count,
        "axiom_not_applicable_count":
            axiom_not_applicable_count,
        "random_seed":
            DEFAULT_RANDOM_SEED,
        "absolute_tolerance":
            ABSOLUTE_TOLERANCE,
        "relative_tolerance":
            RELATIVE_TOLERANCE,
        "input_scale": "[0, 1]",
        "structural_scenarios_included":
            False,
        "behavioral_rankings_generated":
            False,
        "goal_model_fitted":
            False,
        "prediction_data_used":
            False,
        "production_repository_changed":
            False,
        "production_runtime_changed":
            False,
        "interpretation_boundary": (
            "This benchmark evaluates mathematical and "
            "football-behavioral responses under controlled "
            "synthetic player populations. It does not measure "
            "predictive performance on real matches."
        ),
        "outputs": [
            POPULATION_PATH.name,
            SCENARIO_REGISTRY_PATH.name,
            PARAMETER_REGISTRY_PATH.name,
            SCALAR_RESULT_PATH.name,
            DISTRIBUTION_RESULT_PATH.name,
            AXIOM_RESULT_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    scenarios: tuple[
        SyntheticScenario,
        ...,
    ],
    specifications: tuple[
        AggregationSpecification,
        ...,
    ],
    scalar_results: pd.DataFrame,
    distribution_results: pd.DataFrame,
    axiom_results: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    status_counts = (
        scalar_results[
            "comparison_status"
        ]
        .value_counts()
        .to_dict()
    )

    distribution_status_counts = (
        distribution_results[
            "comparison_status"
        ]
        .value_counts()
        .to_dict()
    )

    axiom_status_counts = (
        axiom_results[
            "status"
        ]
        .value_counts()
        .to_dict()
    )

    axiom_pass_count = int(
        axiom_results[
            "status"
        ].eq("pass").sum()
    )

    axiom_failure_count = int(
        axiom_results[
            "status"
        ].eq("fail").sum()
    )

    axiom_not_applicable_count = int(
        axiom_results[
            "status"
        ].eq("not_applicable").sum()
    )

    report = f"""# Study 089B — Synthetic Aggregation Benchmark

## Purpose

Evaluate the frozen Version 2B aggregation specifications under
controlled synthetic player populations.

## Methodological boundary

This study:

- uses no real player or match data;
- fits no goal model;
- changes no production repository;
- changes no production runtime;
- evaluates one-dimensional synthetic populations only;
- defers structural role scenarios;
- does not yet produce behavioral rankings.

## Benchmark design

- Synthetic scenarios: {len(scenarios)}
- Aggregation specifications: {len(specifications)}
- Scalar comparison rows: {len(scalar_results)}
- Distribution-diagnostic rows: {len(distribution_results)}

## Scalar evaluation statuses

{json.dumps(status_counts, indent=2)}

## Distribution evaluation statuses

{json.dumps(distribution_status_counts, indent=2)}

## Mathematical axioms

- Total axiom evaluation rows: {len(axiom_results)}
- Passing rows: {axiom_pass_count}
- Failing rows: {axiom_failure_count}
- Not-applicable rows: {axiom_not_applicable_count}

## Axiom evaluation statuses

{json.dumps(axiom_status_counts, indent=2)}

## Applicability

Scalar not-applicable rows:

{axiom_not_applicable_count}

These rows represent valid minimum-population incompatibilities, not
mathematical failures. Replacement-depth specifications require ten
players and therefore cannot be evaluated on every five-player
scenario.

## Interpretation

The raw outputs describe:

- mathematical validity;
- local stability;
- elite responsiveness;
- fringe sensitivity;
- distribution separation;
- replacement quality;
- rank-boundary behavior;
- scale response.

Behavioral differences are not automatically classified as good or
bad. They encode competing football hypotheses.

## Result

**OVERALL RESULT: {metadata["status"]}**

The study generated raw scenario-level evidence without introducing
real football data or modifying production behavior.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 089B — SYNTHETIC AGGREGATION BENCHMARK"
    )
    print("=" * 88)

    scenarios = build_scenario_registry()
    specifications = (
        build_aggregation_specifications()
    )

    scalar_results, distribution_results = (
        build_benchmark_results(
            scenarios=scenarios,
            specifications=specifications,
        )
    )

    axiom_results = build_axiom_results(
        scenarios=scenarios,
        specifications=specifications,
    )
    validate_benchmark_outputs(
        scenarios=scenarios,
        specifications=specifications,
        scalar_results=scalar_results,
        distribution_results=(
            distribution_results
        ),
    )

    metadata = build_metadata(
        scenarios=scenarios,
        specifications=specifications,
        scalar_results=scalar_results,
        distribution_results=(
            distribution_results
        ),
        axiom_results=axiom_results,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        population_registry_records(
            scenarios
        )
    ).to_csv(
        POPULATION_PATH,
        index=False,
    )

    pd.DataFrame(
        scenario_registry_records(
            scenarios
        )
    ).to_csv(
        SCENARIO_REGISTRY_PATH,
        index=False,
    )

    pd.DataFrame(
        aggregation_specification_records(
            specifications
        )
    ).to_csv(
        PARAMETER_REGISTRY_PATH,
        index=False,
    )

    scalar_results.to_csv(
        SCALAR_RESULT_PATH,
        index=False,
    )

    distribution_results.to_csv(
        DISTRIBUTION_RESULT_PATH,
        index=False,
    )

    axiom_results.to_csv(
        AXIOM_RESULT_PATH,
        index=False,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        scenarios=scenarios,
        specifications=specifications,
        scalar_results=scalar_results,
        distribution_results=(
            distribution_results
        ),
        axiom_results=axiom_results,
        metadata=metadata,
    )

    print()
    print("Benchmark population")
    print("-" * 88)
    print(
        f"  Scenarios: {len(scenarios)}"
    )
    print(
        "  Aggregation specifications: "
        f"{len(specifications)}"
    )
    print(
        "  Scalar result rows: "
        f"{len(scalar_results)}"
    )
    print(
        "  Distribution result rows: "
        f"{len(distribution_results)}"
    )

    print()
    print("Scalar result statuses")
    print("-" * 88)
    print(
        scalar_results[
            "comparison_status"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Axiom summary")
    print("-" * 88)

    axiom_summary = (
        axiom_results
        .groupby(
            "specification_id",
            as_index=False,
        )
        .agg(
            axiom_rows=(
                "scenario_id",
                "size",
            ),
            axiom_passes=(
                "status",
                lambda values: int(
                    values.eq("pass").sum()
                ),
            ),
            axiom_failures=(
                "status",
                lambda values: int(
                    values.eq("fail").sum()
                ),
            ),
            not_applicable_rows=(
                "status",
                lambda values: int(
                    values.eq(
                        "not_applicable"
                    ).sum()
                ),
            ),
        )
    )

    print(
        axiom_summary.to_string(
            index=False
        )
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Scenario registry loading: PASS")
    print("  Aggregation registry loading: PASS")
    print("  Scalar dispatch: PASS")
    print("  Distribution dispatch: PASS")
    print("  Result-row counts: PASS")
    print("  Duplicate-result audit: PASS")
    print("  Runtime evaluation failures: NONE")
    print("  Goal-model fitting: NONE")
    print("  Prediction data usage: NONE")
    print("  Production mutation: NONE")

    print()
    print("=" * 88)
    print(
        f"OVERALL RESULT: {metadata['status']}"
    )
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
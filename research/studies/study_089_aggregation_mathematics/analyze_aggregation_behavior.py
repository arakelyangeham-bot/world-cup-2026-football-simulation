# analyze_aggregation_behavior.py

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_089_aggregation_mathematics"
)

SCALAR_RESULT_PATH = (
    INPUT_DIRECTORY
    / "aggregation_scenario_results.csv"
)

PARAMETER_REGISTRY_PATH = (
    INPUT_DIRECTORY
    / "aggregation_parameter_registry.csv"
)

BEHAVIOR_METRIC_PATH = (
    INPUT_DIRECTORY
    / "aggregation_behavior_metrics.csv"
)

BEHAVIOR_SUMMARY_PATH = (
    INPUT_DIRECTORY
    / "aggregation_behavior_summary.csv"
)

BEHAVIOR_METADATA_PATH = (
    INPUT_DIRECTORY
    / "study_089b2_behavior_metadata.json"
)

BEHAVIOR_REPORT_PATH = (
    INPUT_DIRECTORY
    / "study_089b2_behavior_report.md"
)


REQUIRED_SCALAR_COLUMNS = {
    "scenario_id",
    "scenario_family",
    "scenario_name",
    "evaluated_property",
    "specification_id",
    "aggregation_family",
    "display_name",
    "output_type",
    "comparison_status",
    "value_a",
    "value_b",
    "delta_b_minus_a",
    "absolute_delta",
    "relative_delta",
    "observed_direction",
}


PRIMARY_SCENARIO_METRICS = {
    "ST-001": (
        "stability",
        "weakest_primary_downgrade",
        "absolute_delta",
    ),
    "ST-002": (
        "stability",
        "elite_primary_downgrade",
        "absolute_delta",
    ),
    "ST-003": (
        "stability",
        "threshold_order_swap",
        "absolute_delta",
    ),
    "ER-001": (
        "elite_responsiveness",
        "elite_player_removal",
        "absolute_delta",
    ),
    "ER-002": (
        "elite_responsiveness",
        "ordinary_starter_removal",
        "absolute_delta",
    ),
    "ER-003": (
        "elite_responsiveness",
        "superstar_addition",
        "delta_b_minus_a",
    ),
    "ER-004": (
        "fringe_sensitivity",
        "weak_fringe_addition",
        "absolute_delta",
    ),
    "DP-001": (
        "depth_behavior",
        "weak_fringe_roster_expansion",
        "absolute_delta",
    ),
    "DP-002": (
        "depth_behavior",
        "replacement_unit_improvement",
        "delta_b_minus_a",
    ),
    "DP-003": (
        "scale_behavior",
        "uniform_population_improvement",
        "delta_b_minus_a",
    ),
    "RB-001": (
        "rank_boundary",
        "single_threshold_crossing",
        "absolute_delta",
    ),
    "RB-002": (
        "rank_boundary",
        "tied_boundary_permutation",
        "absolute_delta",
    ),
    "RB-003": (
        "rank_boundary",
        "clustered_threshold_perturbation",
        "absolute_delta",
    ),
    "SC-001": (
        "scale_behavior",
        "uniform_additive_shift",
        "delta_b_minus_a",
    ),
    "SC-002": (
        "scale_behavior",
        "uniform_multiplicative_shift",
        "relative_delta",
    ),
    "SC-003": (
        "scale_behavior",
        "normalized_scale_equivalence",
        "absolute_delta",
    ),
}


def require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(
        required_columns - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: "
            f"{missing}"
        )


def safe_ratio(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    if numerator is None or denominator is None:
        return None

    if pd.isna(numerator) or pd.isna(denominator):
        return None

    denominator_value = float(denominator)

    if math.isclose(
        denominator_value,
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return None

    return float(
        float(numerator)
        / denominator_value
    )


def selected_metric_value(
    row: pd.Series,
    *,
    source_column: str,
) -> float | None:
    value = row[source_column]

    if pd.isna(value):
        return None

    return float(value)


def build_direct_behavior_metrics(
    scalar_results: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    selected = scalar_results.loc[
        scalar_results[
            "scenario_id"
        ].isin(PRIMARY_SCENARIO_METRICS)
    ].copy()

    for row in selected.itertuples(
        index=False
    ):
        (
            metric_family,
            metric_name,
            source_column,
        ) = PRIMARY_SCENARIO_METRICS[
            row.scenario_id
        ]

        raw_value = getattr(
            row,
            source_column,
        )

        metric_value = (
            None
            if pd.isna(raw_value)
            else float(raw_value)
        )

        records.append(
            {
                "specification_id":
                    row.specification_id,
                "aggregation_family":
                    row.aggregation_family,
                "display_name":
                    row.display_name,
                "output_type":
                    row.output_type,
                "metric_family":
                    metric_family,
                "metric_name":
                    metric_name,
                "scenario_id":
                    row.scenario_id,
                "scenario_name":
                    row.scenario_name,
                "comparison_status":
                    row.comparison_status,
                "metric_value":
                    metric_value,
                "source_column":
                    source_column,
                "observed_direction":
                    row.observed_direction,
                "interpretation": (
                    "Smaller absolute values indicate greater "
                    "stability or lower sensitivity."
                    if source_column == "absolute_delta"
                    else (
                        "Positive values indicate the modified "
                        "population received a larger output."
                    )
                ),
            }
        )

    return pd.DataFrame(records)


def metric_lookup(
    direct_metrics: pd.DataFrame,
    *,
    specification_id: str,
    metric_name: str,
) -> float | None:
    rows = direct_metrics.loc[
        direct_metrics[
            "specification_id"
        ].eq(specification_id)
        & direct_metrics[
            "metric_name"
        ].eq(metric_name)
        & direct_metrics[
            "comparison_status"
        ].eq("evaluated")
    ]

    if rows.empty:
        return None

    value = rows.iloc[0][
        "metric_value"
    ]

    if pd.isna(value):
        return None

    return float(value)


def build_derived_behavior_metrics(
    direct_metrics: pd.DataFrame,
    parameter_registry: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for specification in (
        parameter_registry
        .sort_values("specification_id")
        .itertuples(index=False)
    ):
        specification_id = (
            specification.specification_id
        )

        weakest_downgrade = metric_lookup(
            direct_metrics,
            specification_id=specification_id,
            metric_name=(
                "weakest_primary_downgrade"
            ),
        )

        elite_downgrade = metric_lookup(
            direct_metrics,
            specification_id=specification_id,
            metric_name=(
                "elite_primary_downgrade"
            ),
        )

        elite_removal = metric_lookup(
            direct_metrics,
            specification_id=specification_id,
            metric_name="elite_player_removal",
        )

        ordinary_removal = metric_lookup(
            direct_metrics,
            specification_id=specification_id,
            metric_name=(
                "ordinary_starter_removal"
            ),
        )

        superstar_addition = metric_lookup(
            direct_metrics,
            specification_id=specification_id,
            metric_name="superstar_addition",
        )

        fringe_addition = metric_lookup(
            direct_metrics,
            specification_id=specification_id,
            metric_name="weak_fringe_addition",
        )

        threshold_values = [
            metric_lookup(
                direct_metrics,
                specification_id=specification_id,
                metric_name=metric_name,
            )
            for metric_name in (
                "single_threshold_crossing",
                "tied_boundary_permutation",
                "clustered_threshold_perturbation",
            )
        ]

        finite_threshold_values = [
            float(value)
            for value in threshold_values
            if value is not None
            and not pd.isna(value)
        ]

        derived_metrics = (
            (
                "elite_responsiveness",
                "elite_to_weakest_local_sensitivity_ratio",
                safe_ratio(
                    elite_downgrade,
                    weakest_downgrade,
                ),
                (
                    "Values above one indicate that a small "
                    "elite-player downgrade matters more than "
                    "an equally sized downgrade to the weakest "
                    "primary contributor."
                ),
            ),
            (
                "elite_responsiveness",
                "elite_to_ordinary_removal_ratio",
                safe_ratio(
                    elite_removal,
                    ordinary_removal,
                ),
                (
                    "Values above one indicate stronger response "
                    "to elite-player removal than ordinary-player "
                    "removal."
                ),
            ),
            (
                "elite_responsiveness",
                "superstar_to_fringe_addition_ratio",
                safe_ratio(
                    abs(superstar_addition)
                    if superstar_addition is not None
                    else None,
                    fringe_addition,
                ),
                (
                    "Large values indicate strong superstar "
                    "responsiveness relative to weak-fringe "
                    "sensitivity."
                ),
            ),
            (
                "rank_boundary",
                "mean_rank_boundary_sensitivity",
                (
                    float(
                        sum(finite_threshold_values)
                        / len(finite_threshold_values)
                    )
                    if finite_threshold_values
                    else None
                ),
                (
                    "Mean absolute output change across the three "
                    "rank-boundary scenarios."
                ),
            ),
            (
                "rank_boundary",
                "maximum_rank_boundary_sensitivity",
                (
                    max(finite_threshold_values)
                    if finite_threshold_values
                    else None
                ),
                (
                    "Largest absolute output change across the "
                    "rank-boundary scenarios."
                ),
            ),
        )

        for (
            metric_family,
            metric_name,
            metric_value,
            interpretation,
        ) in derived_metrics:
            records.append(
                {
                    "specification_id":
                        specification_id,
                    "aggregation_family":
                        specification.aggregation_family,
                    "display_name":
                        specification.display_name,
                    "output_type":
                        specification.output_type,
                    "metric_family":
                        metric_family,
                    "metric_name":
                        metric_name,
                    "scenario_id":
                        None,
                    "scenario_name":
                        None,
                    "comparison_status": (
                        "evaluated"
                        if metric_value is not None
                        else "not_available"
                    ),
                    "metric_value":
                        metric_value,
                    "source_column":
                        "derived",
                    "observed_direction":
                        None,
                    "interpretation":
                        interpretation,
                }
            )

    return pd.DataFrame(records)


def build_behavior_summary(
    behavior_metrics: pd.DataFrame,
    parameter_registry: pd.DataFrame,
) -> pd.DataFrame:
    evaluated = behavior_metrics.loc[
        behavior_metrics[
            "comparison_status"
        ].eq("evaluated")
    ].copy()

    metric_wide = (
        evaluated
        .pivot_table(
            index="specification_id",
            columns="metric_name",
            values="metric_value",
            aggfunc="first",
        )
        .reset_index()
    )

    summary = parameter_registry.merge(
        metric_wide,
        on="specification_id",
        how="left",
        validate="one_to_one",
    )

    return (
        summary
        .sort_values(
            [
                "output_type",
                "aggregation_family",
                "specification_id",
            ]
        )
        .reset_index(drop=True)
    )


def validate_behavior_outputs(
    *,
    behavior_metrics: pd.DataFrame,
    behavior_summary: pd.DataFrame,
    parameter_registry: pd.DataFrame,
) -> None:
    if behavior_metrics.empty:
        raise AssertionError(
            "Behavior metric output is empty."
        )

    if behavior_summary.empty:
        raise AssertionError(
            "Behavior summary output is empty."
        )

    if behavior_summary[
        "specification_id"
    ].duplicated().any():
        raise AssertionError(
            "Behavior summary contains duplicate "
            "specification IDs."
        )

    expected_specifications = set(
        parameter_registry[
            "specification_id"
        ]
    )

    observed_specifications = set(
        behavior_summary[
            "specification_id"
        ]
    )

    if observed_specifications != expected_specifications:
        raise AssertionError(
            "Behavior summary does not cover the complete "
            "aggregation specification registry."
        )

    invalid_metric_values = (
        behavior_metrics[
            "metric_value"
        ]
        .dropna()
        .map(
            lambda value: not math.isfinite(
                float(value)
            )
        )
    )

    if invalid_metric_values.any():
        raise AssertionError(
            "Behavior metrics contain non-finite values."
        )


def build_metadata(
    *,
    behavior_metrics: pd.DataFrame,
    behavior_summary: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "study_id": "089B2",
        "study_name": (
            "Synthetic Aggregation Behavioral Analysis"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "behavior_metric_row_count":
            len(behavior_metrics),
        "aggregation_summary_row_count":
            len(behavior_summary),
        "ranking_generated": False,
        "composite_score_generated": False,
        "real_football_data_used": False,
        "goal_model_fitted": False,
        "production_runtime_changed": False,
        "interpretation_boundary": (
            "Metrics describe synthetic behavioral responses. "
            "They do not determine predictive superiority."
        ),
        "outputs": [
            BEHAVIOR_METRIC_PATH.name,
            BEHAVIOR_SUMMARY_PATH.name,
            BEHAVIOR_METADATA_PATH.name,
            BEHAVIOR_REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    behavior_metrics: pd.DataFrame,
    behavior_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    evaluated_count = int(
        behavior_metrics[
            "comparison_status"
        ].eq("evaluated").sum()
    )

    unavailable_count = int(
        behavior_metrics[
            "comparison_status"
        ].ne("evaluated").sum()
    )

    metric_family_counts = (
        behavior_metrics[
            "metric_family"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    report = f"""# Study 089B2 — Aggregation Behavioral Analysis

## Purpose

Transform the raw Study 089B synthetic benchmark output into
interpretable behavioral metrics.

## Methodological boundary

This analysis:

- uses only the frozen synthetic benchmark outputs;
- introduces no new aggregation formulas;
- changes no scenario definitions;
- creates no composite score;
- produces no overall aggregation ranking;
- uses no real player or match data;
- makes no claim about predictive superiority.

## Output coverage

- Behavior metric rows: {len(behavior_metrics)}
- Evaluated metric rows: {evaluated_count}
- Unavailable or not-applicable rows: {unavailable_count}
- Aggregation summary rows: {len(behavior_summary)}

## Metric families

{json.dumps(metric_family_counts, indent=2)}

## Interpretation

The outputs describe dimensions such as:

- local stability;
- elite-player responsiveness;
- fringe-player sensitivity;
- replacement-unit response;
- rank-boundary sensitivity;
- additive and multiplicative scale behavior.

A larger value is not universally better.

For example:

- larger elite responsiveness may be desirable;
- larger fringe sensitivity may be undesirable;
- larger rank-boundary sensitivity may indicate instability;
- depth metrics answer a different question from primary-strength
  aggregators.

## Result

**OVERALL RESULT: {metadata["status"]}**

The behavioral metric layer was generated successfully without
ranking or promoting any aggregation method.
"""

    BEHAVIOR_REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 089B2 — AGGREGATION BEHAVIORAL ANALYSIS"
    )
    print("=" * 88)

    scalar_results = pd.read_csv(
        SCALAR_RESULT_PATH
    )

    parameter_registry = pd.read_csv(
        PARAMETER_REGISTRY_PATH
    )

    require_columns(
        scalar_results,
        REQUIRED_SCALAR_COLUMNS,
        frame_name="Scalar benchmark results",
    )

    direct_metrics = (
        build_direct_behavior_metrics(
            scalar_results
        )
    )

    derived_metrics = (
        build_derived_behavior_metrics(
            direct_metrics,
            parameter_registry,
        )
    )

    behavior_metrics = pd.concat(
        [
            direct_metrics,
            derived_metrics,
        ],
        ignore_index=True,
    )

    behavior_summary = (
        build_behavior_summary(
            behavior_metrics,
            parameter_registry,
        )
    )

    validate_behavior_outputs(
        behavior_metrics=behavior_metrics,
        behavior_summary=behavior_summary,
        parameter_registry=parameter_registry,
    )

    metadata = build_metadata(
        behavior_metrics=behavior_metrics,
        behavior_summary=behavior_summary,
    )

    behavior_metrics.to_csv(
        BEHAVIOR_METRIC_PATH,
        index=False,
    )

    behavior_summary.to_csv(
        BEHAVIOR_SUMMARY_PATH,
        index=False,
    )

    BEHAVIOR_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        behavior_metrics=behavior_metrics,
        behavior_summary=behavior_summary,
        metadata=metadata,
    )

    print()
    print("Behavior metric coverage")
    print("-" * 88)
    print(
        behavior_metrics[
            "metric_family"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Evaluation statuses")
    print("-" * 88)
    print(
        behavior_metrics[
            "comparison_status"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Output summary")
    print("-" * 88)
    print(
        f"  Behavior metric rows: "
        f"{len(behavior_metrics)}"
    )
    print(
        f"  Aggregation summary rows: "
        f"{len(behavior_summary)}"
    )
    print("  Composite score: NOT GENERATED")
    print("  Overall ranking: NOT GENERATED")
    print("  Real football data: NOT USED")
    print("  Production behavior: UNCHANGED")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{INPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
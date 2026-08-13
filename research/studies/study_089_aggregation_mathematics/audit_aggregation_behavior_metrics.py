# audit_aggregation_behavior_metrics.py

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_089_aggregation_mathematics"
)

BEHAVIOR_METRIC_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_behavior_metrics.csv"
)

BEHAVIOR_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_behavior_summary.csv"
)

DISTRIBUTION_RESULT_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_distribution_results.csv"
)

AUDITED_BEHAVIOR_METRIC_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_behavior_metrics_audited.csv"
)

AUDITED_SCALAR_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_scalar_behavior_summary.csv"
)

DISTRIBUTION_BEHAVIOR_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_distribution_behavior_summary.csv"
)

AUDIT_METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_089b2a_behavior_audit_metadata.json"
)

AUDIT_REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_089b2a_behavior_audit_report.md"
)


ABSOLUTE_TOLERANCE = 1e-12
RELATIVE_TOLERANCE = 1e-12


REQUIRED_BEHAVIOR_COLUMNS = {
    "specification_id",
    "aggregation_family",
    "display_name",
    "output_type",
    "metric_family",
    "metric_name",
    "comparison_status",
    "metric_value",
}

REQUIRED_SUMMARY_COLUMNS = {
    "specification_id",
    "aggregation_family",
    "display_name",
    "output_type",
}

REQUIRED_DISTRIBUTION_COLUMNS = {
    "scenario_id",
    "scenario_family",
    "specification_id",
    "aggregation_family",
    "output_type",
    "diagnostic_name",
    "comparison_status",
    "value_a",
    "value_b",
    "delta_b_minus_a",
    "absolute_delta",
    "relative_delta",
    "observed_direction",
}


def require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    *,
    frame_name: str,
) -> None:
    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required columns: "
            f"{missing_columns}"
        )


def finite_or_none(
    value: object,
) -> float | None:
    if value is None or pd.isna(value):
        return None

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def close_to_zero(
    value: float | None,
) -> bool:
    if value is None:
        return False

    return math.isclose(
        value,
        0.0,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=ABSOLUTE_TOLERANCE,
    )


def metric_value_lookup(
    behavior_metrics: pd.DataFrame,
    *,
    specification_id: str,
    metric_name: str,
) -> float | None:
    selected = behavior_metrics.loc[
        behavior_metrics[
            "specification_id"
        ].eq(specification_id)
        & behavior_metrics[
            "metric_name"
        ].eq(metric_name)
        & behavior_metrics[
            "comparison_status"
        ].eq("evaluated")
    ]

    if selected.empty:
        return None

    return finite_or_none(
        selected.iloc[0]["metric_value"]
    )


def build_fringe_immunity_metrics(
    behavior_metrics: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    specification_metadata = (
        behavior_metrics[
            [
                "specification_id",
                "aggregation_family",
                "display_name",
                "output_type",
            ]
        ]
        .drop_duplicates()
        .sort_values("specification_id")
    )

    for specification in (
        specification_metadata
        .itertuples(index=False)
    ):
        superstar_addition = metric_value_lookup(
            behavior_metrics,
            specification_id=(
                specification.specification_id
            ),
            metric_name="superstar_addition",
        )

        fringe_addition = metric_value_lookup(
            behavior_metrics,
            specification_id=(
                specification.specification_id
            ),
            metric_name="weak_fringe_addition",
        )

        superstar_responsive = (
            superstar_addition is not None
            and superstar_addition
            > ABSOLUTE_TOLERANCE
        )

        fringe_immune = (
            superstar_responsive
            and fringe_addition is not None
            and close_to_zero(fringe_addition)
        )

        if (
            superstar_addition is None
            or fringe_addition is None
        ):
            comparison_status = "not_available"
            metric_value = None
            interpretation = (
                "Fringe immunity could not be evaluated because "
                "one or more source metrics were unavailable."
            )
        else:
            comparison_status = "evaluated"
            metric_value = float(fringe_immune)
            interpretation = (
                "A value of 1 indicates positive superstar "
                "responsiveness together with zero weak-fringe "
                "sensitivity under the frozen scenarios. "
                "A value of 0 indicates that this strict condition "
                "was not satisfied."
            )

        records.append(
            {
                "specification_id":
                    specification.specification_id,
                "aggregation_family":
                    specification.aggregation_family,
                "display_name":
                    specification.display_name,
                "output_type":
                    specification.output_type,
                "metric_family":
                    "fringe_sensitivity",
                "metric_name":
                    "fringe_immunity",
                "scenario_id":
                    None,
                "scenario_name":
                    None,
                "comparison_status":
                    comparison_status,
                "metric_value":
                    metric_value,
                "source_column":
                    "derived_boolean",
                "observed_direction":
                    None,
                "interpretation":
                    interpretation,
            }
        )

    return pd.DataFrame(records)


def remove_misleading_zero_denominator_ratio(
    behavior_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve the ratio where it is finite and meaningful.

    When weak-fringe sensitivity is exactly zero, the ratio is
    undefined and fringe_immunity becomes the explicit interpretation.
    """

    output = behavior_metrics.copy()

    target_mask = output[
        "metric_name"
    ].eq("superstar_to_fringe_addition_ratio")

    for index in output.index[target_mask]:
        specification_id = str(
            output.at[
                index,
                "specification_id",
            ]
        )

        fringe_addition = metric_value_lookup(
            output,
            specification_id=specification_id,
            metric_name="weak_fringe_addition",
        )

        if close_to_zero(fringe_addition):
            output.at[
                index,
                "comparison_status",
            ] = "undefined_zero_denominator"

            output.at[
                index,
                "metric_value",
            ] = None

            output.at[
                index,
                "interpretation",
            ] = (
                "Undefined because weak-fringe sensitivity is "
                "zero. Consult fringe_immunity instead of "
                "interpreting this as missing behavioral evidence."
            )

    return output


def build_scalar_behavior_summary(
    *,
    behavior_metrics: pd.DataFrame,
    original_summary: pd.DataFrame,
) -> pd.DataFrame:
    scalar_registry = original_summary.loc[
        original_summary[
            "output_type"
        ].ne("distribution_diagnostics")
    ][
        [
            "specification_id",
            "aggregation_family",
            "display_name",
            "output_type",
            "parameterization",
            "description",
            "historical_control",
        ]
    ].copy()

    evaluated = behavior_metrics.loc[
        behavior_metrics[
            "comparison_status"
        ].isin(
            {
                "evaluated",
                "undefined_zero_denominator",
            }
        )
    ].copy()

    metric_wide = (
        evaluated
        .pivot_table(
            index="specification_id",
            columns="metric_name",
            values="metric_value",
            aggfunc="first",
            dropna=False,
        )
        .reset_index()
    )

    summary = scalar_registry.merge(
        metric_wide,
        on="specification_id",
        how="left",
        validate="one_to_one",
    )

    fringe_immunity_rows = (
        behavior_metrics.loc[
            behavior_metrics[
                "metric_name"
            ].eq("fringe_immunity")
        ][
            [
                "specification_id",
                "metric_value",
            ]
        ]
        .rename(
            columns={
                "metric_value":
                    "fringe_immunity"
            }
        )
    )

    if "fringe_immunity" not in summary.columns:
        summary = summary.merge(
            fringe_immunity_rows,
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


def build_distribution_behavior_summary(
    distribution_results: pd.DataFrame,
) -> pd.DataFrame:
    selected_scenarios = {
        "DS-001",
        "DS-002",
        "DS-003",
        "DS-004",
        "ER-004",
        "DP-001",
        "DP-002",
        "DP-003",
        "RB-002",
        "SC-001",
        "SC-002",
        "SC-003",
    }

    selected = distribution_results.loc[
        distribution_results[
            "scenario_id"
        ].isin(selected_scenarios)
    ].copy()

    selected["metric_name"] = (
        selected["scenario_id"]
        .astype(str)
        + "__"
        + selected["diagnostic_name"]
        .astype(str)
    )

    summary = (
        selected
        .pivot_table(
            index=[
                "specification_id",
                "aggregation_family",
                "output_type",
            ],
            columns="metric_name",
            values="delta_b_minus_a",
            aggfunc="first",
        )
        .reset_index()
    )

    scenario_status = (
        selected
        .groupby(
            [
                "specification_id",
                "scenario_id",
            ],
            as_index=False,
        )
        .agg(
            comparison_status=(
                "comparison_status",
                "first",
            )
        )
    )

    evaluated_scenario_count = (
        scenario_status
        .groupby(
            "specification_id",
            as_index=False,
        )
        .agg(
            evaluated_distribution_scenarios=(
                "comparison_status",
                lambda values: int(
                    values.eq("evaluated").sum()
                ),
            ),
            not_applicable_distribution_scenarios=(
                "comparison_status",
                lambda values: int(
                    values.eq(
                        "not_applicable"
                    ).sum()
                ),
            ),
            failed_distribution_scenarios=(
                "comparison_status",
                lambda values: int(
                    values.eq("failed").sum()
                ),
            ),
        )
    )

    summary = summary.merge(
        evaluated_scenario_count,
        on="specification_id",
        how="left",
        validate="one_to_one",
    )

    return (
        summary
        .sort_values("specification_id")
        .reset_index(drop=True)
    )


def validate_audited_outputs(
    *,
    audited_metrics: pd.DataFrame,
    scalar_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
) -> None:
    if audited_metrics.empty:
        raise AssertionError(
            "Audited behavior metric output is empty."
        )

    if scalar_summary.empty:
        raise AssertionError(
            "Scalar behavior summary is empty."
        )

    if distribution_summary.empty:
        raise AssertionError(
            "Distribution behavior summary is empty."
        )

    if scalar_summary[
        "specification_id"
    ].duplicated().any():
        raise AssertionError(
            "Scalar summary contains duplicate specification IDs."
        )

    if distribution_summary[
        "specification_id"
    ].duplicated().any():
        raise AssertionError(
            "Distribution summary contains duplicate "
            "specification IDs."
        )

    distribution_in_scalar = scalar_summary[
        "output_type"
    ].eq("distribution_diagnostics").any()

    if distribution_in_scalar:
        raise AssertionError(
            "Distribution diagnostics leaked into the scalar "
            "behavior summary."
        )

    fringe_rows = audited_metrics.loc[
        audited_metrics[
            "metric_name"
        ].eq("fringe_immunity")
    ]

    if fringe_rows.empty:
        raise AssertionError(
            "Audited metrics do not contain fringe_immunity."
        )

    finite_metrics = (
        audited_metrics[
            "metric_value"
        ]
        .dropna()
        .map(
            lambda value: math.isfinite(
                float(value)
            )
        )
    )

    if not finite_metrics.all():
        raise AssertionError(
            "Audited behavior metrics contain non-finite values."
        )


def build_metadata(
    *,
    audited_metrics: pd.DataFrame,
    scalar_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
) -> dict[str, Any]:
    status_counts = (
        audited_metrics[
            "comparison_status"
        ]
        .value_counts()
        .to_dict()
    )

    return {
        "study_id": "089B2A",
        "study_name": (
            "Aggregation Behavioral Metric Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "audited_behavior_metric_row_count":
            len(audited_metrics),
        "scalar_summary_row_count":
            len(scalar_summary),
        "distribution_summary_row_count":
            len(distribution_summary),
        "behavior_metric_status_counts":
            status_counts,
        "fringe_immunity_added": True,
        "zero_denominator_ratio_policy": (
            "Ratios with zero fringe sensitivity remain null and "
            "are labelled undefined_zero_denominator. "
            "fringe_immunity carries the substantive meaning."
        ),
        "distribution_behavior_separated":
            True,
        "ranking_generated":
            False,
        "composite_score_generated":
            False,
        "real_football_data_used":
            False,
        "goal_model_fitted":
            False,
        "production_runtime_changed":
            False,
        "outputs": [
            AUDITED_BEHAVIOR_METRIC_PATH.name,
            AUDITED_SCALAR_SUMMARY_PATH.name,
            DISTRIBUTION_BEHAVIOR_SUMMARY_PATH.name,
            AUDIT_METADATA_PATH.name,
            AUDIT_REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    audited_metrics: pd.DataFrame,
    scalar_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    status_counts = (
        audited_metrics[
            "comparison_status"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    fringe_immunity_rows = audited_metrics.loc[
        audited_metrics[
            "metric_name"
        ].eq("fringe_immunity")
        & audited_metrics[
            "metric_value"
        ].eq(1.0)
    ]

    report = f"""# Study 089B2A — Behavioral Metric Audit

## Purpose

Audit the first Study 089B2 behavioral metric layer before any
interpretive ranking or real-data benchmark.

## Corrections

### Explicit fringe immunity

A positive superstar response combined with zero weak-fringe response
is now represented by `fringe_immunity = 1`.

The corresponding superstar-to-fringe ratio remains undefined because
its denominator is zero. It is no longer treated as generic missing
evidence.

### Scalar and distribution separation

Scalar-strength and depth specifications are summarized separately
from the distribution-diagnostic specification.

Distribution behavior is now written to its own summary rather than
appearing as an empty scalar-summary row.

### Applicability reporting

Counts now refer directly to behavioral metric statuses rather than
reusing axiom applicability totals.

## Coverage

- Audited behavior metric rows: {len(audited_metrics)}
- Scalar summary rows: {len(scalar_summary)}
- Distribution summary rows: {len(distribution_summary)}
- Specifications with strict fringe immunity:
  {len(fringe_immunity_rows)}

## Behavioral metric statuses

{json.dumps(status_counts, indent=2)}

## Methodological boundary

This audit:

- changes no aggregation formula;
- changes no synthetic scenario;
- changes no frozen aggregation specification;
- produces no composite score;
- produces no overall ranking;
- uses no real football data;
- makes no claim about predictive superiority.

## Result

**OVERALL RESULT: {metadata["status"]}**

The behavioral evidence is now structured for interpretation without
conflating undefined ratios, scalar strength, and distribution
diagnostics.
"""

    AUDIT_REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 089B2A — AGGREGATION BEHAVIORAL METRIC AUDIT"
    )
    print("=" * 88)

    behavior_metrics = pd.read_csv(
        BEHAVIOR_METRIC_PATH
    )

    behavior_summary = pd.read_csv(
        BEHAVIOR_SUMMARY_PATH
    )

    distribution_results = pd.read_csv(
        DISTRIBUTION_RESULT_PATH
    )

    require_columns(
        behavior_metrics,
        REQUIRED_BEHAVIOR_COLUMNS,
        frame_name="Behavior metrics",
    )

    require_columns(
        behavior_summary,
        REQUIRED_SUMMARY_COLUMNS,
        frame_name="Behavior summary",
    )

    require_columns(
        distribution_results,
        REQUIRED_DISTRIBUTION_COLUMNS,
        frame_name="Distribution results",
    )

    corrected_metrics = (
        remove_misleading_zero_denominator_ratio(
            behavior_metrics
        )
    )

    fringe_immunity_metrics = (
        build_fringe_immunity_metrics(
            corrected_metrics
        )
    )

    audited_metrics = pd.concat(
        [
            corrected_metrics,
            fringe_immunity_metrics,
        ],
        ignore_index=True,
    )

    scalar_summary = build_scalar_behavior_summary(
        behavior_metrics=audited_metrics,
        original_summary=behavior_summary,
    )

    distribution_summary = (
        build_distribution_behavior_summary(
            distribution_results
        )
    )

    validate_audited_outputs(
        audited_metrics=audited_metrics,
        scalar_summary=scalar_summary,
        distribution_summary=distribution_summary,
    )

    metadata = build_metadata(
        audited_metrics=audited_metrics,
        scalar_summary=scalar_summary,
        distribution_summary=distribution_summary,
    )

    audited_metrics.to_csv(
        AUDITED_BEHAVIOR_METRIC_PATH,
        index=False,
    )

    scalar_summary.to_csv(
        AUDITED_SCALAR_SUMMARY_PATH,
        index=False,
    )

    distribution_summary.to_csv(
        DISTRIBUTION_BEHAVIOR_SUMMARY_PATH,
        index=False,
    )

    AUDIT_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        audited_metrics=audited_metrics,
        scalar_summary=scalar_summary,
        distribution_summary=distribution_summary,
        metadata=metadata,
    )

    print()
    print("Behavior status counts")
    print("-" * 88)
    print(
        audited_metrics[
            "comparison_status"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Output summary")
    print("-" * 88)
    print(
        f"  Audited metric rows: "
        f"{len(audited_metrics)}"
    )
    print(
        f"  Scalar summary rows: "
        f"{len(scalar_summary)}"
    )
    print(
        f"  Distribution summary rows: "
        f"{len(distribution_summary)}"
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
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
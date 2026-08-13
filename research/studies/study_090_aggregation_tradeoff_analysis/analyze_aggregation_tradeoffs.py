# analyze_aggregation_tradeoffs.py

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

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_090_aggregation_tradeoff_analysis"
)

SCALAR_SUMMARY_PATH = (
    INPUT_DIRECTORY
    / "aggregation_scalar_behavior_summary.csv"
)

DISTRIBUTION_SUMMARY_PATH = (
    INPUT_DIRECTORY
    / "aggregation_distribution_behavior_summary.csv"
)

AUDITED_METRIC_PATH = (
    INPUT_DIRECTORY
    / "aggregation_behavior_metrics_audited.csv"
)

TRADEOFF_PROFILE_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_tradeoff_profiles.csv"
)

FAMILY_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_family_tradeoffs.csv"
)

QUESTION_TABLE_PATH = (
    OUTPUT_DIRECTORY
    / "aggregation_research_question_table.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_090_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_090_report.md"
)


ABSOLUTE_TOLERANCE = 1e-12


REQUIRED_SCALAR_COLUMNS = {
    "specification_id",
    "aggregation_family",
    "display_name",
    "output_type",
    "parameterization",
    "historical_control",
}

REQUIRED_METRIC_COLUMNS = {
    "specification_id",
    "aggregation_family",
    "display_name",
    "output_type",
    "metric_family",
    "metric_name",
    "comparison_status",
    "metric_value",
}


PROFILE_METRICS = (
    "elite_to_weakest_local_sensitivity_ratio",
    "elite_to_ordinary_removal_ratio",
    "superstar_addition",
    "weak_fringe_addition",
    "fringe_immunity",
    "weakest_primary_downgrade",
    "threshold_order_swap",
    "mean_rank_boundary_sensitivity",
    "maximum_rank_boundary_sensitivity",
    "replacement_unit_improvement",
    "weak_fringe_roster_expansion",
    "uniform_population_improvement",
    "uniform_additive_shift",
    "uniform_multiplicative_shift",
)


RESEARCH_QUESTIONS = {
    "elite_dependence": (
        "How strongly does the method depend on elite talent?",
        (
            "elite_to_weakest_local_sensitivity_ratio",
            "elite_to_ordinary_removal_ratio",
            "superstar_addition",
        ),
    ),
    "local_stability": (
        "How stable is the method under small player changes?",
        (
            "weakest_primary_downgrade",
            "threshold_order_swap",
        ),
    ),
    "rank_boundary_stability": (
        "How sensitive is the method to rank-threshold crossings?",
        (
            "mean_rank_boundary_sensitivity",
            "maximum_rank_boundary_sensitivity",
        ),
    ),
    "fringe_immunity": (
        "Can weak fringe additions alter the representation?",
        (
            "weak_fringe_addition",
            "fringe_immunity",
        ),
    ),
    "depth_behavior": (
        "Does the method respond to replacement-unit quality?",
        (
            "replacement_unit_improvement",
            "weak_fringe_roster_expansion",
        ),
    ),
    "scale_behavior": (
        "Does the method respond predictably to scale changes?",
        (
            "uniform_population_improvement",
            "uniform_additive_shift",
            "uniform_multiplicative_shift",
        ),
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
            f"{frame_name} is missing required columns: {missing}"
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


def metric_lookup(
    metrics: pd.DataFrame,
    *,
    specification_id: str,
    metric_name: str,
) -> float | None:
    rows = metrics.loc[
        metrics["specification_id"].eq(
            specification_id
        )
        & metrics["metric_name"].eq(
            metric_name
        )
        & metrics["comparison_status"].isin(
            {
                "evaluated",
                "undefined_zero_denominator",
            }
        )
    ]

    if rows.empty:
        return None

    return finite_or_none(
        rows.iloc[0]["metric_value"]
    )


def qualitative_elite_profile(
    elite_ratio: float | None,
) -> str:
    if elite_ratio is None:
        return "not_available"

    if elite_ratio < 1.25:
        return "low"

    if elite_ratio < 2.0:
        return "moderate"

    if elite_ratio < 4.0:
        return "high"

    return "very_high"


def qualitative_boundary_profile(
    boundary_sensitivity: float | None,
) -> str:
    if boundary_sensitivity is None:
        return "not_available"

    if boundary_sensitivity <= 1e-10:
        return "negligible"

    if boundary_sensitivity <= 1e-4:
        return "very_low"

    if boundary_sensitivity <= 1e-3:
        return "low"

    if boundary_sensitivity <= 1e-2:
        return "moderate"

    return "high"


def qualitative_fringe_profile(
    fringe_immunity: float | None,
    fringe_sensitivity: float | None,
) -> str:
    if fringe_immunity is not None and fringe_immunity >= 1.0:
        return "immune"

    if fringe_sensitivity is None:
        return "not_available"

    if abs(fringe_sensitivity) <= ABSOLUTE_TOLERANCE:
        return "immune"

    if abs(fringe_sensitivity) <= 0.001:
        return "very_low"

    if abs(fringe_sensitivity) <= 0.01:
        return "low"

    return "material"


def infer_football_philosophy(
    *,
    aggregation_family: str,
    elite_profile: str,
    fringe_profile: str,
    output_type: str,
) -> str:
    if output_type == "depth_dropoff":
        return (
            "Encodes squad fragility by measuring the gap between "
            "the primary group and its immediate replacements."
        )

    if aggregation_family == "replacement_group_mean":
        return (
            "Treats replacement quality as a separate component of "
            "team strength rather than blending it into the starting core."
        )

    if aggregation_family == "arithmetic_mean":
        return (
            "Assumes broadly equal contribution across the complete "
            "included population, making squad depth and fringe quality "
            "part of the central representation."
        )

    if aggregation_family == "top_k_mean":
        return (
            "Treats the selected primary group equally and ignores "
            "players below the top-k threshold."
        )

    if aggregation_family == "rank_weighted_top_k":
        return (
            f"Encodes {elite_profile.replace('_', ' ')} elite emphasis "
            "through explicit rank weights while retaining a fixed "
            "primary-group boundary."
        )

    if aggregation_family == "star_influence_top_k":
        return (
            f"Adds a direct superstar channel to the primary-group mean, "
            f"producing {elite_profile.replace('_', ' ')} elite dependence."
        )

    if aggregation_family == "power_mean_top_k":
        return (
            "Rewards superior players smoothly through nonlinear averaging "
            "without assigning explicit rank identities."
        )

    if aggregation_family == "softmax_weighted_top_k":
        return (
            f"Allows the strongest players to dominate endogenously, "
            f"creating {elite_profile.replace('_', ' ')} elite dependence."
        )

    if aggregation_family == "ability_power_weighted_mean":
        return (
            "Applies elite-sensitive weighting across the whole population, "
            f"so its fringe profile is {fringe_profile.replace('_', ' ')} "
            "rather than fully immune."
        )

    return (
        "Represents a distinct aggregation hypothesis requiring separate "
        "football interpretation."
    )


def infer_likely_squad_fit(
    *,
    elite_profile: str,
    fringe_profile: str,
    output_type: str,
) -> str:
    if output_type == "depth_strength":
        return (
            "Most informative for competitions where rotation, injuries, "
            "and replacement quality materially influence outcomes."
        )

    if output_type == "depth_dropoff":
        return (
            "Useful as a vulnerability indicator for squads whose first "
            "unit is much stronger than their immediate replacements."
        )

    if elite_profile in {
        "high",
        "very_high",
    }:
        if fringe_profile == "immune":
            return (
                "Favors top-heavy teams whose outcomes are driven by a "
                "small elite core and should not be diluted by weak fringe players."
            )

        return (
            "Favors star-led squads but also allows lower-ranked players "
            "to influence the representation."
        )

    if elite_profile == "moderate":
        return (
            "Fits teams combining meaningful elite influence with collective "
            "contribution from the rest of the primary unit."
        )

    return (
        "Fits balanced or system-oriented teams where no single player "
        "should dominate the team-strength representation."
    )


def build_tradeoff_profiles(
    scalar_summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for specification in scalar_summary.itertuples(
        index=False
    ):
        metric_values = {
            metric_name: metric_lookup(
                metrics,
                specification_id=(
                    specification.specification_id
                ),
                metric_name=metric_name,
            )
            for metric_name in PROFILE_METRICS
        }

        elite_ratio = metric_values[
            "elite_to_weakest_local_sensitivity_ratio"
        ]

        boundary_sensitivity = metric_values[
            "mean_rank_boundary_sensitivity"
        ]

        fringe_immunity = metric_values[
            "fringe_immunity"
        ]

        fringe_sensitivity = metric_values[
            "weak_fringe_addition"
        ]

        elite_profile = qualitative_elite_profile(
            elite_ratio
        )

        boundary_profile = (
            qualitative_boundary_profile(
                boundary_sensitivity
            )
        )

        fringe_profile = qualitative_fringe_profile(
            fringe_immunity,
            fringe_sensitivity,
        )

        philosophy = infer_football_philosophy(
            aggregation_family=(
                specification.aggregation_family
            ),
            elite_profile=elite_profile,
            fringe_profile=fringe_profile,
            output_type=specification.output_type,
        )

        squad_fit = infer_likely_squad_fit(
            elite_profile=elite_profile,
            fringe_profile=fringe_profile,
            output_type=specification.output_type,
        )

        record: dict[str, object] = {
            "specification_id":
                specification.specification_id,
            "aggregation_family":
                specification.aggregation_family,
            "display_name":
                specification.display_name,
            "output_type":
                specification.output_type,
            "parameterization":
                specification.parameterization,
            "historical_control":
                specification.historical_control,
            "elite_dependence_profile":
                elite_profile,
            "rank_boundary_profile":
                boundary_profile,
            "fringe_profile":
                fringe_profile,
            "football_philosophy":
                philosophy,
            "likely_squad_fit":
                squad_fit,
        }

        record.update(metric_values)

        records.append(record)

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "output_type",
                "aggregation_family",
                "specification_id",
            ]
        )
        .reset_index(drop=True)
    )


def build_family_summary(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in PROFILE_METRICS
        if column in profiles.columns
    ]

    aggregation_map: dict[
        str,
        str | list[str],
    ] = {
        column: "mean"
        for column in numeric_columns
    }

    aggregation_map.update(
        {
            "specification_id": "count",
            "historical_control": "max",
        }
    )

    summary = (
        profiles
        .groupby(
            [
                "aggregation_family",
                "output_type",
            ],
            as_index=False,
        )
        .agg(aggregation_map)
        .rename(
            columns={
                "specification_id":
                    "specification_count"
            }
        )
    )

    elite_order = {
        "not_available": 0,
        "low": 1,
        "moderate": 2,
        "high": 3,
        "very_high": 4,
    }

    profile_modes = (
        profiles
        .groupby(
            [
                "aggregation_family",
                "output_type",
            ],
            as_index=False,
        )
        .agg(
            representative_elite_profile=(
                "elite_dependence_profile",
                lambda values: max(
                    values,
                    key=lambda value: elite_order.get(
                        value,
                        -1,
                    ),
                ),
            ),
            representative_fringe_profile=(
                "fringe_profile",
                lambda values: values.mode().iloc[0],
            ),
            representative_boundary_profile=(
                "rank_boundary_profile",
                lambda values: values.mode().iloc[0],
            ),
        )
    )

    return (
        summary.merge(
            profile_modes,
            on=[
                "aggregation_family",
                "output_type",
            ],
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            [
                "output_type",
                "aggregation_family",
            ]
        )
        .reset_index(drop=True)
    )


def build_research_question_table(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for question_id, (
        question_text,
        metrics,
    ) in RESEARCH_QUESTIONS.items():
        for profile in profiles.itertuples(
            index=False
        ):
            values = {
                metric_name: getattr(
                    profile,
                    metric_name,
                    None,
                )
                for metric_name in metrics
            }

            available_count = sum(
                value is not None
                and not pd.isna(value)
                for value in values.values()
            )

            records.append(
                {
                    "research_question_id":
                        question_id,
                    "research_question":
                        question_text,
                    "specification_id":
                        profile.specification_id,
                    "aggregation_family":
                        profile.aggregation_family,
                    "display_name":
                        profile.display_name,
                    "output_type":
                        profile.output_type,
                    "available_metric_count":
                        available_count,
                    "expected_metric_count":
                        len(metrics),
                    "metrics":
                        json.dumps(
                            values,
                            sort_keys=True,
                            default=str,
                        ),
                }
            )

    return pd.DataFrame(records)


def validate_outputs(
    *,
    profiles: pd.DataFrame,
    family_summary: pd.DataFrame,
    question_table: pd.DataFrame,
    scalar_summary: pd.DataFrame,
) -> None:
    if profiles.empty:
        raise AssertionError(
            "Aggregation tradeoff profiles are empty."
        )

    if family_summary.empty:
        raise AssertionError(
            "Aggregation family summary is empty."
        )

    if question_table.empty:
        raise AssertionError(
            "Research-question table is empty."
        )

    if profiles[
        "specification_id"
    ].duplicated().any():
        raise AssertionError(
            "Tradeoff profiles contain duplicate specification IDs."
        )

    if set(profiles["specification_id"]) != set(
        scalar_summary["specification_id"]
    ):
        raise AssertionError(
            "Tradeoff profiles do not cover the full scalar registry."
        )

    expected_question_rows = (
        len(RESEARCH_QUESTIONS)
        * len(profiles)
    )

    if len(question_table) != expected_question_rows:
        raise AssertionError(
            "Unexpected research-question row count. "
            f"Expected {expected_question_rows}, "
            f"received {len(question_table)}."
        )

    numeric_values = profiles.select_dtypes(
        include="number"
    )

    non_finite = numeric_values.map(
        lambda value: (
            False
            if pd.isna(value)
            else not math.isfinite(float(value))
        )
    )

    if non_finite.any().any():
        raise AssertionError(
            "Tradeoff profiles contain non-finite numeric values."
        )


def build_metadata(
    *,
    profiles: pd.DataFrame,
    family_summary: pd.DataFrame,
    question_table: pd.DataFrame,
    distribution_summary: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "study_id": "090",
        "study_name": (
            "Aggregation Tradeoff Analysis"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "aggregation_profile_count":
            len(profiles),
        "aggregation_family_count":
            profiles[
                "aggregation_family"
            ].nunique(),
        "family_summary_row_count":
            len(family_summary),
        "research_question_count":
            len(RESEARCH_QUESTIONS),
        "research_question_row_count":
            len(question_table),
        "distribution_summary_row_count":
            len(distribution_summary),
        "ranking_generated":
            False,
        "composite_score_generated":
            False,
        "predictive_superiority_claimed":
            False,
        "real_football_data_used":
            False,
        "goal_model_fitted":
            False,
        "production_runtime_changed":
            False,
        "interpretation_boundary": (
            "Profiles describe synthetic football assumptions and "
            "tradeoffs. They do not establish predictive superiority."
        ),
        "outputs": [
            TRADEOFF_PROFILE_PATH.name,
            FAMILY_SUMMARY_PATH.name,
            QUESTION_TABLE_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    profiles: pd.DataFrame,
    family_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    elite_counts = (
        profiles[
            "elite_dependence_profile"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    fringe_counts = (
        profiles[
            "fringe_profile"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    boundary_counts = (
        profiles[
            "rank_boundary_profile"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    primary_profiles = profiles.loc[
        profiles[
            "output_type"
        ].eq("primary_strength")
    ].copy()

    elite_examples = (
        primary_profiles
        .sort_values(
            "elite_to_weakest_local_sensitivity_ratio",
            ascending=False,
            na_position="last",
        )
        .head(5)[
            [
                "display_name",
                "elite_to_weakest_local_sensitivity_ratio",
                "fringe_profile",
                "rank_boundary_profile",
            ]
        ]
        .to_dict(orient="records")
    )

    report = f"""# Study 090 — Aggregation Tradeoff Analysis

## Purpose

Characterize the football philosophy and synthetic behavioral tradeoffs
encoded by each frozen aggregation specification.

## Methodological boundary

This study:

- introduces no new aggregation formula;
- changes no Study 089 scenario or specification;
- uses only audited synthetic outputs;
- creates no composite score;
- creates no overall leaderboard;
- uses no real match or player data;
- makes no claim about predictive superiority.

## Coverage

- Aggregation profiles: {len(profiles)}
- Aggregation families: {profiles["aggregation_family"].nunique()}
- Family-summary rows: {len(family_summary)}
- Distribution-summary rows consumed: {len(distribution_summary)}
- Research questions: {len(RESEARCH_QUESTIONS)}

## Elite-dependence profiles

{json.dumps(elite_counts, indent=2)}

## Fringe profiles

{json.dumps(fringe_counts, indent=2)}

## Rank-boundary profiles

{json.dumps(boundary_counts, indent=2)}

## Most elite-responsive primary specifications

{json.dumps(elite_examples, indent=2)}

## Interpretation principles

The profiles should be read as football hypotheses.

Examples:

- high elite dependence means the method assigns disproportionate
  importance to superior players;
- fringe immunity means weak players below the useful selection boundary
  do not dilute team strength;
- rank-boundary sensitivity measures how strongly a method reacts when
  players cross a selection threshold;
- depth-strength and depth-dropoff specifications answer different
  questions from primary-strength aggregators.

No profile is universally superior.

A method may be desirable for one football context and undesirable for
another. Predictive studies must determine whether the encoded assumptions
improve real-world forecasting.

## Result

**OVERALL RESULT: {metadata["status"]}**

Study 090 successfully converted the audited synthetic evidence into
interpretable aggregation tradeoff profiles without producing a ranking.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 090 — AGGREGATION TRADEOFF ANALYSIS"
    )
    print("=" * 88)

    scalar_summary = pd.read_csv(
        SCALAR_SUMMARY_PATH
    )

    distribution_summary = pd.read_csv(
        DISTRIBUTION_SUMMARY_PATH
    )

    audited_metrics = pd.read_csv(
        AUDITED_METRIC_PATH
    )

    require_columns(
        scalar_summary,
        REQUIRED_SCALAR_COLUMNS,
        frame_name="Scalar behavior summary",
    )

    require_columns(
        audited_metrics,
        REQUIRED_METRIC_COLUMNS,
        frame_name="Audited behavior metrics",
    )

    profiles = build_tradeoff_profiles(
        scalar_summary,
        audited_metrics,
    )

    family_summary = build_family_summary(
        profiles
    )

    question_table = build_research_question_table(
        profiles
    )

    validate_outputs(
        profiles=profiles,
        family_summary=family_summary,
        question_table=question_table,
        scalar_summary=scalar_summary,
    )

    metadata = build_metadata(
        profiles=profiles,
        family_summary=family_summary,
        question_table=question_table,
        distribution_summary=(
            distribution_summary
        ),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles.to_csv(
        TRADEOFF_PROFILE_PATH,
        index=False,
    )

    family_summary.to_csv(
        FAMILY_SUMMARY_PATH,
        index=False,
    )

    question_table.to_csv(
        QUESTION_TABLE_PATH,
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
        profiles=profiles,
        family_summary=family_summary,
        distribution_summary=(
            distribution_summary
        ),
        metadata=metadata,
    )

    print()
    print("Profile coverage")
    print("-" * 88)
    print(
        profiles[
            "aggregation_family"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Elite-dependence profiles")
    print("-" * 88)
    print(
        profiles[
            "elite_dependence_profile"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Fringe profiles")
    print("-" * 88)
    print(
        profiles[
            "fringe_profile"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Output summary")
    print("-" * 88)
    print(
        f"  Aggregation profiles: "
        f"{len(profiles)}"
    )
    print(
        f"  Family summary rows: "
        f"{len(family_summary)}"
    )
    print(
        f"  Research-question rows: "
        f"{len(question_table)}"
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
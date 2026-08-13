#benchmark_role_suitability_signals

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research.player_intelligence.player_contribution import (
    build_player_contribution,
)
from research.player_intelligence.player_schema import (
    LineupAssignment,
    Player,
    PlayerIdentity,
    PlayerRatings,
    RoleRatings,
)
from research.player_intelligence.role_suitability import (
    build_role_suitability_signals,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_098_role_suitability"
    / "study_098b"
)

SCENARIO_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "role_suitability_scenario_results.csv"
)

SIGNAL_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "role_suitability_signal_summary.csv"
)

AXIOM_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "role_suitability_axiom_results.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_098b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_098B_REPORT.md"
)


SIGNAL_COLUMNS = (
    "raw_rating_gap",
    "absolute_rating_gap",
    "symmetric_relative_gap",
    "reciprocal_rank_score",
    "strongest_role_indicator",
    "positive_scale_ratio",
)


@dataclass(frozen=True)
class SyntheticScenario:
    scenario_id: str
    scenario_group: str
    description: str

    assigned_role: str
    role_ratings: dict[str, float | None]

    expected_rank: int | None
    expected_resolution: bool

    scale_family: str
    assignment_quality: str


def make_player(
    *,
    scenario: SyntheticScenario,
) -> Player:
    role_kwargs = {
        "GK": None,
        "CB": None,
        "FB": None,
        "DM": None,
        "CM": None,
        "AM": None,
        "WM": None,
        "W": None,
        "ST": None,
    }

    role_kwargs.update(
        scenario.role_ratings
    )

    finite_values = [
        float(value)
        for value in role_kwargs.values()
        if value is not None
        and math.isfinite(
            float(value)
        )
    ]

    overall = (
        sum(finite_values)
        / len(finite_values)
        if finite_values
        else 0.0
    )

    return Player(
        identity=PlayerIdentity(
            player_id=(
                f"synthetic-{scenario.scenario_id}"
            ),
            name=(
                f"Synthetic {scenario.scenario_id}"
            ),
            national_team="Synthetic Team",
        ),
        ratings=PlayerRatings(
            overall=overall,
            attack=overall,
            midfield=overall,
            defense=overall,
            goalkeeper=0.0,
        ),
        role_ratings=RoleRatings(
            **role_kwargs
        ),
    )


def build_scenarios() -> tuple[
    SyntheticScenario,
    ...,
]:
    return (
        SyntheticScenario(
            scenario_id="positive_rank_1",
            scenario_group="positive_scale",
            description=(
                "Assigned role is strongest on a positive scale."
            ),
            assigned_role="CM",
            role_ratings={
                "CM": 0.90,
                "AM": 0.80,
                "DM": 0.70,
            },
            expected_rank=1,
            expected_resolution=True,
            scale_family="positive",
            assignment_quality="strongest",
        ),
        SyntheticScenario(
            scenario_id="positive_rank_2",
            scenario_group="positive_scale",
            description=(
                "Assigned role is second strongest on a positive scale."
            ),
            assigned_role="AM",
            role_ratings={
                "CM": 0.90,
                "AM": 0.80,
                "DM": 0.70,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="positive",
            assignment_quality="second_best",
        ),
        SyntheticScenario(
            scenario_id="positive_rank_3",
            scenario_group="positive_scale",
            description=(
                "Assigned role is third strongest on a positive scale."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": 0.90,
                "AM": 0.80,
                "DM": 0.70,
            },
            expected_rank=3,
            expected_resolution=True,
            scale_family="positive",
            assignment_quality="third_best",
        ),
        SyntheticScenario(
            scenario_id="positive_large_gap",
            scenario_group="positive_scale",
            description=(
                "Assigned role is much weaker than the strongest role."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": 1.00,
                "DM": 0.20,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="positive",
            assignment_quality="severe_compromise",
        ),
        SyntheticScenario(
            scenario_id="positive_small_gap",
            scenario_group="positive_scale",
            description=(
                "Assigned role is slightly below the strongest role."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": 1.00,
                "DM": 0.95,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="positive",
            assignment_quality="mild_compromise",
        ),
        SyntheticScenario(
            scenario_id="near_zero_same_sign",
            scenario_group="near_zero",
            description=(
                "Both ratings are small and positive."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": 0.02,
                "DM": 0.01,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="near_zero_positive",
            assignment_quality="second_best",
        ),
        SyntheticScenario(
            scenario_id="opposite_sign",
            scenario_group="mixed_sign",
            description=(
                "Assigned role is negative while the strongest role "
                "is positive."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": 0.05,
                "DM": -0.10,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="mixed_sign",
            assignment_quality="severe_compromise",
        ),
        SyntheticScenario(
            scenario_id="negative_rank_1",
            scenario_group="negative_scale",
            description=(
                "Assigned role is strongest when all ratings are negative."
            ),
            assigned_role="CM",
            role_ratings={
                "CM": -0.10,
                "DM": -0.20,
                "AM": -0.30,
            },
            expected_rank=1,
            expected_resolution=True,
            scale_family="negative",
            assignment_quality="strongest",
        ),
        SyntheticScenario(
            scenario_id="negative_rank_2",
            scenario_group="negative_scale",
            description=(
                "Assigned role is second strongest when all ratings "
                "are negative."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": -0.10,
                "DM": -0.20,
                "AM": -0.30,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="negative",
            assignment_quality="second_best",
        ),
        SyntheticScenario(
            scenario_id="zero_tie",
            scenario_group="ties",
            description=(
                "Assigned role is tied strongest at zero but ranks second "
                "under the stable role-vocabulary tie break."
            ),
            assigned_role="CM",
            role_ratings={
                "CM": 0.0,
                "DM": 0.0,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="zero",
            assignment_quality="tied_strongest",
        ),
        SyntheticScenario(
            scenario_id="positive_tie_first_role",
            scenario_group="ties",
            description=(
                "Assigned role is one of two tied strongest roles."
            ),
            assigned_role="CM",
            role_ratings={
                "CM": 0.80,
                "AM": 0.80,
                "DM": 0.60,
            },
            expected_rank=1,
            expected_resolution=True,
            scale_family="positive",
            assignment_quality="tied_strongest",
        ),
        SyntheticScenario(
            scenario_id="positive_tie_second_role",
            scenario_group="ties",
            description=(
                "Assigned role is tied strongest but ordered second "
                "by stable vocabulary."
            ),
            assigned_role="AM",
            role_ratings={
                "CM": 0.80,
                "AM": 0.80,
                "DM": 0.60,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="positive",
            assignment_quality="tied_strongest",
        ),
        SyntheticScenario(
            scenario_id="missing_assigned_role",
            scenario_group="missingness",
            description=(
                "Assigned role has no available rating."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": 0.80,
                "AM": 0.70,
            },
            expected_rank=None,
            expected_resolution=False,
            scale_family="missing",
            assignment_quality="unresolved",
        ),
        SyntheticScenario(
            scenario_id="single_role",
            scenario_group="degenerate",
            description=(
                "Player has only one rated role and is assigned to it."
            ),
            assigned_role="ST",
            role_ratings={
                "ST": 0.90,
            },
            expected_rank=1,
            expected_resolution=True,
            scale_family="positive",
            assignment_quality="strongest",
        ),
        SyntheticScenario(
            scenario_id="scale_small",
            scenario_group="scale_equivalence",
            description=(
                "Small-scale version of an identical proportional gap."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": 0.20,
                "DM": 0.10,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="positive_small",
            assignment_quality="second_best",
        ),
        SyntheticScenario(
            scenario_id="scale_large",
            scenario_group="scale_equivalence",
            description=(
                "Large-scale version of an identical proportional gap."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": 2.00,
                "DM": 1.00,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="positive_large",
            assignment_quality="second_best",
        ),
        SyntheticScenario(
            scenario_id="shift_original",
            scenario_group="translation_sensitivity",
            description=(
                "Original positive ratings before constant translation."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": 0.80,
                "DM": 0.60,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="positive",
            assignment_quality="second_best",
        ),
        SyntheticScenario(
            scenario_id="shift_translated",
            scenario_group="translation_sensitivity",
            description=(
                "Same rating difference after adding one to all roles."
            ),
            assigned_role="DM",
            role_ratings={
                "CM": 1.80,
                "DM": 1.60,
            },
            expected_rank=2,
            expected_resolution=True,
            scale_family="positive_shifted",
            assignment_quality="second_best",
        ),
    )


def evaluate_scenario(
    scenario: SyntheticScenario,
) -> dict[str, Any]:
    player = make_player(
        scenario=scenario
    )

    selection_rating = (
        scenario.role_ratings.get(
            scenario.assigned_role
        )
    )

    assignment = LineupAssignment(
        slot="TEST1",
        tactical_role=(
            scenario.assigned_role
        ),
        player=player,
        selection_rating=(
            float(selection_rating)
            if selection_rating is not None
            else 0.0
        ),
    )

    contribution = (
        build_player_contribution(
            assignment
        )
    )

    signals = (
        build_role_suitability_signals(
            contribution
        )
    )

    result = {
        **asdict(scenario),
        **asdict(signals),
    }

    result[
        "rank_matches_expectation"
    ] = (
        signals.assigned_role_rank
        == scenario.expected_rank
    )

    resolved_signals = (
        signals.raw_rating_gap,
        signals.absolute_rating_gap,
        signals.symmetric_relative_gap,
        signals.reciprocal_rank_score,
        signals.strongest_role_indicator,
    )

    observed_resolution = all(
        value is not None
        for value in resolved_signals
    )

    result[
        "resolution_matches_expectation"
    ] = (
        observed_resolution
        == scenario.expected_resolution
    )

    result[
        "all_resolved_signals_finite"
    ] = all(
        value is None
        or math.isfinite(
            float(value)
        )
        for value in resolved_signals
    )

    result[
        "contribution_adjusted"
    ] = (
        signals.contextual_adjustment_applied
    )

    return result


def build_scenario_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            evaluate_scenario(
                scenario
            )
            for scenario in build_scenarios()
        ]
    )


def build_signal_summary(
    scenario_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for signal in SIGNAL_COLUMNS:
        values = pd.to_numeric(
            scenario_results[signal],
            errors="coerce",
        )

        resolved = values.dropna()

        rows.append(
            {
                "signal": signal,
                "scenario_count": len(
                    scenario_results
                ),
                "resolved_count": int(
                    values.notna().sum()
                ),
                "unresolved_count": int(
                    values.isna().sum()
                ),
                "finite_resolved_count": int(
                    resolved.map(
                        math.isfinite
                    ).sum()
                ),
                "minimum": (
                    float(
                        resolved.min()
                    )
                    if not resolved.empty
                    else None
                ),
                "maximum": (
                    float(
                        resolved.max()
                    )
                    if not resolved.empty
                    else None
                ),
                "mean": (
                    float(
                        resolved.mean()
                    )
                    if not resolved.empty
                    else None
                ),
                "bounded_zero_one": bool(
                    not resolved.empty
                    and resolved.between(
                        0.0,
                        1.0,
                        inclusive="both",
                    ).all()
                ),
                "supports_negative_scale": bool(
                    scenario_results.loc[
                        scenario_results[
                            "scenario_group"
                        ].eq(
                            "negative_scale"
                        ),
                        signal,
                    ]
                    .notna()
                    .all()
                ),
                "supports_missing_assignment": bool(
                    scenario_results.loc[
                        scenario_results[
                            "scenario_id"
                        ].eq(
                            "missing_assigned_role"
                        ),
                        signal,
                    ]
                    .isna()
                    .all()
                ),
            }
        )

    return pd.DataFrame(rows)


def axiom_row(
    *,
    axiom_id: str,
    signal: str,
    passed: bool | None,
    observed: str,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "axiom_id": axiom_id,
        "signal": signal,
        "status": (
            "pass"
            if passed is True
            else "fail"
            if passed is False
            else "not_applicable"
        ),
        "observed": observed,
        "interpretation": interpretation,
    }


def scenario_value(
    results: pd.DataFrame,
    scenario_id: str,
    signal: str,
) -> float | None:
    value = results.loc[
        results[
            "scenario_id"
        ].eq(
            scenario_id
        ),
        signal,
    ].iloc[0]

    if pd.isna(value):
        return None

    return float(value)


def build_axiom_results(
    scenario_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    penalty_signals = (
        "raw_rating_gap",
        "absolute_rating_gap",
        "symmetric_relative_gap",
    )

    score_signals = (
        "reciprocal_rank_score",
        "strongest_role_indicator",
        "positive_scale_ratio",
    )

    for signal in penalty_signals:
        first = scenario_value(
            scenario_results,
            "positive_rank_1",
            signal,
        )
        second = scenario_value(
            scenario_results,
            "positive_rank_2",
            signal,
        )
        third = scenario_value(
            scenario_results,
            "positive_rank_3",
            signal,
        )

        passed = (
            first is not None
            and second is not None
            and third is not None
            and first <= second <= third
        )

        rows.append(
            axiom_row(
                axiom_id="positive_rank_monotonicity",
                signal=signal,
                passed=passed,
                observed=(
                    f"rank1={first}, "
                    f"rank2={second}, "
                    f"rank3={third}"
                ),
                interpretation=(
                    "Penalty-like signals should not decrease as "
                    "the assigned role becomes weaker."
                ),
            )
        )

    for signal in score_signals:
        first = scenario_value(
            scenario_results,
            "positive_rank_1",
            signal,
        )
        second = scenario_value(
            scenario_results,
            "positive_rank_2",
            signal,
        )
        third = scenario_value(
            scenario_results,
            "positive_rank_3",
            signal,
        )

        passed = (
            first is not None
            and second is not None
            and third is not None
            and first >= second >= third
        )

        rows.append(
            axiom_row(
                axiom_id="positive_rank_monotonicity",
                signal=signal,
                passed=passed,
                observed=(
                    f"rank1={first}, "
                    f"rank2={second}, "
                    f"rank3={third}"
                ),
                interpretation=(
                    "Suitability-score signals should not increase "
                    "as the assigned role becomes weaker."
                ),
            )
        )

    for signal in SIGNAL_COLUMNS:
        negative_value = scenario_value(
            scenario_results,
            "negative_rank_2",
            signal,
        )

        expected_supported = (
            signal
            != "positive_scale_ratio"
        )

        passed = (
            negative_value is not None
            if expected_supported
            else negative_value is None
        )

        rows.append(
            axiom_row(
                axiom_id="negative_scale_behavior",
                signal=signal,
                passed=passed,
                observed=(
                    f"value={negative_value}"
                ),
                interpretation=(
                    "Signals should either support negative scales "
                    "or explicitly remain unresolved by design."
                ),
            )
        )

    for signal in SIGNAL_COLUMNS:
        small = scenario_value(
            scenario_results,
            "scale_small",
            signal,
        )
        large = scenario_value(
            scenario_results,
            "scale_large",
            signal,
        )

        scale_invariant_signals = {
            "symmetric_relative_gap",
            "reciprocal_rank_score",
            "strongest_role_indicator",
            "positive_scale_ratio",
        }

        if signal in scale_invariant_signals:
            passed = (
                small is not None
                and large is not None
                and math.isclose(
                    small,
                    large,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        else:
            passed = (
                small is not None
                and large is not None
                and not math.isclose(
                    small,
                    large,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )

        rows.append(
            axiom_row(
                axiom_id="multiplicative_scale_response",
                signal=signal,
                passed=passed,
                observed=(
                    f"small={small}, large={large}"
                ),
                interpretation=(
                    "Normalized and rank-based signals should remain "
                    "unchanged under proportional rescaling, while raw "
                    "rating gaps should preserve native scale."
                ),
            )
        )

    for signal in SIGNAL_COLUMNS:
        original = scenario_value(
            scenario_results,
            "shift_original",
            signal,
        )
        translated = scenario_value(
            scenario_results,
            "shift_translated",
            signal,
        )

        translation_invariant_signals = {
            "raw_rating_gap",
            "absolute_rating_gap",
            "reciprocal_rank_score",
            "strongest_role_indicator",
        }

        if signal in translation_invariant_signals:
            passed = (
                original is not None
                and translated is not None
                and math.isclose(
                    original,
                    translated,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        else:
            passed = (
                original is not None
                and translated is not None
                and not math.isclose(
                    original,
                    translated,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )

        rows.append(
            axiom_row(
                axiom_id="translation_response",
                signal=signal,
                passed=passed,
                observed=(
                    f"original={original}, "
                    f"translated={translated}"
                ),
                interpretation=(
                    "Raw differences and rank signals are translation "
                    "invariant. Ratio-style signals generally are not."
                ),
            )
        )

    for signal in SIGNAL_COLUMNS:
        missing = scenario_value(
            scenario_results,
            "missing_assigned_role",
            signal,
        )

        rows.append(
            axiom_row(
                axiom_id="missing_assignment_resolution",
                signal=signal,
                passed=(
                    missing is None
                ),
                observed=(
                    f"value={missing}"
                ),
                interpretation=(
                    "A signal must remain unresolved when the assigned "
                    "role has no rating."
                ),
            )
        )

    return pd.DataFrame(rows)


def validate_results(
    *,
    scenario_results: pd.DataFrame,
    signal_summary: pd.DataFrame,
    axiom_results: pd.DataFrame,
) -> None:
    if scenario_results.empty:
        raise AssertionError(
            "Synthetic scenario results are empty."
        )

    if scenario_results[
        "scenario_id"
    ].duplicated().any():
        raise AssertionError(
            "Synthetic scenario IDs are not unique."
        )

    if not scenario_results[
        "rank_matches_expectation"
    ].all():
        failures = scenario_results.loc[
            ~scenario_results[
                "rank_matches_expectation"
            ]
        ]

        raise AssertionError(
            "Assigned-role rank expectations failed:\n"
            + failures[
                [
                    "scenario_id",
                    "expected_rank",
                    "assigned_role_rank",
                ]
            ].to_string(index=False)
        )

    if not scenario_results[
        "resolution_matches_expectation"
    ].all():
        failures = scenario_results.loc[
            ~scenario_results[
                "resolution_matches_expectation"
            ]
        ]

        raise AssertionError(
            "Signal-resolution expectations failed:\n"
            + failures.to_string(
                index=False
            )
        )

    if not scenario_results[
        "all_resolved_signals_finite"
    ].all():
        raise AssertionError(
            "At least one resolved signal is non-finite."
        )

    if scenario_results[
        "contribution_adjusted"
    ].any():
        raise AssertionError(
            "Study 098B unexpectedly applied a contextual adjustment."
        )

    if signal_summary.empty:
        raise AssertionError(
            "Signal summary is empty."
        )

    failed_axioms = axiom_results.loc[
        axiom_results[
            "status"
        ].eq("fail")
    ]

    if not failed_axioms.empty:
        raise AssertionError(
            "One or more synthetic signal axioms failed:\n"
            + failed_axioms.to_string(
                index=False
            )
        )


def build_metadata(
    *,
    scenario_results: pd.DataFrame,
    signal_summary: pd.DataFrame,
    axiom_results: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "study_id": "098B",
        "study_name": (
            "Synthetic Role-Suitability Benchmark"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "scenario_count": len(
            scenario_results
        ),
        "scenario_group_count": int(
            scenario_results[
                "scenario_group"
            ].nunique()
        ),
        "signal_count": len(
            SIGNAL_COLUMNS
        ),
        "axiom_count": len(
            axiom_results
        ),
        "axiom_pass_count": int(
            axiom_results[
                "status"
            ].eq("pass").sum()
        ),
        "axiom_fail_count": int(
            axiom_results[
                "status"
            ].eq("fail").sum()
        ),
        "axiom_not_applicable_count": int(
            axiom_results[
                "status"
            ].eq(
                "not_applicable"
            ).sum()
        ),
        "all_ranks_matched_expectation": True,
        "all_resolution_states_matched_expectation": True,
        "all_resolved_signals_finite": True,
        "contextual_adjustment_applied": False,
        "player_contributions_modified": False,
        "team_representation_modified": False,
        "aggregation_modified": False,
        "production_runtime_changed": False,
        "interpretation_boundary": (
            "Synthetic diagnostic comparison of role-suitability "
            "signals. No signal modifies player contribution or team "
            "strength."
        ),
        "outputs": [
            SCENARIO_RESULTS_PATH.name,
            SIGNAL_SUMMARY_PATH.name,
            AXIOM_RESULTS_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    scenario_results: pd.DataFrame,
    signal_summary: pd.DataFrame,
    axiom_results: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    report = f"""# Study 098B — Synthetic Role-Suitability Benchmark

## Status

**PASS**

## Purpose

Compare candidate role-suitability signals under controlled synthetic
assignments before any signal is allowed to modify player contribution.

## Design

- Scenarios: {metadata["scenario_count"]}
- Scenario groups: {metadata["scenario_group_count"]}
- Signals: {metadata["signal_count"]}
- Axiom evaluations: {metadata["axiom_count"]}

Every scenario holds the player object and role-rating population fixed
within that scenario and changes only the assignment conditions being
evaluated.

## Methodological boundary

This study:

- applies no contextual adjustment;
- changes no player contribution;
- changes no team representation;
- changes no aggregation behavior;
- generates no repository;
- fits no goal model;
- changes no production runtime.

## Signal summary

{signal_summary.to_markdown(index=False)}

## Synthetic scenarios

{scenario_results.to_markdown(index=False)}

## Axiom results

{axiom_results.to_markdown(index=False)}

## Interpretation

The benchmark characterizes signal behavior. It does not select a
production penalty and does not claim predictive superiority.

Signals may encode different useful properties:

- raw gaps preserve the native player-rating scale;
- symmetric relative gaps provide bounded cross-scale comparison;
- rank-based scores are stable and highly interpretable;
- strongest-role indicators provide a simple binary control;
- positive-scale ratios retain historical interpretability but are
  deliberately unresolved when the strongest rating is non-positive.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 098B — SYNTHETIC ROLE-SUITABILITY BENCHMARK"
    )
    print("=" * 88)

    scenario_results = (
        build_scenario_results()
    )

    signal_summary = (
        build_signal_summary(
            scenario_results
        )
    )

    axiom_results = (
        build_axiom_results(
            scenario_results
        )
    )

    validate_results(
        scenario_results=(
            scenario_results
        ),
        signal_summary=(
            signal_summary
        ),
        axiom_results=(
            axiom_results
        ),
    )

    metadata = build_metadata(
        scenario_results=(
            scenario_results
        ),
        signal_summary=(
            signal_summary
        ),
        axiom_results=(
            axiom_results
        ),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenario_results.to_csv(
        SCENARIO_RESULTS_PATH,
        index=False,
    )

    signal_summary.to_csv(
        SIGNAL_SUMMARY_PATH,
        index=False,
    )

    axiom_results.to_csv(
        AXIOM_RESULTS_PATH,
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
        scenario_results=(
            scenario_results
        ),
        signal_summary=(
            signal_summary
        ),
        axiom_results=(
            axiom_results
        ),
        metadata=metadata,
    )

    print()
    print("Benchmark summary")
    print("-" * 88)
    print(
        f"  Scenarios: "
        f"{metadata['scenario_count']}"
    )
    print(
        f"  Signals: "
        f"{metadata['signal_count']}"
    )
    print(
        f"  Axiom evaluations: "
        f"{metadata['axiom_count']}"
    )
    print(
        f"  Axiom passes: "
        f"{metadata['axiom_pass_count']}"
    )
    print(
        f"  Axiom failures: "
        f"{metadata['axiom_fail_count']}"
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Scenario identities unique: PASS")
    print("  Assigned-role ranks correct: PASS")
    print("  Signal resolution states correct: PASS")
    print("  Resolved signals finite: PASS")
    print("  Synthetic axioms: PASS")
    print("  Contextual adjustment applied: NO")
    print("  PlayerContribution modified: NO")
    print("  TeamRepresentation modified: NO")
    print("  Aggregation modified: NO")
    print("  Production behavior changed: NO")

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
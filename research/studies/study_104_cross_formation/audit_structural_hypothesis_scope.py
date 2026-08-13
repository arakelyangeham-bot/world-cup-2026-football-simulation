#audit_structural_hypothesis_scope

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.player_intelligence.structural_responsibility_generator import (
    INITIAL_STRUCTURAL_HYPOTHESES,
    StructuralHypothesisStatus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_104_cross_formation"
    / "study_104c"
)

SCOPE_REGISTER_PATH = (
    OUTPUT_DIRECTORY
    / "structural_hypothesis_scope_register.csv"
)

FORMATION_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "structural_hypothesis_formation_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_104c_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_104C_REPORT.md"
)


EXPECTED_SCOPE_BY_RULE = {
    "dm_protects_cb_v1": {
        "4-3-3",
    },
    "dm_supports_cm_v1": {
        "4-3-3",
    },
    "cb_covers_fb_v1": {
        "4-2-3-1",
        "4-3-3",
    },
    "cm_supports_w_v1": {
        "4-3-3",
    },
    "dm_connects_cb_cm_v1": {
        "4-3-3",
    },
}

def build_scope_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_id":
                    hypothesis.rule_id,
                "responsibility_type":
                    hypothesis
                    .responsibility_type.value,
                "hypothesis_status":
                    hypothesis.status.value,
                "supported_formations":
                    "|".join(
                        hypothesis
                        .scope
                        .supported_formations
                    ),
                "supported_formation_count":
                    len(
                        hypothesis
                        .scope
                        .supported_formations
                    ),
                "supported_source_role_counts":
                    "|".join(
                        str(value)
                        for value
                        in hypothesis
                        .scope
                        .supported_source_role_counts
                    ),
                "supported_target_role_counts":
                    "|".join(
                        str(value)
                        for value
                        in hypothesis
                        .scope
                        .supported_target_role_counts
                    ),
                "unit_type":
                    hypothesis
                    .scope
                    .unit_type.value,
                "formation_general":
                    hypothesis
                    .scope
                    .formation_general,
                "requires_single_source_role":
                    hypothesis
                    .scope
                    .requires_single_source_role,
                "requires_multiple_source_roles":
                    hypothesis
                    .scope
                    .requires_multiple_source_roles,
                "empirically_validated":
                    False,
                "production_eligible":
                    False,
            }
            for hypothesis
            in INITIAL_STRUCTURAL_HYPOTHESES
        ]
    )

def validate_scope_register(
    register: pd.DataFrame,
) -> None:
    if register.empty:
        raise AssertionError(
            "Hypothesis scope register is empty."
        )

    if register[
        "rule_id"
    ].duplicated().any():
        raise AssertionError(
            "Hypothesis scope register contains "
            "duplicate rule IDs."
        )

    observed = {
        str(row.rule_id): set(
            str(
                row.supported_formations
            ).split("|")
        )
        for row
        in register.itertuples(
            index=False
        )
    }

    if observed != EXPECTED_SCOPE_BY_RULE:
        raise AssertionError(
            "Registered hypothesis scopes differ "
            "from the completed formation audits."
        )

    single_pivot = register.loc[
        register[
            "rule_id"
        ].eq(
            "dm_protects_cb_v1"
        )
    ].iloc[0]

    if (
        bool(
            single_pivot[
                "formation_general"
            ]
        )
        or not bool(
            single_pivot[
                "requires_single_source_role"
            ]
        )
    ):
        raise AssertionError(
            "Single-pivot DM protection scope is invalid."
        )

    cb_coverage = register.loc[
        register[
            "rule_id"
        ].eq(
            "cb_covers_fb_v1"
        )
    ].iloc[0]

    if not bool(
        cb_coverage[
            "formation_general"
        ]
    ):
        raise AssertionError(
            "CB-to-FB coverage was not marked "
            "formation-general."
        )

    if register[
        "empirically_validated"
    ].any():
        raise AssertionError(
            "A hypothesis scope was incorrectly marked "
            "empirically validated."
        )

    if register[
        "production_eligible"
    ].any():
        raise AssertionError(
            "A hypothesis scope was incorrectly marked "
            "production eligible."
        )

def build_formation_summary(
    register: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row in register.itertuples(
        index=False
    ):
        formations = str(
            row.supported_formations
        ).split("|")

        for formation in formations:
            rows.append(
                {
                    "formation": formation,
                    "rule_id": row.rule_id,
                    "responsibility_type":
                        row.responsibility_type,
                    "hypothesis_status":
                        row.hypothesis_status,
                    "unit_type":
                        row.unit_type,
                    "formation_general":
                        row.formation_general,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "formation",
                "rule_id",
            ]
        )
        .reset_index(drop=True)
    )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 104C — STRUCTURAL HYPOTHESIS "
        "SCOPE FOUNDATIONS"
    )
    print("=" * 88)

    register = build_scope_register()

    validate_scope_register(
        register
    )

    formation_summary = (
        build_formation_summary(
            register
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    register.to_csv(
        SCOPE_REGISTER_PATH,
        index=False,
    )

    formation_summary.to_csv(
        FORMATION_SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "study_id": "104C",
        "study_name": (
            "Structural Hypothesis Scope Foundations"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "hypothesis_count": len(
            register
        ),
        "registered_formation_count": int(
            formation_summary[
                "formation"
            ].nunique()
        ),
        "formation_general_hypothesis_count": int(
            register[
                "formation_general"
            ].sum()
        ),
        "single_pivot_scoped_hypothesis_count": int(
            register[
                "requires_single_source_role"
            ].sum()
        ),
        "double_pivot_hypothesis_promoted": False,
        "scope_filtering_enabled": False,
        "generation_behavior_changed": False,
        "structural_rules_changed": False,
        "football_graph_created": False,
        "diagnostics_generated": False,
        "weights_created": False,
        "team_strength_changed": False,
        "repository_changed": False,
        "simulation_run": False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "Scope metadata records intended hypothesis "
            "applicability. It does not validate football truth "
            "or activate formation-aware generation."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = f"""# Study 104C — Structural Hypothesis Scope Foundations

## Status

**PASS**

## Purpose

Register explicit formation and role-population scope for each
expert-authored structural hypothesis.

## Scope register

{register.to_markdown(index=False)}

## Formation summary

{formation_summary.to_markdown(index=False)}

## Important boundary

The current DM-to-CB protection hypothesis is scoped to a single-pivot
4-3-3. Study 104B demonstrated that applying it mechanically to a
4-2-3-1 produces a shared-protection interpretation, but no
double-pivot protection hypothesis has been promoted.

Scope metadata is diagnostic only. Formation-aware rule activation is
not enabled in this study.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("Scope summary")
    print("-" * 88)
    print(
        "  Hypotheses registered: "
        f"{metadata['hypothesis_count']}"
    )
    print(
        "  Formations represented: "
        f"{metadata['registered_formation_count']}"
    )
    print(
        "  Formation-general hypotheses: "
        f"{metadata['formation_general_hypothesis_count']}"
    )
    print(
        "  Single-pivot-scoped hypotheses: "
        f"{metadata['single_pivot_scoped_hypothesis_count']}"
    )
    print(
        "  DM protection scoped to 4-3-3 single pivot: PASS"
    )
    print(
        "  CB coverage supports both formations: PASS"
    )
    print(
        "  Double-pivot hypothesis promoted: NO"
    )
    print(
        "  Scope filtering enabled: NO"
    )
    print(
        "  Generation behavior changed: NO"
    )
    print(
        "  Football graph created: NO"
    )
    print(
        "  Team strength changed: NO"
    )
    print(
        "  Simulation run: NO"
    )

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
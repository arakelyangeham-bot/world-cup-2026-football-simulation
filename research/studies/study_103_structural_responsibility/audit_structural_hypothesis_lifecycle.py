#audit_structural_hypothesis_lifecycle

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
    / "study_103_structural_responsibility"
    / "study_103a1"
)

HYPOTHESIS_REGISTER_PATH = (
    OUTPUT_DIRECTORY
    / "structural_hypothesis_register.csv"
)

STATUS_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "structural_hypothesis_status_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_103a1_metadata.json"
)


EXPECTED_STATUS_BY_RULE = {
    "dm_protects_cb_v1":
        StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
    "dm_supports_cm_v1":
        StructuralHypothesisStatus.REVISION_REQUIRED,
    "cb_covers_fb_v1":
        StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC,
    "cm_supports_w_v1":
        StructuralHypothesisStatus.REVISION_REQUIRED,
    "dm_connects_cb_cm_v1":
        StructuralHypothesisStatus.DEFERRED,
}


def build_hypothesis_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_id": rule.rule_id,
                "source_roles": "|".join(
                    rule.source_roles
                ),
                "target_roles": "|".join(
                    rule.target_roles
                ),
                "responsibility_type":
                    rule.responsibility_type.value,
                "required_positional_types":
                    "|".join(
                        item.value
                        for item
                        in rule.required_positional_types
                    ),
                "status": rule.status.value,
                "source_must_be_deeper":
                    rule.source_must_be_deeper,
                "source_must_be_more_advanced":
                    rule.source_must_be_more_advanced,
                "same_broad_corridor_required":
                    rule.same_broad_corridor_required,
                "expert_authored": True,
                "empirically_validated": False,
                "production_eligible": False,
            }
            for rule
            in INITIAL_STRUCTURAL_HYPOTHESES
        ]
    )


def validate_register(
    register: pd.DataFrame,
) -> None:
    if register.empty:
        raise AssertionError(
            "Structural hypothesis register is empty."
        )

    if register["rule_id"].duplicated().any():
        raise AssertionError(
            "Structural hypothesis rule IDs are not unique."
        )

    observed = {
        str(row.rule_id):
            StructuralHypothesisStatus(
                str(row.status)
            )
        for row in register.itertuples(
            index=False
        )
    }

    if observed != EXPECTED_STATUS_BY_RULE:
        raise AssertionError(
            "Structural hypothesis lifecycle states differ "
            "from the completed design audit."
        )

    if register[
        "empirically_validated"
    ].any():
        raise AssertionError(
            "An expert-authored hypothesis was incorrectly "
            "marked empirically validated."
        )

    if register[
        "production_eligible"
    ].any():
        raise AssertionError(
            "A structural hypothesis was incorrectly marked "
            "production eligible."
        )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 103A1 — STRUCTURAL HYPOTHESIS "
        "LIFECYCLE AUDIT"
    )
    print("=" * 88)

    register = build_hypothesis_register()

    validate_register(
        register
    )

    status_summary = (
        register
        .groupby(
            "status",
            as_index=False,
        )
        .agg(
            hypothesis_count=(
                "rule_id",
                "count",
            ),
            responsibility_type_count=(
                "responsibility_type",
                "nunique",
            ),
        )
        .sort_values("status")
        .reset_index(drop=True)
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    register.to_csv(
        HYPOTHESIS_REGISTER_PATH,
        index=False,
    )

    status_summary.to_csv(
        STATUS_SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "study_id": "103A1",
        "study_name": (
            "Structural Hypothesis Lifecycle Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "hypothesis_count": len(register),
        "active_diagnostic_count": int(
            register[
                "status"
            ].eq(
                StructuralHypothesisStatus
                .ACTIVE_DIAGNOSTIC
                .value
            ).sum()
        ),
        "revision_required_count": int(
            register[
                "status"
            ].eq(
                StructuralHypothesisStatus
                .REVISION_REQUIRED
                .value
            ).sum()
        ),
        "deferred_count": int(
            register[
                "status"
            ].eq(
                StructuralHypothesisStatus
                .DEFERRED
                .value
            ).sum()
        ),
        "expert_authored": True,
        "empirically_validated_count": 0,
        "production_eligible_count": 0,
        "relationships_generated": False,
        "football_graph_created": False,
        "team_strength_changed": False,
        "simulation_run": False,
        "interpretation_boundary": (
            "Lifecycle status records research maturity only. "
            "It does not validate the football truth of any "
            "structural hypothesis."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Lifecycle summary")
    print("-" * 88)
    print(
        f"  Hypotheses: "
        f"{metadata['hypothesis_count']}"
    )
    print(
        "  Active diagnostic: "
        f"{metadata['active_diagnostic_count']}"
    )
    print(
        "  Revision required: "
        f"{metadata['revision_required_count']}"
    )
    print(
        "  Deferred: "
        f"{metadata['deferred_count']}"
    )
    print(
        "  Unique rule IDs: PASS"
    )
    print(
        "  Audit dispositions preserved: PASS"
    )
    print(
        "  Empirically validated hypotheses: 0"
    )
    print(
        "  Production-eligible hypotheses: 0"
    )
    print(
        "  Relationships generated: NO"
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
#validate_club_repository_schema

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.modeling.football_feature_registry import (
    get_club_goal_model_feature_spec,
)
from research.baselines.club_goal_model import (
    CURRENT_CLUB_GOAL_MODEL,
)
from scripts.club_team_repository_loader import (
    load_club_team_repository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Replace this path only if your current production club
# repository is stored elsewhere.
CLUB_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "team_repositories"
    / "premier_league_validation_repository.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_071a_club_repository_schema_alignment"
)

REPOSITORY_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "club_repository_schema_audit.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


EXPECTED_CANONICAL_FIELDS = {
    "attack",
    "midfield",
    "defense",
    "gk",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
    "squad_quality",
    "evidence_score",
    "poisson_attack",
    "poisson_defense",
    "rating_prior",
    "opta_rating",
    "rating_prior_source",
}


EXPECTED_BASELINE_SOURCE_FIELDS = {
    "attack",
    "defense",
    "attack_depth",
}


def validate_source_csv() -> pd.DataFrame:
    if not CLUB_REPOSITORY_PATH.exists():
        raise FileNotFoundError(
            "Configured production club repository does "
            "not exist: "
            f"{CLUB_REPOSITORY_PATH}"
        )

    dataframe = pd.read_csv(
        CLUB_REPOSITORY_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Configured production club repository is "
            "empty."
        )

    expected_csv_columns = {
        "club",
        "att_composite",
        "mid_composite",
        "def_composite",
        "gk_composite",
        "poisson_attack_adj",
        "poisson_defense_adj",
        "attack_depth",
        "midfield_depth",
        "defense_depth",
        "squad_quality",
        "evidence_score",
        "opta_rating",
        "rating_prior",
    }

    missing = (
        expected_csv_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Production club repository CSV is missing "
            "columns required by the aligned runtime "
            f"schema: {sorted(missing)}"
        )

    return dataframe


def build_repository_audit(
    source: pd.DataFrame,
    repository: dict[str, dict],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for club, entry in repository.items():
        missing_fields = sorted(
            EXPECTED_CANONICAL_FIELDS
            - set(entry)
        )

        numeric_fields = [
            field
            for field in EXPECTED_CANONICAL_FIELDS
            if field != "rating_prior_source"
        ]

        numeric_values = np.asarray(
            [
                float(entry[field])
                for field in numeric_fields
                if field in entry
            ],
            dtype=float,
        )

        finite_values_pass = bool(
            len(numeric_values)
            == len(numeric_fields)
            and np.isfinite(
                numeric_values
            ).all()
        )

        records.append(
            {
                "club": club,
                "missing_canonical_fields":
                    "|".join(missing_fields),
                "canonical_field_contract_pass":
                    not missing_fields,
                "finite_numeric_values_pass":
                    finite_values_pass,
                "rating_prior_source":
                    entry.get(
                        "rating_prior_source"
                    ),
                "rating_prior_source_pass":
                    entry.get(
                        "rating_prior_source"
                    )
                    == "opta_power_rating",
            }
        )

    audit = pd.DataFrame(records)

    if len(audit) != len(source):
        raise AssertionError(
            "Loaded repository row count differs from "
            "the source CSV row count."
        )

    if not audit[
        "canonical_field_contract_pass"
    ].all():
        raise AssertionError(
            "One or more loaded club entries failed the "
            "canonical schema contract."
        )

    if not audit[
        "finite_numeric_values_pass"
    ].all():
        raise AssertionError(
            "One or more loaded club entries contain "
            "missing or non-finite numeric values."
        )

    if not audit[
        "rating_prior_source_pass"
    ].all():
        raise AssertionError(
            "One or more loaded club entries contain an "
            "unexpected rating-prior source."
        )

    return audit


def validate_baseline_runtime_inputs(
    repository: dict[str, dict],
) -> None:
    CURRENT_CLUB_GOAL_MODEL.validate()

    specification = (
        CURRENT_CLUB_GOAL_MODEL
        .get_feature_specification()
    )

    registered = (
        get_club_goal_model_feature_spec(
            specification.name
        )
    )

    if registered != specification:
        raise AssertionError(
            "Baseline feature specification does not "
            "match the registered specification."
        )

    missing_by_club: dict[str, list[str]] = {}

    for club, entry in repository.items():
        missing = sorted(
            EXPECTED_BASELINE_SOURCE_FIELDS
            - set(entry)
        )

        if missing:
            missing_by_club[club] = missing

    if missing_by_club:
        raise AssertionError(
            "Repository cannot supply all direct inputs "
            "needed by the Version 1 live observation "
            f"builder: {missing_by_club}"
        )


def build_metadata(
    source: pd.DataFrame,
    repository: dict[str, dict],
    audit: pd.DataFrame,
) -> dict[str, object]:
    return {
        "study_id": "071A",
        "study_name": (
            "Club Repository Schema Alignment"
        ),
        "repository_path": str(
            CLUB_REPOSITORY_PATH
        ),
        "source_rows": int(
            len(source)
        ),
        "loaded_clubs": int(
            len(repository)
        ),
        "canonical_fields": sorted(
            EXPECTED_CANONICAL_FIELDS
        ),
        "baseline_source_fields": sorted(
            EXPECTED_BASELINE_SOURCE_FIELDS
        ),
        "row_preservation_pass": (
            len(source)
            == len(repository)
        ),
        "canonical_schema_pass": bool(
            audit[
                "canonical_field_contract_pass"
            ].all()
        ),
        "finite_numeric_values_pass": bool(
            audit[
                "finite_numeric_values_pass"
            ].all()
        ),
        "baseline_runtime_input_pass": True,
        "rating_prior_source_pass": bool(
            audit[
                "rating_prior_source_pass"
            ].all()
        ),
        "overall_result": "PASS",
    }


def write_report(
    metadata: dict[str, object],
) -> None:
    report = f"""# Study 071A — Club Repository Schema Alignment

## Purpose

Align the runtime club repository loader with the complete
validated team-representation schema already stored in the
club repository.

## Repository

- Path: `{metadata["repository_path"]}`
- Source rows: {metadata["source_rows"]}
- Loaded clubs: {metadata["loaded_clubs"]}

## Canonical runtime fields

{chr(10).join(
    f"- `{field}`"
    for field in metadata["canonical_fields"]
)}

## Version 1 direct repository inputs

{chr(10).join(
    f"- `{field}`"
    for field in metadata["baseline_source_fields"]
)}

`rating_prior_diff` is not taken from the static repository.
The production Live Match Observation Builder will derive it
from prediction-date ClubElo ratings.

## Validation

- Source file existence: PASS
- Required source columns: PASS
- Source row preservation: PASS
- Unique normalized club names: PASS
- Canonical runtime schema: PASS
- Finite numeric values: PASS
- Version 1 repository inputs: PASS
- Rating-prior metadata: PASS
- Baseline Registry integration: PASS
- Feature Registry integration: PASS

## Result

**OVERALL RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    source = validate_source_csv()

    repository = load_club_team_repository(
        CLUB_REPOSITORY_PATH
    )

    audit = build_repository_audit(
        source=source,
        repository=repository,
    )

    validate_baseline_runtime_inputs(
        repository
    )

    metadata = build_metadata(
        source=source,
        repository=repository,
        audit=audit,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        REPOSITORY_AUDIT_PATH,
        index=False,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(metadata)

    print(
        "Study 071A — Club Repository Schema Alignment"
    )
    print("=" * 76)
    print()
    print(
        f"Repository: {CLUB_REPOSITORY_PATH}"
    )
    print(
        f"Source rows: {len(source)}"
    )
    print(
        f"Loaded clubs: {len(repository)}"
    )
    print()
    print("Required source columns: PASS")
    print("Source row preservation: PASS")
    print("Unique normalized club names: PASS")
    print("Canonical runtime schema: PASS")
    print("Finite numeric values: PASS")
    print("Version 1 repository inputs: PASS")
    print("Rating-prior metadata: PASS")
    print("Baseline Registry integration: PASS")
    print("Feature Registry integration: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
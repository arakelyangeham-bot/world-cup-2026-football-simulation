#build_robust_premier_league_repository

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.production.build_premier_league_club_repository_v1 import (
    REPRESENTATION_FIELDS,
    build_team_appearance_rows,
    select_production_representations,
    validate_representation_consistency,
    validate_repository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_OBSERVATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_096_premier_league_robust_candidate"
    / "study_096b_clubelo_enrichment"
    / "full_squad_observations_robust_zscore_with_clubelo.csv"
)

FROZEN_GLOBAL_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_071a_premier_league_club_repository_v1"
    / "premier_league_club_repository_v1.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_096_premier_league_robust_candidate"
    / "study_096c_runtime_repository"
)

REPOSITORY_PATH = (
    OUTPUT_DIRECTORY
    / "premier_league_club_repository_robust_zscore.csv"
)

REPRESENTATION_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "club_representation_selection_audit.csv"
)

CONTROL_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "repository_control_comparison.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_096c_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_096C_RESULTS.md"
)

def load_candidate_observations() -> pd.DataFrame:
    if not SOURCE_OBSERVATION_PATH.exists():
        raise FileNotFoundError(
            "Study 096B candidate observations do not exist: "
            f"{SOURCE_OBSERVATION_PATH}"
        )

    dataframe = pd.read_csv(
        SOURCE_OBSERVATION_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Study 096B candidate observations are empty."
        )

    required_columns = {
        "event_id",
        "date",
        "home_team",
        "away_team",
    }

    for prefix in ("home", "away"):
        for field in REPRESENTATION_FIELDS:
            required_columns.add(
                f"{prefix}_{field}"
            )

        required_columns.update(
            {
                f"{prefix}_representation_type",
                f"{prefix}_representation_source",
                f"{prefix}_representation_season_id",
                f"{prefix}_representation_player_count",
                (
                    f"{prefix}_representation_"
                    "available_player_count"
                ),
            }
        )

    missing = required_columns - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Study 096B observations are missing repository "
            f"columns: {sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    if dataframe["event_id"].duplicated().any():
        raise ValueError(
            "Study 096B observations contain duplicate "
            "event IDs."
        )

    return dataframe

def build_control_comparison(
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not FROZEN_GLOBAL_REPOSITORY_PATH.exists():
        raise FileNotFoundError(
            "Frozen FM002 repository does not exist: "
            f"{FROZEN_GLOBAL_REPOSITORY_PATH}"
        )

    frozen = pd.read_csv(
        FROZEN_GLOBAL_REPOSITORY_PATH,
        low_memory=False,
    )

    if frozen.empty:
        raise ValueError(
            "Frozen FM002 repository is empty."
        )

    if candidate["club"].duplicated().any():
        raise AssertionError(
            "Candidate repository contains duplicate clubs."
        )

    if frozen["club"].duplicated().any():
        raise AssertionError(
            "Frozen repository contains duplicate clubs."
        )

    candidate_indexed = (
        candidate
        .set_index("club")
        .sort_index()
    )

    frozen_indexed = (
        frozen
        .set_index("club")
        .sort_index()
    )

    if not candidate_indexed.index.equals(
        frozen_indexed.index
    ):
        only_candidate = sorted(
            set(candidate_indexed.index)
            - set(frozen_indexed.index)
        )

        only_frozen = sorted(
            set(frozen_indexed.index)
            - set(candidate_indexed.index)
        )

        raise AssertionError(
            "Candidate and frozen repository club "
            "populations differ. "
            f"Only candidate={only_candidate}, "
            f"only frozen={only_frozen}."
        )

    rows: list[dict[str, object]] = []

    changed_fields: list[str] = []

    for club in candidate_indexed.index:
        for field in REPRESENTATION_FIELDS:
            candidate_value = float(
                candidate_indexed.loc[
                    club,
                    field,
                ]
            )

            frozen_value = float(
                frozen_indexed.loc[
                    club,
                    field,
                ]
            )

            difference = (
                candidate_value
                - frozen_value
            )

            changed = abs(
                difference
            ) > 1e-12

            if changed:
                changed_fields.append(
                    field
                )

            rows.append(
                {
                    "club": club,
                    "field": field,
                    "frozen_global_value":
                        frozen_value,
                    "robust_candidate_value":
                        candidate_value,
                    "robust_minus_global":
                        difference,
                    "changed":
                        changed,
                }
            )

    comparison = pd.DataFrame(
        rows
    )

    if not comparison["changed"].any():
        raise AssertionError(
            "The robust runtime repository is identical to "
            "the frozen global repository."
        )

    return comparison, {
        "club_population_equal":
            True,
        "club_count":
            len(candidate),
        "comparison_row_count":
            len(comparison),
        "changed_value_count":
            int(
                comparison[
                    "changed"
                ].sum()
            ),
        "changed_field_count":
            len(
                set(changed_fields)
            ),
        "changed_fields":
            sorted(
                set(changed_fields)
            ),
    }

def write_report(
    metadata: dict[str, object],
) -> None:
    changed_fields = "\n".join(
        f"- `{field}`"
        for field in metadata[
            "changed_fields"
        ]
    )

    report = f"""# Study 096C — Robust Premier League Runtime Repository

## Status

**PASS**

## Purpose

Build an isolated Premier League runtime repository from the
Study 096B `robust_zscore` ClubElo-enriched observations.

## Input

`{metadata["source_observation_path"]}`

## Output

`{metadata["repository_path"]}`

## Population

- Source observations:
  {metadata["source_observation_count"]}
- Team appearances:
  {metadata["team_appearance_count"]}
- Runtime clubs:
  {metadata["club_count"]}
- Club population equal to frozen FM002:
  {metadata["club_population_equal"]}

## Robust propagation

- Changed repository values:
  {metadata["changed_value_count"]}
- Changed representation fields:
  {metadata["changed_field_count"]}

{changed_fields}

## Selection policy

For each club, the final chronological full-squad
representation is selected using the established Study 071A
policy.

ClubElo is not stored in this repository. Prediction-date
rating priors continue to be resolved separately.

## Validation

- Study 096B source loading: PASS
- Home/away appearance projection: PASS
- Representation consistency audit: PASS
- Latest representation selection: PASS
- Unique-club contract: PASS
- Finite-value contract: PASS
- Frozen FM002 club population match: PASS
- Robust representation differences present: PASS
- Production defaults modified: NO
- Canonical files overwritten: NO

## Result

**OVERALL RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 096C — ROBUST PREMIER LEAGUE "
        "RUNTIME REPOSITORY"
    )
    print("=" * 88)

    observations = (
        load_candidate_observations()
    )

    appearances = (
        build_team_appearance_rows(
            observations
        )
    )

    audit = (
        validate_representation_consistency(
            appearances
        )
    )

    repository = (
        select_production_representations(
            appearances
        )
    )

    repository[
        "repository_version"
    ] = "096C-1.0"

    repository[
        "repository_scope"
    ] = (
        "premier_league_robust_candidate"
    )

    validate_repository(
        repository
    )

    numeric_values = repository[
        list(
            REPRESENTATION_FIELDS
        )
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise AssertionError(
            "Candidate runtime repository contains "
            "non-finite representation values."
        )

    comparison, comparison_summary = (
        build_control_comparison(
            repository
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    repository.to_csv(
        REPOSITORY_PATH,
        index=False,
    )

    audit.to_csv(
        REPRESENTATION_AUDIT_PATH,
        index=False,
    )

    comparison.to_csv(
        CONTROL_COMPARISON_PATH,
        index=False,
    )

    metadata = {
        "study_id":
            "096C",
        "study_name": (
            "Robust Premier League Runtime Repository"
        ),
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "status":
            "PASS",

        "source_observation_path":
            str(
                SOURCE_OBSERVATION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        "repository_path":
            str(
                REPOSITORY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        "frozen_global_repository_path":
            str(
                FROZEN_GLOBAL_REPOSITORY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "source_observation_count":
            len(observations),
        "team_appearance_count":
            len(appearances),

        **comparison_summary,

        "player_representation_transformation":
            "robust_zscore",
        "repository_version":
            "096C-1.0",
        "repository_scope":
            "premier_league_robust_candidate",

        "production_defaults_modified":
            False,
        "canonical_files_overwritten":
            False,

        "outputs": [
            REPOSITORY_PATH.name,
            REPRESENTATION_AUDIT_PATH.name,
            CONTROL_COMPARISON_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        metadata
    )

    print()
    print("Runtime repository population")
    print("-" * 88)
    print(
        f"  Source observations: "
        f"{len(observations)}"
    )
    print(
        f"  Team appearances: "
        f"{len(appearances)}"
    )
    print(
        f"  Runtime clubs: "
        f"{len(repository)}"
    )
    print(
        "  Frozen FM002 club population: MATCH"
    )
    print(
        "  Changed repository values: "
        f"{comparison_summary['changed_value_count']}"
    )
    print(
        "  Changed representation fields: "
        f"{comparison_summary['changed_field_count']}"
    )

    print()
    print("Validation summary")
    print("  Candidate observation loading: PASS")
    print("  Team-appearance projection: PASS")
    print("  Representation consistency audit: PASS")
    print("  Latest representation selection: PASS")
    print("  Runtime repository schema: PASS")
    print("  Unique-club contract: PASS")
    print("  Finite-value contract: PASS")
    print("  Frozen club population comparison: PASS")
    print("  Robust representation propagation: PASS")
    print("  Production defaults modified: NO")
    print("  Canonical files overwritten: NO")

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
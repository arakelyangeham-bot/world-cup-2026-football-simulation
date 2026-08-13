#build_robust_clubelo_enriched_observations

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.data_pipeline.build_clubelo_enriched_observations import (
    ENRICHMENT_COLUMNS,
    PRIOR_REQUIRED_COLUMNS,
    SOURCE_REQUIRED_COLUMNS,
    build_enriched_observations,
    build_population_audit,
    load_csv,
    normalize_dates,
    remove_rating_prior_placeholders,
    validate_population_audit,
    validate_rating_priors,
    validate_source_observations,
    validate_temporal_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_OBSERVATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_096_premier_league_robust_candidate"
    / "study_096a_observations"
    / "full_squad_observations_robust_zscore.csv"
)

RATING_PRIOR_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_059_rating_prior_provider"
    / "match_rating_priors.csv"
)

FROZEN_GLOBAL_ENRICHED_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_060_clubelo_enriched_observations"
    / "full_squad_observations_with_clubelo.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_096_premier_league_robust_candidate"
    / "study_096b_clubelo_enrichment"
)

OUTPUT_OBSERVATION_PATH = (
    OUTPUT_DIRECTORY
    / "full_squad_observations_robust_zscore_with_clubelo.csv"
)

MERGE_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "clubelo_enrichment_audit.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "study_096b_summary.json"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_096b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_096B_RESULTS.md"
)

def validate_inputs() -> None:
    required_paths = {
        "robust Study 096A observations":
            SOURCE_OBSERVATION_PATH,
        "fixture-date ClubElo priors":
            RATING_PRIOR_PATH,
        "frozen global enriched observations":
            FROZEN_GLOBAL_ENRICHED_PATH,
    }

    missing = {
        label: path
        for label, path in required_paths.items()
        if not path.exists()
    }

    if missing:
        details = "\n".join(
            f"- {label}: {path}"
            for label, path in missing.items()
        )

        raise FileNotFoundError(
            "Study 096B is missing required inputs:\n"
            f"{details}"
        )

    empty = {
        label: path
        for label, path in required_paths.items()
        if path.stat().st_size <= 0
    }

    if empty:
        details = "\n".join(
            f"- {label}: {path}"
            for label, path in empty.items()
        )

        raise ValueError(
            "Study 096B found empty input artifacts:\n"
            f"{details}"
        )

def validate_against_frozen_global(
    candidate: pd.DataFrame,
) -> dict[str, object]:
    frozen = pd.read_csv(
        FROZEN_GLOBAL_ENRICHED_PATH,
        low_memory=False,
    )

    if frozen.empty:
        raise ValueError(
            "Frozen global enriched observation dataset "
            "is empty."
        )

    candidate = candidate.copy()
    frozen = frozen.copy()

    candidate["event_id"] = pd.to_numeric(
        candidate["event_id"],
        errors="raise",
    ).astype(int)

    frozen["event_id"] = pd.to_numeric(
        frozen["event_id"],
        errors="raise",
    ).astype(int)

    if candidate["event_id"].duplicated().any():
        raise AssertionError(
            "Candidate enriched observations contain "
            "duplicate event IDs."
        )

    if frozen["event_id"].duplicated().any():
        raise AssertionError(
            "Frozen global enriched observations contain "
            "duplicate event IDs."
        )

    candidate_indexed = (
        candidate
        .set_index("event_id")
        .sort_index()
    )

    frozen_indexed = (
        frozen
        .set_index("event_id")
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
            "Candidate and frozen enriched event populations "
            "differ. "
            f"Only candidate={only_candidate[:20]}, "
            f"only frozen={only_frozen[:20]}."
        )

    identity_and_target_columns = (
        "date",
        "home_team",
        "home_team_id",
        "away_team",
        "away_team_id",
        "home_score",
        "away_score",
        "total_goals",
        "goal_difference",
        "result",
    )

    for column in identity_and_target_columns:
        left = candidate_indexed[column]
        right = frozen_indexed[column]

        if column == "date":
            left_values = pd.to_datetime(
                left,
                errors="raise",
                utc=True,
            )

            right_values = pd.to_datetime(
                right,
                errors="raise",
                utc=True,
            )

            equal = left_values.equals(
                right_values
            )

        elif pd.api.types.is_numeric_dtype(left):
            equal = np.allclose(
                pd.to_numeric(
                    left,
                    errors="raise",
                ).to_numpy(dtype=float),
                pd.to_numeric(
                    right,
                    errors="raise",
                ).to_numpy(dtype=float),
                atol=0.0,
                rtol=0.0,
            )

        else:
            equal = (
                left.fillna("<missing>")
                .astype(str)
                .equals(
                    right.fillna("<missing>")
                    .astype(str)
                )
            )

        if not equal:
            raise AssertionError(
                "Candidate and frozen enriched datasets "
                f"differ in matched column {column!r}."
            )

    prior_columns = (
        "home_rating_prior",
        "away_rating_prior",
        "rating_prior_diff",
        "rating_prior_source",
        "rating_prior_available",
        "home_rating_effective_from",
        "home_rating_effective_to",
        "away_rating_effective_from",
        "away_rating_effective_to",
    )

    for column in prior_columns:
        left = candidate_indexed[column]
        right = frozen_indexed[column]

        if column in {
            "home_rating_prior",
            "away_rating_prior",
            "rating_prior_diff",
        }:
            equal = np.allclose(
                pd.to_numeric(
                    left,
                    errors="raise",
                ).to_numpy(dtype=float),
                pd.to_numeric(
                    right,
                    errors="raise",
                ).to_numpy(dtype=float),
                atol=1e-12,
                rtol=0.0,
            )
        else:
            equal = (
                left.fillna("<missing>")
                .astype(str)
                .equals(
                    right.fillna("<missing>")
                    .astype(str)
                )
            )

        if not equal:
            raise AssertionError(
                "Candidate and frozen enriched datasets "
                f"differ in ClubElo column {column!r}."
            )

    representation_fields = (
        "attack",
        "midfield",
        "defense",
        "goalkeeper",
        "attack_depth",
        "midfield_depth",
        "defense_depth",
        "squad_quality",
        "evidence_score",
    )

    changed_representation_columns: list[str] = []

    for prefix in (
        "home",
        "away",
    ):
        for field in representation_fields:
            column = f"{prefix}_{field}"

            left = pd.to_numeric(
                candidate_indexed[column],
                errors="raise",
            ).to_numpy(dtype=float)

            right = pd.to_numeric(
                frozen_indexed[column],
                errors="raise",
            ).to_numpy(dtype=float)

            if not np.allclose(
                left,
                right,
                atol=1e-12,
                rtol=0.0,
            ):
                changed_representation_columns.append(
                    column
                )

    if not changed_representation_columns:
        raise AssertionError(
            "No representation columns differ between the "
            "robust candidate and frozen global dataset. "
            "This suggests that the candidate ratings were "
            "not propagated into the observations."
        )

    return {
        "frozen_global_row_count":
            len(frozen),
        "candidate_row_count":
            len(candidate),
        "event_population_equal":
            True,
        "match_identity_equal":
            True,
        "target_population_equal":
            True,
        "clubelo_prior_values_equal":
            True,
        "changed_representation_column_count":
            len(
                changed_representation_columns
            ),
        "changed_representation_columns":
            changed_representation_columns,
    }

def build_candidate_summary(
    *,
    source_observations: pd.DataFrame,
    priors: pd.DataFrame,
    enriched: pd.DataFrame,
    population_audit: pd.DataFrame,
    temporal_audit: pd.DataFrame,
    frozen_comparison: dict[str, object],
) -> dict[str, object]:
    rating_difference_valid = bool(
        np.allclose(
            enriched[
                "rating_prior_diff"
            ].to_numpy(dtype=float),
            (
                enriched[
                    "home_rating_prior"
                ].to_numpy(dtype=float)
                - enriched[
                    "away_rating_prior"
                ].to_numpy(dtype=float)
            ),
            atol=1e-10,
            rtol=0.0,
        )
    )

    if not rating_difference_valid:
        raise AssertionError(
            "Candidate rating-prior differences failed "
            "arithmetic validation."
        )

    return {
        "study":
            "study_096b_robust_clubelo_enrichment",
        "source_observation_path":
            str(
                SOURCE_OBSERVATION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        "rating_prior_path":
            str(
                RATING_PRIOR_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        "output_observation_path":
            str(
                OUTPUT_OBSERVATION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "source_observation_rows":
            len(source_observations),
        "rating_prior_rows":
            len(priors),
        "enriched_observation_rows":
            len(enriched),
        "unique_events":
            int(
                enriched[
                    "event_id"
                ].nunique()
            ),

        "rating_prior_source_values":
            sorted(
                enriched[
                    "rating_prior_source"
                ]
                .astype(str)
                .unique()
                .tolist()
            ),

        "population_validation_pass":
            bool(
                population_audit[
                    "population_validation_pass"
                ].all()
            ),
        "temporal_validity_pass":
            bool(
                temporal_audit[
                    "temporal_validity_pass"
                ].all()
            ),
        "rating_difference_validation_pass":
            rating_difference_valid,

        "home_rating_prior_mean":
            float(
                enriched[
                    "home_rating_prior"
                ].mean()
            ),
        "away_rating_prior_mean":
            float(
                enriched[
                    "away_rating_prior"
                ].mean()
            ),
        "rating_prior_difference_mean":
            float(
                enriched[
                    "rating_prior_diff"
                ].mean()
            ),

        **frozen_comparison,
    }

def write_report(
    metadata: dict[str, object],
) -> None:
    changed_columns = "\n".join(
        f"- `{column}`"
        for column in metadata[
            "changed_representation_columns"
        ]
    )

    report = f"""# Study 096B — Robust ClubElo-Enriched Observation Candidate

## Status

**PASS**

## Purpose

Attach the frozen fixture-date ClubElo priors from Study 059
to the isolated Premier League `robust_zscore` observation
candidate produced by Study 096A.

## Inputs

- Robust observations:
  `{metadata["source_observation_path"]}`
- ClubElo priors:
  `{metadata["rating_prior_path"]}`

## Output

`{metadata["output_observation_path"]}`

## Population

- Source observations:
  {metadata["source_observation_rows"]}
- Rating-prior rows:
  {metadata["rating_prior_rows"]}
- Enriched observations:
  {metadata["enriched_observation_rows"]}
- Unique events:
  {metadata["unique_events"]}

## Frozen FM002 control

- Event population equal:
  {metadata["event_population_equal"]}
- Match identities equal:
  {metadata["match_identity_equal"]}
- Targets equal:
  {metadata["target_population_equal"]}
- ClubElo values equal:
  {metadata["clubelo_prior_values_equal"]}
- Changed player-representation columns:
  {metadata["changed_representation_column_count"]}

{changed_columns}

## Validation

- Study 096A source loading: PASS
- ClubElo prior loading: PASS
- Exact event population match: PASS
- Match-date and team agreement: PASS
- Complete ClubElo coverage: PASS
- Rating-difference arithmetic: PASS
- ClubElo source provenance: PASS
- Historical temporal validity: PASS
- Source row preservation: PASS
- Frozen FM002 identity and targets preserved: PASS
- Frozen FM002 ClubElo priors preserved: PASS
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
        "STUDY 096B — ROBUST CLUB ELO-ENRICHED "
        "OBSERVATION CANDIDATE"
    )
    print("=" * 88)

    validate_inputs()

    observations = load_csv(
        path=SOURCE_OBSERVATION_PATH,
        required_columns=SOURCE_REQUIRED_COLUMNS,
        label="Study 096A robust observations",
    )

    priors = load_csv(
        path=RATING_PRIOR_PATH,
        required_columns=PRIOR_REQUIRED_COLUMNS,
        label="Study 059 rating priors",
    )

    observations, priors = normalize_dates(
        observations=observations,
        priors=priors,
    )

    validate_source_observations(
        observations
    )

    source_observations = (
        observations.copy()
    )

    observations = (
        remove_rating_prior_placeholders(
            observations
        )
    )

    validate_rating_priors(
        priors
    )

    population_audit = build_population_audit(
        observations=observations,
        priors=priors,
    )

    validate_population_audit(
        audit=population_audit,
        expected_rows=len(observations),
    )

    enriched = build_enriched_observations(
        observations=observations,
        priors=priors,
    )

    temporal_audit = validate_temporal_provenance(
        enriched
    )

    frozen_comparison = (
        validate_against_frozen_global(
            enriched
        )
    )

    summary = build_candidate_summary(
        source_observations=(
            source_observations
        ),
        priors=priors,
        enriched=enriched,
        population_audit=population_audit,
        temporal_audit=temporal_audit,
        frozen_comparison=frozen_comparison,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched_output = enriched.copy()

    enriched_output["date"] = (
        pd.to_datetime(
            enriched_output["date"],
            errors="raise",
            utc=True,
        )
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    enriched_output.to_csv(
        OUTPUT_OBSERVATION_PATH,
        index=False,
    )

    output_audit = population_audit.merge(
        temporal_audit[
            [
                "event_id",
                "home_temporal_validity_pass",
                "away_temporal_validity_pass",
                "temporal_validity_pass",
            ]
        ],
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    output_audit.to_csv(
        MERGE_AUDIT_PATH,
        index=False,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "study_id":
            "096B",
        "study_name": (
            "Robust ClubElo-Enriched Observation Candidate"
        ),
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "status":
            "PASS",

        **summary,

        "player_representation_transformation":
            "robust_zscore",
        "rating_prior_source":
            "clubelo",
        "production_defaults_modified":
            False,
        "canonical_files_overwritten":
            False,

        "outputs": [
            OUTPUT_OBSERVATION_PATH.name,
            MERGE_AUDIT_PATH.name,
            SUMMARY_PATH.name,
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
    print("Candidate enrichment population")
    print("-" * 88)
    print(
        "  Source observations: "
        f"{len(source_observations)}"
    )
    print(
        "  Rating-prior rows: "
        f"{len(priors)}"
    )
    print(
        "  Enriched observations: "
        f"{len(enriched)}"
    )
    print(
        "  Unique events: "
        f"{enriched['event_id'].nunique()}"
    )

    print()
    print("Frozen FM002 comparison")
    print("-" * 88)
    print("  Event population: MATCH")
    print("  Match identities: MATCH")
    print("  Targets: MATCH")
    print("  ClubElo prior values: MATCH")
    print(
        "  Changed representation columns: "
        f"{frozen_comparison['changed_representation_column_count']}"
    )

    print()
    print("Validation summary")
    print("  Robust source loading: PASS")
    print("  ClubElo prior loading: PASS")
    print("  Population matching: PASS")
    print("  ClubElo enrichment: PASS")
    print("  Rating-difference arithmetic: PASS")
    print("  Historical temporal validity: PASS")
    print("  Frozen FM002 control comparison: PASS")
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
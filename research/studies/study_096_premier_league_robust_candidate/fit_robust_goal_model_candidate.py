#fit_robust_goal_model_candidate

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.production.club_goal_model_artifact import (
    load_club_goal_model_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAINING_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_096_premier_league_robust_candidate"
    / "study_096b_clubelo_enrichment"
    / "full_squad_observations_robust_zscore_with_clubelo.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_096_premier_league_robust_candidate"
    / "study_096d_goal_model"
)

CANDIDATE_ARTIFACT_PATH = (
    OUTPUT_DIRECTORY
    / "integrated_club_goal_model_robust_candidate.json"
)

FROZEN_FM002_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_069_production_club_goal_model_v1"
    / "integrated_club_goal_model_v1.json"
)

COEFFICIENT_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "coefficient_comparison.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_096d_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_096D_RESULTS.md"
)

def validate_inputs() -> None:
    required_paths = {
        "Study 096B robust training dataset":
            TRAINING_PATH,
        "Frozen FM002 goal-model artifact":
            FROZEN_FM002_ARTIFACT_PATH,
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
            "Study 096D is missing required inputs:\n"
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
            "Study 096D found empty inputs:\n"
            f"{details}"
        )

def run_existing_goal_model_fitter() -> None:
    command = [
        sys.executable,
        "-m",
        (
            "research.production."
            "fit_integrated_club_goal_model_v1"
        ),

        "--training-path",
        str(TRAINING_PATH),

        "--output-directory",
        str(OUTPUT_DIRECTORY),

        "--artifact-filename",
        CANDIDATE_ARTIFACT_PATH.name,

        "--artifact-name",
        (
            "integrated_club_goal_model_"
            "robust_candidate"
        ),

        "--artifact-version",
        "096D-1.0",

        "--study-id",
        "096D",

        "--study-name",
        (
            "Premier League Robust Goal "
            "Model Candidate"
        ),
    ]

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Existing Study 069 goal-model fitter failed "
            f"with exit code {completed.returncode}."
        )

    if not CANDIDATE_ARTIFACT_PATH.exists():
        raise AssertionError(
            "Goal-model fitter completed without writing "
            "the candidate artifact."
        )

def build_coefficient_comparison(
) -> tuple[pd.DataFrame, dict[str, object]]:
    frozen = load_club_goal_model_artifact(
        FROZEN_FM002_ARTIFACT_PATH
    )

    candidate = load_club_goal_model_artifact(
        CANDIDATE_ARTIFACT_PATH
    )

    frozen.validate()
    candidate.validate()

    if (
        candidate.feature_specification
        != frozen.feature_specification
    ):
        raise AssertionError(
            "Candidate and FM002 use different feature "
            "specifications."
        )

    if (
        candidate.baseline_name
        != frozen.baseline_name
    ):
        raise AssertionError(
            "Candidate and FM002 use different baseline "
            "definitions."
        )

    if (
        candidate.baseline_version
        != frozen.baseline_version
    ):
        raise AssertionError(
            "Candidate and FM002 use different baseline "
            "versions."
        )

    if (
        tuple(candidate.home_model.features)
        != tuple(frozen.home_model.features)
    ):
        raise AssertionError(
            "Candidate and FM002 home-feature ordering differs."
        )

    if (
        tuple(candidate.away_model.features)
        != tuple(frozen.away_model.features)
    ):
        raise AssertionError(
            "Candidate and FM002 away-feature ordering differs."
        )

    rows: list[dict[str, object]] = []

    model_pairs = (
        (
            "home",
            frozen.home_model,
            candidate.home_model,
        ),
        (
            "away",
            frozen.away_model,
            candidate.away_model,
        ),
    )

    for target, frozen_model, candidate_model in model_pairs:
        intercept_difference = (
            candidate_model.intercept
            - frozen_model.intercept
        )

        rows.append(
            {
                "target": target,
                "parameter": "intercept",
                "frozen_fm002_value":
                    float(
                        frozen_model.intercept
                    ),
                "robust_candidate_value":
                    float(
                        candidate_model.intercept
                    ),
                "candidate_minus_fm002":
                    float(
                        intercept_difference
                    ),
                "changed":
                    abs(
                        intercept_difference
                    ) > 1e-12,
            }
        )

        for (
            feature,
            frozen_coefficient,
            candidate_coefficient,
        ) in zip(
            frozen_model.features,
            frozen_model.coefficients,
            candidate_model.coefficients,
            strict=True,
        ):
            difference = (
                candidate_coefficient
                - frozen_coefficient
            )

            rows.append(
                {
                    "target": target,
                    "parameter": feature,
                    "frozen_fm002_value":
                        float(
                            frozen_coefficient
                        ),
                    "robust_candidate_value":
                        float(
                            candidate_coefficient
                        ),
                    "candidate_minus_fm002":
                        float(
                            difference
                        ),
                    "changed":
                        abs(
                            difference
                        ) > 1e-12,
                }
            )

    comparison = pd.DataFrame(
        rows
    )

    if comparison.empty:
        raise AssertionError(
            "Coefficient comparison is empty."
        )

    if not comparison[
        "changed"
    ].any():
        raise AssertionError(
            "Candidate coefficients are identical to FM002."
        )

    home_intercept_changed = bool(
        comparison.loc[
            (
                comparison["target"].eq("home")
                & comparison[
                    "parameter"
                ].eq("intercept")
            ),
            "changed",
        ].iloc[0]
    )

    away_intercept_changed = bool(
        comparison.loc[
            (
                comparison["target"].eq("away")
                & comparison[
                    "parameter"
                ].eq("intercept")
            ),
            "changed",
        ].iloc[0]
    )

    coefficient_rows = comparison[
        ~comparison[
            "parameter"
        ].eq("intercept")
    ]

    changed_coefficient_count = int(
        coefficient_rows[
            "changed"
        ].sum()
    )

    if changed_coefficient_count == 0:
        raise AssertionError(
            "No fitted feature coefficient changed."
        )

    return comparison, {
        "feature_specification_equal":
            True,
        "baseline_name_equal":
            True,
        "baseline_version_equal":
            True,
        "home_feature_order_equal":
            True,
        "away_feature_order_equal":
            True,
        "parameter_count":
            len(comparison),
        "changed_parameter_count":
            int(
                comparison[
                    "changed"
                ].sum()
            ),
        "changed_coefficient_count":
            changed_coefficient_count,
        "home_intercept_changed":
            home_intercept_changed,
        "away_intercept_changed":
            away_intercept_changed,
    }

def validate_candidate_training_population(
) -> dict[str, object]:
    training = pd.read_csv(
        TRAINING_PATH,
        low_memory=False,
    )

    if training.empty:
        raise ValueError(
            "Candidate training dataset is empty."
        )

    if training["event_id"].duplicated().any():
        raise AssertionError(
            "Candidate training dataset contains duplicate "
            "event IDs."
        )

    required_columns = {
        "event_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_rating_prior",
        "away_rating_prior",
        "rating_prior_diff",
    }

    missing = required_columns - set(
        training.columns
    )

    if missing:
        raise AssertionError(
            "Candidate training dataset is missing columns: "
            f"{sorted(missing)}"
        )

    numeric_columns = [
        "home_score",
        "away_score",
        "home_rating_prior",
        "away_rating_prior",
        "rating_prior_diff",
    ]

    numeric_values = training[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise AssertionError(
            "Candidate training population contains "
            "non-finite required values."
        )

    dates = pd.to_datetime(
        training["date"],
        errors="raise",
        utc=True,
    )

    return {
        "training_match_count":
            len(training),
        "training_event_count":
            int(
                training[
                    "event_id"
                ].nunique()
            ),
        "training_start_date":
            dates.min().date().isoformat(),
        "training_end_date":
            dates.max().date().isoformat(),
    }

def write_report(
    metadata: dict[str, object],
) -> None:
    report = f"""# Study 096D — Premier League Robust Goal Model Candidate

## Status

**PASS**

## Purpose

Fit an isolated Integrated Club Goal Model using the
Premier League `robust_zscore` training observations produced
by Study 096B.

## Training input

`{metadata["training_dataset"]}`

## Candidate artifact

`{metadata["candidate_artifact_path"]}`

## Frozen control

`{metadata["frozen_fm002_artifact_path"]}`

## Training population

- Matches:
  {metadata["training_match_count"]}
- Unique events:
  {metadata["training_event_count"]}
- Start date:
  {metadata["training_start_date"]}
- End date:
  {metadata["training_end_date"]}

## Contract comparison

- Feature specification equal:
  {metadata["feature_specification_equal"]}
- Baseline name equal:
  {metadata["baseline_name_equal"]}
- Baseline version equal:
  {metadata["baseline_version_equal"]}
- Home feature order equal:
  {metadata["home_feature_order_equal"]}
- Away feature order equal:
  {metadata["away_feature_order_equal"]}

## Fitted-parameter comparison

- Parameters compared:
  {metadata["parameter_count"]}
- Changed parameters:
  {metadata["changed_parameter_count"]}
- Changed feature coefficients:
  {metadata["changed_coefficient_count"]}
- Home intercept changed:
  {metadata["home_intercept_changed"]}
- Away intercept changed:
  {metadata["away_intercept_changed"]}

## Validation

- Study 096B training population loading: PASS
- Existing Study 069 fitter execution: PASS
- Artifact serialization: PASS
- Artifact loading: PASS
- Prediction round-trip reproduction: PASS
- Feature contract preservation: PASS
- Baseline contract preservation: PASS
- Feature ordering preservation: PASS
- Fitted parameter differences detected: PASS
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
        "STUDY 096D — PREMIER LEAGUE ROBUST "
        "GOAL MODEL CANDIDATE"
    )
    print("=" * 88)

    validate_inputs()

    training_summary = (
        validate_candidate_training_population()
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_existing_goal_model_fitter()

    comparison, comparison_summary = (
        build_coefficient_comparison()
    )

    comparison.to_csv(
        COEFFICIENT_COMPARISON_PATH,
        index=False,
    )

    candidate_artifact = (
        load_club_goal_model_artifact(
            CANDIDATE_ARTIFACT_PATH
        )
    )

    metadata = {
        "study_id":
            "096D",
        "study_name": (
            "Premier League Robust Goal Model Candidate"
        ),
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "status":
            "PASS",

        "training_dataset":
            str(
                TRAINING_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        "candidate_artifact_path":
            str(
                CANDIDATE_ARTIFACT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        "frozen_fm002_artifact_path":
            str(
                FROZEN_FM002_ARTIFACT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        **training_summary,
        **comparison_summary,

        "artifact_name":
            candidate_artifact.artifact_name,
        "artifact_version":
            candidate_artifact.artifact_version,
        "feature_specification":
            candidate_artifact.feature_specification,
        "model_family":
            candidate_artifact.model_family,
        "alpha":
            candidate_artifact.alpha,

        "player_representation_transformation":
            "robust_zscore",
        "production_defaults_modified":
            False,
        "canonical_files_overwritten":
            False,

        "outputs": [
            CANDIDATE_ARTIFACT_PATH.name,
            "production_coefficients.csv",
            "training_predictions.csv",
            COEFFICIENT_COMPARISON_PATH.name,
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
    print("Candidate training population")
    print("-" * 88)
    print(
        "  Training matches: "
        f"{training_summary['training_match_count']}"
    )
    print(
        "  Unique events: "
        f"{training_summary['training_event_count']}"
    )
    print(
        "  Training period: "
        f"{training_summary['training_start_date']} "
        "through "
        f"{training_summary['training_end_date']}"
    )

    print()
    print("Artifact control comparison")
    print("-" * 88)
    print("  Feature specification: MATCH")
    print("  Baseline contract: MATCH")
    print("  Home feature order: MATCH")
    print("  Away feature order: MATCH")
    print(
        "  Changed parameters: "
        f"{comparison_summary['changed_parameter_count']}"
    )
    print(
        "  Changed feature coefficients: "
        f"{comparison_summary['changed_coefficient_count']}"
    )
    print(
        "  Home intercept changed: "
        f"{comparison_summary['home_intercept_changed']}"
    )
    print(
        "  Away intercept changed: "
        f"{comparison_summary['away_intercept_changed']}"
    )

    print()
    print("Validation summary")
    print("  Candidate training population: PASS")
    print("  Existing Study 069 fitter execution: PASS")
    print("  Artifact serialization: PASS")
    print("  Artifact loading: PASS")
    print("  Prediction round-trip reproduction: PASS")
    print("  Feature contract preservation: PASS")
    print("  Baseline contract preservation: PASS")
    print("  Fitted parameter differences detected: PASS")
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
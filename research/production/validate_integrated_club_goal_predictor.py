#validate_integrated_club_goal_predictor

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.production.club_goal_model_artifact import (
    load_club_goal_model_artifact,
)
from simulation.integrated_club_goal_predictor import (
    DEFAULT_CLUB_GOAL_MODEL_ARTIFACT,
    IntegratedClubGoalPredictor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_069_production_club_goal_model_v1"
    / "training_predictions.csv"
)

TRAINING_OBSERVATIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_060_clubelo_enriched_observations"
    / "full_squad_observations_with_clubelo.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_070_production_goal_model_interface"
)

VALIDATION_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "prediction_interface_validation.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


def load_validation_population() -> pd.DataFrame:
    if not TRAINING_OBSERVATIONS_PATH.exists():
        raise FileNotFoundError(
            "Study 060 observations do not exist: "
            f"{TRAINING_OBSERVATIONS_PATH}"
        )

    if not TRAINING_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Study 069 training predictions do not exist: "
            f"{TRAINING_PREDICTIONS_PATH}"
        )

    observations = pd.read_csv(
        TRAINING_OBSERVATIONS_PATH,
        low_memory=False,
    )

    predictions = pd.read_csv(
        TRAINING_PREDICTIONS_PATH,
        low_memory=False,
    )

    required_prediction_columns = {
        "event_id",
        "pred_home_goals",
        "pred_away_goals",
    }

    missing_predictions = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_predictions:
        raise ValueError(
            "Training prediction file is missing "
            f"columns: {sorted(missing_predictions)}"
        )

    merged = observations.merge(
        predictions[
            [
                "event_id",
                "pred_home_goals",
                "pred_away_goals",
            ]
        ],
        on="event_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(observations):
        raise AssertionError(
            "Validation merge did not preserve the full "
            "observation population."
        )

    return merged


def validate_prediction_interface(
    population: pd.DataFrame,
) -> pd.DataFrame:
    artifact = load_club_goal_model_artifact(
        DEFAULT_CLUB_GOAL_MODEL_ARTIFACT
    )

    predictor = IntegratedClubGoalPredictor(
        enforce_post_training_prediction_date=False,
    )

    required_features = (
        predictor.required_features
    )

    expected_required_features = tuple(
        dict.fromkeys(
            (
                *artifact.home_model.features,
                *artifact.away_model.features,
            )
        )
    )

    if (
        required_features
        != expected_required_features
    ):
        raise AssertionError(
            "Predictor required-feature contract differs "
            "from the artifact contract."
        )

    records: list[dict[str, object]] = []

    for row in population.to_dict(
        orient="records"
    ):
        feature_values = {
            feature: float(row[feature])
            for feature in required_features
        }

        prediction = predictor.predict_features(
            feature_values
        )

        records.append(
            {
                "event_id": row["event_id"],
                "expected_home_goals":
                    float(row["pred_home_goals"]),
                "expected_away_goals":
                    float(row["pred_away_goals"]),
                "interface_home_goals":
                    prediction.lambda_home,
                "interface_away_goals":
                    prediction.lambda_away,
                "home_difference":
                    prediction.lambda_home
                    - float(row["pred_home_goals"]),
                "away_difference":
                    prediction.lambda_away
                    - float(row["pred_away_goals"]),
                "artifact_name":
                    prediction.artifact_name,
                "artifact_version":
                    prediction.artifact_version,
                "baseline_version":
                    prediction.baseline_version,
                "feature_specification":
                    prediction.feature_specification,
                "training_end_date":
                    prediction.training_end_date,
            }
        )

    results = pd.DataFrame(records)

    if not np.allclose(
        results["interface_home_goals"],
        results["expected_home_goals"],
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError(
            "Production predictor does not reproduce "
            "Study 069 home predictions."
        )

    if not np.allclose(
        results["interface_away_goals"],
        results["expected_away_goals"],
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError(
            "Production predictor does not reproduce "
            "Study 069 away predictions."
        )

    return results


def validate_missing_feature_contract() -> None:
    predictor = IntegratedClubGoalPredictor(
        enforce_post_training_prediction_date=False,
    )

    incomplete_features = {
        feature: 0.0
        for feature in predictor.required_features
        if feature != "rating_prior_diff"
    }

    try:
        predictor.predict_features(
            incomplete_features
        )
    except KeyError:
        return

    raise AssertionError(
        "Predictor accepted a feature mapping with a "
        "missing required feature."
    )


def validate_training_cutoff_contract() -> None:
    predictor = IntegratedClubGoalPredictor(
        enforce_post_training_prediction_date=True,
    )

    feature_values = {
        feature: 0.0
        for feature in predictor.required_features
    }

    try:
        predictor.predict_features(
            feature_values,
            prediction_date=(
                predictor.model.training_end_date
            ),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Predictor accepted a prediction date on its "
            "training cutoff."
        )

    predictor.predict_features(
        feature_values,
        prediction_date="2025-05-26",
    )


def build_metadata(
    results: pd.DataFrame,
) -> dict[str, object]:
    return {
        "study_id": "070",
        "study_name": (
            "Production Goal Model Interface"
        ),
        "artifact_path": str(
            DEFAULT_CLUB_GOAL_MODEL_ARTIFACT
        ),
        "validation_match_count": len(results),
        "maximum_home_prediction_difference":
            float(
                results[
                    "home_difference"
                ].abs().max()
            ),
        "maximum_away_prediction_difference":
            float(
                results[
                    "away_difference"
                ].abs().max()
            ),
        "artifact_loading_pass": True,
        "required_feature_contract_pass": True,
        "prediction_reproduction_pass": True,
        "missing_feature_rejection_pass": True,
        "training_cutoff_guard_pass": True,
        "overall_result": "PASS",
    }


def write_report(
    metadata: dict[str, object],
) -> None:
    report = f"""# Study 070 — Production Goal Model Interface

## Purpose

Introduce the runtime interface used to evaluate the frozen
Integrated Club Goal Model v1 without fitting a model or
loading research benchmark infrastructure.

## Architecture

```text
Production JSON artifact
        ↓
ProductionGoalModel
        ↓
IntegratedClubGoalPredictor
        ↓
Expected home and away goals
Validation population
Matches: {metadata["validation_match_count"]}
Source predictions: Study 069
Artifact: integrated_club_goal_model_v1.json
Validation
Artifact loading: PASS
Artifact metadata exposure: PASS
Required feature contract: PASS
Missing feature rejection: PASS
Training-cutoff guard: PASS
Home prediction reproduction: PASS
Away prediction reproduction: PASS
Maximum reproduction differences
Home:
{metadata["maximum_home_prediction_difference"]:.16e}
Away:
{metadata["maximum_away_prediction_difference"]:.16e}
Boundary established

The production predictor:

performs no fitting;
does not load the benchmark engine;
does not depend on the training dataset during prediction;
receives a complete feature mapping;
returns scalar expected goals;
records artifact and baseline provenance;
protects against accidental retrospective use through its
optional training-cutoff guard.

Live club feature assembly remains outside this study.

Result

OVERALL RESULT: PASS
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

def main() -> None:
    population = load_validation_population()

    results = validate_prediction_interface(
        population
    )

    validate_missing_feature_contract()
    validate_training_cutoff_contract()

    metadata = build_metadata(
        results
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        VALIDATION_RESULTS_PATH,
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
        "Study 070 — Production Goal Model Interface"
    )
    print("=" * 76)
    print()
    print(
        f"Artifact: "
        f"{DEFAULT_CLUB_GOAL_MODEL_ARTIFACT.name}"
    )
    print(
        f"Validation matches: "
        f"{len(results)}"
    )
    print(
        "Maximum home prediction difference: "
        f"{metadata['maximum_home_prediction_difference']:.16e}"
    )
    print(
        "Maximum away prediction difference: "
        f"{metadata['maximum_away_prediction_difference']:.16e}"
    )
    print()
    print("Artifact loading: PASS")
    print("Artifact metadata exposure: PASS")
    print("Required feature contract: PASS")
    print("Missing feature rejection: PASS")
    print("Training-cutoff guard: PASS")
    print("Home prediction reproduction: PASS")
    print("Away prediction reproduction: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )

if __name__ == "__main__":
    main()
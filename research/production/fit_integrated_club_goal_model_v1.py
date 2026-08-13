#fit_integrated_club_goal_model_v1

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from research.baselines.club_goal_model import (
    CURRENT_CLUB_GOAL_MODEL,
)
from research.production.club_goal_model_artifact import (
    ClubGoalModelArtifact,
    GoalModelTargetArtifact,
    load_club_goal_model_artifact,
    write_club_goal_model_artifact,
)
from simulation.goal_models import (
    PoissonGoalModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAINING_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_060_clubelo_enriched_observations"
    / "full_squad_observations_with_clubelo.csv"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_069_production_club_goal_model_v1"
)

DEFAULT_ARTIFACT_FILENAME = (
    "integrated_club_goal_model_v1.json"
)

DEFAULT_ARTIFACT_NAME = (
    "integrated_club_goal_model"
)

DEFAULT_ARTIFACT_VERSION = "1.0"

MODEL_ALPHA = 0.0


MODEL_ALPHA = 0.0
ARTIFACT_VERSION = "1.0"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and serialize an integrated Poisson club "
            "goal-model artifact."
        )
    )

    parser.add_argument(
        "--training-path",
        type=Path,
        default=DEFAULT_TRAINING_PATH,
        help=(
            "Club observation dataset used to fit the model."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory receiving the artifact and audit outputs."
        ),
    )

    parser.add_argument(
        "--artifact-filename",
        default=DEFAULT_ARTIFACT_FILENAME,
        help=(
            "Filename of the serialized JSON artifact."
        ),
    )

    parser.add_argument(
        "--artifact-name",
        default=DEFAULT_ARTIFACT_NAME,
        help=(
            "Logical artifact name stored inside the JSON."
        ),
    )

    parser.add_argument(
        "--artifact-version",
        default=DEFAULT_ARTIFACT_VERSION,
        help=(
            "Artifact version stored inside the JSON."
        ),
    )

    parser.add_argument(
        "--study-id",
        default="069",
        help="Study identifier written to metadata.",
    )

    parser.add_argument(
        "--study-name",
        default=(
            "Production Integrated Club Goal Model v1"
        ),
        help="Study name written to metadata.",
    )

    return parser.parse_args()

def load_training_dataset(
    training_path: Path,
) -> pd.DataFrame:
    if not training_path.exists():
        raise FileNotFoundError(
            "Production training dataset does not exist: "
            f"{training_path}"
        )

    dataframe = pd.read_csv(
        training_path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Production training dataset is empty."
        )

    specification = (
        CURRENT_CLUB_GOAL_MODEL
        .get_feature_specification()
    )

    required_columns = {
        "event_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        *specification.required_columns(),
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Production training dataset is missing "
            f"columns: {sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    numeric_columns = [
        "home_score",
        "away_score",
        *specification.required_columns(),
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    if dataframe[
        numeric_columns
    ].isna().any().any():
        raise ValueError(
            "Production training data contain missing "
            "required numeric values."
        )

    if not np.isfinite(
        dataframe[
            numeric_columns
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Production training data contain non-finite "
            "required values."
        )

    if dataframe["event_id"].duplicated().any():
        raise ValueError(
            "Production training data contain duplicate "
            "event IDs."
        )

    return (
        dataframe
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )


def fit_production_model(
    training: pd.DataFrame,
) -> PoissonGoalModel:
    specification = (
        CURRENT_CLUB_GOAL_MODEL
        .get_feature_specification()
    )

    model = PoissonGoalModel(
        name=(
            "integrated_club_goal_model_"
            "production_v1"
        ),
        home_features=list(
            specification.home_features
        ),
        away_features=list(
            specification.away_features
        ),
        alpha=MODEL_ALPHA,
    )

    model.fit(training)

    return model


def build_artifact(
    model: PoissonGoalModel,
    training: pd.DataFrame,
    *,
    training_path: Path,
    artifact_name: str,
    artifact_version: str,
) -> ClubGoalModelArtifact:
    baseline = CURRENT_CLUB_GOAL_MODEL

    return ClubGoalModelArtifact(
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        baseline_name=baseline.name,
        baseline_version=baseline.version,
        feature_specification=(
            baseline.feature_specification
        ),
        model_family="poisson_log_link",
        alpha=MODEL_ALPHA,
        training_dataset=(
            str(
                training_path.relative_to(
                    PROJECT_ROOT
                )
            )
            if training_path.is_relative_to(
                PROJECT_ROOT
            )
            else str(training_path)
        ),
        training_match_count=len(training),
        training_start_date=(
            training["date"]
            .min()
            .date()
            .isoformat()
        ),
        training_end_date=(
            training["date"]
            .max()
            .date()
            .isoformat()
        ),
        fitted_at=datetime.now(
            timezone.utc
        ).isoformat(),
        home_model=GoalModelTargetArtifact(
            target="home_score",
            features=tuple(
                model.home_features
            ),
            intercept=float(
                model.home_model.intercept_
            ),
            coefficients=tuple(
                float(value)
                for value in (
                    model.home_model.coef_
                )
            ),
        ),
        away_model=GoalModelTargetArtifact(
            target="away_score",
            features=tuple(
                model.away_features
            ),
            intercept=float(
                model.away_model.intercept_
            ),
            coefficients=tuple(
                float(value)
                for value in (
                    model.away_model.coef_
                )
            ),
        ),
    )


def validate_round_trip(
    model: PoissonGoalModel,
    artifact: ClubGoalModelArtifact,
    training: pd.DataFrame,
) -> pd.DataFrame:
    sklearn_prediction = model.predict(
        training
    )

    artifact_home: list[float] = []
    artifact_away: list[float] = []

    specification = (
        CURRENT_CLUB_GOAL_MODEL
        .get_feature_specification()
    )

    required_features = (
        specification.required_columns()
    )

    for row in training[
        list(required_features)
    ].to_dict(
        orient="records"
    ):
        home_goals, away_goals = (
            artifact.predict(row)
        )

        artifact_home.append(
            home_goals
        )

        artifact_away.append(
            away_goals
        )

    sklearn_home = np.asarray(
        sklearn_prediction.pred_home_goals,
        dtype=float,
    )

    sklearn_away = np.asarray(
        sklearn_prediction.pred_away_goals,
        dtype=float,
    )

    artifact_home_array = np.asarray(
        artifact_home,
        dtype=float,
    )

    artifact_away_array = np.asarray(
        artifact_away,
        dtype=float,
    )

    if not np.allclose(
        sklearn_home,
        artifact_home_array,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError(
            "Serialized home model does not reproduce "
            "the fitted scikit-learn model."
        )

    if not np.allclose(
        sklearn_away,
        artifact_away_array,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError(
            "Serialized away model does not reproduce "
            "the fitted scikit-learn model."
        )

    output = training[
        [
            "event_id",
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    output[
        "pred_home_goals"
    ] = artifact_home_array

    output[
        "pred_away_goals"
    ] = artifact_away_array

    output[
        "home_prediction_difference"
    ] = (
        artifact_home_array
        - sklearn_home
    )

    output[
        "away_prediction_difference"
    ] = (
        artifact_away_array
        - sklearn_away
    )

    return output


def build_coefficient_table(
    artifact: ClubGoalModelArtifact,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for model in (
        artifact.home_model,
        artifact.away_model,
    ):
        records.append(
            {
                "target": model.target,
                "feature": "intercept",
                "coefficient":
                    model.intercept,
            }
        )

        for feature, coefficient in zip(
            model.features,
            model.coefficients,
        ):
            records.append(
                {
                    "target": model.target,
                    "feature": feature,
                    "coefficient":
                        coefficient,
                }
            )

    return pd.DataFrame(records)


def build_metadata(
    artifact: ClubGoalModelArtifact,
    predictions: pd.DataFrame,
    *,
    artifact_path: Path,
    study_id: str,
    study_name: str,
) -> dict[str, object]:
    return {
        "study_id": study_id,
        "study_name": study_name,
        "artifact_path": str(
            artifact_path
        ),
        "artifact_name":
            artifact.artifact_name,
        "artifact_version":
            artifact.artifact_version,
        "baseline_name":
            artifact.baseline_name,
        "baseline_version":
            artifact.baseline_version,
        "feature_specification":
            artifact.feature_specification,
        "model_family":
            artifact.model_family,
        "alpha":
            artifact.alpha,
        "training_dataset":
            artifact.training_dataset,
        "training_match_count":
            artifact.training_match_count,
        "training_start_date":
            artifact.training_start_date,
        "training_end_date":
            artifact.training_end_date,
        "home_features": list(
            artifact.home_model.features
        ),
        "away_features": list(
            artifact.away_model.features
        ),
        "maximum_home_round_trip_difference":
            float(
                predictions[
                    "home_prediction_difference"
                ].abs().max()
            ),
        "maximum_away_round_trip_difference":
            float(
                predictions[
                    "away_prediction_difference"
                ].abs().max()
            ),
        "artifact_validation_pass": True,
        "round_trip_validation_pass": True,
    }


def write_report(
    artifact: ClubGoalModelArtifact,
    metadata: dict[str, object],
    *,
    report_path: Path,
) -> None:
    report = f"""# Study 069 — Production Integrated Club Goal Model v1

## Purpose

Fit the evidence-backed Integrated Club Goal Model v1 on the
complete validated Study 060 observation population and store
it as a portable production artifact.

## Model contract

- Baseline: `{artifact.baseline_name}`
- Baseline version: `{artifact.baseline_version}`
- Artifact version: `{artifact.artifact_version}`
- Feature specification:
  `{artifact.feature_specification}`
- Model family: `{artifact.model_family}`
- Alpha: `{artifact.alpha}`

## Training population

- Dataset: `{artifact.training_dataset}`
- Matches: {artifact.training_match_count}
- Start date: {artifact.training_start_date}
- End date: {artifact.training_end_date}

The artifact is intended for predictions after the recorded
training end date.

## Home-goal features

{chr(10).join(f"- `{feature}`" for feature in artifact.home_model.features)}

## Away-goal features

{chr(10).join(f"- `{feature}`" for feature in artifact.away_model.features)}

## Serialization

The artifact stores:

- feature ordering;
- intercepts;
- coefficients;
- model family;
- baseline provenance;
- training data provenance;
- training cutoff.

Expected goals are reproduced using the fitted Poisson
log-link equation.

## Validation

- Baseline Registry integration: PASS
- Feature Registry integration: PASS
- Complete training population: PASS
- Finite fitted parameters: PASS
- JSON artifact round trip: PASS
- Home prediction reproduction: PASS
- Away prediction reproduction: PASS
- Training cutoff recording: PASS

## Maximum reproduction differences

- Home:
  {metadata["maximum_home_round_trip_difference"]:.16e}
- Away:
  {metadata["maximum_away_round_trip_difference"]:.16e}

## Result

**OVERALL RESULT: PASS**
"""

    report_path.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()

    training_path = arguments.training_path
    output_directory = arguments.output_directory

    artifact_path = (
        output_directory
        / arguments.artifact_filename
    )

    training_predictions_path = (
        output_directory
        / "training_predictions.csv"
    )

    coefficients_path = (
        output_directory
        / "production_coefficients.csv"
    )

    metadata_path = (
        output_directory
        / "study_metadata.json"
    )

    report_path = (
        output_directory
        / "study_report.md"
    )

    CURRENT_CLUB_GOAL_MODEL.validate()

    training = load_training_dataset(
        training_path
    )

    model = fit_production_model(
        training
    )

    artifact = build_artifact(
        model=model,
        training=training,
        training_path=training_path,
        artifact_name=(
            arguments.artifact_name
        ),
        artifact_version=(
            arguments.artifact_version
        ),
    )

    artifact.validate()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_club_goal_model_artifact(
        artifact=artifact,
        path=artifact_path,
    )

    loaded_artifact = (
        load_club_goal_model_artifact(
            artifact_path
        )
    )

    predictions = validate_round_trip(
        model=model,
        artifact=loaded_artifact,
        training=training,
    )

    coefficients = build_coefficient_table(
        loaded_artifact
    )

    metadata = build_metadata(
        artifact=loaded_artifact,
        predictions=predictions,
        artifact_path=artifact_path,
        study_id=arguments.study_id,
        study_name=arguments.study_name,
    )

    prediction_output = predictions.copy()

    prediction_output["date"] = (
        pd.to_datetime(
            prediction_output["date"],
            errors="raise",
            utc=True,
        )
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    prediction_output.to_csv(
        training_predictions_path,
        index=False,
    )

    coefficients.to_csv(
        coefficients_path,
        index=False,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        artifact=loaded_artifact,
        metadata=metadata,
        report_path=report_path,
    )

    print(
        f"{arguments.study_id} — "
        f"{arguments.study_name}"
    )
    print("=" * 76)
    print()
    print(
        f"Training dataset: {training_path}"
    )
    print(
        f"Artifact: {artifact_path}"
    )
    print(
        f"Artifact name: "
        f"{loaded_artifact.artifact_name}"
    )
    print(
        f"Artifact version: "
        f"{loaded_artifact.artifact_version}"
    )
    print(
        "Feature specification: "
        f"{loaded_artifact.feature_specification}"
    )
    print(
        f"Training matches: "
        f"{loaded_artifact.training_match_count}"
    )
    print(
        "Training period: "
        f"{loaded_artifact.training_start_date} "
        f"through "
        f"{loaded_artifact.training_end_date}"
    )
    print()
    print("Artifact serialization: PASS")
    print("Artifact loading: PASS")
    print(
        "Prediction round-trip reproduction: PASS"
    )
    print("Training provenance: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
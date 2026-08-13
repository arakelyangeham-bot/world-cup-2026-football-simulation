#standardize_environment_features

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_043_football_environment_discovery"
    / "outputs"
)

DEFAULT_OUTPUT_DIRECTORY = DEFAULT_INPUT_DIRECTORY

FEATURE_MATRIX_FILES = {
    "full": "environment_features_full.csv",
    "reduced": "environment_features_reduced.csv",
}

IDENTITY_COLUMNS = [
    "observation_id",
    "feature_set",
    "competition_key",
    "competition_name",
    "season_start_year",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize Study 043 football-environment "
            "feature matrices."
        )
    )

    parser.add_argument(
        "--input-directory",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    return parser.parse_args()


def load_feature_matrix(
    input_path: Path,
    expected_feature_set: str,
) -> tuple[pd.DataFrame, list[str]]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found:\n{input_path}"
        )

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError(
            f"Feature matrix is empty: {input_path}"
        )

    missing_identity = (
        set(IDENTITY_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_identity:
        raise ValueError(
            f"{input_path.name} is missing identity columns: "
            f"{sorted(missing_identity)}"
        )

    observed_sets = set(
        dataframe["feature_set"]
        .astype(str)
        .unique()
    )

    if observed_sets != {
        expected_feature_set
    }:
        raise ValueError(
            f"{input_path.name} contains unexpected "
            f"feature-set labels: {sorted(observed_sets)}"
        )

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in IDENTITY_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            f"{input_path.name} contains no feature columns."
        )

    dataframe = dataframe.copy()

    for feature in feature_columns:
        dataframe[feature] = pd.to_numeric(
            dataframe[feature],
            errors="raise",
        )

        if dataframe[feature].isna().any():
            raise ValueError(
                f"Feature {feature!r} contains missing values."
            )

    duplicates = dataframe[
        dataframe["observation_id"].duplicated(
            keep=False
        )
    ]

    if not duplicates.empty:
        raise ValueError(
            "Duplicate observation IDs found: "
            f"{duplicates['observation_id'].tolist()}"
        )

    return dataframe, feature_columns


def standardize_matrix(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    feature_set_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    standardized = dataframe[
        IDENTITY_COLUMNS
    ].copy()

    parameter_rows: list[
        dict[str, object]
    ] = []

    for feature in feature_columns:
        mean = float(
            dataframe[feature].mean()
        )

        standard_deviation = float(
            dataframe[feature].std(
                ddof=0
            )
        )

        if standard_deviation <= 0:
            raise ValueError(
                f"Feature {feature!r} has zero variance."
            )

        standardized[feature] = (
            dataframe[feature] - mean
        ) / standard_deviation

        parameter_rows.append(
            {
                "feature_set": feature_set_name,
                "feature_key": feature,
                "mean": mean,
                "standard_deviation_population": (
                    standard_deviation
                ),
                "variance_population": (
                    standard_deviation ** 2
                ),
                "minimum_original": float(
                    dataframe[feature].min()
                ),
                "maximum_original": float(
                    dataframe[feature].max()
                ),
            }
        )

    parameters = pd.DataFrame(
        parameter_rows
    )

    return standardized, parameters


def validate_standardized_matrix(
    standardized: pd.DataFrame,
    feature_columns: list[str],
    tolerance: float = 1e-10,
) -> None:
    for feature in feature_columns:
        mean = float(
            standardized[feature].mean()
        )

        standard_deviation = float(
            standardized[feature].std(
                ddof=0
            )
        )

        if abs(mean) > tolerance:
            raise ValueError(
                f"Standardized feature {feature!r} "
                f"has mean {mean}, expected approximately 0."
            )

        if abs(
            standard_deviation - 1.0
        ) > tolerance:
            raise ValueError(
                f"Standardized feature {feature!r} "
                f"has standard deviation "
                f"{standard_deviation}, expected 1."
            )


def main() -> None:
    arguments = parse_arguments()

    created_utc = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata: dict[str, object] = {
        "study_id": "study_043",
        "phase": "phase_2_feature_standardization",
        "created_utc": created_utc,
        "feature_sets": {},
        "validation_status": "passed",
    }

    print(
        "Study 043 — Environment Feature Standardization"
    )
    print(
        "==============================================="
    )

    for (
        feature_set_name,
        filename,
    ) in FEATURE_MATRIX_FILES.items():
        input_path = (
            arguments.input_directory
            / filename
        )

        dataframe, feature_columns = (
            load_feature_matrix(
                input_path=input_path,
                expected_feature_set=(
                    feature_set_name
                ),
            )
        )

        (
            standardized,
            parameters,
        ) = standardize_matrix(
            dataframe=dataframe,
            feature_columns=feature_columns,
            feature_set_name=(
                feature_set_name
            ),
        )

        validate_standardized_matrix(
            standardized=standardized,
            feature_columns=feature_columns,
        )

        standardized_path = (
            arguments.output_directory
            / (
                "standardized_environment_"
                f"features_{feature_set_name}.csv"
            )
        )

        parameters_path = (
            arguments.output_directory
            / (
                "feature_scaler_parameters_"
                f"{feature_set_name}.csv"
            )
        )

        standardized.to_csv(
            standardized_path,
            index=False,
            encoding="utf-8",
        )

        parameters.to_csv(
            parameters_path,
            index=False,
            encoding="utf-8",
        )

        metadata["feature_sets"][
            feature_set_name
        ] = {
            "input_path": str(
                input_path
            ),
            "standardized_output": str(
                standardized_path
            ),
            "scaler_parameters_output": str(
                parameters_path
            ),
            "observation_count": len(
                standardized
            ),
            "feature_count": len(
                feature_columns
            ),
            "features": feature_columns,
        }

        print()
        print(
            f"Feature set: {feature_set_name}"
        )
        print(
            f"Observations: {len(standardized)}"
        )
        print(
            f"Features: {len(feature_columns)}"
        )
        print(
            f"Standardized output: "
            f"{standardized_path}"
        )
        print(
            f"Scaler parameters: "
            f"{parameters_path}"
        )

    metadata_path = (
        arguments.output_directory
        / "phase_2_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Metadata: {metadata_path}"
    )
    print()
    print("Validation Result")
    print("-----------------")
    print("PASSED")
    print(
        "Study 043 standardized feature matrices "
        "were written successfully."
    )


if __name__ == "__main__":
    main()
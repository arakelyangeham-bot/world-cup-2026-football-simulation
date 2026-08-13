#score_player_attributes.py

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from research.player_intelligence.feature_transformations import (
    FeatureTransformationStrategy,
    get_feature_transformation,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wc_2026_model_features.csv"
)

FEATURE_ATTRIBUTE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "feature_attribute_manifest.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_attribute_scores.csv"
)

DEFAULT_FEATURE_TRANSFORMATION_ID = (
    "robust_zscore"
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build player attribute scores from weighted "
            "canonical-player features using a configurable "
            "feature transformation."
        )
    )

    parser.add_argument(
        "--transformation-id",
        default=DEFAULT_FEATURE_TRANSFORMATION_ID,
        choices=(
            "global_zscore",
            "percentile_normal",
            "robust_zscore",
            "winsorized_zscore",
        ),
        help=(
            "Feature transformation applied before attribute "
            "aggregation. Default: robust_zscore."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Destination CSV path. Defaults to the canonical "
            "production attribute file."
        ),
    )

    parser.add_argument(
        "--features-file",
        type=Path,
        default=FEATURES_FILE,
    )

    parser.add_argument(
        "--feature-attribute-file",
        type=Path,
        default=FEATURE_ATTRIBUTE_FILE,
    )

    return parser.parse_args()


def build_player_attribute_scores(
    *,
    transformation: FeatureTransformationStrategy,
    output_path: Path,
    features_path: Path,
    feature_attribute_path: Path,
) -> pd.DataFrame:
    stats = pd.read_csv(
        features_path,
        low_memory=False,
    )

    feature_manifest = pd.read_csv(
        feature_attribute_path
    )

    required_metadata_columns = [
        "canonical_player_id",
        "player_id",
        "player",
        "minutesPlayed",
        "total_weighted_evidence",
        "evidence_confidence",
        "source_competitions",
        "competition_count",
        "season_count",
    ]

    missing_metadata_columns = [
        column
        for column in required_metadata_columns
        if column not in stats.columns
    ]

    if missing_metadata_columns:
        raise ValueError(
            "Weighted player-feature input is missing "
            "required metadata columns: "
            f"{missing_metadata_columns}"
        )

    if stats[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Weighted player-feature input contains "
            "duplicate canonical player IDs."
        )

    required_manifest_columns = {
        "feature",
        "attribute",
        "weight",
    }

    missing_manifest_columns = (
        required_manifest_columns
        - set(feature_manifest.columns)
    )

    if missing_manifest_columns:
        raise ValueError(
            "Feature-attribute manifest is missing "
            "required columns: "
            f"{sorted(missing_manifest_columns)}"
        )

    feature_manifest["weight"] = pd.to_numeric(
        feature_manifest["weight"],
        errors="raise",
    )

    player_features = sorted(
        feature_manifest[
            "feature"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    missing_features = [
        feature
        for feature in player_features
        if feature not in stats.columns
    ]

    if missing_features:
        raise AssertionError(
            "Weighted player-feature input is missing "
            "manifest features: "
            f"{missing_features}"
        )

    transformed = stats[
        required_metadata_columns
    ].copy()

    for feature in player_features:
        transformed[feature] = (
            transformation(
                stats[feature]
            )
        )

    attribute_rows = {
        column: transformed[column]
        for column in required_metadata_columns
    }

    attributes = sorted(
        feature_manifest[
            "attribute"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    for attribute in attributes:
        specification = feature_manifest.loc[
            feature_manifest[
                "attribute"
            ].eq(attribute)
        ].copy()

        attribute_features = (
            specification[
                "feature"
            ]
            .astype(str)
            .tolist()
        )

        weights = pd.to_numeric(
            specification["weight"],
            errors="raise",
        )

        weighted_values = (
            transformed[
                attribute_features
            ]
            .mul(
                weights.to_numpy(),
                axis=1,
            )
        )

        available_weight = (
            transformed[
                attribute_features
            ]
            .notna()
            .mul(
                weights.to_numpy(),
                axis=1,
            )
            .sum(axis=1)
        )

        weighted_sum = (
            weighted_values.sum(
                axis=1,
                min_count=1,
            )
        )

        attribute_rows[
            f"attribute_{attribute}"
        ] = (
            weighted_sum
            / available_weight.where(
                available_weight.gt(0.0)
            )
        )

    output = pd.DataFrame(
        attribute_rows
    )

    if len(output) != len(stats):
        raise AssertionError(
            "Player population changed during "
            "attribute construction."
        )

    if output[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Player attribute output contains duplicate "
            "canonical player IDs."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    return output

def main() -> None:
    arguments = parse_arguments()


    transformation = (
        get_feature_transformation(
            arguments.transformation_id
        )
    )

    output = build_player_attribute_scores(
        transformation=transformation,
        output_path=arguments.output_path,
        features_path=arguments.features_file,
        feature_attribute_path=(
            arguments.feature_attribute_file
        ),
    )

    print(
        "Feature transformation: "
        f"{transformation.metadata.transformation_id}"
    )

    print(
        f"Player rows: {len(output)}"
    )

    print(
        f"Output: {arguments.output_path}"
    )


if __name__ == "__main__":
    main()
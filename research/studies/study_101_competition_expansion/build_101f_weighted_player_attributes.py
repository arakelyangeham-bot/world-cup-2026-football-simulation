#build_101f_weighted_player_attributes

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.player_intelligence.feature_transformations import (
    get_feature_transformation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "study_101f_weighted_player_features.csv"
)

FEATURE_ATTRIBUTE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "feature_attribute_manifest.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "study_101f_weighted_player_attributes.csv"
)

METADATA_COLUMNS = [
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

def main() -> None:
    features = pd.read_csv(
        FEATURES_FILE,
        low_memory=False,
    )

    manifest = pd.read_csv(
        FEATURE_ATTRIBUTE_FILE
    )

    required_manifest_columns = {
        "feature",
        "attribute",
        "weight",
    }

    missing_manifest_columns = (
        required_manifest_columns
        - set(manifest.columns)
    )

    if missing_manifest_columns:
        raise ValueError(
            "Feature-attribute manifest is missing "
            "required columns: "
            f"{sorted(missing_manifest_columns)}"
        )

    if features[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Weighted player features contain duplicate "
            "canonical player IDs."
        )

    player_features = sorted(
        manifest[
            "feature"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    missing_features = [
        feature
        for feature in player_features
        if feature not in features.columns
    ]

    if missing_features:
        raise AssertionError(
            "Weighted player-feature input is missing "
            f"manifest features: {missing_features}"
        )

    transformation = (
        get_feature_transformation(
            "robust_zscore"
        )
    )

    missing_metadata_columns = [
        column
        for column in METADATA_COLUMNS
        if column not in features.columns
    ]

    if missing_metadata_columns:
        raise AssertionError(
            "Weighted player-feature input is missing "
            "required metadata columns: "
            f"{missing_metadata_columns}"
        )

    transformed = features[
        METADATA_COLUMNS
    ].copy()

    for feature in player_features:
        transformed[feature] = (
            transformation(
                features[feature]
            )
        )

    attribute_rows = {
        column: transformed[column]
        for column in METADATA_COLUMNS
    }

    attributes = sorted(
        manifest[
            "attribute"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    for attribute in attributes:
        specification = manifest.loc[
            manifest[
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

    if len(output) != len(features):
        raise AssertionError(
            "Player population changed during "
            "attribute construction."
        )

    if output[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Weighted player attributes contain "
            "duplicate canonical IDs."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Weighted player rows: "
        f"{len(features):,}"
    )

    print(
        f"Transformed features: "
        f"{len(player_features)}"
    )

    print(
        f"Attributes: "
        f"{len(attributes)}"
    )

    print(
        f"Output rows: "
        f"{len(output):,}"
    )

    print(
        "Unique canonical IDs: "
        f"{output['canonical_player_id'].nunique():,}"
    )

    print(
        f"Wrote: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
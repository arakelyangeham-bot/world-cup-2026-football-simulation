#build_weighted_player_features

from __future__ import annotations

from pathlib import Path

import pandas as pd
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical_player_features.csv"
)

DEFAULT_COMPETITION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "competition_manifest.csv"
)

DEFAULT_COMPETITION_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "competition_feature_manifest.csv"
)

DEFAULT_FEATURE_ATTRIBUTE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "feature_attribute_manifest.csv"
)

DEFAULT_REGISTRY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical_player_registry.csv"
)

DEFAULT_CONFIDENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "confidence_manifest.csv"
)

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wc_2026_model_features.csv"
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate canonical competition-season "
            "features into weighted canonical-player "
            "feature representations."
        )
    )

    parser.add_argument(
        "--features-file",
        type=Path,
        default=DEFAULT_FEATURES_FILE,
    )

    parser.add_argument(
        "--competition-file",
        type=Path,
        default=DEFAULT_COMPETITION_FILE,
    )

    parser.add_argument(
        "--competition-feature-file",
        type=Path,
        default=DEFAULT_COMPETITION_FEATURE_FILE,
    )

    parser.add_argument(
        "--feature-attribute-file",
        type=Path,
        default=DEFAULT_FEATURE_ATTRIBUTE_FILE,
    )

    parser.add_argument(
        "--registry-file",
        type=Path,
        default=DEFAULT_REGISTRY_FILE,
    )

    parser.add_argument(
        "--confidence-file",
        type=Path,
        default=DEFAULT_CONFIDENCE_FILE,
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )

    return parser.parse_args()

def main() -> None:

    arguments = parse_arguments()

    features_file = arguments.features_file
    competition_file = arguments.competition_file
    competition_feature_file = (
        arguments.competition_feature_file
    )
    feature_attribute_file = (
        arguments.feature_attribute_file
    )
    registry_file = arguments.registry_file
    confidence_file = arguments.confidence_file
    output_file = arguments.output_file

    features = pd.read_csv(
        features_file,
        dtype={"season_year": str},
        low_memory=False,
    )

    competitions = pd.read_csv(
        competition_file,
        dtype={"season_year": str},
        low_memory=False,
    )

    availability = pd.read_csv(
        competition_feature_file,
        dtype={"season_year": str},
        low_memory=False,
    )

    feature_manifest = pd.read_csv(
        feature_attribute_file
    )

    canonical_registry = pd.read_csv(
        registry_file,
        low_memory=False,
    )

    confidence_config = pd.read_csv(
        confidence_file
    )

    confidence_params = dict(
        zip(
            confidence_config["parameter"],
            confidence_config["value"],
        )
    )

    full_confidence_minutes = float(
        confidence_params.get(
            "full_confidence_minutes",
            1800,
        )
    )

    if canonical_registry[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Canonical registry contains duplicate "
            "canonical player IDs."
        )

    player_features = sorted(
        feature_manifest[
            "feature"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    missing_player_features = [
        feature
        for feature in player_features
        if feature not in features.columns
    ]

    if missing_player_features:
        raise AssertionError(
            "Player Intelligence features are missing "
            "from competition-season evidence: "
            f"{missing_player_features}"
        )
    
    canonical_key = [
        "competition_id",
        "season_id",
        "canonical_player_id",
    ]

    if features.duplicated(
        canonical_key
    ).any():
        raise AssertionError(
            "Feature input contains duplicate "
            "canonical evidence keys."
        )

    competition_keys = competitions[
        [
            "competition_id",
            "season_id",
        ]
    ].drop_duplicates()

    coverage = features.merge(
        competition_keys,
        on=[
            "competition_id",
            "season_id",
        ],
        how="left",
        indicator=True,
    )

    invalid_scopes = int(
        coverage["_merge"]
        .ne("both")
        .sum()
    )

    if invalid_scopes:
        raise AssertionError(
            "Feature input contains competition-season "
            f"scopes absent from the manifest: {invalid_scopes}"
        )

    features = features.merge(
        competitions[
            [
                "competition_id",
                "season_id",
                "recency_weight",
                "competition_importance",
            ]
        ],
        on=[
            "competition_id",
            "season_id",
        ],
        how="left",
        validate="many_to_one",
    )

    if features[
        [
            "recency_weight",
            "competition_importance",
        ]
    ].isna().any().any():
        raise AssertionError(
            "Historical weighting metadata is missing "
            "after competition-manifest merge."
        )

    features["minutesPlayed"] = pd.to_numeric(
        features["minutesPlayed"],
        errors="coerce",
    ).fillna(0.0)

    features["recency_weight"] = pd.to_numeric(
        features["recency_weight"],
        errors="raise",
    )

    features["competition_importance"] = pd.to_numeric(
        features["competition_importance"],
        errors="raise",
    )

    features["row_weight"] = (
        features["minutesPlayed"]
        * features["recency_weight"]
        * features["competition_importance"]
    )

    if features["row_weight"].lt(0.0).any():
        raise AssertionError(
            "Historical row weights must not be negative."
        )

    positive_weight_rows = int(
        features["row_weight"].gt(0.0).sum()
    )

    zero_weight_rows = int(
        features["row_weight"].eq(0.0).sum()
    )

    print()
    print("Historical evidence weighting")
    print("-" * 88)
    print(
        f"Positive-weight rows: "
        f"{positive_weight_rows:,}"
    )
    print(
        f"Zero-weight rows: "
        f"{zero_weight_rows:,}"
    )
    print(
        "Row-weight minimum: "
        f"{features['row_weight'].min():.6f}"
    )
    print(
        "Row-weight median: "
        f"{features['row_weight'].median():.6f}"
    )
    print(
        "Row-weight maximum: "
        f"{features['row_weight'].max():.6f}"
    )

    weighted_rows = []

    for canonical_player_id, group in features.groupby(
        "canonical_player_id",
        sort=True,
    ):
        competitions_used = sorted(
            set(
                group[
                    "competition"
                ]
                .dropna()
                .astype(str)
            )
        )

        row = {
            "canonical_player_id":
                canonical_player_id,

            "minutesPlayed":
                float(
                    pd.to_numeric(
                        group["minutesPlayed"],
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .sum()
                ),

            "total_weighted_evidence":
                float(
                    pd.to_numeric(
                        group["row_weight"],
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .sum()
                ),

            "source_competitions":
                "; ".join(
                    competitions_used
                ),

            "competition_count":
                int(
                    group[
                        "competition_id"
                    ].nunique()
                ),

            "season_count":
                int(
                    group[
                        "season_id"
                    ].nunique()
                ),
        }

        row["evidence_confidence"] = min(
            row["total_weighted_evidence"]
            / full_confidence_minutes,
            1.0,
        )
        for feature in player_features:
            if feature not in group.columns:
                row[feature] = float("nan")
                continue

            feature_group = group[
                [
                    "competition",
                    "season_year",
                    feature,
                    "row_weight",
                ]
            ].copy()

            feature_availability = (
                availability.loc[
                    availability[
                        "feature"
                    ].eq(feature),
                    [
                        "competition",
                        "season_year",
                        "available",
                    ],
                ]
                .copy()
            )

            feature_group = feature_group.merge(
                feature_availability,
                on=[
                    "competition",
                    "season_year",
                ],
                how="left",
                validate="many_to_one",
            )

            feature_group["available"] = (
                feature_group["available"]
                .fillna(True)
            )

            values = pd.to_numeric(
                feature_group[feature],
                errors="coerce",
            )

            weights = pd.to_numeric(
                feature_group["row_weight"],
                errors="coerce",
            )

            valid = (
                values.notna()
                & weights.notna()
                & weights.gt(0.0)
                & feature_group["available"]
            )

            if valid.any():
                row[feature] = (
                    (values.loc[valid] * weights.loc[valid]).sum()
                    / weights.loc[valid].sum()
                )
            else:
                row[feature] = float("nan")

        weighted_rows.append(row)

    weighted_player_features = pd.DataFrame(
        weighted_rows
    )

    if len(weighted_player_features) != (
        features["canonical_player_id"].nunique()
    ):
        raise AssertionError(
            "Weighted player-feature population mismatch."
        )

    if weighted_player_features[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Weighted player features contain duplicate "
            "canonical player IDs."
        )

    weighted_player_features = (
        weighted_player_features.merge(
            canonical_registry[
                [
                    "canonical_player_id",
                    "player_id",
                    "player",
                ]
            ],
            on="canonical_player_id",
            how="left",
            validate="one_to_one",
        )
    )

    metadata_columns = [
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

    expected_output_columns = (
        metadata_columns
        + player_features
    )

    missing_output_columns = [
        column
        for column in expected_output_columns
        if column not in weighted_player_features.columns
    ]

    if missing_output_columns:
        raise AssertionError(
            "Weighted player-feature output is missing "
            f"expected columns: {missing_output_columns}"
        )

    weighted_player_features = (
        weighted_player_features[
            expected_output_columns
        ]
        .copy()
    )

    if len(
        weighted_player_features.columns
    ) != (
        len(metadata_columns)
        + len(player_features)
    ):
        raise AssertionError(
            "Weighted player-feature output contains "
            "unexpected columns."
        )

    if weighted_player_features[
        [
            "player_id",
            "player",
        ]
    ].isna().all(axis=1).any():
        raise AssertionError(
            "Canonical registry enrichment failed for "
            "one or more weighted players."
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weighted_player_features.to_csv(
        output_file,
        index=False,
    )

    print()
    print("Weighted canonical player features")
    print("-" * 88)
    print(
        f"Output rows: "
        f"{len(weighted_player_features):,}"
    )
    print(
        "Unique canonical players: "
        f"{weighted_player_features['canonical_player_id'].nunique():,}"
    )
    print(
        f"Player Intelligence features: "
        f"{len(player_features)}"
    )
    print(
        f"Wrote: {output_file}"
    )
    print(
        f"Evidence feature rows: "
        f"{len(features):,}"
    )

    print(
        "Canonical players: "
        f"{features['canonical_player_id'].nunique():,}"
    )

    print(
        "Duplicate canonical evidence keys: 0"
    )

    print(
        "Invalid competition-season scopes: 0"
    )

    print(
        f"Competition manifest rows: "
        f"{len(competitions):,}"
    )

    print(
        f"Feature-availability rows: "
        f"{len(availability):,}"
    )


if __name__ == "__main__":
    main()
#build_101f_weighted_player_features

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "study_101f_canonical_competition_season_features.csv"
)

COMPETITION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "study_101d_expanded_player_intelligence"
    / "candidate_competition_manifest.csv"
)

COMPETITION_FEATURE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "study_101d_expanded_player_intelligence"
    / "candidate_competition_feature_manifest.csv"
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
    / "study_101f_weighted_player_features.csv"
)

CANONICAL_REGISTRY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "study_101f_canonical_player_registry.csv"
)

def main() -> None:
    features = pd.read_csv(
        FEATURES_FILE,
        dtype={"season_year": str},
        low_memory=False,
    )

    competitions = pd.read_csv(
        COMPETITION_FILE,
        dtype={"season_year": str},
        low_memory=False,
    )

    availability = pd.read_csv(
        COMPETITION_FEATURE_FILE,
        dtype={"season_year": str},
        low_memory=False,
    )

    feature_manifest = pd.read_csv(
        FEATURE_ATTRIBUTE_FILE
    )

    canonical_registry = pd.read_csv(
        CANONICAL_REGISTRY_FILE,
        low_memory=False,
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

    dlamini = features.loc[
        features[
            "canonical_player_id"
        ].eq(1218855)
    ].copy()

    dlamini_columns = [
        column
        for column in [
            "canonical_player_id",
            "player_id",
            "competition",
            "season_year",
            "minutesPlayed",
            "recency_weight",
            "competition_importance",
            "row_weight",
        ]
        if column in dlamini.columns
    ]

    print()
    print("Dlamini historical evidence weights")
    print("-" * 88)
    print(
        dlamini[
            dlamini_columns
        ].sort_values(
            [
                "season_year",
                "competition",
            ]
        ).to_string(
            index=False
        )
    )

    print(
        "Dlamini total weighted evidence: "
        f"{dlamini['row_weight'].sum():.6f}"
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
            / 1800.0,
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

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weighted_player_features.to_csv(
        OUTPUT_FILE,
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
        f"Wrote: {OUTPUT_FILE}"
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
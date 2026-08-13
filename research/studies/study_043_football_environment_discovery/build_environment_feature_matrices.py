#build_environment_feature_matrices

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "research"
    / "datasets"
    / "league_season_repository"
    / "league_season_repository.csv"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_043_football_environment_discovery"
    / "outputs"
)

FULL_FEATURES = [
    "goals_per_match",
    "home_goals_per_match",
    "away_goals_per_match",
    "both_teams_to_score_rate",
    "draw_rate",
    "one_goal_margin_rate",
    "three_plus_goal_margin_rate",
]

REDUCED_FEATURES = [
    "goals_per_match",
    "both_teams_to_score_rate",
    "draw_rate",
    "one_goal_margin_rate",
    "three_plus_goal_margin_rate",
]

IDENTITY_COLUMNS = [
    "competition_key",
    "competition_name",
    "season_start_year",
]

RATE_FEATURES = {
    "both_teams_to_score_rate",
    "draw_rate",
    "one_goal_margin_rate",
    "three_plus_goal_margin_rate",
}

FEATURE_LABELS = {
    "goals_per_match": "Goals per match",
    "home_goals_per_match": "Home goals per match",
    "away_goals_per_match": "Away goals per match",
    "both_teams_to_score_rate": "Both teams to score rate",
    "draw_rate": "Draw rate",
    "one_goal_margin_rate": "One-goal-margin rate",
    "three_plus_goal_margin_rate": (
        "Three-plus-goal-margin rate"
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the canonical league-season repository and "
            "build Study 043 environment feature matrices."
        )
    )

    parser.add_argument(
        "--repository",
        type=Path,
        default=DEFAULT_REPOSITORY_PATH,
        help=(
            "Path to the canonical League-Season Repository."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for Study 043 Phase 1 outputs.",
    )

    return parser.parse_args()


def load_repository(
    repository_path: Path,
) -> pd.DataFrame:
    if not repository_path.exists():
        raise FileNotFoundError(
            "League-Season Repository was not found:\n"
            f"{repository_path}"
        )

    repository = pd.read_csv(
        repository_path
    )

    if repository.empty:
        raise ValueError(
            "League-Season Repository is empty."
        )

    required_columns = (
        set(IDENTITY_COLUMNS)
        | set(FULL_FEATURES)
    )

    missing_columns = (
        required_columns
        - set(repository.columns)
    )

    if missing_columns:
        raise ValueError(
            "League-Season Repository is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    repository = repository.copy()

    repository["competition_key"] = (
        repository["competition_key"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    repository["competition_name"] = (
        repository["competition_name"]
        .astype(str)
        .str.strip()
    )

    repository["season_start_year"] = (
        pd.to_numeric(
            repository["season_start_year"],
            errors="raise",
        )
        .astype(int)
    )

    for feature in FULL_FEATURES:
        repository[feature] = pd.to_numeric(
            repository[feature],
            errors="raise",
        )

    return repository


def validate_primary_key(
    repository: pd.DataFrame,
) -> None:
    duplicates = repository[
        repository.duplicated(
            subset=[
                "competition_key",
                "season_start_year",
            ],
            keep=False,
        )
    ]

    if not duplicates.empty:
        preview = (
            duplicates[
                [
                    "competition_key",
                    "season_start_year",
                ]
            ]
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate league-season primary keys found: "
            f"{preview}"
        )


def validate_identity_values(
    repository: pd.DataFrame,
) -> None:
    if repository[
        "competition_key"
    ].eq("").any():
        raise ValueError(
            "One or more rows have an empty competition key."
        )

    if repository[
        "competition_name"
    ].eq("").any():
        raise ValueError(
            "One or more rows have an empty competition name."
        )


def validate_feature_values(
    repository: pd.DataFrame,
) -> None:
    for feature in FULL_FEATURES:
        missing_count = int(
            repository[feature]
            .isna()
            .sum()
        )

        if missing_count:
            raise ValueError(
                f"Feature {feature!r} contains "
                f"{missing_count} missing value(s)."
            )

        if not repository[feature].map(
            lambda value: pd.notna(value)
        ).all():
            raise ValueError(
                f"Feature {feature!r} contains invalid values."
            )

    for feature in RATE_FEATURES:
        invalid = repository[
            ~repository[feature].between(
                0.0,
                1.0,
                inclusive="both",
            )
        ]

        if not invalid.empty:
            raise ValueError(
                f"Rate feature {feature!r} contains values "
                "outside the interval [0, 1]."
            )

    positive_features = [
        "goals_per_match",
        "home_goals_per_match",
        "away_goals_per_match",
    ]

    for feature in positive_features:
        if (
            repository[feature] < 0
        ).any():
            raise ValueError(
                f"Feature {feature!r} contains negative values."
            )


def validate_feature_relationships(
    repository: pd.DataFrame,
    tolerance: float = 1e-9,
) -> None:
    reconstructed_total = (
        repository["home_goals_per_match"]
        + repository["away_goals_per_match"]
    )

    inconsistent = repository[
        (
            reconstructed_total
            - repository["goals_per_match"]
        ).abs()
        > tolerance
    ]

    if not inconsistent.empty:
        preview = inconsistent[
            [
                "competition_key",
                "season_start_year",
                "goals_per_match",
                "home_goals_per_match",
                "away_goals_per_match",
            ]
        ].to_dict("records")

        raise ValueError(
            "Goals-per-match values are inconsistent with "
            "home plus away scoring components: "
            f"{preview}"
        )


def build_feature_audit(
    repository: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for feature in FULL_FEATURES:
        series = repository[feature]

        rows.append(
            {
                "feature_key": feature,
                "feature_name": (
                    FEATURE_LABELS[feature]
                ),
                "included_in_full": True,
                "included_in_reduced": (
                    feature in REDUCED_FEATURES
                ),
                "observation_count": int(
                    series.count()
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
                "unique_value_count": int(
                    series.nunique()
                ),
                "minimum": float(
                    series.min()
                ),
                "maximum": float(
                    series.max()
                ),
                "mean": float(
                    series.mean()
                ),
                "standard_deviation_sample": float(
                    series.std(ddof=1)
                ),
                "variance_sample": float(
                    series.var(ddof=1)
                ),
                "is_rate_feature": (
                    feature in RATE_FEATURES
                ),
            }
        )

    return pd.DataFrame(rows)


def build_feature_matrix(
    repository: pd.DataFrame,
    feature_columns: list[str],
    feature_set_name: str,
) -> pd.DataFrame:
    matrix = repository[
        IDENTITY_COLUMNS
        + feature_columns
    ].copy()

    matrix.insert(
        0,
        "feature_set",
        feature_set_name,
    )

    matrix["observation_id"] = (
        matrix["competition_key"]
        + "_"
        + matrix[
            "season_start_year"
        ].astype(str)
    )

    ordered_columns = [
        "observation_id",
        "feature_set",
        *IDENTITY_COLUMNS,
        *feature_columns,
    ]

    return (
        matrix[ordered_columns]
        .sort_values(
            [
                "competition_key",
                "season_start_year",
            ]
        )
        .reset_index(drop=True)
    )


def build_feature_correlation_matrix(
    repository: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    correlation = repository[
        feature_columns
    ].corr(
        method="pearson"
    )

    correlation.index.name = "feature_key"

    return correlation.reset_index()


def build_metadata(
    repository: pd.DataFrame,
    audit: pd.DataFrame,
    repository_path: Path,
    created_utc: str,
) -> dict[str, object]:
    return {
        "study_id": "study_043",
        "phase": (
            "phase_1_dataset_audit_and_"
            "feature_construction"
        ),
        "created_utc": created_utc,
        "repository_path": str(
            repository_path
        ),
        "observation_count": len(
            repository
        ),
        "competition_count": int(
            repository[
                "competition_key"
            ].nunique()
        ),
        "season_count": int(
            repository[
                "season_start_year"
            ].nunique()
        ),
        "season_start_years": [
            int(year)
            for year in sorted(
                repository[
                    "season_start_year"
                ].unique()
            )
        ],
        "full_feature_count": len(
            FULL_FEATURES
        ),
        "reduced_feature_count": len(
            REDUCED_FEATURES
        ),
        "full_features": (
            FULL_FEATURES
        ),
        "reduced_features": (
            REDUCED_FEATURES
        ),
        "feature_audit_rows": len(
            audit
        ),
        "primary_key": [
            "competition_key",
            "season_start_year",
        ],
        "validation_status": "passed",
    }


def main() -> None:
    arguments = parse_arguments()

    created_utc = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    repository = load_repository(
        arguments.repository
    )

    validate_primary_key(
        repository
    )

    validate_identity_values(
        repository
    )

    validate_feature_values(
        repository
    )

    validate_feature_relationships(
        repository
    )

    audit = build_feature_audit(
        repository
    )

    full_matrix = build_feature_matrix(
        repository=repository,
        feature_columns=FULL_FEATURES,
        feature_set_name="full",
    )

    reduced_matrix = build_feature_matrix(
        repository=repository,
        feature_columns=REDUCED_FEATURES,
        feature_set_name="reduced",
    )

    full_correlation = (
        build_feature_correlation_matrix(
            repository=repository,
            feature_columns=FULL_FEATURES,
        )
    )

    reduced_correlation = (
        build_feature_correlation_matrix(
            repository=repository,
            feature_columns=REDUCED_FEATURES,
        )
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        arguments.output_directory
        / "environment_feature_audit.csv"
    )

    full_matrix_path = (
        arguments.output_directory
        / "environment_features_full.csv"
    )

    reduced_matrix_path = (
        arguments.output_directory
        / "environment_features_reduced.csv"
    )

    full_correlation_path = (
        arguments.output_directory
        / "environment_feature_correlations_full.csv"
    )

    reduced_correlation_path = (
        arguments.output_directory
        / "environment_feature_correlations_reduced.csv"
    )

    metadata_path = (
        arguments.output_directory
        / "phase_1_metadata.json"
    )

    audit.to_csv(
        audit_path,
        index=False,
        encoding="utf-8",
    )

    full_matrix.to_csv(
        full_matrix_path,
        index=False,
        encoding="utf-8",
    )

    reduced_matrix.to_csv(
        reduced_matrix_path,
        index=False,
        encoding="utf-8",
    )

    full_correlation.to_csv(
        full_correlation_path,
        index=False,
        encoding="utf-8",
    )

    reduced_correlation.to_csv(
        reduced_correlation_path,
        index=False,
        encoding="utf-8",
    )

    metadata = build_metadata(
        repository=repository,
        audit=audit,
        repository_path=arguments.repository,
        created_utc=created_utc,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Study 043 — Environment Feature Construction"
    )
    print(
        "============================================"
    )
    print(
        f"Repository: "
        f"{arguments.repository}"
    )
    print(
        f"Observations: "
        f"{len(repository)}"
    )
    print(
        "Competitions: "
        f"{repository['competition_key'].nunique()}"
    )
    print(
        "Seasons: "
        f"{repository['season_start_year'].nunique()}"
    )
    print(
        f"Full features: "
        f"{len(FULL_FEATURES)}"
    )
    print(
        f"Reduced features: "
        f"{len(REDUCED_FEATURES)}"
    )
    print()

    print("Feature Audit")
    print("-------------")
    print(
        audit[
            [
                "feature_key",
                "included_in_full",
                "included_in_reduced",
                "minimum",
                "maximum",
                "standard_deviation_sample",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Outputs")
    print("-------")
    print(f"audit: {audit_path}")
    print(f"full_features: {full_matrix_path}")
    print(
        f"reduced_features: "
        f"{reduced_matrix_path}"
    )
    print(
        f"full_correlations: "
        f"{full_correlation_path}"
    )
    print(
        f"reduced_correlations: "
        f"{reduced_correlation_path}"
    )
    print(f"metadata: {metadata_path}")
    print()

    print("Validation Result")
    print("-----------------")
    print("PASSED")
    print(
        "Study 043 environment feature matrices "
        "were written successfully."
    )


if __name__ == "__main__":
    main()
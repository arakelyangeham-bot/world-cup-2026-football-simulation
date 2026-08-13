#finish_101d_expanded_player_intelligence

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STUDY_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "study_101d_expanded_player_intelligence"
)

CANDIDATE_STATS = (
    STUDY_ROOT
    / "candidate_player_stats.csv"
)

CANDIDATE_MANIFEST = (
    STUDY_ROOT
    / "candidate_competition_manifest.csv"
)

CANDIDATE_FEATURES = (
    STUDY_ROOT
    / "candidate_model_features.csv"
)

CANONICAL_COMPETITION_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "competition_feature_manifest.csv"
)

FEATURE_ATTRIBUTE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "feature_attribute_manifest.csv"
)

PLAYER_REGISTRY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_registry.csv"
)

CANONICAL_ATTRIBUTES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_attribute_scores.csv"
)

CANONICAL_RATINGS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_ratings.csv"
)

CANDIDATE_COMPETITION_FEATURES = (
    STUDY_ROOT
    / "candidate_competition_feature_manifest.csv"
)

CANDIDATE_ATTRIBUTES = (
    STUDY_ROOT
    / "candidate_player_attribute_scores.csv"
)

CANDIDATE_RATINGS = (
    STUDY_ROOT
    / "candidate_player_ratings.csv"
)

RATING_COMPARISON = (
    STUDY_ROOT
    / "canonical_vs_candidate_ratings.csv"
)

RATING_SUMMARY = (
    STUDY_ROOT
    / "rating_stability_summary.csv"
)

DOMESTIC_LEAGUES = {
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "Eredivisie",
    "Liga Portugal",
}

def build_candidate_competition_features() -> None:
    canonical = pd.read_csv(
        CANONICAL_COMPETITION_FEATURES,
        dtype={"season_year": str},
    )

    stats = pd.read_csv(
        CANDIDATE_STATS,
        dtype={"season_year": str},
        low_memory=False,
    )

    availability_features = sorted(
        canonical["feature"]
        .dropna()
        .astype(str)
        .unique()
    )

    canonical_international = (
        canonical.loc[
            ~canonical["competition"].isin(
                DOMESTIC_LEAGUES
            )
        ]
        .copy()
    )

    domestic_stats = stats.loc[
        stats["competition"].isin(
            DOMESTIC_LEAGUES
        )
    ].copy()

    rows: list[dict[str, object]] = []

    for (
        competition,
        season_year,
    ), group in domestic_stats.groupby(
        [
            "competition",
            "season_year",
        ],
        sort=True,
    ):
        for feature in availability_features:
            if feature in group.columns:
                coverage = float(
                    group[feature]
                    .notna()
                    .mean()
                )
            else:
                coverage = 0.0

            #
            # Preserve the existing manifest's practical
            # availability semantics: a feature must have
            # meaningful population coverage.
            #
            available = coverage >= 0.10

            rows.append(
                {
                    "competition":
                        competition,
                    "season_year":
                        str(season_year),
                    "feature":
                        feature,
                    "coverage":
                        coverage,
                    "available":
                        available,
                }
            )

    domestic = pd.DataFrame(rows)

    expected_scopes = (
        len(DOMESTIC_LEAGUES) * 5
    )

    observed_scopes = (
        domestic[
            [
                "competition",
                "season_year",
            ]
        ]
        .drop_duplicates()
    )

    if len(observed_scopes) != expected_scopes:
        raise AssertionError(
            "Expected 35 domestic availability scopes, "
            f"found {len(observed_scopes)}."
        )

    if domestic["feature"].nunique() != len(
        availability_features
    ):
        raise AssertionError(
            "Candidate domestic availability feature "
            "population is incomplete."
        )

    candidate = pd.concat(
        [
            canonical_international,
            domestic,
        ],
        ignore_index=True,
        sort=False,
    )

    duplicate_count = int(
        candidate.duplicated(
            [
                "competition",
                "season_year",
                "feature",
            ]
        ).sum()
    )

    if duplicate_count:
        raise AssertionError(
            "Candidate competition-feature manifest "
            f"contains {duplicate_count} duplicate keys."
        )

    candidate.to_csv(
        CANDIDATE_COMPETITION_FEATURES,
        index=False,
    )

    print()
    print("Candidate competition-feature manifest")
    print("-" * 88)
    print(
        f"Availability features: "
        f"{len(availability_features)}"
    )
    print(
        f"Domestic scopes: "
        f"{len(observed_scopes)}"
    )
    print(
        f"Domestic rows: {len(domestic)}"
    )
    print(
        f"International rows preserved: "
        f"{len(canonical_international)}"
    )
    print(
        f"Total rows: {len(candidate)}"
    )

def run_command(
    command: list[str],
    label: str,
) -> None:
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code "
            f"{completed.returncode}."
        )


def run_scoring_pipeline() -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "scripts.score_player_attributes",
            "--transformation-id",
            "robust_zscore",
            "--features-file",
            str(CANDIDATE_FEATURES),
            "--competition-file",
            str(CANDIDATE_MANIFEST),
            "--competition-feature-file",
            str(CANDIDATE_COMPETITION_FEATURES),
            "--registry-file",
            str(PLAYER_REGISTRY),
            "--feature-attribute-file",
            str(FEATURE_ATTRIBUTE_MANIFEST),
            "--output-path",
            str(CANDIDATE_ATTRIBUTES),
        ],
        "101D — ROBUST PLAYER ATTRIBUTES",
    )

    run_command(
        [
            sys.executable,
            "-m",
            "scripts.build_player_ratings_v4",
            "--attribute-path",
            str(CANDIDATE_ATTRIBUTES),
            "--output-path",
            str(CANDIDATE_RATINGS),
        ],
        "101D — V4 PLAYER ROLE RATINGS",
    )

def build_rating_comparison() -> None:
    canonical = pd.read_csv(
        CANONICAL_RATINGS,
        low_memory=False,
    )

    candidate = pd.read_csv(
        CANDIDATE_RATINGS,
        low_memory=False,
    )

    key = (
        "canonical_player_id"
        if (
            "canonical_player_id"
            in canonical.columns
            and "canonical_player_id"
            in candidate.columns
        )
        else "player_id"
    )

    canonical_rating_columns = [
        column
        for column in canonical.columns
        if column.startswith(
            "rating_"
        )
    ]

    candidate_rating_columns = [
        column
        for column in candidate.columns
        if column.startswith(
            "rating_"
        )
    ]

    common_rating_columns = sorted(
        set(canonical_rating_columns)
        & set(candidate_rating_columns)
    )

    if not common_rating_columns:
        raise AssertionError(
            "No common role-rating columns found."
        )

    canonical_subset = canonical[
        [key]
        + common_rating_columns
    ].copy()

    candidate_subset = candidate[
        [key]
        + common_rating_columns
    ].copy()

    comparison = candidate_subset.merge(
        canonical_subset,
        on=key,
        how="left",
        suffixes=(
            "_candidate",
            "_canonical",
        ),
        indicator=True,
    )

    comparison["population_status"] = np.where(
        comparison["_merge"].eq("both"),
        "matched_existing",
        "new_candidate_player",
    )

    delta_columns: list[str] = []

    for rating_column in common_rating_columns:
        candidate_column = (
            f"{rating_column}_candidate"
        )

        canonical_column = (
            f"{rating_column}_canonical"
        )

        delta_column = (
            f"delta_{rating_column}"
        )

        comparison[delta_column] = (
            pd.to_numeric(
                comparison[candidate_column],
                errors="coerce",
            )
            - pd.to_numeric(
                comparison[canonical_column],
                errors="coerce",
            )
        )

        delta_columns.append(
            delta_column
        )

    comparison[
        "mean_absolute_role_rating_change"
    ] = (
        comparison[delta_columns]
        .abs()
        .mean(axis=1)
    )

    comparison[
        "maximum_absolute_role_rating_change"
    ] = (
        comparison[delta_columns]
        .abs()
        .max(axis=1)
    )

    comparison = comparison.drop(
        columns=["_merge"]
    )

    comparison.to_csv(
        RATING_COMPARISON,
        index=False,
    )

    matched = comparison.loc[
        comparison[
            "population_status"
        ].eq("matched_existing")
    ].copy()

    summary_rows = [
        {
            "metric":
                "candidate_players",
            "value":
                len(candidate),
        },
        {
            "metric":
                "canonical_players",
            "value":
                len(canonical),
        },
        {
            "metric":
                "matched_existing_players",
            "value":
                len(matched),
        },
        {
            "metric":
                "new_candidate_players",
            "value":
                int(
                    comparison[
                        "population_status"
                    ]
                    .eq(
                        "new_candidate_player"
                    )
                    .sum()
                ),
        },
        {
            "metric":
                "mean_absolute_role_rating_change",
            "value":
                float(
                    matched[
                        "mean_absolute_role_rating_change"
                    ].mean()
                ),
        },
        {
            "metric":
                "median_absolute_role_rating_change",
            "value":
                float(
                    matched[
                        "mean_absolute_role_rating_change"
                    ].median()
                ),
        },
        {
            "metric":
                "p90_absolute_role_rating_change",
            "value":
                float(
                    matched[
                        "mean_absolute_role_rating_change"
                    ].quantile(0.90)
                ),
        },
        {
            "metric":
                "maximum_observed_role_rating_change",
            "value":
                float(
                    matched[
                        "maximum_absolute_role_rating_change"
                    ].max()
                ),
        },
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        RATING_SUMMARY,
        index=False,
    )

    print()
    print("Canonical vs expanded Player Intelligence")
    print("-" * 88)
    print(
        summary.to_string(
            index=False
        )
    )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 101D — EXPANDED PLAYER "
        "INTELLIGENCE SCORING"
    )
    print("=" * 88)

    required = [
        CANDIDATE_STATS,
        CANDIDATE_MANIFEST,
        CANDIDATE_FEATURES,
        CANONICAL_COMPETITION_FEATURES,
        FEATURE_ATTRIBUTE_MANIFEST,
        PLAYER_REGISTRY,
        CANONICAL_RATINGS,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Study 101D inputs:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    build_candidate_competition_features()
    run_scoring_pipeline()
    build_rating_comparison()

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Candidate attributes: "
        f"{CANDIDATE_ATTRIBUTES}"
    )
    print(
        f"Candidate ratings: "
        f"{CANDIDATE_RATINGS}"
    )
    print(
        f"Rating comparison: "
        f"{RATING_COMPARISON}"
    )
    print(
        f"Stability summary: "
        f"{RATING_SUMMARY}"
    )


if __name__ == "__main__":
    main()
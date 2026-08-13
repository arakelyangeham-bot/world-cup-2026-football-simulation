#audit_101a_feature_coverage

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANONICAL_STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_stats.csv"
)

PILOT_STATS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_101a_competition_expansion"
    / "pilot_24_25_player_stats.csv"
)

FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "feature_attribute_manifest.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_101a_competition_expansion"
    / "feature_coverage"
)

FEATURE_COVERAGE_PATH = (
    OUTPUT_DIRECTORY
    / "model_feature_coverage_24_25.csv"
)

ATTRIBUTE_COVERAGE_PATH = (
    OUTPUT_DIRECTORY
    / "attribute_family_coverage_24_25.csv"
)

SEASON_YEAR = "24/25"

REFERENCE_COMPETITIONS = {
    "Premier League",
    "Bundesliga",
}

CANDIDATE_COMPETITIONS = {
    "Eredivisie",
    "Liga Portugal",
}

def source_stat_for_feature(
    feature: str,
) -> str:
    if feature.endswith("_per90"):
        return feature.removesuffix(
            "_per90"
        )

    return feature

def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    canonical = pd.read_csv(
        CANONICAL_STATS_PATH,
        dtype={"season_year": str},
        low_memory=False,
    )

    pilot = pd.read_csv(
        PILOT_STATS_PATH,
        dtype={"season_year": str},
        low_memory=False,
    )

    manifest = pd.read_csv(
        FEATURE_MANIFEST_PATH,
        low_memory=False,
    )

    if canonical.empty:
        raise ValueError(
            "Canonical player-stat dataset is empty."
        )

    if pilot.empty:
        raise ValueError(
            "Study 101A pilot player-stat dataset is empty."
        )

    required_manifest_columns = {
        "feature",
        "attribute",
        "weight",
    }

    missing = (
        required_manifest_columns
        - set(manifest.columns)
    )

    if missing:
        raise ValueError(
            "Feature-attribute manifest is missing columns: "
            f"{sorted(missing)}"
        )

    canonical = canonical.loc[
        canonical["season_year"].astype(str).eq(
            SEASON_YEAR
        )
        & canonical["competition"].isin(
            REFERENCE_COMPETITIONS
        )
    ].copy()

    pilot = pilot.loc[
        pilot["season_year"].astype(str).eq(
            SEASON_YEAR
        )
        & pilot["competition"].isin(
            CANDIDATE_COMPETITIONS
        )
    ].copy()

    observed_reference_competitions = set(
        canonical["competition"]
    )

    if observed_reference_competitions != (
        REFERENCE_COMPETITIONS
    ):
        raise AssertionError(
            "Canonical 24/25 domestic reference population "
            "mismatch. "
            "Expected="
            f"{sorted(REFERENCE_COMPETITIONS)}, "
            "observed="
            f"{sorted(observed_reference_competitions)}."
        )

    observed_candidates = set(
        pilot["competition"]
    )

    if observed_candidates != (
        CANDIDATE_COMPETITIONS
    ):
        raise AssertionError(
            "Candidate competition population mismatch. "
            f"Observed={sorted(observed_candidates)}"
        )

    combined = pd.concat(
        [
            canonical,
            pilot,
        ],
        ignore_index=True,
        sort=False,
    )

    competition_counts = (
        combined.groupby(
            "competition"
        )["player_id"]
        .nunique()
    )

    minimum_population = 400

    undersized_competitions = (
        competition_counts.loc[
            competition_counts
            < minimum_population
        ]
    )

    if not undersized_competitions.empty:
        raise AssertionError(
            "One or more Study 101A competition populations "
            "are unexpectedly small:\n"
            f"{undersized_competitions.to_string()}"
        )

    return combined, manifest

def build_feature_coverage(
    stats: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    manifest = (
        manifest[
            [
                "feature",
                "attribute",
                "weight",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    for competition, group in (
        stats.groupby(
            "competition",
            sort=True,
        )
    ):
        player_rows = len(group)

        for feature_row in (
            manifest.itertuples(
                index=False
            )
        ):
            feature = str(
                feature_row.feature
            )

            source_stat = (
                source_stat_for_feature(
                    feature
                )
            )

            structurally_present = (
                source_stat
                in group.columns
            )

            if structurally_present:
                values = pd.to_numeric(
                    group[source_stat],
                    errors="coerce",
                )

                non_null_rows = int(
                    values.notna().sum()
                )

            else:
                non_null_rows = 0

            coverage_rate = (
                non_null_rows / player_rows
                if player_rows
                else 0.0
            )

            rows.append(
                {
                    "competition":
                        competition,
                    "season_year":
                        SEASON_YEAR,
                    "player_rows":
                        player_rows,
                    "feature":
                        feature,
                    "source_stat":
                        source_stat,
                    "attribute":
                        feature_row.attribute,
                    "feature_weight":
                        float(
                            feature_row.weight
                        ),
                    "structurally_present":
                        structurally_present,
                    "non_null_rows":
                        non_null_rows,
                    "coverage_rate":
                        coverage_rate,
                }
            )

    return pd.DataFrame(
        rows
    )

def build_attribute_coverage(
    feature_coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (
        competition,
        attribute,
    ), group in feature_coverage.groupby(
        [
            "competition",
            "attribute",
        ],
        sort=True,
    ):
        weights = group[
            "feature_weight"
        ]

        weighted_coverage = (
            (
                group["coverage_rate"]
                * weights
            ).sum()
            / weights.sum()
        )

        rows.append(
            {
                "competition":
                    competition,
                "season_year":
                    SEASON_YEAR,
                "attribute":
                    attribute,
                "feature_count":
                    len(group),
                "structurally_present_features":
                    int(
                        group[
                            "structurally_present"
                        ].sum()
                    ),
                "minimum_feature_coverage":
                    float(
                        group[
                            "coverage_rate"
                        ].min()
                    ),
                "mean_feature_coverage":
                    float(
                        group[
                            "coverage_rate"
                        ].mean()
                    ),
                "weighted_feature_coverage":
                    float(
                        weighted_coverage
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 101A — COMPETITION EXPANSION "
        "MODEL-FEATURE COVERAGE"
    )
    print("=" * 88)

    print()
    print("Comparison design")
    print("-" * 88)
    print(
        "  Reference competitions: "
        f"{sorted(REFERENCE_COMPETITIONS)}"
    )
    print(
        "  Candidate competitions: "
        f"{sorted(CANDIDATE_COMPETITIONS)}"
    )
    print(
        f"  Shared season: {SEASON_YEAR}"
    )

    stats, manifest = load_inputs()

    feature_coverage = (
        build_feature_coverage(
            stats,
            manifest,
        )
    )

    attribute_coverage = (
        build_attribute_coverage(
            feature_coverage
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_coverage.to_csv(
        FEATURE_COVERAGE_PATH,
        index=False,
    )

    attribute_coverage.to_csv(
        ATTRIBUTE_COVERAGE_PATH,
        index=False,
    )

    print()
    print("Population")
    print("-" * 88)

    population = (
        stats.groupby("competition")
        .agg(
            rows=("player_id", "size"),
            players=("player_id", "nunique"),
        )
        .sort_index()
    )

    print(
        population.to_string()
    )
    
    print()
    print("Attribute-family weighted coverage")
    print("-" * 88)

    display = (
        attribute_coverage
        .pivot(
            index="attribute",
            columns="competition",
            values="weighted_feature_coverage",
        )
        .sort_index()
    )

    print(
        display.to_string(
            float_format=lambda value: (
                f"{value:.3f}"
            )
        )
    )

    missing_structure = (
        feature_coverage.loc[
            ~feature_coverage[
                "structurally_present"
            ],
            [
                "competition",
                "feature",
                "source_stat",
                "attribute",
            ],
        ]
    )

    print()
    print("Structurally unavailable model features")
    print("-" * 88)

    if missing_structure.empty:
        print("None")
    else:
        print(
            missing_structure.to_string(
                index=False
            )
        )

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Feature coverage: "
        f"{FEATURE_COVERAGE_PATH}"
    )
    print(
        f"Attribute coverage: "
        f"{ATTRIBUTE_COVERAGE_PATH}"
    )


if __name__ == "__main__":
    main()
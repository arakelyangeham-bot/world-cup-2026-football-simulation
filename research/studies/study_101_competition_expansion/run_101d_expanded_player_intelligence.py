#run_101d_expanded_player_intelligence

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STUDY_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "study_101d_expanded_player_intelligence"
)

SOURCE_STUDY_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "study_101a_competition_expansion"
)

SEVEN_LEAGUE_STATS = (
    PROJECT_ROOT
    / "outputs"
    / "study_101c_seven_league_evidence"
    / "seven_league_player_stats.csv"
)

CANONICAL_STATS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_stats.csv"
)

CANONICAL_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "competition_manifest.csv"
)

CANDIDATE_MANIFEST_SOURCE = (
    SOURCE_STUDY_ROOT
    / "candidate_competition_manifest.csv"
)

STAT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "stat_manifest.csv"
)

PROFILES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_profiles.csv"
)

CONFIDENCE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "confidence_manifest.csv"
)

CANDIDATE_STATS = (
    STUDY_ROOT
    / "candidate_player_stats.csv"
)

CANDIDATE_MANIFEST = (
    STUDY_ROOT
    / "candidate_competition_manifest.csv"
)

CANDIDATE_PLAYER_DATASET = (
    STUDY_ROOT
    / "candidate_player_dataset.csv"
)

CANDIDATE_MODEL_FEATURES = (
    STUDY_ROOT
    / "candidate_model_features.csv"
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

TASK_KEY = [
    "competition_id",
    "season_id",
    "player_id",
]


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


def validate_inputs() -> None:
    required = [
        SEVEN_LEAGUE_STATS,
        CANONICAL_STATS,
        CANONICAL_MANIFEST,
        CANDIDATE_MANIFEST_SOURCE,
        STAT_MANIFEST,
        PROFILES,
        CONFIDENCE_MANIFEST,
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


def build_candidate_evidence() -> None:
    canonical = pd.read_csv(
        CANONICAL_STATS,
        dtype={"season_year": str},
        low_memory=False,
    )

    domestic = pd.read_csv(
        SEVEN_LEAGUE_STATS,
        dtype={"season_year": str},
        low_memory=False,
    )

    #
    # Preserve canonical international evidence,
    # but replace all domestic club-league evidence
    # with the validated seven-league artifact.
    #
    canonical_international = canonical.loc[
        ~canonical["competition"].isin(
            DOMESTIC_LEAGUES
        )
    ].copy()

    combined = pd.concat(
        [
            canonical_international,
            domestic,
        ],
        ignore_index=True,
        sort=False,
    )

    duplicate_count = int(
        combined.duplicated(
            TASK_KEY
        ).sum()
    )

    if duplicate_count:
        raise AssertionError(
            "Candidate evidence contains "
            f"{duplicate_count} duplicate task keys."
        )

    observed_domestic = set(
        combined.loc[
            combined["competition"].isin(
                DOMESTIC_LEAGUES
            ),
            "competition",
        ]
    )

    if observed_domestic != DOMESTIC_LEAGUES:
        raise AssertionError(
            "Candidate domestic population mismatch. "
            f"Observed={sorted(observed_domestic)}"
        )

    STUDY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        CANDIDATE_STATS,
        index=False,
    )

    print()
    print("Candidate evidence universe")
    print("-" * 88)
    print(
        f"Canonical international rows: "
        f"{len(canonical_international)}"
    )
    print(
        f"Seven-league domestic rows: "
        f"{len(domestic)}"
    )
    print(
        f"Combined rows: {len(combined)}"
    )
    print(
        f"Unique players: "
        f"{combined['player_id'].nunique()}"
    )
    print("Duplicate task keys: 0")


def build_candidate_manifest() -> None:
    canonical = pd.read_csv(
        CANONICAL_MANIFEST,
        dtype={"season_year": str},
    )

    expanded = pd.read_csv(
        CANDIDATE_MANIFEST_SOURCE,
        dtype={"season_year": str},
    )

    canonical_international = (
        canonical.loc[
            ~canonical["competition"].isin(
                DOMESTIC_LEAGUES
            )
        ]
        .copy()
    )

    expanded_domestic = (
        expanded.loc[
            expanded["competition"].isin(
                DOMESTIC_LEAGUES
            )
        ]
        .copy()
    )

    candidate = pd.concat(
        [
            canonical_international,
            expanded_domestic,
        ],
        ignore_index=True,
        sort=False,
    )

    duplicate_count = int(
        candidate.duplicated(
            [
                "competition_id",
                "season_id",
            ]
        ).sum()
    )

    if duplicate_count:
        raise AssertionError(
            "Candidate competition manifest contains "
            f"{duplicate_count} duplicate source scopes."
        )

    domestic_scope_count = len(
        candidate.loc[
            candidate["competition"].isin(
                DOMESTIC_LEAGUES
            )
        ]
    )

    if domestic_scope_count != 35:
        raise AssertionError(
            "Expected 35 domestic competition-season "
            f"scopes, found {domestic_scope_count}."
        )

    candidate.to_csv(
        CANDIDATE_MANIFEST,
        index=False,
    )

    print()
    print("Candidate competition manifest")
    print("-" * 88)
    print(
        f"Domestic scopes: "
        f"{domestic_scope_count}"
    )
    print(
        f"International scopes: "
        f"{len(canonical_international)}"
    )
    print(
        f"Total scopes: {len(candidate)}"
    )


def run_preprocessing() -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "scripts.aggregate_player_history_v2",
            "--stats-file",
            str(CANDIDATE_STATS),
            "--competition-manifest-file",
            str(CANDIDATE_MANIFEST),
            "--stat-manifest-file",
            str(STAT_MANIFEST),
            "--profiles-file",
            str(PROFILES),
            "--output-file",
            str(CANDIDATE_PLAYER_DATASET),
        ],
        "101D — HISTORICAL PLAYER AGGREGATION",
    )

    run_command(
        [
            sys.executable,
            "-m",
            "scripts.sofascore_feature_engineering",
            "--input-file",
            str(CANDIDATE_PLAYER_DATASET),
            "--confidence-file",
            str(CONFIDENCE_MANIFEST),
            "--output-file",
            str(CANDIDATE_MODEL_FEATURES),
        ],
        "101D — PLAYER FEATURE ENGINEERING",
    )


def validate_preprocessing() -> None:
    players = pd.read_csv(
        CANDIDATE_PLAYER_DATASET,
        low_memory=False,
    )

    features = pd.read_csv(
        CANDIDATE_MODEL_FEATURES,
        low_memory=False,
    )

    if players.empty:
        raise AssertionError(
            "Candidate player dataset is empty."
        )

    if features.empty:
        raise AssertionError(
            "Candidate model-feature dataset is empty."
        )

    if players["player_id"].duplicated().any():
        raise AssertionError(
            "Candidate player dataset contains "
            "duplicate player IDs."
        )

    if features["player_id"].duplicated().any():
        raise AssertionError(
            "Candidate model-feature dataset contains "
            "duplicate player IDs."
        )

    if set(players["player_id"]) != set(
        features["player_id"]
    ):
        raise AssertionError(
            "Player population changed during "
            "feature engineering."
        )

    per90_columns = [
        column
        for column in features.columns
        if column.endswith("_per90")
    ]

    if not per90_columns:
        raise AssertionError(
            "No per-90 model features were generated."
        )

    print()
    print("Preprocessing validation")
    print("-" * 88)
    print(
        f"Aggregated players: {len(players)}"
    )
    print(
        f"Feature rows: {len(features)}"
    )
    print(
        f"Per-90 features: {len(per90_columns)}"
    )
    print(
        "Player population preservation: PASS"
    )
    print(
        "Unique-player contract: PASS"
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 101D — EXPANDED PLAYER "
        "INTELLIGENCE CANDIDATE"
    )
    print("=" * 88)

    validate_inputs()
    build_candidate_evidence()
    build_candidate_manifest()
    run_preprocessing()
    validate_preprocessing()

    print()
    print("=" * 88)
    print(
        "PREPROCESSING RESULT: PASS"
    )
    print("=" * 88)
    print()
    print(
        f"Candidate evidence: {CANDIDATE_STATS}"
    )
    print(
        f"Candidate player dataset: "
        f"{CANDIDATE_PLAYER_DATASET}"
    )
    print(
        f"Candidate model features: "
        f"{CANDIDATE_MODEL_FEATURES}"
    )


if __name__ == "__main__":
    main()
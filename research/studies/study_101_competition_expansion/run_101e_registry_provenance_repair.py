#run_101e_registry_provenance_repair

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STUDY_101D_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "study_101d_expanded_player_intelligence"
)

STUDY_101E_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "study_101e_registry_provenance_repair"
)

CANDIDATE_PLAYER_DATASET = (
    STUDY_101D_ROOT
    / "candidate_player_dataset.csv"
)

CANDIDATE_MODEL_FEATURES = (
    STUDY_101D_ROOT
    / "candidate_model_features.csv"
)

CANDIDATE_COMPETITION_MANIFEST = (
    STUDY_101D_ROOT
    / "candidate_competition_manifest.csv"
)

CANDIDATE_COMPETITION_FEATURE_MANIFEST = (
    STUDY_101D_ROOT
    / "candidate_competition_feature_manifest.csv"
)

FEATURE_ATTRIBUTE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "feature_attribute_manifest.csv"
)

ROLE_ATTRIBUTE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "role_attribute_manifest.csv"
)

CANDIDATE_REGISTRY = (
    STUDY_101E_ROOT
    / "candidate_player_registry.csv"
)

REPAIRED_ATTRIBUTES = (
    STUDY_101E_ROOT
    / "candidate_player_attribute_scores.csv"
)

REPAIRED_RATINGS = (
    STUDY_101E_ROOT
    / "candidate_player_ratings.csv"
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


def validate_inputs() -> None:
    required = [
        CANDIDATE_PLAYER_DATASET,
        CANDIDATE_MODEL_FEATURES,
        CANDIDATE_COMPETITION_MANIFEST,
        CANDIDATE_COMPETITION_FEATURE_MANIFEST,
        FEATURE_ATTRIBUTE_MANIFEST,
        ROLE_ATTRIBUTE_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Study 101E inputs:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


def run_repair_pipeline() -> None:
    STUDY_101E_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_command(
        [
            sys.executable,
            "-m",
            "scripts.build_player_registry",
            "--input-file",
            str(CANDIDATE_PLAYER_DATASET),
            "--output-file",
            str(CANDIDATE_REGISTRY),
        ],
        "101E — EXPANDED PLAYER REGISTRY",
    )

    run_command(
        [
            sys.executable,
            "-m",
            "scripts.score_player_attributes",
            "--transformation-id",
            "robust_zscore",
            "--features-file",
            str(CANDIDATE_MODEL_FEATURES),
            "--competition-file",
            str(CANDIDATE_COMPETITION_MANIFEST),
            "--competition-feature-file",
            str(
                CANDIDATE_COMPETITION_FEATURE_MANIFEST
            ),
            "--registry-file",
            str(CANDIDATE_REGISTRY),
            "--feature-attribute-file",
            str(FEATURE_ATTRIBUTE_MANIFEST),
            "--output-path",
            str(REPAIRED_ATTRIBUTES),
        ],
        "101E — PROVENANCE-PRESERVING ATTRIBUTES",
    )

    run_command(
        [
            sys.executable,
            "-m",
            "scripts.build_player_ratings_v4",
            "--attribute-path",
            str(REPAIRED_ATTRIBUTES),
            "--registry-path",
            str(CANDIDATE_REGISTRY),
            "--role-attribute-path",
            str(ROLE_ATTRIBUTE_MANIFEST),
            "--output-path",
            str(REPAIRED_RATINGS),
        ],
        "101E — EXPANDED V4 ROLE RATINGS",
    )


def validate_outputs() -> None:
    dataset = pd.read_csv(
        CANDIDATE_PLAYER_DATASET,
        low_memory=False,
    )

    registry = pd.read_csv(
        CANDIDATE_REGISTRY,
        low_memory=False,
    )

    attributes = pd.read_csv(
        REPAIRED_ATTRIBUTES,
        low_memory=False,
    )

    ratings = pd.read_csv(
        REPAIRED_RATINGS,
        low_memory=False,
    )

    expected_players = int(
        dataset["player_id"].nunique()
    )

    if expected_players != 14930:
        raise AssertionError(
            "Study 101E expected the frozen Study 101D "
            f"population of 14,930 players; found "
            f"{expected_players}."
        )

    if len(registry) != expected_players:
        raise AssertionError(
            "Registry population mismatch: "
            f"{len(registry)} vs {expected_players}."
        )

    if len(attributes) != expected_players:
        raise AssertionError(
            "Attribute population mismatch: "
            f"{len(attributes)} vs {expected_players}."
        )

    if len(ratings) != expected_players:
        raise AssertionError(
            "Rating population mismatch: "
            f"{len(ratings)} vs {expected_players}."
        )

    if registry[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Candidate registry contains duplicate "
            "canonical player IDs."
        )

    if attributes[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Candidate attributes contain duplicate "
            "canonical player IDs."
        )

    if ratings[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Candidate ratings contain duplicate "
            "canonical player IDs."
        )

    role_columns = [
        column
        for column in ratings.columns
        if (
            column.startswith("rating_")
            and not column.startswith(
                "raw_rating_"
            )
        )
    ]

    if not role_columns:
        raise AssertionError(
            "No V4 role-rating columns were produced."
        )

    usable_roles = (
        ratings[role_columns]
        .notna()
        .any(axis=1)
    )

    usable_role_count = int(
        usable_roles.sum()
    )

    #
    # Historical provenance should no longer have
    # collapsed to one competition/season per player.
    #
    multi_competition_players = int(
        pd.to_numeric(
            attributes["competition_count"],
            errors="coerce",
        )
        .gt(1)
        .sum()
    )

    multi_season_players = int(
        pd.to_numeric(
            attributes["season_count"],
            errors="coerce",
        )
        .gt(1)
        .sum()
    )

    if multi_season_players == 0:
        raise AssertionError(
            "Historical season provenance still appears "
            "collapsed: no player has season_count > 1."
        )

    print()
    print("Study 101E validation")
    print("-" * 88)
    print(
        f"Aggregated player population: "
        f"{expected_players}"
    )
    print(
        f"Registry rows: {len(registry)}"
    )
    print(
        f"Attribute rows: {len(attributes)}"
    )
    print(
        f"Rating rows: {len(ratings)}"
    )
    print(
        f"Players with >=1 usable role rating: "
        f"{usable_role_count}"
    )
    print(
        f"Players with >1 source competition: "
        f"{multi_competition_players}"
    )
    print(
        f"Players with >1 source season: "
        f"{multi_season_players}"
    )
    print(
        "Unique-player contracts: PASS"
    )
    print(
        "Historical provenance preservation: PASS"
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 101E — EXPANDED REGISTRY "
        "AND PROVENANCE REPAIR"
    )
    print("=" * 88)

    validate_inputs()
    run_repair_pipeline()
    validate_outputs()

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)

    print()
    print(
        f"Candidate registry: "
        f"{CANDIDATE_REGISTRY}"
    )
    print(
        f"Repaired attributes: "
        f"{REPAIRED_ATTRIBUTES}"
    )
    print(
        f"Repaired ratings: "
        f"{REPAIRED_RATINGS}"
    )


if __name__ == "__main__":
    main()
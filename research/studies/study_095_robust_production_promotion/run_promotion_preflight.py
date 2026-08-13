#run_promotion_preflight

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_095_robust_production_promotion"
)

LEGACY_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "legacy_global"
)

PREFLIGHT_PATH = (
    OUTPUT_DIRECTORY
    / "artifact_preflight.csv"
)

HASH_PATH = (
    OUTPUT_DIRECTORY
    / "artifact_hashes_before_promotion.csv"
)

BACKUP_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "legacy_backup_audit.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_095a_metadata.json"
)


ARTIFACTS = {
    "canonical_player_attributes": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "player_attribute_scores.csv"
    ),
    "canonical_player_ratings": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "player_ratings.csv"
    ),
    "canonical_bundesliga_repository": (
        PROJECT_ROOT
        / "outputs"
        / "study_078_bundesliga_production_repository"
        / "bundesliga_club_repository_v1.csv"
    ),
    "canonical_goal_model": (
        PROJECT_ROOT
        / "outputs"
        / "study_069_production_club_goal_model_v1"
        / "integrated_club_goal_model_v1.json"
    ),
    "robust_player_attributes": (
        PROJECT_ROOT
        / "outputs"
        / "study_092_representation_calibration"
        / "study_092c1"
        / "player_attribute_scores_robust_zscore.csv"
    ),
    "robust_player_ratings": (
        PROJECT_ROOT
        / "outputs"
        / "study_092_representation_calibration"
        / "study_092c1"
        / "player_ratings_robust_zscore.csv"
    ),
    "robust_bundesliga_repository": (
        PROJECT_ROOT
        / "outputs"
        / "study_092_representation_calibration"
        / "study_092c1"
        / "club_repositories"
        / "bundesliga_club_repository_robust_zscore.csv"
    ),
    "robust_goal_model": (
        PROJECT_ROOT
        / "outputs"
        / "study_093_production_candidate_validation"
        / "paired_goal_model_artifacts"
        / "robust_zscore"
        / "integrated_club_goal_model_robust_zscore.json"
    ),
}


LEGACY_BACKUPS = {
    "canonical_player_attributes": (
        LEGACY_DIRECTORY
        / "player_attribute_scores_global_zscore.csv"
    ),
    "canonical_player_ratings": (
        LEGACY_DIRECTORY
        / "player_ratings_global_zscore.csv"
    ),
    "canonical_bundesliga_repository": (
        LEGACY_DIRECTORY
        / "bundesliga_club_repository_global_zscore.csv"
    ),
    "canonical_goal_model": (
        LEGACY_DIRECTORY
        / "integrated_club_goal_model_global_zscore.json"
    ),
}


def relative_path(path: Path) -> str:
    try:
        return str(
            path.relative_to(PROJECT_ROOT)
        )
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def build_preflight_table() -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for artifact_name, path in ARTIFACTS.items():
        exists = path.exists()

        records.append(
            {
                "artifact_name": artifact_name,
                "artifact_role": (
                    "robust_candidate"
                    if artifact_name.startswith("robust_")
                    else "current_canonical"
                ),
                "path": relative_path(path),
                "exists": exists,
                "file_size_bytes": (
                    path.stat().st_size
                    if exists
                    else None
                ),
            }
        )

    return pd.DataFrame(records)


def validate_preflight(
    preflight: pd.DataFrame,
) -> None:
    missing = preflight.loc[
        ~preflight["exists"]
    ]

    if not missing.empty:
        raise FileNotFoundError(
            "Promotion preflight found missing artifacts:\n"
            + missing[
                [
                    "artifact_name",
                    "path",
                ]
            ].to_string(index=False)
        )

    if preflight[
        "file_size_bytes"
    ].le(0).any():
        raise ValueError(
            "Promotion preflight found an empty artifact."
        )


def build_hash_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact_name": artifact_name,
                "path": relative_path(path),
                "sha256": sha256(path),
                "file_size_bytes":
                    path.stat().st_size,
            }
            for artifact_name, path
            in ARTIFACTS.items()
        ]
    )


def create_legacy_backups() -> None:
    LEGACY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for artifact_name, backup_path in (
        LEGACY_BACKUPS.items()
    ):
        source_path = ARTIFACTS[
            artifact_name
        ]

        shutil.copy2(
            source_path,
            backup_path,
        )


def build_backup_audit() -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for artifact_name, backup_path in (
        LEGACY_BACKUPS.items()
    ):
        source_path = ARTIFACTS[
            artifact_name
        ]

        source_hash = sha256(
            source_path
        )

        backup_exists = backup_path.exists()

        backup_hash = (
            sha256(backup_path)
            if backup_exists
            else None
        )

        records.append(
            {
                "artifact_name":
                    artifact_name,
                "source_path":
                    relative_path(
                        source_path
                    ),
                "backup_path":
                    relative_path(
                        backup_path
                    ),
                "backup_exists":
                    backup_exists,
                "source_sha256":
                    source_hash,
                "backup_sha256":
                    backup_hash,
                "byte_identical":
                    (
                        backup_exists
                        and source_hash
                        == backup_hash
                    ),
            }
        )

    return pd.DataFrame(records)


def validate_backups(
    audit: pd.DataFrame,
) -> None:
    if not audit[
        "backup_exists"
    ].all():
        raise AssertionError(
            "One or more legacy backups were not created."
        )

    if not audit[
        "byte_identical"
    ].all():
        failures = audit.loc[
            ~audit[
                "byte_identical"
            ]
        ]

        raise AssertionError(
            "One or more legacy backups differ from their "
            "canonical sources:\n"
            + failures.to_string(
                index=False
            )
        )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 095A — ROBUST PRODUCTION "
        "PROMOTION PREFLIGHT"
    )
    print("=" * 88)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    preflight = build_preflight_table()

    preflight.to_csv(
        PREFLIGHT_PATH,
        index=False,
    )

    print()
    print("Artifact preflight")
    print("-" * 88)
    print(
        preflight.to_string(
            index=False
        )
    )

    validate_preflight(
        preflight
    )

    hashes = build_hash_table()

    hashes.to_csv(
        HASH_PATH,
        index=False,
    )

    create_legacy_backups()

    backup_audit = (
        build_backup_audit()
    )

    backup_audit.to_csv(
        BACKUP_AUDIT_PATH,
        index=False,
    )

    validate_backups(
        backup_audit
    )

    metadata = {
        "study_id": "095A",
        "study_name": (
            "Robust Production Promotion Preflight"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "artifact_count": len(
            ARTIFACTS
        ),
        "canonical_artifact_count": 4,
        "robust_candidate_artifact_count": 4,
        "legacy_backup_count": len(
            LEGACY_BACKUPS
        ),
        "all_artifacts_present": True,
        "all_artifacts_nonempty": True,
        "all_legacy_backups_created": True,
        "all_legacy_backups_byte_identical": True,
        "canonical_artifacts_modified": False,
        "robust_artifacts_promoted": False,
        "configuration_defaults_changed": False,
        "promotion_scope": {
            "representation_default": (
                "approved_but_not_applied"
            ),
            "bundesliga_runtime_candidate": (
                "approved_but_not_applied"
            ),
            "world_cup_runtime": (
                "unchanged"
            ),
        },
        "outputs": [
            PREFLIGHT_PATH.name,
            HASH_PATH.name,
            BACKUP_AUDIT_PATH.name,
            relative_path(
                LEGACY_DIRECTORY
            ),
            METADATA_PATH.name,
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Legacy backup audit")
    print("-" * 88)
    print(
        backup_audit[
            [
                "artifact_name",
                "backup_exists",
                "byte_identical",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Validation summary")
    print("  Canonical artifacts present: PASS")
    print("  Robust candidate artifacts present: PASS")
    print("  Artifact files nonempty: PASS")
    print("  SHA-256 identities recorded: PASS")
    print("  Legacy Global backups created: PASS")
    print("  Legacy backups byte-identical: PASS")
    print("  Canonical artifacts modified: NO")
    print("  Robust candidates promoted: NO")
    print("  World Cup runtime changed: NO")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
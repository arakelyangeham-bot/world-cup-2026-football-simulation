#validate_club_identity_registry

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.football_identity.club_identity_registry import (
    CLUB_IDENTITIES,
    get_club_identity,
    list_registered_observation_names,
)
from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBSERVATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_048_club_observation_dataset"
    / "full_squad_observations.csv"
)

CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "clubelo_histories"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_058_club_identity_registry"
)


def load_observation_clubs() -> tuple[str, ...]:
    if not OBSERVATION_PATH.exists():
        raise FileNotFoundError(
            "Observation dataset does not exist: "
            f"{OBSERVATION_PATH}"
        )

    dataframe = pd.read_csv(
        OBSERVATION_PATH,
        usecols=[
            "home_team",
            "away_team",
        ],
        low_memory=False,
    )

    clubs = pd.concat(
        [
            dataframe["home_team"],
            dataframe["away_team"],
        ],
        ignore_index=True,
    )

    clubs = (
        clubs
        .dropna()
        .astype(str)
        .str.strip()
    )

    clubs = clubs[
        clubs.ne("")
    ]

    return tuple(
        sorted(
            clubs.unique().tolist()
        )
    )


def validate_registry_integrity() -> None:
    if not CLUB_IDENTITIES:
        raise AssertionError(
            "Club identity registry is empty."
        )

    observation_names = [
        identity.observation_name
        for identity in (
            CLUB_IDENTITIES.values()
        )
    ]

    canonical_names = [
        identity.canonical_name
        for identity in (
            CLUB_IDENTITIES.values()
        )
    ]

    lookup_names = [
        identity.clubelo_lookup_name
        for identity in (
            CLUB_IDENTITIES.values()
        )
    ]

    if len(observation_names) != len(
        set(observation_names)
    ):
        raise AssertionError(
            "Duplicate observation names detected."
        )

    if len(canonical_names) != len(
        set(canonical_names)
    ):
        raise AssertionError(
            "Duplicate canonical names detected."
        )

    if len(lookup_names) != len(
        set(lookup_names)
    ):
        raise AssertionError(
            "Duplicate ClubElo lookup names detected."
        )

    for key, identity in (
        CLUB_IDENTITIES.items()
    ):
        if key != identity.observation_name:
            raise AssertionError(
                "Registry key and observation name "
                f"differ: {key!r} vs "
                f"{identity.observation_name!r}"
            )

        if not identity.canonical_name.strip():
            raise AssertionError(
                f"{key}: empty canonical name."
            )

        if not (
            identity.clubelo_lookup_name.strip()
        ):
            raise AssertionError(
                f"{key}: empty ClubElo lookup name."
            )


def build_resolution_audit(
    observation_clubs: tuple[str, ...],
    repository: ClubEloRepository,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for observation_name in (
        observation_clubs
    ):
        record: dict[str, object] = {
            "observation_name":
                observation_name,
            "canonical_name": "",
            "clubelo_lookup_name": "",
            "clubelo_resolved_name": "",
            "country": "",
            "history_row_count": 0,
            "history_start": "",
            "history_end": "",
            "identity_registered": False,
            "clubelo_resolution_pass": False,
            "country_validation_pass": False,
            "overall_validation_pass": False,
            "error_type": "",
            "error_message": "",
        }

        try:
            identity = get_club_identity(
                observation_name
            )

            record.update(
                {
                    "canonical_name":
                        identity.canonical_name,
                    "clubelo_lookup_name":
                        identity.clubelo_lookup_name,
                    "identity_registered":
                        True,
                }
            )

            history = repository.get_history(
                club_name=(
                    identity.clubelo_lookup_name
                ),
                refresh=False,
            )

            resolved_names = (
                history["Club"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            if len(resolved_names) != 1:
                raise AssertionError(
                    "ClubElo history resolved to "
                    "an unexpected number of source "
                    f"names: {resolved_names}"
                )

            countries = (
                history["Country"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            if len(countries) != 1:
                raise AssertionError(
                    "ClubElo history contains "
                    "multiple country values: "
                    f"{countries}"
                )

            country_pass = (
                countries[0]
                == identity.expected_country
            )

            record.update(
                {
                    "clubelo_resolved_name":
                        resolved_names[0],
                    "country":
                        countries[0],
                    "history_row_count":
                        int(len(history)),
                    "history_start": (
                        history["From"]
                        .min()
                        .date()
                        .isoformat()
                    ),
                    "history_end": (
                        history["To"]
                        .max()
                        .date()
                        .isoformat()
                    ),
                    "clubelo_resolution_pass":
                        True,
                    "country_validation_pass":
                        country_pass,
                    "overall_validation_pass":
                        country_pass,
                }
            )

        except Exception as error:
            record["error_type"] = (
                type(error).__name__
            )

            record["error_message"] = str(
                error
            )

        records.append(record)

        status = (
            "PASS"
            if record[
                "overall_validation_pass"
            ]
            else "FAIL"
        )

        print(
            f"{observation_name}: {status}"
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "observation_name"
        )
        .reset_index(drop=True)
    )


def main() -> None:
    validate_registry_integrity()

    observation_clubs = (
        load_observation_clubs()
    )

    registered_names = set(
        list_registered_observation_names()
    )

    observation_name_set = set(
        observation_clubs
    )

    missing_registry_entries = sorted(
        observation_name_set
        - registered_names
    )

    extra_registry_entries = sorted(
        registered_names
        - observation_name_set
    )

    if missing_registry_entries:
        raise AssertionError(
            "Observation clubs are missing from "
            "the identity registry: "
            f"{missing_registry_entries}"
        )

    repository = ClubEloRepository(
        cache_directory=CACHE_DIRECTORY
    )

    print("Club Identity Registry Validation")
    print("=" * 76)
    print()
    print(
        "Observation clubs: "
        f"{len(observation_clubs)}"
    )
    print(
        "Registered identities: "
        f"{len(CLUB_IDENTITIES)}"
    )
    print()

    print("ClubElo Resolution")
    print("-" * 76)

    resolution_audit = (
        build_resolution_audit(
            observation_clubs=(
                observation_clubs
            ),
            repository=repository,
        )
    )

    failures = resolution_audit[
        ~resolution_audit[
            "overall_validation_pass"
        ]
    ]

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolution_audit.to_csv(
        OUTPUT_DIRECTORY
        / "club_identity_resolution.csv",
        index=False,
    )

    failures.to_csv(
        OUTPUT_DIRECTORY
        / "unresolved_club_identities.csv",
        index=False,
    )

    summary = {
        "observation_club_count":
            len(observation_clubs),
        "registered_identity_count":
            len(CLUB_IDENTITIES),
        "extra_registry_entry_count":
            len(extra_registry_entries),
        "extra_registry_entries":
            extra_registry_entries,
        "clubelo_resolution_count": int(
            resolution_audit[
                "clubelo_resolution_pass"
            ].sum()
        ),
        "country_validation_count": int(
            resolution_audit[
                "country_validation_pass"
            ].sum()
        ),
        "unresolved_count":
            len(failures),
        "complete_coverage_pass":
            failures.empty,
    }

    with (
        OUTPUT_DIRECTORY
        / "identity_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    metadata = {
        "study_id": "058",
        "study_name": (
            "Canonical Club Identity Registry"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "PASS"
            if failures.empty
            else "FAIL"
        ),
        **summary,
        "output_files": [
            "club_identity_resolution.csv",
            "unresolved_club_identities.csv",
            "identity_summary.json",
            "study_metadata.json",
        ],
    }

    with (
        OUTPUT_DIRECTORY
        / "study_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    if not failures.empty:
        print()
        print("Resolution Failures")
        print("-" * 76)
        print(
            failures[
                [
                    "observation_name",
                    "clubelo_lookup_name",
                    "error_type",
                    "error_message",
                ]
            ].to_string(index=False)
        )

        raise AssertionError(
            "Canonical club identity validation "
            "did not achieve complete coverage."
        )

    print()
    print("Resolution Table")
    print("-" * 76)
    print(
        resolution_audit[
            [
                "observation_name",
                "canonical_name",
                "clubelo_lookup_name",
                "clubelo_resolved_name",
                "country",
            ]
        ].to_string(index=False)
    )

    print()
    print("Registry integrity: PASS")
    print("Observation coverage: PASS")
    print("ClubElo resolution: PASS")
    print("Country validation: PASS")
    print("Duplicate lookup prevention: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        "Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
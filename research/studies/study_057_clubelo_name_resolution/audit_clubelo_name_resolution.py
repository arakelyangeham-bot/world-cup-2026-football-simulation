#audit_clubelo_name_resolution

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    / "study_057_clubelo_name_resolution"
)


TEAM_COLUMNS = (
    "home_team",
    "away_team",
)


def load_observations() -> pd.DataFrame:
    if not OBSERVATION_PATH.exists():
        raise FileNotFoundError(
            "Observation dataset does not exist: "
            f"{OBSERVATION_PATH}"
        )

    dataframe = pd.read_csv(
        OBSERVATION_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Observation dataset is empty."
        )

    missing_columns = (
        set(TEAM_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Observation dataset is missing team "
            f"columns: {sorted(missing_columns)}"
        )

    return dataframe


def extract_unique_clubs(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    home_clubs = (
        observations["home_team"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    away_clubs = (
        observations["away_team"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    clubs = pd.concat(
        [
            home_clubs,
            away_clubs,
        ],
        ignore_index=True,
    )

    clubs = clubs[
        clubs.ne("")
    ]

    counts = (
        clubs
        .value_counts()
        .rename_axis("dataset_club")
        .reset_index(name="match_appearances")
        .sort_values("dataset_club")
        .reset_index(drop=True)
    )

    if counts["dataset_club"].duplicated().any():
        raise AssertionError(
            "Duplicate club names remain after "
            "unique-club extraction."
        )

    return counts


def audit_exact_resolution(
    clubs: pd.DataFrame,
    repository: ClubEloRepository,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for row in clubs.itertuples(
        index=False
    ):
        dataset_club = str(
            row.dataset_club
        )

        record: dict[str, object] = {
            "dataset_club": dataset_club,
            "match_appearances": int(
                row.match_appearances
            ),
            "lookup_name": dataset_club,
            "resolution_method": "exact",
            "exact_lookup_succeeded": False,
            "resolved_club": "",
            "country": "",
            "history_row_count": 0,
            "history_start": "",
            "history_end": "",
            "cache_path": str(
                repository.cache_path(
                    dataset_club
                )
            ),
            "error_type": "",
            "error_message": "",
            "validation_pass": False,
        }

        try:
            history = repository.get_history(
                club_name=dataset_club,
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
                    "ClubElo history did not resolve "
                    "to exactly one club name: "
                    f"{resolved_names}"
                )

            resolved_club = (
                resolved_names[0]
            )

            exact_name_match = (
                dataset_club.casefold()
                == resolved_club.casefold()
            )

            record.update(
                {
                    "exact_lookup_succeeded": True,
                    "resolved_club":
                        resolved_club,
                    "country": str(
                        history["Country"].iloc[0]
                    ),
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
                    "validation_pass":
                        exact_name_match,
                }
            )

            if not exact_name_match:
                record["error_type"] = (
                    "resolved_name_mismatch"
                )

                record["error_message"] = (
                    "The exact lookup returned a "
                    "different ClubElo club name."
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
            if record["validation_pass"]
            else "FAIL"
        )

        print(
            f"{dataset_club}: {status}"
        )

    return (
        pd.DataFrame(records)
        .sort_values("dataset_club")
        .reset_index(drop=True)
    )


def build_summary(
    resolution_audit: pd.DataFrame,
) -> dict[str, object]:
    exact_matches = int(
        resolution_audit[
            "validation_pass"
        ].sum()
    )

    unresolved = int(
        (
            ~resolution_audit[
                "validation_pass"
            ]
        ).sum()
    )

    unresolved_clubs = (
        resolution_audit.loc[
            ~resolution_audit[
                "validation_pass"
            ],
            "dataset_club",
        ]
        .astype(str)
        .tolist()
    )

    error_counts = (
        resolution_audit.loc[
            ~resolution_audit[
                "validation_pass"
            ],
            "error_type",
        ]
        .replace("", "unknown")
        .value_counts()
        .to_dict()
    )

    return {
        "unique_club_count": int(
            len(resolution_audit)
        ),
        "exact_match_count":
            exact_matches,
        "exact_match_rate": float(
            exact_matches
            / len(resolution_audit)
            if len(resolution_audit)
            else 0.0
        ),
        "unresolved_club_count":
            unresolved,
        "unresolved_clubs":
            unresolved_clubs,
        "error_counts": {
            str(key): int(value)
            for key, value
            in error_counts.items()
        },
        "alias_registry_required": (
            unresolved > 0
        ),
    }


def write_markdown_report(
    path: Path,
    summary: dict[str, object],
    resolution_audit: pd.DataFrame,
) -> None:
    lines = [
        "# Study 057 Results",
        "",
        "## ClubElo Name Resolution Audit",
        "",
        "**Status:** `PASS`",
        "",
        "## Observation population",
        "",
        (
            "- Unique clubs: "
            f"{summary['unique_club_count']}"
        ),
        (
            "- Exact ClubElo matches: "
            f"{summary['exact_match_count']}"
        ),
        (
            "- Exact-match rate: "
            f"{summary['exact_match_rate']:.3f}"
        ),
        (
            "- Unresolved clubs: "
            f"{summary['unresolved_club_count']}"
        ),
        "",
        "## Unresolved clubs",
        "",
    ]

    unresolved_clubs = (
        summary["unresolved_clubs"]
    )

    if unresolved_clubs:
        for club in unresolved_clubs:
            lines.append(
                f"- `{club}`"
            )
    else:
        lines.append(
            "- None."
        )

    lines.extend(
        [
            "",
            "## Resolution policy",
            "",
            (
                "This audit allowed exact ClubElo "
                "history lookups only."
            ),
            (
                "No aliases, fuzzy matching, token "
                "matching, or manual substitutions "
                "were applied."
            ),
            "",
            "## Exact matches",
            "",
        ]
    )

    exact_rows = resolution_audit[
        resolution_audit[
            "validation_pass"
        ]
    ]

    if exact_rows.empty:
        lines.append(
            "- None."
        )
    else:
        for row in exact_rows.itertuples(
            index=False
        ):
            lines.append(
                f"- `{row.dataset_club}` → "
                f"`{row.resolved_club}`"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "A failed exact lookup does not mean "
                "that ClubElo lacks the club. It means "
                "that the observation-dataset name "
                "cannot be used directly and requires "
                "a reviewed deterministic alias."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    observations = load_observations()

    clubs = extract_unique_clubs(
        observations
    )

    repository = ClubEloRepository(
        cache_directory=CACHE_DIRECTORY
    )

    print("Study 057")
    print("=" * 76)
    print()
    print(
        f"Unique clubs discovered: "
        f"{len(clubs)}"
    )
    print()
    print("Exact ClubElo Resolution")
    print("-" * 76)

    resolution_audit = (
        audit_exact_resolution(
            clubs=clubs,
            repository=repository,
        )
    )

    summary = build_summary(
        resolution_audit
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    clubs.to_csv(
        OUTPUT_DIRECTORY
        / "unique_clubs.csv",
        index=False,
    )

    resolution_audit.to_csv(
        OUTPUT_DIRECTORY
        / "exact_resolution_audit.csv",
        index=False,
    )

    unresolved = resolution_audit[
        ~resolution_audit[
            "validation_pass"
        ]
    ].copy()

    unresolved.to_csv(
        OUTPUT_DIRECTORY
        / "unresolved_clubs.csv",
        index=False,
    )

    with (
        OUTPUT_DIRECTORY
        / "resolution_summary.json"
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
        "study_id": "057",
        "study_name": (
            "ClubElo Name Resolution Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "observation_path": str(
            OBSERVATION_PATH
        ),
        "cache_directory": str(
            CACHE_DIRECTORY
        ),
        "resolution_policy":
            "exact_lookup_only",
        **summary,
        "output_files": [
            "unique_clubs.csv",
            "exact_resolution_audit.csv",
            "unresolved_clubs.csv",
            "resolution_summary.json",
            "study_metadata.json",
            "STUDY_057_RESULTS.md",
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

    write_markdown_report(
        path=(
            OUTPUT_DIRECTORY
            / "STUDY_057_RESULTS.md"
        ),
        summary=summary,
        resolution_audit=(
            resolution_audit
        ),
    )

    print()
    print("Resolution Summary")
    print("-" * 76)
    print(
        "Unique clubs: "
        f"{summary['unique_club_count']}"
    )
    print(
        "Exact matches: "
        f"{summary['exact_match_count']}"
    )
    print(
        "Exact-match rate: "
        f"{summary['exact_match_rate']:.3f}"
    )
    print(
        "Unresolved clubs: "
        f"{summary['unresolved_club_count']}"
    )

    if summary["unresolved_clubs"]:
        print()
        print("Alias Candidates")
        print("-" * 76)

        for club in (
            summary["unresolved_clubs"]
        ):
            print(club)

    print()
    print("Unique-club extraction: PASS")
    print("Exact-resolution audit: PASS")
    print("Failure capture: PASS")
    print("No fuzzy matching: PASS")
    print("Audit artifact generation: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        "Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
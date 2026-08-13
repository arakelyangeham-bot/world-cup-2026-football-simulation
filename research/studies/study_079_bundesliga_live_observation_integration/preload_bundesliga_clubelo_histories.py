#preload_bundesliga_clubelo_histories

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)

from research.studies.study_079_bundesliga_live_observation_integration.audit_bundesliga_clubelo_cache import (
    BUNDESLIGA_CLUBELO_NAME_OVERRIDES,
    CLUBELO_CACHE_DIRECTORY,
)


MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2.0


@dataclass(frozen=True)
class PreloadResult:
    production_club: str
    clubelo_lookup_name: str
    status: str
    cache_path: Path | None
    resolved_club: str | None
    row_count: int | None
    error: str | None


def preload_one_history(
    *,
    repository: ClubEloRepository,
    production_club: str,
    clubelo_lookup_name: str,
) -> PreloadResult:
    """
    Download, validate, and cache one ClubElo history.

    Existing cache files are reused. Failed downloads are retried
    up to MAX_ATTEMPTS times.
    """

    expected_path = repository.cache_path(
        clubelo_lookup_name
    )

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        try:
            existed_before = expected_path.exists()

            dataframe = repository.get_history(
                club_name=clubelo_lookup_name,
                refresh=False,
            )

            if dataframe.empty:
                raise ValueError(
                    "Resolved ClubElo history is empty."
                )

            unique_resolved_clubs = (
                dataframe["Club"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            if len(unique_resolved_clubs) != 1:
                raise ValueError(
                    "ClubElo history contains an unexpected "
                    "number of resolved club names: "
                    f"{unique_resolved_clubs}"
                )

            if not expected_path.exists():
                repository.save_history(
                    club_name=clubelo_lookup_name,
                    dataframe=dataframe,
                )

            status = (
                "EXISTING"
                if existed_before
                else "DOWNLOADED"
            )

            return PreloadResult(
                production_club=production_club,
                clubelo_lookup_name=(
                    clubelo_lookup_name
                ),
                status=status,
                cache_path=expected_path,
                resolved_club=(
                    unique_resolved_clubs[0]
                ),
                row_count=len(dataframe),
                error=None,
            )

        except Exception as error:
            if attempt < MAX_ATTEMPTS:
                print(
                    "  Retry "
                    f"{attempt}/{MAX_ATTEMPTS - 1} for "
                    f"{clubelo_lookup_name!r}: "
                    f"{type(error).__name__}: {error}"
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

                continue

            return PreloadResult(
                production_club=production_club,
                clubelo_lookup_name=(
                    clubelo_lookup_name
                ),
                status="FAILED",
                cache_path=None,
                resolved_club=None,
                row_count=None,
                error=(
                    f"{type(error).__name__}: {error}"
                ),
            )

    raise AssertionError(
        "Unreachable preload state."
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 079 — PRELOAD BUNDESLIGA "
        "CLUBELO HISTORIES"
    )
    print("=" * 88)

    repository = ClubEloRepository(
        cache_directory=(
            CLUBELO_CACHE_DIRECTORY
        )
    )

    print()
    print(
        "ClubElo cache directory: "
        f"{CLUBELO_CACHE_DIRECTORY}"
    )
    print(
        "Configured Bundesliga clubs: "
        f"{len(BUNDESLIGA_CLUBELO_NAME_OVERRIDES)}"
    )

    results: list[PreloadResult] = []

    print()
    print("Acquiring histories")
    print("-" * 88)

    for production_club in sorted(
        BUNDESLIGA_CLUBELO_NAME_OVERRIDES,
        key=str.casefold,
    ):
        lookup_name = (
            BUNDESLIGA_CLUBELO_NAME_OVERRIDES[
                production_club
            ]
        )

        print(
            f"{production_club:<25} "
            f"-> {lookup_name:<16}",
            end="",
            flush=True,
        )

        result = preload_one_history(
            repository=repository,
            production_club=production_club,
            clubelo_lookup_name=lookup_name,
        )

        results.append(result)

        if result.status == "FAILED":
            print(
                f"FAILED — {result.error}"
            )
        else:
            print(
                f"{result.status} — "
                f"{result.resolved_club} — "
                f"{result.row_count} rows"
            )

    successful = [
        result
        for result in results
        if result.status
        in {
            "DOWNLOADED",
            "EXISTING",
        }
    ]

    failed = [
        result
        for result in results
        if result.status == "FAILED"
    ]

    downloaded = [
        result
        for result in results
        if result.status == "DOWNLOADED"
    ]

    existing = [
        result
        for result in results
        if result.status == "EXISTING"
    ]

    print()
    print("Validation summary")
    print(
        f"  Requested histories: {len(results)}"
    )
    print(
        f"  Downloaded histories: {len(downloaded)}"
    )
    print(
        f"  Existing histories: {len(existing)}"
    )
    print(
        f"  Successful histories: {len(successful)}"
    )
    print(
        f"  Failed histories: {len(failed)}"
    )

    if failed:
        print()
        print("Failed aliases")
        print("-" * 88)

        for result in failed:
            print(
                f"  {result.production_club}"
                f" -> "
                f"{result.clubelo_lookup_name}"
                f" -> "
                f"{result.error}"
            )

        print()
        print("=" * 88)
        print(
            "OVERALL RESULT: PARTIAL COVERAGE"
        )
        print("=" * 88)

        raise SystemExit(1)

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
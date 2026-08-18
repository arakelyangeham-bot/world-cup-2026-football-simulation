#preload_bundesliga_clubelo_histories

from __future__ import annotations

from pathlib import Path

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)

from research.studies.study_079_bundesliga_live_observation_integration.audit_bundesliga_clubelo_cache import (
    BUNDESLIGA_CLUBELO_NAME_OVERRIDES,
    CLUBELO_CACHE_DIRECTORY,
)

from research.production.domestic_clubelo_cache import (
    PreloadResult,
    preload_one_history,
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
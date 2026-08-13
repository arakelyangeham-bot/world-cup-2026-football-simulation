#audit_bundesliga_clubelo_cache

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)

# Adjust this import only if the module lives elsewhere.
from simulation.live_match_observation_builder import (
    ProductionClubRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BUNDESLIGA_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_078_bundesliga_production_repository"
    / "bundesliga_club_repository_v1.csv"
)

# Replace this path only if your existing ClubElo histories are
# stored somewhere else.
CLUBELO_CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "clubelo_histories"
)


# These are candidate ClubElo API/cache names.
#
# The audit does not assume they are correct. It reports the exact
# cache path produced for each alias and whether that file exists.
BUNDESLIGA_CLUBELO_NAME_OVERRIDES = {
    "1. FC Heidenheim": "Heidenheim",
    "1. FC Union Berlin": "UnionBerlin",
    "1. FSV Mainz 05": "Mainz",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia M'gladbach": "Gladbach",
    "Eintracht Frankfurt": "Frankfurt",
    "FC Augsburg": "Augsburg",
    "FC Bayern München": "Bayern",
    "FC St. Pauli": "StPauli",
    "Holstein Kiel": "Holstein",
    "RB Leipzig": "RBLeipzig",
    "SC Freiburg": "Freiburg",
    "SV Werder Bremen": "Werder",
    "TSG Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart",
    "VfL Bochum 1848": "Bochum",
    "VfL Wolfsburg": "Wolfsburg",
}


@dataclass(frozen=True)
class CacheAuditResult:
    production_club: str
    clubelo_lookup_name: str
    cache_path: Path
    cache_exists: bool


def validate_override_coverage(
    production_clubs: tuple[str, ...],
) -> None:
    production_set = set(production_clubs)
    override_set = set(
        BUNDESLIGA_CLUBELO_NAME_OVERRIDES
    )

    missing_overrides = sorted(
        production_set - override_set,
        key=str.casefold,
    )

    extra_overrides = sorted(
        override_set - production_set,
        key=str.casefold,
    )

    if missing_overrides:
        raise ValueError(
            "ClubElo override table is missing Bundesliga "
            f"clubs: {missing_overrides}"
        )

    if extra_overrides:
        raise ValueError(
            "ClubElo override table contains clubs that are "
            "not present in the production repository: "
            f"{extra_overrides}"
        )


def audit_cache(
    *,
    production_repository: ProductionClubRepository,
    clubelo_repository: ClubEloRepository,
) -> tuple[CacheAuditResult, ...]:
    production_clubs = (
        production_repository.list_clubs()
    )

    validate_override_coverage(
        production_clubs
    )

    results: list[CacheAuditResult] = []

    for production_club in production_clubs:
        lookup_name = (
            BUNDESLIGA_CLUBELO_NAME_OVERRIDES[
                production_club
            ]
        )

        cache_path = (
            clubelo_repository.cache_path(
                lookup_name
            )
        )

        results.append(
            CacheAuditResult(
                production_club=production_club,
                clubelo_lookup_name=lookup_name,
                cache_path=cache_path,
                cache_exists=cache_path.exists(),
            )
        )

    return tuple(results)


def print_audit(
    results: tuple[CacheAuditResult, ...],
) -> None:
    print()
    print("ClubElo cache audit")
    print("-" * 88)

    for result in results:
        status = (
            "PASS"
            if result.cache_exists
            else "MISSING"
        )

        print(
            f"{status:<8} "
            f"{result.production_club:<25} "
            f"-> {result.clubelo_lookup_name:<16} "
            f"{result.cache_path.name}"
        )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 079 — BUNDESLIGA CLUBELO CACHE AUDIT"
    )
    print("=" * 88)

    production_repository = (
        ProductionClubRepository(
            repository_path=(
                BUNDESLIGA_REPOSITORY_PATH
            )
        )
    )

    clubelo_repository = ClubEloRepository(
        cache_directory=(
            CLUBELO_CACHE_DIRECTORY
        )
    )

    production_clubs = (
        production_repository.list_clubs()
    )

    print()
    print(
        "Production repository: "
        f"{BUNDESLIGA_REPOSITORY_PATH}"
    )
    print(
        "ClubElo cache directory: "
        f"{CLUBELO_CACHE_DIRECTORY}"
    )
    print(
        f"Bundesliga clubs: {len(production_clubs)}"
    )

    results = audit_cache(
        production_repository=(
            production_repository
        ),
        clubelo_repository=(
            clubelo_repository
        ),
    )

    print_audit(results)

    existing = tuple(
        result
        for result in results
        if result.cache_exists
    )

    missing = tuple(
        result
        for result in results
        if not result.cache_exists
    )

    print()
    print("Validation summary")
    print(
        f"  Production clubs: {len(results)}"
    )
    print(
        f"  Existing cache files: {len(existing)}"
    )
    print(
        f"  Missing cache files: {len(missing)}"
    )

    if missing:
        print()
        print("Missing ClubElo histories")

        for result in missing:
            print(
                "  "
                f"{result.production_club} "
                f"-> {result.clubelo_lookup_name} "
                f"-> {result.cache_path}"
            )

        print()
        print("=" * 88)
        print(
            "OVERALL RESULT: INCOMPLETE CACHE COVERAGE"
        )
        print("=" * 88)

        raise SystemExit(1)

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
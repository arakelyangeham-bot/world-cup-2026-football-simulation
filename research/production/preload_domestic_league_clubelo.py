#preload_domestic_league_clubelo

from __future__ import annotations

import argparse

import pandas as pd

from research.adapters.football_model_adapter import (
    CLUBELO_CACHE_DIRECTORY,
)
from research.production.domestic_clubelo_cache import (
    preload_one_history,
)
from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)
from simulation.domestic_league_configs import (
    DOMESTIC_LEAGUE_CONFIGS,
)

from research.production.domestic_league_onboarding import (
    DOMESTIC_LEAGUE_ONBOARDING_SPECS,
)

from research.studies.study_079_bundesliga_live_observation_integration.audit_bundesliga_clubelo_cache import (
    BUNDESLIGA_CLUBELO_NAME_OVERRIDES,
)

from research.production.domestic_clubelo_identity import (
    LA_LIGA_CLUBELO_NAME_OVERRIDES,
    SERIE_A_2026_27_CLUBELO_NAME_OVERRIDES,
    build_clubelo_lookup_candidates,
)

BUNDESLIGA_2026_27_CLUBELO_NAME_OVERRIDES = {
    **BUNDESLIGA_CLUBELO_NAME_OVERRIDES,
    "1. FC Köln": "Koeln",
    "FC Schalke 04": "Schalke",
    "Hamburger SV": "Hamburg",
    "SC Paderborn 07": "Paderborn",
    "SV 07 Elversberg": "Elversberg",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preload ClubElo histories for a configured "
            "domestic league."
        )
    )

    parser.add_argument(
        "--competition",
        choices=sorted(DOMESTIC_LEAGUE_CONFIGS),
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the ClubElo lookup candidate chain "
            "without making requests or writing caches."
        ),
    )

    return parser.parse_args()

def main() -> None:
    arguments = parse_arguments()

    config = DOMESTIC_LEAGUE_CONFIGS[
        arguments.competition
    ]

    onboarding_spec = (
        DOMESTIC_LEAGUE_ONBOARDING_SPECS.get(
            arguments.competition
        )
    )

    repository_frame = pd.read_csv(
        config.repository_path,
        low_memory=False,
    )

    repository_clubs = sorted(
        repository_frame["club"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    team_slugs_by_club: dict[str, str] = {}

    if (
        onboarding_spec is not None
        and onboarding_spec.target_participants_path.exists()
    ):
        participants = pd.read_csv(
            onboarding_spec.target_participants_path,
            low_memory=False,
        )

        required_identity_columns = {
            "team",
            "team_slug",
        }

        if required_identity_columns.issubset(
            participants.columns
        ):
            identity_rows = (
                participants[
                    [
                        "team",
                        "team_slug",
                    ]
                ]
                .dropna()
                .drop_duplicates()
            )

            team_slugs_by_club = {
                str(row["team"]): str(row["team_slug"])
                for _, row in identity_rows.iterrows()
            }

    clubs = repository_clubs

    name_overrides_by_competition = {
        "la_liga": (
            LA_LIGA_CLUBELO_NAME_OVERRIDES
        ),
        "bundesliga": (
            BUNDESLIGA_2026_27_CLUBELO_NAME_OVERRIDES
        ),
        "serie_a": (
            SERIE_A_2026_27_CLUBELO_NAME_OVERRIDES
        ),
    }

    name_overrides = (
        name_overrides_by_competition.get(
            arguments.competition,
            {},
        )
    )

    repository = ClubEloRepository(
        cache_directory=CLUBELO_CACHE_DIRECTORY,
    )

    results = []

    print()
    print(
        f"{config.competition_name.upper()} "
        "CLUBELO CACHE PRELOAD"
    )
    print("=" * 72)

    print(
        "Cache directory:",
        CLUBELO_CACHE_DIRECTORY,
    )
    print("Clubs:", len(clubs))

    print()

    for index, production_club in enumerate(
        clubs,
        start=1,
    ):
        explicit_lookup = name_overrides.get(
            production_club
        )

        slug_lookup = team_slugs_by_club.get(
            production_club
        )

        lookup_candidates = build_clubelo_lookup_candidates(
            production_club=production_club,
            explicit_lookup=explicit_lookup,
            team_slug=slug_lookup,
        )

        if arguments.dry_run:
            print(
                f"[{index}/{len(clubs)}] "
                f"{production_club} "
                f"-> "
                + " -> ".join(lookup_candidates)
            )
            continue

        result = None

        for lookup_name in lookup_candidates:
            print(
                f"[{index}/{len(clubs)}] "
                f"{production_club} "
                f"-> {lookup_name}"
            )

            candidate_result = preload_one_history(
                repository=repository,
                production_club=production_club,
                clubelo_lookup_name=lookup_name,
            )

            result = candidate_result

            if candidate_result.status != "FAILED":
                break

        results.append(result)

        if result.status == "FAILED":
            print(
                "  FAILED:",
                result.error,
            )
        else:
            print(
                f"  {result.status}: "
                f"{result.resolved_club} "
                f"({result.row_count} rows)"
            )

    if arguments.dry_run:
        print()
        print("Dry run complete. No ClubElo requests were made.")
        return

    successful = [
        result
        for result in results
        if result.status in {
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
    print("PRELOAD SUMMARY")
    print("=" * 72)

    print(
        "Requested histories:",
        len(results),
    )
    print(
        "Downloaded histories:",
        len(downloaded),
    )
    print(
        "Existing histories:",
        len(existing),
    )
    print(
        "Successful histories:",
        len(successful),
    )
    print(
        "Failed histories:",
        len(failed),
    )

    if failed:
        print()
        print("FAILURES")
        print("=" * 72)

        for result in failed:
            print(
                result.production_club,
                "->",
                result.clubelo_lookup_name,
                "|",
                result.error,
            )

        raise SystemExit(1)

    print()
    print(
        "Domestic ClubElo preload: PASS"
    )


if __name__ == "__main__":
    main()
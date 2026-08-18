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


LA_LIGA_CLUBELO_NAME_OVERRIDES = {
    "Athletic Club": "Bilbao",
    "Atlético Madrid": "Atletico",
    "Celta Vigo": "Celta",
    "Deportivo Alavés": "Alaves",
    "Deportivo de A Coruña": "Depor",
    "FC Barcelona": "Barcelona",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "Rayo Vallecano": "rayovallecano",
    "Real Betis": "Betis",
    "Real Madrid": "realmadrid",
    "Real Racing Club": "Santander",
    "Real Sociedad": "Sociedad",
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

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    config = DOMESTIC_LEAGUE_CONFIGS[
        arguments.competition
    ]

    repository_frame = pd.read_csv(
        config.repository_path,
        low_memory=False,
    )

    clubs = sorted(
        repository_frame["club"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    name_overrides = {}

    if arguments.competition == "la_liga":
        name_overrides = (
            LA_LIGA_CLUBELO_NAME_OVERRIDES
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
        lookup_name = name_overrides.get(
            production_club,
            production_club,
        )

        print(
            f"[{index}/{len(clubs)}] "
            f"{production_club} "
            f"-> {lookup_name}"
        )

        result = preload_one_history(
            repository=repository,
            production_club=production_club,
            clubelo_lookup_name=lookup_name,
        )

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
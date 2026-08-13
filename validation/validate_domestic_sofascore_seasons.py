#validate_domestic_sofascore_seasons

from __future__ import annotations

import argparse

from shared.competition_registry import get_competition
from shared.sofascore_season_loader import (
    load_sofascore_seasons,
)




EXPECTED_SEASONS = {
    "premier_league": {
        "unique_tournament_id": 17,
        "season_ids": {
            2021: 37036,
            2022: 41886,
            2023: 52186,
            2024: 61627,
            2025: 76986,
        },
    },
    "la_liga": {
        "unique_tournament_id": 8,
        "season_ids": {
            2021: 37223,
            2022: 42409,
            2023: 52376,
            2024: 61643,
            2025: 77559,
        },
    },
    "serie_a": {
        "unique_tournament_id": 23,
        "season_ids": {
            2021: 37475,
            2022: 42415,
            2023: 52760,
            2024: 63515,
            2025: 76457,
        },
    },
    "bundesliga": {
        "unique_tournament_id": 35,
        "season_ids": {
            2021: 37166,
            2022: 42268,
            2023: 52608,
            2024: 63516,
            2025: 77333,
        },
    },
    "ligue_1": {
        "unique_tournament_id": 34,
        "season_ids": {
            2021: 37167,
            2022: 42273,
            2023: 52571,
            2024: 61736,
            2025: 77356,
        },
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate registered Sofascore seasons "
            "for one domestic competition."
        )
    )

    parser.add_argument(
        "--competition",
        required=True,
        choices=sorted(EXPECTED_SEASONS),
        help="Domestic competition key to validate.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    seasons = load_sofascore_seasons()

    competition_key = arguments.competition
    expected = EXPECTED_SEASONS[competition_key]

    competition = get_competition(
        competition_key
    )

    registered_seasons = sorted(
        [
            season
            for season in seasons
            if season.competition_key == competition_key
        ],
        key=lambda season: season.year,
    )

    expected_season_ids = expected[
        "season_ids"
    ]

    print("Domestic Sofascore Season Validation")
    print("====================================")
    print(f"Competition: {competition.display_name}")
    print(f"Competition key: {competition.key}")
    print(
        f"Registered seasons: "
        f"{len(registered_seasons)}"
    )
    print()

    if competition.category != "domestic_league":
        raise ValueError(
            f"{competition_key!r} is not registered "
            "as a domestic league."
        )

    if len(registered_seasons) != len(
        expected_season_ids
    ):
        raise ValueError(
            f"Expected {len(expected_season_ids)} seasons, "
            f"but found {len(registered_seasons)}."
        )

    observed_years = {
        season.year
        for season in registered_seasons
    }

    if observed_years != set(expected_season_ids):
        raise ValueError(
            "Registered years do not match expectations. "
            f"Observed: {sorted(observed_years)}"
        )

    for season in registered_seasons:
        expected_season_id = (
            expected_season_ids[season.year]
        )

        if (
            season.unique_tournament_id
            != expected["unique_tournament_id"]
        ):
            raise ValueError(
                f"{season.dataset_id} has unique tournament "
                f"ID {season.unique_tournament_id}; expected "
                f"{expected['unique_tournament_id']}."
            )

        if season.season_id != expected_season_id:
            raise ValueError(
                f"{season.dataset_id} has season ID "
                f"{season.season_id}; expected "
                f"{expected_season_id}."
            )

        output_filename = (
            competition.filename_pattern.format(
                year=season.year,
            )
        )

        print(f"Dataset ID: {season.dataset_id}")
        print(
            "Unique tournament ID: "
            f"{season.unique_tournament_id}"
        )
        print(f"Season ID: {season.season_id}")
        print(f"Output filename: {output_filename}")
        print()

    print("Validation Result")
    print("-----------------")
    print("PASSED")
    print(
        f"All {competition.display_name} "
        "Sofascore season checks passed."
    )


if __name__ == "__main__":
    main()
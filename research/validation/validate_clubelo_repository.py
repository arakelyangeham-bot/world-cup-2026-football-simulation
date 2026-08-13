#validate_clubelo_repository

from __future__ import annotations

from pathlib import Path

import numpy as np

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "clubelo_histories"
)


TEST_CLUB = "Arsenal"

TEST_DATES = (
    "2023-08-10",
    "2023-08-11",
    "2023-08-12",
    "2024-01-01",
    "2024-05-19",
)


def main() -> None:
    repository = ClubEloRepository(
        cache_directory=CACHE_DIRECTORY
    )

    downloaded = repository.get_history(
        TEST_CLUB,
        refresh=True,
    )

    if downloaded.empty:
        raise AssertionError(
            "Downloaded ClubElo history is empty."
        )

    cache_path = repository.cache_path(
        TEST_CLUB
    )

    if not cache_path.exists():
        raise AssertionError(
            "ClubElo cache file was not created."
        )

    cached = repository.get_history(
        TEST_CLUB,
        refresh=False,
    )

    comparison_columns = [
        "Rank",
        "Club",
        "Country",
        "Level",
        "Elo",
        "From",
        "To",
    ]

    if len(downloaded) != len(cached):
        raise AssertionError(
            "Downloaded and cached history "
            "row counts differ."
        )

    for column in comparison_columns:
        left = downloaded[column]
        right = cached[column]

        if column in {
            "Rank",
            "Level",
            "Elo",
        }:
            equal = np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
            ).all()
        else:
            equal = (
                left.astype(str)
                .eq(
                    right.astype(str)
                )
                .all()
            )

        if not equal:
            raise AssertionError(
                "Downloaded and cached "
                f"histories differ in {column!r}."
            )

    results = []

    for test_date in TEST_DATES:
        result = repository.resolve_rating(
            club_name=TEST_CLUB,
            prediction_date=test_date,
        )

        if not result.temporal_validity_pass:
            raise AssertionError(
                f"{test_date}: temporal validity failed."
            )

        if not (
            result.effective_from
            <= result.prediction_date
            <= result.effective_to
        ):
            raise AssertionError(
                f"{test_date}: resolved date is "
                "outside the returned interval."
            )

        if not np.isfinite(
            result.rating
        ):
            raise AssertionError(
                f"{test_date}: non-finite Elo rating."
            )

        results.append(result)

    first_result = results[0]

    if not np.isclose(
        first_result.rating,
        1919.381104,
        atol=1e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "The Arsenal rating for 2023-08-10 "
            "does not match the verified ClubElo "
            "dated-endpoint result."
        )

    if (
        first_result.effective_from.isoformat()
        != "2023-06-11"
    ):
        raise AssertionError(
            "Unexpected Arsenal interval start "
            "for 2023-08-10."
        )

    if (
        first_result.effective_to.isoformat()
        != "2023-08-12"
    ):
        raise AssertionError(
            "Unexpected Arsenal interval end "
            "for 2023-08-10."
        )

    print("ClubElo Repository Validation")
    print("=" * 72)
    print()
    print(
        f"Club: {TEST_CLUB}"
    )
    print(
        "Downloaded history rows: "
        f"{len(downloaded)}"
    )
    print(
        f"Cache path: {cache_path}"
    )
    print()

    print("Resolved Ratings")
    print("-" * 72)

    for result in results:
        print(
            f"{result.prediction_date}: "
            f"Elo={result.rating:.6f}, "
            f"interval="
            f"{result.effective_from}"
            f" to "
            f"{result.effective_to}"
        )

    print()
    print("Download validation: PASS")
    print("Cache round-trip: PASS")
    print("Interval integrity: PASS")
    print("Historical resolution: PASS")
    print("Temporal validity: PASS")
    print("Verified 2023 rating: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
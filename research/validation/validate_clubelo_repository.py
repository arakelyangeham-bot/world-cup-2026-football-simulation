#validate_clubelo_repository

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.rating_priors.clubelo_repository import (
    ClubEloDownloadError,
    ClubEloRepository,
)

from research.production.domestic_clubelo_cache import (
    preload_one_history,
)

from unittest.mock import patch
from urllib.error import HTTPError, URLError
import socket

from tempfile import TemporaryDirectory

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

class FakeHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def read(self) -> bytes:
        return self.payload

with TemporaryDirectory() as temporary_directory:
    temporary_path = Path(temporary_directory)

    timeout_repository = ClubEloRepository(
        cache_directory=temporary_path / "timeout_cache",
        request_timeout_seconds=2.0,
    )

    successful_payload = (
        b"Rank,Club,Country,Level,Elo,From,To\n"
        b"1,Test Club,ENG,1,1900.0,2026-01-01,2026-01-31\n"
        b"2,Test Club,ENG,1,1895.0,2026-02-01,2026-02-28\n"
    )

    with patch(
        "research.rating_priors.clubelo_repository.urlopen",
        side_effect=TimeoutError("timed out"),
    ):
        try:
            timeout_repository.download_history(
                "Test Club"
            )
        except ClubEloDownloadError as exc:
            assert exc.category == "TIMEOUT"
            assert exc.club_name == "Test Club"
            assert "api.clubelo.com" in exc.url
        else:
            raise AssertionError(
                "Expected ClubElo timeout to raise "
                "ClubEloDownloadError."
            )

    http_repository = ClubEloRepository(
        cache_directory=temporary_path / "http_cache",
        request_timeout_seconds=2.0,
    )

    http_error = HTTPError(
        url="http://api.clubelo.com/Test%20Club",
        code=502,
        msg="Bad Gateway",
        hdrs=None,
        fp=None,
    )

    with patch(
        "research.rating_priors.clubelo_repository.urlopen",
        side_effect=http_error,
    ):
        try:
            http_repository.download_history(
                "Test Club"
            )
        except ClubEloDownloadError as exc:
            assert exc.category == "HTTP_5XX"
            assert "502" in str(exc)
        else:
            raise AssertionError(
                "Expected HTTP 502 to raise "
                "ClubEloDownloadError."
            )

    network_repository = ClubEloRepository(
        cache_directory=temporary_path / "network_cache",
        request_timeout_seconds=2.0,
    )

    network_error = URLError(
        socket.gaierror(
            11001,
            "getaddrinfo failed",
        )
    )

    with patch(
        "research.rating_priors.clubelo_repository.urlopen",
        side_effect=network_error,
    ):
        try:
            network_repository.download_history(
                "Test Club"
            )
        except ClubEloDownloadError as exc:
            assert exc.category == "NETWORK"
        else:
            raise AssertionError(
                "Expected DNS/network failure to raise "
                "ClubEloDownloadError."
            )

    successful_repository = ClubEloRepository(
        cache_directory=(
            temporary_path / "successful_cache"
        ),
        request_timeout_seconds=2.0,
    )

    with patch(
        "research.rating_priors.clubelo_repository.urlopen",
        return_value=FakeHTTPResponse(
            successful_payload
        ),
    ) as mocked_urlopen:
        successful_history = (
            successful_repository.get_history(
                "Test Club"
            )
        )

    assert len(successful_history) == 2

    assert successful_history[
        "Elo"
    ].tolist() == [
        1900.0,
        1895.0,
    ]

    mocked_urlopen.assert_called_once_with(
        "http://api.clubelo.com/Test%20Club",
        timeout=2.0,
    )

    successful_cache_path = (
        successful_repository.cache_path(
            "Test Club"
        )
    )

    assert successful_cache_path.exists()

    with patch(
        "research.rating_priors.clubelo_repository.urlopen",
        side_effect=AssertionError(
            "Network should not be used for cached history."
        ),
    ):
        cached_history = (
            successful_repository.get_history(
                "Test Club"
            )
        )

    assert len(cached_history) == 2


def main() -> None:
    repository = ClubEloRepository(
        cache_directory=CACHE_DIRECTORY
    )

    history = repository.get_history(
        TEST_CLUB,
        refresh=False,
    )

    if history.empty:
        raise AssertionError(
            "ClubElo history is empty."
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

    if len(history) != len(cached):
        raise AssertionError(
            "Downloaded and cached history "
            "row counts differ."
        )

    for column in comparison_columns:
        left = history[column]
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
        f"{len(history)}"
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
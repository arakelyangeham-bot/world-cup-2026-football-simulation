#domestic_clubelo_cache

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
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

    The ClubElo lookup name may differ from the production club name.

    The final cache is written under the production club name so that
    downstream simulation code can resolve ratings using its canonical
    domestic-league club identity.
    """

    expected_path = repository.cache_path(
        production_club
    )

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        try:
            existed_before = expected_path.exists()

            uses_alias = (
                clubelo_lookup_name != production_club
            )

            if uses_alias:
                dataframe = repository.get_history(
                    clubelo_lookup_name,
                    refresh=False,
                )
            else:
                dataframe = repository.get_history(
                    production_club,
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

            #
            # If ClubElo required an alias, persist the validated
            # history under the production club identity as well.
            #
            if (
                uses_alias
                or not expected_path.exists()
            ):
                repository.save_history(
                    club_name=production_club,
                    dataframe=dataframe,
                )

            status = (
                "EXISTING"
                if existed_before
                else "DOWNLOADED"
            )

            return PreloadResult(
                production_club=production_club,
                clubelo_lookup_name=clubelo_lookup_name,
                status=status,
                cache_path=expected_path,
                resolved_club=unique_resolved_clubs[0],
                row_count=len(dataframe),
                error=None,
            )

        except Exception as error:
            if attempt < MAX_ATTEMPTS:
                time.sleep(
                    RETRY_DELAY_SECONDS
                )
                continue

            return PreloadResult(
                production_club=production_club,
                clubelo_lookup_name=clubelo_lookup_name,
                status="FAILED",
                cache_path=(
                    expected_path
                    if expected_path.exists()
                    else None
                ),
                resolved_club=None,
                row_count=None,
                error=(
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            )

    raise AssertionError(
        "Unreachable preload state."
    )
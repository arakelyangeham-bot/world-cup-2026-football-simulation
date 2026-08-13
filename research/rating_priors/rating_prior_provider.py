#rating_prior_provider

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import pandas as pd

from research.football_identity.club_identity_registry import (
    get_club_identity,
)
from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)


@dataclass(frozen=True)
class RatingPriorRequest:
    """
    Generic request for a temporally valid team rating.

    team_name may be either an observation-dataset name or
    a canonical project name registered by the identity layer.
    """

    team_name: str
    prediction_date: (
        str
        | date
        | datetime
        | pd.Timestamp
    )


@dataclass(frozen=True)
class RatingPriorResult:
    """
    Generic historical rating-prior result.

    This object deliberately contains no ClubElo-specific
    implementation details beyond source provenance.
    """

    requested_team_name: str
    canonical_team_name: str
    source_team_name: str

    rating: float
    source: str

    prediction_date: date
    effective_from: date
    effective_to: date

    temporal_validity_pass: bool
    rating_available: bool

    source_rank: float | None = None
    source_country: str | None = None
    source_level: int | None = None


class RatingPriorProvider(Protocol):
    """
    Interface implemented by historical rating providers.
    """

    @property
    def source_name(self) -> str:
        ...

    def get_rating_prior(
        self,
        request: RatingPriorRequest,
    ) -> RatingPriorResult:
        ...


class ClubEloRatingPriorProvider:
    """
    Rating-prior provider backed by ClubElo interval histories.

    Responsibilities:
    - resolve project club identity;
    - translate to the ClubElo lookup identifier;
    - request the rating valid on the prediction date;
    - return a generic RatingPriorResult.

    It does not download or validate raw ClubElo data itself.
    Those responsibilities remain inside ClubEloRepository.
    """

    def __init__(
        self,
        repository: ClubEloRepository,
    ) -> None:
        self.repository = repository

    @property
    def source_name(self) -> str:
        return "clubelo"

    def get_rating_prior(
        self,
        request: RatingPriorRequest,
    ) -> RatingPriorResult:
        identity = get_club_identity(
            request.team_name
        )

        source_result = (
            self.repository.resolve_rating(
                club_name=(
                    identity.clubelo_lookup_name
                ),
                prediction_date=(
                    request.prediction_date
                ),
            )
        )

        if not source_result.temporal_validity_pass:
            raise AssertionError(
                "ClubElo repository returned a rating "
                "that failed temporal validation."
            )

        if source_result.source != self.source_name:
            raise AssertionError(
                "ClubElo provider received an unexpected "
                f"source label: {source_result.source!r}"
            )

        return RatingPriorResult(
            requested_team_name=(
                request.team_name
            ),
            canonical_team_name=(
                identity.canonical_name
            ),
            source_team_name=(
                source_result.resolved_club
            ),
            rating=source_result.rating,
            source=source_result.source,
            prediction_date=(
                source_result.prediction_date
            ),
            effective_from=(
                source_result.effective_from
            ),
            effective_to=(
                source_result.effective_to
            ),
            temporal_validity_pass=True,
            rating_available=True,
            source_rank=source_result.rank,
            source_country=(
                source_result.country
            ),
            source_level=source_result.level,
        )
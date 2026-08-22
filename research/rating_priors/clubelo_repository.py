#clubelo_repository

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import socket

import pandas as pd


CLUBELO_BASE_URL = "http://api.clubelo.com"




@dataclass(frozen=True)
class ClubEloRatingInterval:
    """
    One ClubElo rating interval.

    The rating is valid from effective_from through
    effective_to, inclusive.
    """

    club: str
    country: str
    level: int
    rank: float | None
    elo: float
    effective_from: date
    effective_to: date
    source: str = "clubelo"

    def contains(
        self,
        prediction_date: date,
    ) -> bool:
        return (
            self.effective_from
            <= prediction_date
            <= self.effective_to
        )


@dataclass(frozen=True)
class ClubEloRatingResult:
    """
    Result returned by a temporally valid repository lookup.
    """

    requested_club: str
    resolved_club: str
    rating: float
    rank: float | None
    country: str
    level: int
    prediction_date: date
    effective_from: date
    effective_to: date
    source: str
    temporal_validity_pass: bool

class ClubEloDownloadError(RuntimeError):
    """
    Classified failure while acquiring a ClubElo history.
    """

    def __init__(
        self,
        *,
        club_name: str,
        url: str,
        category: str,
        message: str,
    ) -> None:
        super().__init__(message)

        self.club_name = club_name
        self.url = url
        self.category = category

class ClubEloRepository:
    """
    Cached repository of ClubElo club-history intervals.
    """

    REQUIRED_COLUMNS = {
        "Rank",
        "Club",
        "Country",
        "Level",
        "Elo",
        "From",
        "To",
    }

    def __init__(
        self,
        cache_directory: Path,
        request_timeout_seconds: float = 15.0,
    ) -> None:
        self.cache_directory = Path(
            cache_directory
        )

        self.cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be positive."
            )

        self.request_timeout_seconds = (
            float(request_timeout_seconds)
        )

        self._memory_cache: dict[
            str,
            pd.DataFrame,
        ] = {}

    @staticmethod
    def normalize_lookup_key(
        club_name: str,
    ) -> str:
        normalized = (
            club_name
            .strip()
            .casefold()
        )

        if not normalized:
            raise ValueError(
                "Club name cannot be empty."
            )

        return normalized

    @staticmethod
    def cache_file_stem(
        club_name: str,
    ) -> str:
        characters: list[str] = []

        for character in (
            club_name.strip()
        ):
            if character.isalnum():
                characters.append(
                    character.lower()
                )
            else:
                characters.append("_")

        stem = "".join(characters)

        while "__" in stem:
            stem = stem.replace(
                "__",
                "_",
            )

        stem = stem.strip("_")

        if not stem:
            raise ValueError(
                "Club name produced an empty "
                "cache-file stem."
            )

        return stem

    def cache_path(
        self,
        club_name: str,
    ) -> Path:
        return (
            self.cache_directory
            / (
                f"{self.cache_file_stem(club_name)}"
                "_clubelo_history.csv"
            )
        )

    @staticmethod
    def build_history_url(
        club_name: str,
    ) -> str:
        encoded_name = quote(
            club_name.strip(),
            safe="",
        )

        return (
            f"{CLUBELO_BASE_URL}/"
            f"{encoded_name}"
        )

    @staticmethod
    def parse_prediction_date(
        value: (
            str
            | date
            | datetime
            | pd.Timestamp
        ),
    ) -> date:
        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, pd.Timestamp):
            return value.date()

        if isinstance(value, date):
            return value

        parsed = pd.to_datetime(
            value,
            errors="raise",
        )

        return parsed.date()

    def download_history(
        self,
        club_name: str,
    ) -> pd.DataFrame:
        """
        Download one club's complete ClubElo history.
        """
        url = self.build_history_url(
            club_name
        )

        try:
            with urlopen(
                url,
                timeout=self.request_timeout_seconds,
            ) as response:
                payload = response.read()

        except HTTPError as exc:
            if 500 <= exc.code <= 599:
                category = "HTTP_5XX"
            else:
                category = "HTTP_ERROR"

            raise ClubEloDownloadError(
                club_name=club_name,
                url=url,
                category=category,
                message=(
                    f"ClubElo request failed for "
                    f"{club_name!r}: HTTP {exc.code} "
                    f"{exc.reason}"
                ),
            ) from exc

        except (
            TimeoutError,
            socket.timeout,
        ) as exc:
            raise ClubEloDownloadError(
                club_name=club_name,
                url=url,
                category="TIMEOUT",
                message=(
                    "ClubElo request timed out for "
                    f"{club_name!r} after "
                    f"{self.request_timeout_seconds:.1f} seconds."
                ),
            ) from exc

        except URLError as exc:
            reason = exc.reason

            if isinstance(
                reason,
                (
                    TimeoutError,
                    socket.timeout,
                ),
            ):
                category = "TIMEOUT"
            else:
                category = "NETWORK"

            raise ClubEloDownloadError(
                club_name=club_name,
                url=url,
                category=category,
                message=(
                    "ClubElo network request failed for "
                    f"{club_name!r}: {reason}"
                ),
            ) from exc

        dataframe = pd.read_csv(
            BytesIO(payload),
            low_memory=False,
        )

        return self.normalize_history(
            dataframe=dataframe,
            requested_club=club_name,
        )

    def normalize_history(
        self,
        dataframe: pd.DataFrame,
        requested_club: str,
    ) -> pd.DataFrame:
        """
        Validate and normalize a raw ClubElo history.
        """
        if dataframe.empty:
            raise ValueError(
                "ClubElo returned an empty "
                f"history for {requested_club!r}."
            )

        missing_columns = (
            self.REQUIRED_COLUMNS
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "ClubElo history is missing "
                f"columns: {sorted(missing_columns)}"
            )

        output = dataframe[
            [
                "Rank",
                "Club",
                "Country",
                "Level",
                "Elo",
                "From",
                "To",
            ]
        ].copy()

        output["Club"] = (
            output["Club"]
            .astype(str)
            .str.strip()
        )

        output["Country"] = (
            output["Country"]
            .astype(str)
            .str.strip()
        )

        output["Rank"] = pd.to_numeric(
            output["Rank"],
            errors="coerce",
        )

        output["Level"] = pd.to_numeric(
            output["Level"],
            errors="raise",
        ).astype(int)

        output["Elo"] = pd.to_numeric(
            output["Elo"],
            errors="raise",
        )

        output["From"] = pd.to_datetime(
            output["From"],
            errors="raise",
        ).dt.normalize()

        output["To"] = pd.to_datetime(
            output["To"],
            errors="raise",
        ).dt.normalize()

        if output[
            [
                "Club",
                "Country",
                "Level",
                "Elo",
                "From",
                "To",
            ]
        ].isna().any().any():
            raise ValueError(
                "ClubElo history contains missing "
                "required values."
            )

        if (
            output["From"]
            > output["To"]
        ).any():
            raise ValueError(
                "ClubElo history contains an "
                "interval whose start is after "
                "its end."
            )

        output = (
            output
            .sort_values(
                [
                    "From",
                    "To",
                ]
            )
            .reset_index(drop=True)
        )

        self.validate_non_overlapping_intervals(
            output
        )

        unique_clubs = (
            output["Club"]
            .dropna()
            .unique()
            .tolist()
        )

        if len(unique_clubs) != 1:
            raise ValueError(
                "ClubElo club-history response "
                "contains multiple club names: "
                f"{unique_clubs}"
            )

        output[
            "requested_club"
        ] = requested_club

        output[
            "source"
        ] = "clubelo"

        return output

    @staticmethod
    def validate_non_overlapping_intervals(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Confirm that rating intervals do not overlap.
        """
        if len(dataframe) <= 1:
            return

        previous_to = (
            dataframe["To"]
            .shift(1)
        )

        overlapping = (
            dataframe["From"]
            <= previous_to
        )

        if overlapping.iloc[1:].any():
            bad_rows = dataframe.loc[
                overlapping.fillna(False),
                [
                    "Club",
                    "From",
                    "To",
                    "Elo",
                ],
            ]

            raise ValueError(
                "ClubElo history contains "
                "overlapping intervals:\n"
                f"{bad_rows.to_string(index=False)}"
            )

    def save_history(
        self,
        club_name: str,
        dataframe: pd.DataFrame,
    ) -> Path:
        path = self.cache_path(
            club_name
        )

        dataframe.to_csv(
            path,
            index=False,
        )

        return path

    def load_cached_history(
        self,
        club_name: str,
    ) -> pd.DataFrame:
        path = self.cache_path(
            club_name
        )

        if not path.exists():
            raise FileNotFoundError(
                "Cached ClubElo history does "
                f"not exist: {path}"
            )

        dataframe = pd.read_csv(
            path,
            low_memory=False,
        )

        return self.normalize_history(
            dataframe=dataframe,
            requested_club=club_name,
        )

    def get_history(
        self,
        club_name: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return one club history, using memory and disk caches.
        """
        lookup_key = (
            self.normalize_lookup_key(
                club_name
            )
        )

        if (
            not refresh
            and lookup_key
            in self._memory_cache
        ):
            return self._memory_cache[
                lookup_key
            ].copy()

        path = self.cache_path(
            club_name
        )

        if path.exists() and not refresh:
            dataframe = (
                self.load_cached_history(
                    club_name
                )
            )
        else:
            dataframe = (
                self.download_history(
                    club_name
                )
            )

            self.save_history(
                club_name=club_name,
                dataframe=dataframe,
            )

        self._memory_cache[
            lookup_key
        ] = dataframe.copy()

        return dataframe.copy()

    def resolve_rating(
        self,
        club_name: str,
        prediction_date: (
            str
            | date
            | datetime
            | pd.Timestamp
        ),
        refresh: bool = False,
    ) -> ClubEloRatingResult:
        """
        Return the ClubElo rating valid on the requested date.
        """
        resolved_date = (
            self.parse_prediction_date(
                prediction_date
            )
        )

        dataframe = self.get_history(
            club_name=club_name,
            refresh=refresh,
        )

        timestamp = pd.Timestamp(
            resolved_date
        )

        matches = dataframe[
            dataframe["From"].le(timestamp)
            & dataframe["To"].ge(timestamp)
        ]

        if matches.empty:
            history_start = (
                dataframe["From"]
                .min()
                .date()
            )

            history_end = (
                dataframe["To"]
                .max()
                .date()
            )

            raise LookupError(
                "No ClubElo interval found for "
                f"{club_name!r} on "
                f"{resolved_date}. "
                f"Available history: "
                f"{history_start} through "
                f"{history_end}."
            )

        if len(matches) != 1:
            raise LookupError(
                "ClubElo interval resolution "
                "returned multiple matches for "
                f"{club_name!r} on "
                f"{resolved_date}."
            )

        row = matches.iloc[0]

        effective_from = (
            row["From"].date()
        )

        effective_to = (
            row["To"].date()
        )

        temporal_validity_pass = (
            effective_from
            <= resolved_date
            <= effective_to
        )

        if not temporal_validity_pass:
            raise AssertionError(
                "Resolved ClubElo interval failed "
                "temporal-validity validation."
            )

        rank_value = row["Rank"]

        rank = (
            None
            if pd.isna(rank_value)
            else float(rank_value)
        )

        return ClubEloRatingResult(
            requested_club=club_name,
            resolved_club=str(
                row["Club"]
            ),
            rating=float(
                row["Elo"]
            ),
            rank=rank,
            country=str(
                row["Country"]
            ),
            level=int(
                row["Level"]
            ),
            prediction_date=(
                resolved_date
            ),
            effective_from=(
                effective_from
            ),
            effective_to=(
                effective_to
            ),
            source="clubelo",
            temporal_validity_pass=True,
        )

    def preload_histories(
        self,
        club_names: Iterable[str],
        refresh: bool = False,
    ) -> dict[str, Path]:
        """
        Download or load several club histories.
        """
        output: dict[str, Path] = {}

        for club_name in club_names:
            dataframe = self.get_history(
                club_name=club_name,
                refresh=refresh,
            )

            path = self.save_history(
                club_name=club_name,
                dataframe=dataframe,
            )

            output[club_name] = path

        return output
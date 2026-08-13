#production_club_repository_builder

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
)
from research.production.production_repository_config import (
    ProductionRepositoryConfig,
)
from research.production.production_repository_schema import (
    ProductionClubRecord,
)


RepresentationProvider = Callable[
    [str],
    TeamRepresentation,
]


PRODUCTION_REPOSITORY_COLUMNS = (
    "club",
    "attack",
    "midfield",
    "defense",
    "goalkeeper",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
    "squad_quality",
    "evidence_score",
    "representation_type",
    "aggregation_profile",
    "player_count",
    "available_player_count",
    "repository_version",
    "repository_scope",
    "representation_season_id",
)


NUMERIC_REPRESENTATION_FIELDS = (
    "attack",
    "midfield",
    "defense",
    "goalkeeper",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
    "squad_quality",
    "evidence_score",
)


class ProductionClubRepositoryBuilder:
    """
    Build an immutable production repository from existing
    football-intelligence representations.

    Responsibilities
    ----------------
    - Consume club names and a representation provider.
    - Serialize TeamRepresentation objects.
    - Validate domain and persistence integrity.
    - Write a deterministic CSV artifact.

    Non-responsibilities
    --------------------
    - Build player repositories.
    - Calculate football intelligence.
    - Resolve ClubElo.
    - Build match observations.
    - Fit prediction models.
    - Simulate matches.
    """

    def __init__(
        self,
        config: ProductionRepositoryConfig,
    ) -> None:
        config.validate()
        self.config = config

    def build(
        self,
        *,
        clubs: Iterable[str],
        representation_provider: RepresentationProvider,
    ) -> pd.DataFrame:
        """
        Build, validate, and write the production repository.

        Parameters
        ----------
        clubs:
            Club names accepted by the representation provider.

        representation_provider:
            Callable that returns one TeamRepresentation for a
            supplied club name.

        Returns
        -------
        pandas.DataFrame
            The validated DataFrame written to disk.
        """

        normalized_clubs = self._prepare_club_names(
            clubs
        )

        records = self._collect_records(
            clubs=normalized_clubs,
            representation_provider=(
                representation_provider
            ),
        )

        self._validate_records(records)

        dataframe = self._create_dataframe(records)
        self._validate_dataframe(dataframe)

        self._write_repository(dataframe)

        return dataframe

    def _prepare_club_names(
        self,
        clubs: Iterable[str],
    ) -> tuple[str, ...]:
        prepared: list[str] = []

        for value in clubs:
            club = str(value).strip()

            if not club:
                raise ValueError(
                    "Club names must not be empty."
                )

            prepared.append(club)

        if not prepared:
            raise ValueError(
                "No clubs were supplied to the production "
                "repository builder."
            )

        duplicates = self._find_duplicates(prepared)

        if duplicates:
            raise ValueError(
                "Duplicate clubs were supplied: "
                f"{duplicates}"
            )

        prepared.sort(
            key=lambda name: name.casefold()
        )

        if (
            self.config.expected_club_count is not None
            and len(prepared)
            != self.config.expected_club_count
        ):
            raise ValueError(
                "Unexpected club count. "
                f"Expected "
                f"{self.config.expected_club_count}, "
                f"received {len(prepared)}."
            )

        return tuple(prepared)

    def _collect_records(
        self,
        *,
        clubs: tuple[str, ...],
        representation_provider: (
            RepresentationProvider
        ),
    ) -> tuple[ProductionClubRecord, ...]:
        records: list[ProductionClubRecord] = []

        for club in clubs:
            representation = (
                representation_provider(club)
            )

            if not isinstance(
                representation,
                TeamRepresentation,
            ):
                raise TypeError(
                    "Representation provider returned an "
                    "unexpected object for "
                    f"{club!r}: "
                    f"{type(representation).__name__}"
                )

            if (
                representation.representation_type
                != self.config.representation_type
            ):
                raise ValueError(
                    "Representation type does not match the "
                    "configured repository contract for "
                    f"{club!r}. Expected "
                    f"{self.config.representation_type!r}, "
                    f"received "
                    f"{representation.representation_type!r}."
                )

            record = (
                ProductionClubRecord
                .from_team_representation(
                    club=club,
                    representation=representation,
                    repository_version=(
                        self.config.repository_version
                    ),
                    repository_scope=(
                        self.config.repository_scope
                    ),
                    representation_season_id=(
                        self.config.season_id
                    ),
                )
            )

            records.append(record)

        return tuple(records)

    def _validate_records(
        self,
        records: tuple[
            ProductionClubRecord,
            ...,
        ],
    ) -> None:
        clubs = [
            record.club
            for record in records
        ]

        duplicates = self._find_duplicates(clubs)

        if duplicates:
            raise ValueError(
                "Duplicate clubs were produced: "
                f"{duplicates}"
            )

        for record in records:
            if not record.club.strip():
                raise ValueError(
                    "Production record contains an empty "
                    "club name."
                )

            for field_name in (
                NUMERIC_REPRESENTATION_FIELDS
            ):
                value = getattr(
                    record,
                    field_name,
                )

                if not math.isfinite(value):
                    raise ValueError(
                        "Production record contains a "
                        "non-finite value. "
                        f"Club={record.club!r}, "
                        f"field={field_name!r}, "
                        f"value={value!r}."
                    )

            if not 0.0 <= record.evidence_score <= 1.0:
                raise ValueError(
                    "evidence_score must be between 0 and 1. "
                    f"Club={record.club!r}, "
                    f"value={record.evidence_score!r}."
                )

            if record.player_count < 0:
                raise ValueError(
                    "player_count must not be negative. "
                    f"Club={record.club!r}."
                )

            if record.available_player_count < 0:
                raise ValueError(
                    "available_player_count must not be "
                    "negative. "
                    f"Club={record.club!r}."
                )

            if (
                record.available_player_count
                > record.player_count
            ):
                raise ValueError(
                    "available_player_count cannot exceed "
                    "player_count. "
                    f"Club={record.club!r}."
                )

            if not record.representation_type.strip():
                raise ValueError(
                    "representation_type must not be empty. "
                    f"Club={record.club!r}."
                )

            if not record.aggregation_profile.strip():
                raise ValueError(
                    "aggregation_profile must not be empty. "
                    f"Club={record.club!r}."
                )

    def _create_dataframe(
        self,
        records: tuple[
            ProductionClubRecord,
            ...,
        ],
    ) -> pd.DataFrame:
        dataframe = pd.DataFrame(
            [
                record.to_dict()
                for record in records
            ],
            columns=PRODUCTION_REPOSITORY_COLUMNS,
        )

        return dataframe.sort_values(
            by="club",
            key=lambda series: (
                series.astype(str).str.casefold()
            ),
            kind="stable",
        ).reset_index(drop=True)

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        expected_columns = list(
            PRODUCTION_REPOSITORY_COLUMNS
        )

        if list(dataframe.columns) != expected_columns:
            raise AssertionError(
                "Production repository columns do not match "
                "the declared schema. "
                f"Expected: {expected_columns}. "
                f"Actual: {list(dataframe.columns)}."
            )

        if dataframe.empty:
            raise ValueError(
                "Production repository DataFrame is empty."
            )

        if dataframe["club"].duplicated().any():
            duplicate_clubs = (
                dataframe.loc[
                    dataframe["club"].duplicated(
                        keep=False
                    ),
                    "club",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                "Production repository contains duplicate "
                f"clubs: {duplicate_clubs}"
            )

        required_columns = [
            "club",
            *NUMERIC_REPRESENTATION_FIELDS,
            "representation_type",
            "aggregation_profile",
            "player_count",
            "available_player_count",
            "repository_version",
            "repository_scope",
            "representation_season_id",
        ]

        if dataframe[
            required_columns
        ].isna().any().any():
            missing_counts = (
                dataframe[
                    required_columns
                ]
                .isna()
                .sum()
            )

            missing_counts = (
                missing_counts[
                    missing_counts > 0
                ]
                .to_dict()
            )

            raise ValueError(
                "Production repository contains missing "
                f"required values: {missing_counts}"
            )

        numeric_values = dataframe[
            list(
                NUMERIC_REPRESENTATION_FIELDS
            )
        ].to_numpy(dtype=float)

        if not pd.notna(
            numeric_values
        ).all():
            raise ValueError(
                "Production repository contains invalid "
                "numeric values."
            )

        if (
            self.config.expected_club_count is not None
            and len(dataframe)
            != self.config.expected_club_count
        ):
            raise ValueError(
                "Written repository would contain an "
                "unexpected number of clubs. "
                f"Expected "
                f"{self.config.expected_club_count}, "
                f"received {len(dataframe)}."
            )

    def _write_repository(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        output_path = Path(
            self.config.output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path: Path | None = None

        try:
            with NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                prefix=(
                    f".{output_path.stem}_"
                ),
                dir=output_path.parent,
                delete=False,
                encoding="utf-8",
                newline="",
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )

                dataframe.to_csv(
                    temporary_file,
                    index=False,
                )

            temporary_path.replace(
                output_path
            )

        except Exception:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

            raise

    @staticmethod
    def _find_duplicates(
        values: Iterable[str],
    ) -> list[str]:
        seen: set[str] = set()
        duplicates: set[str] = set()

        for value in values:
            key = value.casefold()

            if key in seen:
                duplicates.add(value)
            else:
                seen.add(key)

        return sorted(
            duplicates,
            key=str.casefold,
        )
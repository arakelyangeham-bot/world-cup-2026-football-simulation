#team_feature_provider.py

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from shared.team_name_normalizer import (
    normalize_team_name,
)


DateLike = (
    str
    | date
    | datetime
    | pd.Timestamp
)


@runtime_checkable
class TeamFeatureProvider(Protocol):
    """
    Interface implemented by static or historically varying
    team-feature providers.
    """

    @property
    def provider_name(self) -> str:
        """
        Return the stable provenance label for this provider.
        """
        ...

    def get_team_features(
        self,
        request: TeamFeatureRequest,
    ) -> TeamFeatureResult:
        """
        Return the team-feature snapshot valid for the request.

        Implementations must either return a validated
        TeamFeatureResult or raise an explicit error.
        """
        ...



@dataclass(frozen=True)
class TeamFeatureRequest:
    """
    Request for the team representation available at a
    particular prediction date.

    The first provider implementation is static and therefore
    ignores the date when selecting values. The date is still
    mandatory so every caller is already compatible with later
    historically varying providers.
    """

    team_name: str
    prediction_date: DateLike


@dataclass(frozen=True)
class TeamFeatureResult:
    """
    Team-strength snapshot returned by a feature provider.

    The canonical values deliberately match the current
    repository and historical-training contracts.
    """

    requested_team_name: str
    canonical_team_name: str

    prediction_date: date
    source_date: date | None

    attack: float
    midfield: float
    defense: float
    goalkeeper: float

    poisson_attack: float
    poisson_defense: float

    provider_name: str
    representation_type: str
    aggregation_profile: str

    feature_available: bool
    temporal_validity_pass: bool

    repository_path: str | None = None
    repository_version: str | None = None
    repository_scope: str | None = None

    def validate(self) -> None:
        if not self.requested_team_name.strip():
            raise ValueError(
                "requested_team_name must not be empty."
            )

        if not self.canonical_team_name.strip():
            raise ValueError(
                "canonical_team_name must not be empty."
            )

        if not self.provider_name.strip():
            raise ValueError(
                "provider_name must not be empty."
            )

        if not self.representation_type.strip():
            raise ValueError(
                "representation_type must not be empty."
            )

        if not self.aggregation_profile.strip():
            raise ValueError(
                "aggregation_profile must not be empty."
            )

        numeric_values = {
            "attack": self.attack,
            "midfield": self.midfield,
            "defense": self.defense,
            "goalkeeper": self.goalkeeper,
            "poisson_attack": self.poisson_attack,
            "poisson_defense": self.poisson_defense,
        }

        for field_name, value in numeric_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(
                    "Team feature result contains a "
                    "non-finite value. "
                    f"field={field_name!r}, value={value!r}."
                )

        if not self.feature_available:
            raise ValueError(
                "A TeamFeatureResult must represent an "
                "available feature snapshot."
            )

        if not self.temporal_validity_pass:
            raise ValueError(
                "A TeamFeatureResult must pass temporal "
                "validation."
            )

        if (
            self.source_date is not None
            and self.source_date > self.prediction_date
        ):
            raise ValueError(
                "source_date cannot occur after "
                "prediction_date."
            )

    def to_repository_entry(self) -> dict[str, float]:
        """
        Return the canonical runtime repository schema.

        This keeps downstream match-feature and simulation
        interfaces independent of the provider implementation.
        """

        self.validate()

        return {
            "attack": self.attack,
            "midfield": self.midfield,
            "defense": self.defense,
            "gk": self.goalkeeper,
            "poisson_attack": self.poisson_attack,
            "poisson_defense": self.poisson_defense,

            # Compatibility aliases used by existing scripts.
            "att_composite": self.attack,
            "mid_composite": self.midfield,
            "def_composite": self.defense,
            "gk_composite": self.goalkeeper,
            "poisson_attack_adj": self.poisson_attack,
            "poisson_defense_adj": self.poisson_defense,
        }


def normalize_prediction_date(
    value: DateLike,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )
    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return timestamp


class StaticCsvTeamFeatureProvider:
    """
    Compatibility provider backed by one static repository CSV.

    This provider intentionally reproduces the current historical
    training behavior: every requested match date receives values
    from the same repository.

    It is a migration adapter, not a historically faithful provider.
    """

    REQUIRED_COLUMNS = {
        "nation",
        "att_composite",
        "mid_composite",
        "def_composite",
        "gk_composite",
        "poisson_attack_adj",
        "poisson_defense_adj",
    }

    def __init__(
        self,
        repository_path: Path,
        *,
        provider_name: str = (
            "static_csv_team_features_v1"
        ),
        representation_type: str = (
            "static_repository"
        ),
        aggregation_profile: str = (
            "legacy_static"
        ),
        repository_version: str | None = None,
        repository_scope: str | None = None,
        source_date: DateLike | None = None,
    ) -> None:
        self.repository_path = Path(
            repository_path
        )

        self._provider_name = (
            str(provider_name).strip()
        )
        self.representation_type = (
            str(representation_type).strip()
        )
        self.aggregation_profile = (
            str(aggregation_profile).strip()
        )
        self.repository_version = (
            repository_version
        )
        self.repository_scope = (
            repository_scope
        )

        if not self._provider_name:
            raise ValueError(
                "provider_name must not be empty."
            )

        if not self.representation_type:
            raise ValueError(
                "representation_type must not be empty."
            )

        if not self.aggregation_profile:
            raise ValueError(
                "aggregation_profile must not be empty."
            )

        self.source_date = (
            normalize_prediction_date(
                source_date
            ).date()
            if source_date is not None
            else None
        )

        self._repository = (
            self._load_repository()
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def _load_repository(
        self,
    ) -> dict[str, dict[str, float]]:
        if not self.repository_path.exists():
            raise FileNotFoundError(
                "Static team repository does not exist: "
                f"{self.repository_path}"
            )

        dataframe = pd.read_csv(
            self.repository_path
        )

        if dataframe.empty:
            raise ValueError(
                "Static team repository is empty."
            )

        missing_columns = (
            self.REQUIRED_COLUMNS
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "Static team repository is missing "
                f"columns: {sorted(missing_columns)}"
            )

        dataframe = dataframe.copy()

        dataframe["canonical_team_name"] = (
            dataframe["nation"]
            .astype(str)
            .map(normalize_team_name)
        )

        if dataframe[
            "canonical_team_name"
        ].duplicated().any():
            duplicates = sorted(
                dataframe.loc[
                    dataframe[
                        "canonical_team_name"
                    ].duplicated(
                        keep=False
                    ),
                    "canonical_team_name",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                "Static team repository contains "
                "duplicate normalized teams: "
                f"{duplicates}"
            )

        numeric_columns = (
            "att_composite",
            "mid_composite",
            "def_composite",
            "gk_composite",
            "poisson_attack_adj",
            "poisson_defense_adj",
        )

        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise",
            )

            if dataframe[column].isna().any():
                raise ValueError(
                    "Static team repository contains "
                    f"missing values in {column!r}."
                )

            if not dataframe[column].map(
                lambda value: math.isfinite(
                    float(value)
                )
            ).all():
                raise ValueError(
                    "Static team repository contains "
                    f"non-finite values in {column!r}."
                )

        repository: dict[
            str,
            dict[str, float],
        ] = {}

        for row in dataframe.itertuples(
            index=False
        ):
            repository[
                row.canonical_team_name
            ] = {
                "attack":
                    float(row.att_composite),
                "midfield":
                    float(row.mid_composite),
                "defense":
                    float(row.def_composite),
                "goalkeeper":
                    float(row.gk_composite),
                "poisson_attack":
                    float(
                        row.poisson_attack_adj
                    ),
                "poisson_defense":
                    float(
                        row.poisson_defense_adj
                    ),
            }

        return repository

    def get_team_features(
        self,
        request: TeamFeatureRequest,
    ) -> TeamFeatureResult:
        requested_name = str(
            request.team_name
        ).strip()

        if not requested_name:
            raise ValueError(
                "Requested team name must not be empty."
            )

        prediction_timestamp = (
            normalize_prediction_date(
                request.prediction_date
            )
        )

        canonical_name = normalize_team_name(
            requested_name
        )

        try:
            values = self._repository[
                canonical_name
            ]
        except KeyError as error:
            raise KeyError(
                "No static team features exist for "
                f"{canonical_name!r}."
            ) from error

        if (
            self.source_date is not None
            and self.source_date
            > prediction_timestamp.date()
        ):
            raise ValueError(
                "Static repository source date occurs "
                "after the requested prediction date. "
                f"Team={canonical_name!r}, "
                f"source_date={self.source_date}, "
                "prediction_date="
                f"{prediction_timestamp.date()}."
            )

        result = TeamFeatureResult(
            requested_team_name=(
                requested_name
            ),
            canonical_team_name=(
                canonical_name
            ),
            prediction_date=(
                prediction_timestamp.date()
            ),
            source_date=self.source_date,
            attack=values["attack"],
            midfield=values["midfield"],
            defense=values["defense"],
            goalkeeper=values[
                "goalkeeper"
            ],
            poisson_attack=values[
                "poisson_attack"
            ],
            poisson_defense=values[
                "poisson_defense"
            ],
            provider_name=(
                self.provider_name
            ),
            representation_type=(
                self.representation_type
            ),
            aggregation_profile=(
                self.aggregation_profile
            ),
            feature_available=True,
            temporal_validity_pass=True,
            repository_path=str(
                self.repository_path
            ),
            repository_version=(
                self.repository_version
            ),
            repository_scope=(
                self.repository_scope
            ),
        )

        result.validate()

        return result
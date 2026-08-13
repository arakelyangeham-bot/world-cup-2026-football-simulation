#live_match_observation_builder

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from research.baselines.club_goal_model import (
    CURRENT_CLUB_GOAL_MODEL,
)
from shared.team_name_normalizer import (
    normalize_team_name,
)

# Adjust only this import if clubelo_repository.py lives in a
# different package in your project.
from research.rating_priors.clubelo_repository import (
    ClubEloRatingResult,
    ClubEloRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLUB_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_071a_premier_league_club_repository_v1"
    / "premier_league_club_repository_v1.csv"
)


REQUIRED_REPOSITORY_COLUMNS = {
    "club",
    "attack",
    "defense",
    "attack_depth",
}

DEFAULT_CLUBELO_NAME_OVERRIDES = {
    "Aston Villa": "AstonVilla",
    "Brighton & Hove Albion": "Brighton",
    "Crystal Palace": "CrystalPalace",
    "Liverpool FC": "Liverpool",
    "Manchester City": "ManCity",
    "Manchester United": "ManUnited",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "WestHam",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
}

@dataclass(frozen=True)
class ProductionClubRepresentation:
    """
    Runtime football-intelligence representation for one club.
    """

    club: str
    attack: float
    defense: float
    attack_depth: float

    midfield: float | None = None
    goalkeeper: float | None = None
    midfield_depth: float | None = None
    defense_depth: float | None = None
    squad_quality: float | None = None
    evidence_score: float | None = None

    representation_type: str | None = None
    representation_source: str | None = None
    representation_season_id: str | None = None
    repository_version: str | None = None
    repository_scope: str | None = None


@dataclass(frozen=True)
class LiveMatchObservation:
    """
    Complete production observation for one future club match.
    """

    requested_home_team: str
    requested_away_team: str

    home_team: str
    away_team: str
    prediction_date: date

    home_attack: float
    away_attack: float

    home_defense: float
    away_defense: float

    home_attack_depth: float
    away_attack_depth: float
    attack_depth_diff: float

    home_rating_prior: float
    away_rating_prior: float
    rating_prior_diff: float

    home_rating_effective_from: date
    home_rating_effective_to: date

    away_rating_effective_from: date
    away_rating_effective_to: date

    rating_prior_source: str
    repository_version: str | None
    repository_scope: str | None

    def to_feature_mapping(
        self,
    ) -> dict[str, float]:
        """
        Return exactly the feature mapping required by the
        current production club goal model.
        """

        feature_values = {
            "home_attack": self.home_attack,
            "away_attack": self.away_attack,
            "home_defense": self.home_defense,
            "away_defense": self.away_defense,
            "attack_depth_diff":
                self.attack_depth_diff,
            "rating_prior_diff":
                self.rating_prior_diff,
        }

        specification = (
            CURRENT_CLUB_GOAL_MODEL
            .get_feature_specification()
        )

        expected_features = set(
            specification.required_columns()
        )

        actual_features = set(
            feature_values
        )

        if actual_features != expected_features:
            raise AssertionError(
                "Live observation feature mapping does not "
                "match the current club goal-model contract. "
                f"Expected: {sorted(expected_features)}. "
                f"Actual: {sorted(actual_features)}."
            )

        return feature_values


class ProductionClubRepository:
    """
    Read-only runtime repository for validated club
    representations.
    """

    def __init__(
        self,
        repository_path: Path = (
            DEFAULT_CLUB_REPOSITORY_PATH
        ),
    ) -> None:
        self.repository_path = Path(
            repository_path
        )

        self._representations = (
            self._load_repository()
        )

    def _load_repository(
        self,
    ) -> dict[str, ProductionClubRepresentation]:
        if not self.repository_path.exists():
            raise FileNotFoundError(
                "Production club repository does not exist: "
                f"{self.repository_path}"
            )

        dataframe = pd.read_csv(
            self.repository_path,
            low_memory=False,
        )

        if dataframe.empty:
            raise ValueError(
                "Production club repository is empty."
            )

        missing = (
            REQUIRED_REPOSITORY_COLUMNS
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                "Production club repository is missing "
                f"required columns: {sorted(missing)}"
            )

        dataframe = dataframe.copy()

        numeric_columns = [
            column
            for column in (
                "attack",
                "defense",
                "attack_depth",
                "midfield",
                "goalkeeper",
                "midfield_depth",
                "defense_depth",
                "squad_quality",
                "evidence_score",
            )
            if column in dataframe.columns
        ]

        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise",
            )

        if dataframe[
            [
                "attack",
                "defense",
                "attack_depth",
            ]
        ].isna().any().any():
            raise ValueError(
                "Production repository contains missing "
                "Version 1 values."
            )

        required_values = dataframe[
            [
                "attack",
                "defense",
                "attack_depth",
            ]
        ].to_numpy(dtype=float)

        if not np.isfinite(
            required_values
        ).all():
            raise ValueError(
                "Production repository contains non-finite "
                "Version 1 values."
            )

        repository: dict[
            str,
            ProductionClubRepresentation,
        ] = {}

        for _, row in dataframe.iterrows():
            raw_club = str(
                row["club"]
            ).strip()

            club = normalize_team_name(
                raw_club
            )

            if not club:
                raise ValueError(
                    "Club name became empty after "
                    "normalization."
                )

            if club in repository:
                raise ValueError(
                    "Duplicate normalized club in "
                    f"production repository: {club}"
                )

            repository[club] = (
                ProductionClubRepresentation(
                    club=club,
                    attack=float(
                        row["attack"]
                    ),
                    defense=float(
                        row["defense"]
                    ),
                    attack_depth=float(
                        row["attack_depth"]
                    ),
                    midfield=_optional_float(
                        row,
                        "midfield",
                    ),
                    goalkeeper=_optional_float(
                        row,
                        "goalkeeper",
                    ),
                    midfield_depth=_optional_float(
                        row,
                        "midfield_depth",
                    ),
                    defense_depth=_optional_float(
                        row,
                        "defense_depth",
                    ),
                    squad_quality=_optional_float(
                        row,
                        "squad_quality",
                    ),
                    evidence_score=_optional_float(
                        row,
                        "evidence_score",
                    ),
                    representation_type=(
                        _optional_string(
                            row,
                            "representation_type",
                        )
                    ),
                    representation_source=(
                        _optional_string(
                            row,
                            "representation_source",
                        )
                    ),
                    representation_season_id=(
                        _optional_string(
                            row,
                            "representation_season_id",
                        )
                    ),
                    repository_version=(
                        _optional_string(
                            row,
                            "repository_version",
                        )
                    ),
                    repository_scope=(
                        _optional_string(
                            row,
                            "repository_scope",
                        )
                    ),
                )
            )

        return repository

    def list_clubs(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._representations
            )
        )

    def resolve_club(
        self,
        club_name: str,
    ) -> ProductionClubRepresentation:
        normalized = normalize_team_name(
            club_name
        )

        try:
            return self._representations[
                normalized
            ]
        except KeyError as error:
            available = ", ".join(
                self.list_clubs()
            )

            raise KeyError(
                "Club is not present in the production "
                f"repository: {club_name!r}. "
                f"Normalized value: {normalized!r}. "
                f"Available clubs: {available}"
            ) from error


class LiveMatchObservationBuilder:
    """
    Assemble the exact live feature mapping required by
    Integrated Club Goal Model v1.

    This builder performs no model fitting and no football
    intelligence calculations beyond home-minus-away
    differences.
    """

    def __init__(
        self,
        club_repository: ProductionClubRepository,
        clubelo_repository: ClubEloRepository,
        clubelo_name_overrides: (
            Mapping[str, str] | None
        ) = None,
    ) -> None:
        self.club_repository = (
            club_repository
        )

        self.clubelo_repository = (
            clubelo_repository
        )

        self.clubelo_name_overrides = {
            normalize_team_name(key): value
            for key, value in (
                clubelo_name_overrides
                or {}
            ).items()
        }

        self._validate_model_contract()

    def _validate_model_contract(
        self,
    ) -> None:
        CURRENT_CLUB_GOAL_MODEL.validate()

        specification = (
            CURRENT_CLUB_GOAL_MODEL
            .get_feature_specification()
        )

        expected_features = {
            "home_attack",
            "away_attack",
            "home_defense",
            "away_defense",
            "attack_depth_diff",
            "rating_prior_diff",
        }

        registered_features = set(
            specification.required_columns()
        )

        if (
            registered_features
            != expected_features
        ):
            raise RuntimeError(
                "LiveMatchObservationBuilder supports "
                "Integrated Club Goal Model v1 only. "
                "The current baseline feature contract has "
                "changed. "
                f"Registered features: "
                f"{sorted(registered_features)}"
            )

    @staticmethod
    def parse_prediction_date(
        value: str | date | datetime,
    ) -> date:
        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        return date.fromisoformat(
            value
        )

    def _clubelo_lookup_name(
        self,
        canonical_club: str,
    ) -> str:
        return self.clubelo_name_overrides.get(
            canonical_club,
            canonical_club,
        )

    def _resolve_rating(
        self,
        representation: ProductionClubRepresentation,
        prediction_date: date,
    ) -> ClubEloRatingResult:
        clubelo_name = (
            self._clubelo_lookup_name(
                representation.club
            )
        )

        result = (
            self.clubelo_repository
            .resolve_rating(
                club_name=clubelo_name,
                prediction_date=prediction_date,
            )
        )

        if not result.temporal_validity_pass:
            raise AssertionError(
                "Resolved ClubElo rating failed temporal "
                "validation."
            )

        return result

    def build(
        self,
        home_team: str,
        away_team: str,
        prediction_date: (
            str | date | datetime
        ),
    ) -> LiveMatchObservation:
        parsed_date = (
            self.parse_prediction_date(
                prediction_date
            )
        )

        home = (
            self.club_repository
            .resolve_club(
                home_team
            )
        )

        away = (
            self.club_repository
            .resolve_club(
                away_team
            )
        )

        if home.club == away.club:
            raise ValueError(
                "Home and away clubs must differ."
            )

        home_rating = (
            self._resolve_rating(
                representation=home,
                prediction_date=parsed_date,
            )
        )

        away_rating = (
            self._resolve_rating(
                representation=away,
                prediction_date=parsed_date,
            )
        )

        observation = LiveMatchObservation(
            requested_home_team=home_team,
            requested_away_team=away_team,
            home_team=home.club,
            away_team=away.club,
            prediction_date=parsed_date,
            home_attack=home.attack,
            away_attack=away.attack,
            home_defense=home.defense,
            away_defense=away.defense,
            home_attack_depth=(
                home.attack_depth
            ),
            away_attack_depth=(
                away.attack_depth
            ),
            attack_depth_diff=(
                home.attack_depth
                - away.attack_depth
            ),
            home_rating_prior=(
                home_rating.rating
            ),
            away_rating_prior=(
                away_rating.rating
            ),
            rating_prior_diff=(
                home_rating.rating
                - away_rating.rating
            ),
            home_rating_effective_from=(
                home_rating.effective_from
            ),
            home_rating_effective_to=(
                home_rating.effective_to
            ),
            away_rating_effective_from=(
                away_rating.effective_from
            ),
            away_rating_effective_to=(
                away_rating.effective_to
            ),
            rating_prior_source="clubelo",
            repository_version=(
                home.repository_version
            ),
            repository_scope=(
                home.repository_scope
            ),
        )

        # Force contract validation before returning.
        observation.to_feature_mapping()

        return observation


def _optional_float(
    row: pd.Series,
    column: str,
) -> float | None:
    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    return float(value)


def _optional_string(
    row: pd.Series,
    column: str,
) -> str | None:
    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    text = str(value).strip()

    return text or None
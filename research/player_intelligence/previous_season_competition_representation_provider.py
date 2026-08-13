#previous_season_competition_representation_provider

from __future__ import annotations

import pandas as pd

from research.football_observatory.representation_provider import (
    RepresentationRequest,
    RepresentationResult,
)
from research.player_intelligence.competition_team_repository import (
    CompetitionTeamRepository,
)


class PreviousSeasonCompetitionRepresentationProvider:
    """
    Supplies team representations from the season immediately preceding
    the prediction season.

    Version 1 uses CompetitionTeamRepository and Sofascore team IDs.

    This provider deliberately rejects:

    - same-season representations;
    - future-season representations;
    - missing competition-season mappings;
    - teams absent from the representation season.
    """

    def __init__(
        self,
        team_repository: CompetitionTeamRepository | None = None,
        *,
        representation_type: str = "expected_starting_xi",
        formation: str = "4-3-3",
    ) -> None:
        if representation_type not in {
            "full_squad",
            "expected_starting_xi",
        }:
            raise ValueError(
                "Unknown representation type: "
                f"{representation_type!r}."
            )

        self.team_repository = (
            team_repository
            or CompetitionTeamRepository()
        )

        self.representation_type = representation_type
        self.formation = formation

        self._season_lookup: dict[
            tuple[str, int],
            tuple[int, int],
        ] | None = None

    @property
    def provider_name(self) -> str:
        return (
            "previous_season_competition_"
            f"{self.representation_type}"
        )

    @staticmethod
    def _parse_season_start_year(
        season_year: object,
    ) -> int:
        """
        Convert labels such as '23/24' or '2023/24' to 2023.
        """

        text = str(season_year).strip()

        if "/" not in text:
            return int(text)

        first_part = text.split("/", maxsplit=1)[0]

        if len(first_part) == 2:
            return 2000 + int(first_part)

        return int(first_part)

    def _build_season_lookup(
        self,
    ) -> dict[tuple[str, int], tuple[int, int]]:
        seasons = (
            self.team_repository
            .roster_builder
            .list_competition_seasons()
        )

        required_columns = {
            "competition",
            "competition_id",
            "season_id",
            "season_year",
        }

        missing = required_columns - set(seasons.columns)

        if missing:
            raise ValueError(
                "Competition-season table is missing columns: "
                f"{sorted(missing)}"
            )

        lookup: dict[
            tuple[str, int],
            tuple[int, int],
        ] = {}

        for row in seasons.itertuples(index=False):
            competition_key = (
                str(row.competition)
                .strip()
                .lower()
                .replace(" ", "_")
            )

            season_start_year = (
                self._parse_season_start_year(
                    row.season_year
                )
            )

            key = (
                competition_key,
                season_start_year,
            )

            value = (
                int(row.competition_id),
                int(row.season_id),
            )

            existing = lookup.get(key)

            if existing is not None and existing != value:
                raise ValueError(
                    "Competition-season lookup is ambiguous "
                    f"for {key}: {existing} vs {value}."
                )

            lookup[key] = value

        return lookup

    def _get_season_lookup(
        self,
    ) -> dict[tuple[str, int], tuple[int, int]]:
        if self._season_lookup is None:
            self._season_lookup = (
                self._build_season_lookup()
            )

        return self._season_lookup

    def get_representation(
        self,
        request: RepresentationRequest,
    ) -> RepresentationResult:
        expected_source_year = (
            request.prediction_season_start_year - 1
        )

        if (
            request.representation_season_start_year
            != expected_source_year
        ):
            raise ValueError(
                "Previous-season provider requires "
                "representation_season_start_year to equal "
                "prediction_season_start_year - 1. "
                f"Prediction={request.prediction_season_start_year}, "
                "representation="
                f"{request.representation_season_start_year}."
            )

        normalized_competition_key = (
            request.competition_key
            .strip()
            .lower()
        )

        lookup_key = (
            normalized_competition_key,
            request.representation_season_start_year,
        )

        season_lookup = self._get_season_lookup()

        try:
            competition_id, season_id = (
                season_lookup[lookup_key]
            )
        except KeyError as exc:
            available = sorted(season_lookup)

            raise KeyError(
                "No representation season exists for "
                f"{lookup_key}. Available keys include: "
                f"{available[:20]}"
            ) from exc

        teams = (
            self.team_repository
            .roster_builder
            .list_teams(
                competition_id=competition_id,
                season_id=season_id,
            )
        )

        matching_team = teams[
            pd.to_numeric(
                teams["team_id"],
                errors="coerce",
            ).eq(request.team_id)
        ]

        if matching_team.empty:
            raise KeyError(
                "Team was not present in the representation "
                "season. This commonly occurs for promoted clubs. "
                f"team_id={request.team_id}, "
                f"competition={request.competition_key}, "
                "representation season="
                f"{request.representation_season_start_year}."
            )

        representation = (
            self.team_repository
            .get_team_representation(
                competition_id=competition_id,
                season_id=season_id,
                team_id=request.team_id,
                representation_type=(
                    self.representation_type
                ),
                formation=self.formation,
            )
        )

        return RepresentationResult(
            request=request,
            representation=representation,
            representation_type=(
                self.representation_type
            ),
            formation=(
                self.formation
                if self.representation_type
                == "expected_starting_xi"
                else None
            ),
            competition_id=competition_id,
            season_id=season_id,
            source=self.provider_name,
            temporal_validity_pass=True,
        )
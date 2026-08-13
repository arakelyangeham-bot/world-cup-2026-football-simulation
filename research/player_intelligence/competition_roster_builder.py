#competition_roster_builder

from __future__ import annotations

from dataclasses import dataclass

from research.player_intelligence.competition_player_repository import (
    CompetitionPlayerRepository,
)
from research.player_intelligence.player_schema import Squad


@dataclass(frozen=True)
class CompetitionSquadContext:
    competition_id: int
    season_id: int
    team_id: int

    competition: str
    competition_type: str
    season_year: str

    team: str


class CompetitionRosterBuilder:
    """
    Builds Squad objects from competition-season-team membership.

    Membership resolution is delegated to CompetitionPlayerRepository.

    This builder does not:
        - calculate player ratings;
        - infer transfers;
        - select a starting XI;
        - calculate team strength;
        - use profile current_team fields.
    """

    def __init__(
        self,
        repository: CompetitionPlayerRepository | None = None,
    ) -> None:
        self.repository = (
            repository
            or CompetitionPlayerRepository()
        )

    def list_competition_seasons(self):
        return self.repository.list_competition_seasons()

    def list_teams(
        self,
        competition_id: int,
        season_id: int,
    ):
        return self.repository.list_teams(
            competition_id=competition_id,
            season_id=season_id,
        )

    def get_context(
        self,
        competition_id: int,
        season_id: int,
        team_id: int,
    ) -> CompetitionSquadContext:
        membership_rows = (
            self.repository.get_membership_rows(
                competition_id=competition_id,
                season_id=season_id,
                team_id=team_id,
            )
        )

        identity_columns = [
            "competition",
            "competition_type",
            "season_year",
            "team",
        ]

        identity_rows = (
            membership_rows[identity_columns]
            .drop_duplicates()
        )

        if len(identity_rows) != 1:
            raise ValueError(
                "Competition squad identity is not unique for "
                f"competition_id={competition_id}, "
                f"season_id={season_id}, "
                f"team_id={team_id}. "
                f"Distinct identity rows: {len(identity_rows)}"
            )

        identity = identity_rows.iloc[0]

        return CompetitionSquadContext(
            competition_id=competition_id,
            season_id=season_id,
            team_id=team_id,
            competition=str(
                identity["competition"]
            ),
            competition_type=str(
                identity["competition_type"]
            ),
            season_year=str(
                identity["season_year"]
            ),
            team=str(
                identity["team"]
            ),
        )

    def get_squad(
        self,
        competition_id: int,
        season_id: int,
        team_id: int,
        *,
        require_complete_join: bool = True,
    ) -> Squad:
        context = self.get_context(
            competition_id=competition_id,
            season_id=season_id,
            team_id=team_id,
        )

        players = self.repository.get_players_for_team(
            competition_id=competition_id,
            season_id=season_id,
            team_id=team_id,
            require_complete_join=require_complete_join,
        )

        if not players:
            raise ValueError(
                "Cannot build an empty competition squad for "
                f"{context.team}, "
                f"{context.competition} "
                f"{context.season_year}."
            )

        return Squad(
            national_team=context.team,
            players=players,
        )
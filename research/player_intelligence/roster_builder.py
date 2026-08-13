#roster_builder.py

from __future__ import annotations

from research.player_intelligence.player_repository import PlayerRepository
from research.player_intelligence.player_schema import Player, Squad


class RosterBuilder:
    """
    Builds Squad objects from repository-backed player data.

    Version 1 treats the source file's team column as the roster key.
    This works naturally for club-style competition data.

    Later versions can add national-team roster logic separately.
    """

    def __init__(self, repository: PlayerRepository | None = None) -> None:
        self.repository = repository or PlayerRepository()

    def load_players(self) -> tuple[Player, ...]:
        return self.repository.load_players()

    def list_teams(self) -> list[str]:
        teams = {
            player.identity.national_team
            for player in self.load_players()
            if player.identity.national_team != "unknown"
        }

        return sorted(teams)

    def get_squad(self, team_name: str) -> Squad:
        players = tuple(
            player
            for player in self.load_players()
            if player.identity.national_team == team_name
        )

        return Squad(
            national_team=team_name,
            players=players,
        )
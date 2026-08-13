#competition_player_repository

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.player_intelligence.player_repository import (
    PlayerRepository,
)
from research.player_intelligence.player_schema import Player


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MEMBERSHIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_players.csv"
)


REQUIRED_MEMBERSHIP_COLUMNS = {
    "competition",
    "competition_type",
    "competition_id",
    "season_id",
    "season_year",
    "team_id",
    "team",
    "player_id",
    "player",
}


class CompetitionPlayerRepository:
    """
    Resolves Player objects within a competition-season-team context.

    Membership comes from the competition-season player ingestion table.
    Player intelligence comes from PlayerRepository.

    The profile field `current_team` is never used to determine historical
    competition-season membership.
    """

    def __init__(
        self,
        membership_path: Path = DEFAULT_MEMBERSHIP_PATH,
        player_repository: PlayerRepository | None = None,
    ) -> None:
        self.membership_path = membership_path
        self.player_repository = (
            player_repository
            or PlayerRepository()
        )

        self._memberships: pd.DataFrame | None = None
        self._players_by_id: dict[str, Player] | None = None

    def load_memberships(self) -> pd.DataFrame:
        if self._memberships is not None:
            return self._memberships

        if not self.membership_path.exists():
            raise FileNotFoundError(
                "Competition membership file does not exist: "
                f"{self.membership_path}"
            )

        memberships = pd.read_csv(
            self.membership_path,
            dtype={
                "season_year": str,
                "player_id": str,
            },
            low_memory=False,
        )

        if memberships.empty:
            raise ValueError(
                "Competition membership file is empty: "
                f"{self.membership_path}"
            )

        missing = (
            REQUIRED_MEMBERSHIP_COLUMNS
            - set(memberships.columns)
        )

        if missing:
            raise ValueError(
                "Competition membership file is missing "
                f"required columns: {sorted(missing)}"
            )

        duplicate_mask = memberships.duplicated(
            subset=[
                "competition_id",
                "season_id",
                "team_id",
                "player_id",
            ],
            keep=False,
        )

        if duplicate_mask.any():
            duplicate_count = int(
                duplicate_mask.sum()
            )

            raise ValueError(
                "Duplicate competition-season-team-player "
                f"memberships detected: {duplicate_count}"
            )

        self._memberships = memberships
        return memberships

    def load_players_by_id(self) -> dict[str, Player]:
        if self._players_by_id is not None:
            return self._players_by_id

        players = self.player_repository.load_players()

        players_by_id: dict[str, Player] = {}

        for player in players:
            player_id = str(
                player.identity.player_id
            )

            if player_id in players_by_id:
                raise ValueError(
                    "Duplicate Player object ID detected in "
                    f"PlayerRepository: {player_id}"
                )

            players_by_id[player_id] = player

        self._players_by_id = players_by_id
        return players_by_id

    def list_competition_seasons(
        self,
    ) -> pd.DataFrame:
        memberships = self.load_memberships()

        columns = [
            "competition",
            "competition_type",
            "competition_id",
            "season_id",
            "season_year",
        ]

        return (
            memberships[columns]
            .drop_duplicates()
            .sort_values(
                [
                    "competition",
                    "season_year",
                ]
            )
            .reset_index(drop=True)
        )

    def list_teams(
        self,
        competition_id: int,
        season_id: int,
    ) -> pd.DataFrame:
        memberships = self.load_memberships()

        selected = memberships[
            (
                memberships["competition_id"]
                == competition_id
            )
            & (
                memberships["season_id"]
                == season_id
            )
        ]

        if selected.empty:
            raise KeyError(
                "No membership records found for "
                f"competition_id={competition_id}, "
                f"season_id={season_id}."
            )

        return (
            selected[
                [
                    "team_id",
                    "team",
                ]
            ]
            .drop_duplicates()
            .sort_values("team")
            .reset_index(drop=True)
        )

    def get_membership_rows(
        self,
        competition_id: int,
        season_id: int,
        team_id: int,
    ) -> pd.DataFrame:
        memberships = self.load_memberships()

        selected = memberships[
            (
                memberships["competition_id"]
                == competition_id
            )
            & (
                memberships["season_id"]
                == season_id
            )
            & (
                memberships["team_id"]
                == team_id
            )
        ].copy()

        if selected.empty:
            raise KeyError(
                "No player memberships found for "
                f"competition_id={competition_id}, "
                f"season_id={season_id}, "
                f"team_id={team_id}."
            )

        return (
            selected
            .sort_values(
                [
                    "player",
                    "player_id",
                ]
            )
            .reset_index(drop=True)
        )

    def get_players_for_team(
        self,
        competition_id: int,
        season_id: int,
        team_id: int,
        *,
        require_complete_join: bool = True,
    ) -> tuple[Player, ...]:
        membership_rows = self.get_membership_rows(
            competition_id=competition_id,
            season_id=season_id,
            team_id=team_id,
        )

        players_by_id = self.load_players_by_id()

        players: list[Player] = []
        missing_player_ids: list[str] = []

        for player_id in (
            membership_rows["player_id"]
            .astype(str)
            .tolist()
        ):
            player = players_by_id.get(player_id)

            if player is None:
                missing_player_ids.append(player_id)
                continue

            players.append(player)

        if (
            require_complete_join
            and missing_player_ids
        ):
            preview = missing_player_ids[:20]

            raise ValueError(
                "Competition memberships could not be joined "
                "to PlayerRepository objects. "
                f"Missing player IDs: {preview}"
                + (
                    "..."
                    if len(missing_player_ids) > 20
                    else ""
                )
            )

        return tuple(players)
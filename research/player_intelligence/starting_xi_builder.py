#starting_xi_builder.py

from __future__ import annotations

import ast
from dataclasses import dataclass

import pandas as pd

from research.player_intelligence.player_schema import (
    LineupAssignment,
    Player,
    Squad,
    StartingXI,
)

@dataclass(frozen=True)
class LineupSlot:
    slot: str
    role: str


class StartingXIBuilder:
    """
    Builds expected starting lineups from player ratings and a formation.

    This class preserves the existing build_expected_lineups.py selection logic,
    but moves it behind a reusable Player Intelligence interface.
    """

    def __init__(self, formation: str = "4-3-3") -> None:
        self.formation = formation

    def build_for_team(
        self,
        team_df: pd.DataFrame,
        formation_df: pd.DataFrame,
    ) -> list[dict]:
        selected_player_ids = set()
        lineup_rows = []

        for _, slot in formation_df.iterrows():
            role = slot["role"]
            rating_col = f"rating_{role}"

            if rating_col not in team_df.columns:
                continue

            candidates = team_df[
                team_df["eligible_roles_list"].apply(
                    lambda roles: role in roles
                )
                & ~team_df["player_id"].isin(selected_player_ids)
                & team_df[rating_col].notna()
            ].copy()

            if candidates.empty:
                lineup_rows.append(
                    {
                        "slot": slot["slot"],
                        "role": role,
                        "player_id": pd.NA,
                        "player": pd.NA,
                        "rating": pd.NA,
                    }
                )
                continue

            chosen = candidates.sort_values(
                rating_col,
                ascending=False,
            ).iloc[0]

            selected_player_ids.add(chosen["player_id"])

            lineup_rows.append(
                {
                    "slot": slot["slot"],
                    "role": role,
                    "player_id": chosen["player_id"],
                    "player": chosen["player"],
                    "rating": chosen[rating_col],
                }
            )

        return lineup_rows

    def build_for_squad(
        self,
        squad: Squad,
        formation_df: pd.DataFrame,
    ) -> StartingXI:
        """
        Build a StartingXI from the shared object-model Squad.

        This method adapts Player objects into the legacy dataframe
        selection format, reuses build_for_team(), and then maps the
        selected player IDs back to the original Player objects.
        """

        if not squad.players:
            raise ValueError(
                f"Cannot build a starting XI for empty squad "
                f"{squad.national_team!r}."
            )

        self._validate_formation_dataframe(
            formation_df
        )

        team_df = self._squad_to_dataframe(
            squad
        )

        lineup_rows = self.build_for_team(
            team_df=team_df,
            formation_df=formation_df,
        )

        missing_slots = [
            row
            for row in lineup_rows
            if pd.isna(row["player_id"])
        ]

        if missing_slots:
            descriptions = [
                f"{row['slot']} ({row['role']})"
                for row in missing_slots
            ]

            raise ValueError(
                f"Could not fill every lineup slot for "
                f"{squad.national_team!r}. "
                f"Missing slots: {descriptions}"
            )

        players_by_id = {
            str(player.identity.player_id): player
            for player in squad.players
        }

        selected_players: list[Player] = []
        assignments: list[LineupAssignment] = []

        for row in lineup_rows:
            player_id = str(
                row["player_id"]
            )

            try:
                player = players_by_id[
                    player_id
                ]

            except KeyError as exc:
                raise KeyError(
                    "Selected lineup player was not found "
                    "in the source squad: "
                    f"{player_id}"
                ) from exc

            selection_rating = float(
                row["rating"]
            )

            selected_players.append(
                player
            )

            assignments.append(
                LineupAssignment(
                    slot=str(
                        row["slot"]
                    ),
                    tactical_role=str(
                        row["role"]
                    ),
                    player=player,
                    selection_rating=(
                        selection_rating
                    ),
                )
            )

        if len(selected_players) != len(formation_df):
            raise AssertionError(
                "Selected player count does not match "
                "formation slot count. "
                f"Players={len(selected_players)}, "
                f"slots={len(formation_df)}."
            )

        selected_ids = [
            str(player.identity.player_id)
            for player in selected_players
        ]

        if len(selected_ids) != len(set(selected_ids)):
            raise AssertionError(
                "Starting XI contains duplicate players."
            )

        return StartingXI(
            national_team=squad.national_team,
            formation=self.formation,
            players=tuple(
                selected_players
            ),
            assignments=tuple(
                assignments
            ),
        )

    @staticmethod
    def _validate_formation_dataframe(
        formation_df: pd.DataFrame,
    ) -> None:
        required_columns = {
            "slot",
            "role",
        }

        missing = (
            required_columns
            - set(formation_df.columns)
        )

        if missing:
            raise ValueError(
                "Formation dataframe is missing required "
                f"columns: {sorted(missing)}"
            )

        if formation_df.empty:
            raise ValueError(
                "Formation dataframe is empty."
            )

        if formation_df["slot"].isna().any():
            raise ValueError(
                "Formation contains missing slot names."
            )

        if formation_df["role"].isna().any():
            raise ValueError(
                "Formation contains missing roles."
            )

        if formation_df["slot"].duplicated().any():
            duplicates = (
                formation_df.loc[
                    formation_df["slot"].duplicated(
                        keep=False
                    ),
                    "slot",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                "Formation contains duplicate slots: "
                f"{duplicates}"
            )

    @staticmethod
    def _squad_to_dataframe(
        squad: Squad,
    ) -> pd.DataFrame:
        rows = [
            StartingXIBuilder._player_to_row(
                player
            )
            for player in squad.players
            if player.availability.available
        ]

        if not rows:
            raise ValueError(
                f"No available players remain for "
                f"{squad.national_team!r}."
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _player_to_row(
        player: Player,
    ) -> dict:
        role_ratings = player.role_ratings

        role_values = {
            "GK": None,
            "CB": None,
            "FB": None,
            "DM": None,
            "CM": None,
            "AM": None,
            "WM": None,
            "W": None,
            "ST": None,
        }

        if role_ratings is not None:
            role_values = {
                role: getattr(
                    role_ratings,
                    role,
                )
                for role in role_values
            }

        eligible_roles = [
            role
            for role, rating
            in role_values.items()
            if rating is not None
            and not pd.isna(rating)
        ]

        row = {
            "player_id": str(
                player.identity.player_id
            ),
            "player": player.identity.name,
            "eligible_roles_list": eligible_roles,
        }

        for role, rating in role_values.items():
            row[f"rating_{role}"] = rating

        return row


def parse_roles(value) -> list[str]:
    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return []
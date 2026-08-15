#production_club_representation_provider

from __future__ import annotations

import pandas as pd

from research.player_intelligence.player_repository import (
    _row_value,
    build_player_evidence,
    build_role_ratings,
    resolve_player_name,
)
from research.player_intelligence.player_schema import (
    Player,
    PlayerAvailability,
    PlayerIdentity,
    PlayerRatings,
)
from research.player_intelligence.position_normalizer import (
    primary_position,
    secondary_positions,
)

from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
    build_team_representation_from_players,
)


def player_from_v4_rating_row(
    row: pd.Series,
) -> Player:
    """
    Convert one Player Ratings v4 row into the canonical Player model.

    The modern club-production path derives football strength from
    role-specific ratings. The legacy scalar overall rating remains
    zero-valued for compatibility with the established robust club
    repository contract, where squad_quality and evidence_score are
    deliberately non-operative.
    """

    name = resolve_player_name(row)

    player_id = str(
        _row_value(
            row,
            "canonical_player_id",
            _row_value(row, "player_id", ""),
        )
    )

    position_string = _row_value(
        row,
        "position",
        None,
    )

    return Player(
        identity=PlayerIdentity(
            player_id=player_id,
            name=name,
            national_team=str(
                _row_value(
                    row,
                    "country",
                    "",
                )
            ),
            club=_row_value(
                row,
                "current_team",
                None,
            ),
            primary_position=primary_position(
                position_string
            ),
            secondary_positions=secondary_positions(
                position_string
            ),
        ),
        ratings=PlayerRatings(
            overall=0.0,
            attack=0.0,
            midfield=0.0,
            defense=0.0,
            goalkeeper=0.0,
            recent_form=None,
        ),
        availability=PlayerAvailability(
            available=True,
            expected_to_start=None,
            minutes_fit=_row_value(
                row,
                "minutesPlayed",
                None,
            ),
        ),
        role_ratings=build_role_ratings(row),
        evidence=build_player_evidence(row),
    )

def build_players_for_club(
    *,
    club: str,
    membership: pd.DataFrame,
    ratings: pd.DataFrame,
) -> tuple[Player, ...]:
    """
    Build the rated Player population assigned to one club.

    Club membership is authoritative for the target squad.
    The club stored in historical Player Ratings v4 evidence
    is therefore replaced by the supplied season membership.
    """

    club_membership = membership.loc[
        membership["team"].astype(str) == club
    ].copy()

    membership_player_ids = (
        club_membership["player_id"]
        .astype(str)
        .tolist()
    )

    if len(membership_player_ids) != len(
        set(membership_player_ids)
    ):
        raise ValueError(
            "Club membership contains duplicate player IDs. "
            f"Club={club!r}."
        )

    ratings_by_player_id = (
        ratings
        .assign(
            _player_id=(
                ratings["player_id"]
                .astype(str)
            )
        )
        .set_index(
            "_player_id",
            drop=False,
        )
    )

    players: list[Player] = []

    for player_id in membership_player_ids:
        if player_id not in ratings_by_player_id.index:
            raise ValueError(
                "Club membership contains a player with "
                "missing Player Ratings v4 evidence. "
                f"Club={club!r}, "
                f"player_id={player_id!r}."
            )

        rating_row = (
            ratings_by_player_id
            .loc[player_id]
            .copy()
        )

        # Season membership is authoritative for club identity.
        rating_row["current_team"] = club

        players.append(
            player_from_v4_rating_row(
                rating_row
            )
        )

    return tuple(players)

def build_club_representation(
    *,
    club: str,
    membership: pd.DataFrame,
    ratings: pd.DataFrame,
) -> TeamRepresentation:
    """
    Build one full-squad club representation from season membership
    and Player Ratings v4.

    Player selection is delegated to build_players_for_club().
    Football-intelligence aggregation is delegated to the established
    TeamRepresentation builder.
    """

    players = build_players_for_club(
        club=club,
        membership=membership,
        ratings=ratings,
    )

    return build_team_representation_from_players(
        national_team=club,
        players=players,
        representation_type="full_squad",
        aggregation_profile="legacy_top_5",
    )

class ClubRepresentationProvider:
    """
    In-memory callable provider for production club representations.

    Membership and Player Ratings v4 are loaded once and reused across
    club lookups. Calling the provider with a club name returns one
    full-squad TeamRepresentation.
    """

    def __init__(
        self,
        *,
        membership: pd.DataFrame,
        ratings: pd.DataFrame,
    ) -> None:
        self.membership = membership.copy()
        self.ratings = ratings.copy()

    def __call__(
        self,
        club: str,
    ) -> TeamRepresentation:
        return build_club_representation(
            club=club,
            membership=self.membership,
            ratings=self.ratings,
        )
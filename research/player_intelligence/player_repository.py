#player_repository.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.player_schema import (
    Player,
    PlayerAvailability,
    PlayerIdentity,
    PlayerRatings,
    Squad,
    RoleRatings,
    PlayerEvidence
)

from research.player_intelligence.position_normalizer import (
    primary_position,
    secondary_positions,
)

from research.player_intelligence.player_evidence_repository import (
    PlayerEvidenceRepository,
)


DEFAULT_PLAYER_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_ratings.csv"
)


class PlayerRepository:
    def __init__(
        self,
        player_features_path: Path = DEFAULT_PLAYER_FEATURES_PATH,
        evidence_repository: PlayerEvidenceRepository | None = None,
    ) -> None:
        self.player_features_path = player_features_path
        self.evidence_repository = evidence_repository
        self._players: tuple[Player, ...] | None = None

    def load_players(self) -> tuple[Player, ...]:
        if self._players is not None:
            return self._players

        df = pd.read_csv(self.player_features_path)

        histories = (
            self.evidence_repository.load_histories()
            if self.evidence_repository is not None
            else {}
        )

        players = tuple(
            player_from_row(
                row,
                evidence_history=histories.get(str(_row_value(row, "player_id", ""))),
            )
            for _, row in df.iterrows()
    )

        self._players = players
        return players

    def get_players_for_team(self, national_team: str) -> tuple[Player, ...]:
        return tuple(
            player
            for player in self.load_players()
            if player.identity.national_team == national_team
        )

    def get_squad(self, national_team: str) -> Squad:
        return Squad(
            national_team=national_team,
            players=self.get_players_for_team(national_team),
        )


def _row_value(row: pd.Series, column: str, default=None):
    if column not in row or pd.isna(row[column]):
        return default

    return row[column]


def player_from_row(
    row: pd.Series,
    evidence_history=None,
) -> Player:
    name = resolve_player_name(row)

    national_team = resolve_team_name(row)

    player_id = str(
        _row_value(
            row,
            "player_id",
            f"{national_team}:{name}",
        )
    )

    position_string = _row_value(row, "position", None)

    rating = float(_row_value(row, "rating", 0.0))

    return Player(
        identity=PlayerIdentity(
            player_id=player_id,
            name=name,
            national_team=national_team,
            club=_row_value(row, "club", None),
            primary_position=primary_position(position_string),
            secondary_positions=secondary_positions(position_string),
        ),
        ratings=PlayerRatings(
            overall=rating,
            attack=rating,
            midfield=rating,
            defense=rating,
            goalkeeper=rating,
            recent_form=rating,
        ),
        availability=PlayerAvailability(
            available=True,
            expected_to_start=None,
            minutes_fit=_row_value(row, "minutesPlayed", None),
        ),

        role_ratings=build_role_ratings(row),
        evidence=build_player_evidence(row),
        evidence_history=evidence_history,
    )

def resolve_team_name(row: pd.Series) -> str:
    for column in [
        "country",
        "nation",
        "national_team",
        "team",
        "team_name",
        "current_team",
    ]:
        value = _row_value(row, column)
        if value is not None:
            return str(value)

    return "unknown"

def resolve_player_name(row: pd.Series) -> str:
    for column in [
        "player",
        "player_name",
        "name",
        "short_name",
        "known_name",
    ]:
        value = _row_value(row, column)
        if value is not None:
            return str(value)

    return "unknown"

def build_role_ratings(row: pd.Series) -> RoleRatings:
    return RoleRatings(
        GK=_row_value(row, "rating_GK", None),
        CB=_row_value(row, "rating_CB", None),
        FB=_row_value(row, "rating_FB", None),
        DM=_row_value(row, "rating_DM", None),
        CM=_row_value(row, "rating_CM", None),
        AM=_row_value(row, "rating_AM", None),
        WM=_row_value(row, "rating_WM", None),
        W=_row_value(row, "rating_W", None),
        ST=_row_value(row, "rating_ST", None),
    )

def build_player_evidence(row: pd.Series) -> PlayerEvidence:
    return PlayerEvidence(
        minutes_played=_row_value(row, "minutesPlayed", None),
        total_weighted_evidence=_row_value(row, "total_weighted_evidence", None),
        evidence_confidence=_row_value(row, "evidence_confidence", None),
        competition_count=_row_value(row, "competition_count", None),
        season_count=_row_value(row, "season_count", None),
        recency_weight=_row_value(row, "recency_weight", None),
        sample_quality=_row_value(row, "sample_quality", None),
    )
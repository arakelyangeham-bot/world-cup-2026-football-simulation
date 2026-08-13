#player_validation.py

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research.player_intelligence.player_schema import Player


@dataclass(frozen=True)
class PlayerValidationReport:
    total_players: int
    missing_rating_count: int
    zero_rating_count: int
    missing_position_count: int
    missing_team_count: int
    duplicate_player_id_count: int
    duplicate_team_name_count: int


def validate_players(players: tuple[Player, ...]) -> PlayerValidationReport:
    rows = []

    for player in players:
        rows.append(
            {
                "player_id": player.identity.player_id,
                "name": player.identity.name,
                "national_team": player.identity.national_team,
                "primary_position": player.identity.primary_position,
                "overall_rating": player.ratings.overall,
            }
        )

    df = pd.DataFrame(rows)

    return PlayerValidationReport(
        total_players=len(df),
        missing_rating_count=int(df["overall_rating"].isna().sum()),
        zero_rating_count=int((df["overall_rating"] == 0).sum()),
        missing_position_count=int(df["primary_position"].isna().sum()),
        missing_team_count=int(df["national_team"].isna().sum()),
        duplicate_player_id_count=int(df["player_id"].duplicated().sum()),
        duplicate_team_name_count=int(
            df.duplicated(subset=["national_team", "name"]).sum()
        ),
    )


def validation_report_to_dataframe(
    report: PlayerValidationReport,
) -> pd.DataFrame:
    return pd.DataFrame([report.__dict__])
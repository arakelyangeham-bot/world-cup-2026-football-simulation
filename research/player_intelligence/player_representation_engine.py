#player_representation_engine.py

from __future__ import annotations

from dataclasses import dataclass

from research.player_intelligence.player_schema import Player
from research.player_intelligence.player_representation import PlayerRepresentation
from research.player_intelligence.current_ability_estimator import (
    confidence_adjusted_ability,
)


@dataclass(frozen=True)
class PlayerRepresentationDiagnostics:
    player_id: str
    player_name: str

    total_minutes: float
    competition_count: int
    season_count: int

    latest_season: str | None
    recent_minutes: float
    historical_minutes: float
    recency_share: float


def _season_sort_key(season_year: str | None) -> str:
    if season_year is None:
        return ""

    return str(season_year)


def build_player_representation_diagnostics(
    player: Player,
) -> PlayerRepresentationDiagnostics:
    history = player.evidence_history

    if history is None:
        total_minutes = player.evidence.minutes_played or 0.0

        return PlayerRepresentationDiagnostics(
            player_id=player.identity.player_id,
            player_name=player.identity.name,
            total_minutes=total_minutes,
            competition_count=player.evidence.competition_count or 0,
            season_count=player.evidence.season_count or 0,
            latest_season=None,
            recent_minutes=0.0,
            historical_minutes=total_minutes,
            recency_share=0.0,
        )

    entries = history.entries

    total_minutes = history.total_minutes
    competition_count = history.competition_count
    season_count = history.season_count

    seasons = sorted(
        {
            entry.season_year
            for entry in entries
            if entry.season_year is not None
        },
        key=_season_sort_key,
    )

    latest_season = seasons[-1] if seasons else None

    recent_minutes = sum(
        entry.minutes_played or 0.0
        for entry in entries
        if entry.season_year == latest_season
    )

    historical_minutes = total_minutes - recent_minutes

    recency_share = (
        recent_minutes / total_minutes
        if total_minutes > 0
        else 0.0
    )

    return PlayerRepresentationDiagnostics(
        player_id=player.identity.player_id,
        player_name=player.identity.name,
        total_minutes=total_minutes,
        competition_count=competition_count,
        season_count=season_count,
        latest_season=latest_season,
        recent_minutes=recent_minutes,
        historical_minutes=historical_minutes,
        recency_share=recency_share,
    )

def build_player_representation(
    player: Player,
) -> PlayerRepresentation:
    diagnostics = build_player_representation_diagnostics(player)

    return PlayerRepresentation(
        player_id=diagnostics.player_id,
        player_name=diagnostics.player_name,
        current_ability=confidence_adjusted_ability(
            role_ratings=player.role_ratings,
            evidence_confidence=player.evidence.evidence_confidence,
        ),
        evidence_confidence=player.evidence.evidence_confidence or 0.0,
        total_minutes=diagnostics.total_minutes,
        competition_count=diagnostics.competition_count,
        season_count=diagnostics.season_count,
        latest_season=diagnostics.latest_season,
        recency_share=diagnostics.recency_share,
    )
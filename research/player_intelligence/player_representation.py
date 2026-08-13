#player_representation.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerRepresentation:
    player_id: str
    player_name: str

    current_ability: float
    evidence_confidence: float

    total_minutes: float
    competition_count: int
    season_count: int

    latest_season: str | None
    recency_share: float
#competition_evidence_projection.py

from __future__ import annotations


def identity_competition_weight() -> float:
    return 1.0


def competition_count_weight(
    competition_count: int | float | None,
    full_competition_count: float = 3.0,
) -> float:
    if competition_count is None:
        return 0.0

    return max(
        0.0,
        min(float(competition_count) / full_competition_count, 1.0),
    )


def season_count_weight(
    season_count: int | float | None,
    full_season_count: float = 3.0,
) -> float:
    if season_count is None:
        return 0.0

    return max(
        0.0,
        min(float(season_count) / full_season_count, 1.0),
    )


def weighted_evidence_weight(
    total_weighted_evidence: float | None,
    full_weighted_evidence: float = 1800.0,
) -> float:
    if total_weighted_evidence is None:
        return 0.0

    return max(
        0.0,
        min(float(total_weighted_evidence) / full_weighted_evidence, 1.0),
    )


def combined_competition_weight(
    competition_count: int | float | None,
    season_count: int | float | None,
    total_weighted_evidence: float | None,
) -> float:
    weights = [
        competition_count_weight(competition_count),
        season_count_weight(season_count),
        weighted_evidence_weight(total_weighted_evidence),
    ]

    return sum(weights) / len(weights)


def apply_competition_weight(
    rating: float | None,
    weight: float,
) -> float:
    if rating is None:
        return 0.0

    return float(rating) * weight
#current_ability_estimator.py

from __future__ import annotations

from research.player_intelligence.player_schema import RoleRatings


def _values(role_ratings: RoleRatings | None) -> list[float]:
    if role_ratings is None:
        return []

    values = [
        role_ratings.GK,
        role_ratings.CB,
        role_ratings.FB,
        role_ratings.DM,
        role_ratings.CM,
        role_ratings.AM,
        role_ratings.WM,
        role_ratings.W,
        role_ratings.ST,
    ]

    return [
        float(value)
        for value in values
        if value is not None
    ]


def best_role_ability(role_ratings: RoleRatings | None) -> float:
    values = _values(role_ratings)

    if not values:
        return 0.0

    return max(values)


def mean_role_ability(role_ratings: RoleRatings | None) -> float:
    values = _values(role_ratings)

    if not values:
        return 0.0

    return sum(values) / len(values)


def confidence_adjusted_ability(
    role_ratings: RoleRatings | None,
    evidence_confidence: float | None,
) -> float:
    confidence = 0.0 if evidence_confidence is None else float(evidence_confidence)

    confidence = max(0.0, min(confidence, 1.0))

    return best_role_ability(role_ratings) * confidence
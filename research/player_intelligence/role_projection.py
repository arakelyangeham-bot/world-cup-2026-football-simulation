#role_projection.py

from __future__ import annotations

from research.player_intelligence.player_schema import RoleRatings


def _value(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def weighted_role_score(
    role_ratings: RoleRatings | None,
    weights: dict[str, float],
) -> float:
    if role_ratings is None:
        return 0.0

    total_weight = sum(weights.values())

    if total_weight == 0:
        return 0.0

    score = 0.0

    for role, weight in weights.items():
        score += _value(getattr(role_ratings, role, None)) * weight

    return score / total_weight


def project_attack(role_ratings: RoleRatings | None) -> float:
    return weighted_role_score(
        role_ratings,
        {
            "ST": 0.40,
            "W": 0.25,
            "AM": 0.20,
            "CM": 0.10,
            "FB": 0.05,
        },
    )


def project_midfield(role_ratings: RoleRatings | None) -> float:
    return weighted_role_score(
        role_ratings,
        {
            "CM": 0.35,
            "DM": 0.25,
            "AM": 0.20,
            "WM": 0.10,
            "FB": 0.10,
        },
    )


def project_defense(role_ratings: RoleRatings | None) -> float:
    return weighted_role_score(
        role_ratings,
        {
            "CB": 0.40,
            "FB": 0.25,
            "DM": 0.25,
            "GK": 0.10,
        },
    )


def project_goalkeeper(role_ratings: RoleRatings | None) -> float:
    return weighted_role_score(
        role_ratings,
        {
            "GK": 1.00,
        },
    )
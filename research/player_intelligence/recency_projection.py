from __future__ import annotations


def identity_recency_weight() -> float:
    return 1.0


def linear_recency_weight(
    recency_weight: float | None,
) -> float:
    if recency_weight is None:
        return 1.0

    return max(0.0, min(float(recency_weight), 1.0))


def exponential_recency_weight(
    recency_weight: float | None,
    gamma: float = 2.0,
) -> float:
    if recency_weight is None:
        return 1.0

    value = max(0.0, min(float(recency_weight), 1.0))
    return value ** gamma


def blended_recency_weight(
    recency_weight: float | None,
    floor: float = 0.50,
) -> float:
    if recency_weight is None:
        return 1.0

    value = max(0.0, min(float(recency_weight), 1.0))
    return floor + (1.0 - floor) * value


def apply_recency_weight(
    rating: float | None,
    weight: float,
) -> float:
    if rating is None:
        return 0.0

    return float(rating) * weight
#evidence_projection.py

from __future__ import annotations


def identity_evidence_weight() -> float:
    return 1.0


def confidence_weight(
    evidence_confidence: float | None,
) -> float:
    if evidence_confidence is None:
        return 1.0

    return max(0.0, min(float(evidence_confidence), 1.0))


def minutes_weight(
    minutes_played: float | None,
    full_evidence_minutes: float = 1800.0,
) -> float:
    if minutes_played is None:
        return 0.0

    return max(
        0.0,
        min(float(minutes_played) / full_evidence_minutes, 1.0),
    )


def sample_quality_weight(
    sample_quality: str | None,
) -> float:
    if sample_quality == "observed":
        return 1.0

    if sample_quality == "limited_minutes":
        return 0.65

    if sample_quality == "unobserved_or_zero_rating":
        return 0.25

    return 1.0


def combined_evidence_weight(
    evidence_confidence: float | None,
    minutes_played: float | None,
    sample_quality: str | None,
) -> float:
    return (
        confidence_weight(evidence_confidence)
        * minutes_weight(minutes_played)
        * sample_quality_weight(sample_quality)
    )


def apply_evidence_weight(
    rating: float | None,
    weight: float,
) -> float:
    if rating is None:
        return 0.0

    return float(rating) * weight
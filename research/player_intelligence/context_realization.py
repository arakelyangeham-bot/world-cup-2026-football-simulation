#context_realization

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ContextRealizationPolicy:
    """
    Controls how deployment suitability affects realized contribution.

    A strength of zero reproduces the legacy contribution exactly.
    """

    adjustment_strength: float = 0.0
    policy_id: str = "symmetric_gap_linear_v1"

    def __post_init__(self) -> None:
        if not math.isfinite(
            self.adjustment_strength
        ):
            raise ValueError(
                "Adjustment strength must be finite."
            )

        if not (
            0.0
            <= self.adjustment_strength
            <= 1.0
        ):
            raise ValueError(
                "Adjustment strength must lie within [0, 1]."
            )


def symmetric_relative_gap(
    *,
    assigned_rating: float,
    strongest_rating: float,
) -> float:
    """
    Return a bounded deployment gap in [0, 1].
    """

    assigned = float(
        assigned_rating
    )

    strongest = float(
        strongest_rating
    )

    if not math.isfinite(assigned):
        raise ValueError(
            "Assigned rating must be finite."
        )

    if not math.isfinite(strongest):
        raise ValueError(
            "Strongest rating must be finite."
        )

    denominator = (
        abs(strongest)
        + abs(assigned)
    )

    if denominator == 0.0:
        return 0.0

    return float(
        min(
            max(
                abs(
                    strongest
                    - assigned
                )
                / denominator,
                0.0,
            ),
            1.0,
        )
    )


def context_multiplier(
    *,
    gap: float | None,
    policy: ContextRealizationPolicy,
) -> float:
    """
    Convert a role-suitability gap into a contribution multiplier.

    Unresolved gaps receive no adjustment.
    """

    if gap is None:
        return 1.0

    numeric_gap = float(
        gap
    )

    if not math.isfinite(
        numeric_gap
    ):
        raise ValueError(
            "Context gap must be finite when resolved."
        )

    if not (
        0.0
        <= numeric_gap
        <= 1.0
    ):
        raise ValueError(
            "Context gap must lie within [0, 1]."
        )

    multiplier = (
        1.0
        - policy.adjustment_strength
        * numeric_gap
    )

    return float(
        max(
            multiplier,
            0.0,
        )
    )
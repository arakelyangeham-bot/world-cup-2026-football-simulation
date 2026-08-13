#role_suitability

from __future__ import annotations

from dataclasses import dataclass
import math

from research.player_intelligence.context_realization import (
    symmetric_relative_gap,
)


@dataclass(frozen=True)
class RoleSuitabilitySignals:
    """
    Diagnostic role-suitability signals for one player assignment.

    Study 098A introduces no player-contribution adjustment.

    These signals describe the relationship between the player's
    assigned role and strongest rated role.
    """

    assigned_role: str
    strongest_role: str | None

    assigned_role_rating: float | None
    strongest_role_rating: float | None

    assigned_role_rank: int | None

    raw_rating_gap: float | None
    absolute_rating_gap: float | None
    symmetric_relative_gap: float | None

    reciprocal_rank_score: float | None
    strongest_role_indicator: float | None

    positive_scale_ratio: float | None

    signal_profile: str = (
        "role_suitability_diagnostics_v1"
    )

    contextual_adjustment_applied: bool = False

    def __post_init__(self) -> None:
        optional_numeric_values = (
            self.raw_rating_gap,
            self.absolute_rating_gap,
            self.symmetric_relative_gap,
            self.reciprocal_rank_score,
            self.strongest_role_indicator,
            self.positive_scale_ratio,
        )

        for value in optional_numeric_values:
            if value is None:
                continue

            if not math.isfinite(value):
                raise ValueError(
                    "Role-suitability signals must be finite "
                    "when resolved."
                )

        if (
            self.assigned_role_rank is not None
            and self.assigned_role_rank < 1
        ):
            raise ValueError(
                "Assigned role rank must be at least one."
            )

        if (
            self.symmetric_relative_gap is not None
            and not (
                0.0
                <= self.symmetric_relative_gap
                <= 1.0
            )
        ):
            raise ValueError(
                "Symmetric relative gap must lie within "
                "[0, 1]."
            )

        if (
            self.reciprocal_rank_score is not None
            and not (
                0.0
                < self.reciprocal_rank_score
                <= 1.0
            )
        ):
            raise ValueError(
                "Reciprocal-rank score must lie within "
                "(0, 1]."
            )

        if (
            self.strongest_role_indicator is not None
            and self.strongest_role_indicator
            not in {
                0.0,
                1.0,
            }
        ):
            raise ValueError(
                "Strongest-role indicator must be zero or one."
            )


def symmetric_relative_gap(
    *,
    assigned_rating: float,
    strongest_rating: float,
) -> float:
    """
    Return a bounded difference between assigned and strongest ratings.

    Formula:

        abs(strongest - assigned)
        /
        (
            abs(strongest)
            + abs(assigned)
        )

    This remains meaningful when ratings are negative or close to zero.

    If both values are zero, the gap is defined as zero.
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

    result = (
        abs(
            strongest
            - assigned
        )
        / denominator
    )

    return float(
        min(
            max(
                result,
                0.0,
            ),
            1.0,
        )
    )


def build_role_suitability_signals(
    contribution: PlayerContribution,
) -> RoleSuitabilitySignals:
    """
    Build diagnostic suitability signals from one contribution.

    No attack, midfield, defense, or goalkeeper value is modified.
    """

    assigned_rating = (
        contribution.assigned_role_rating
    )

    strongest_rating = (
        contribution.strongest_role_rating
    )

    assigned_rank = (
        contribution.assigned_role_rank
    )

    if (
        assigned_rating is None
        or strongest_rating is None
    ):
        raw_gap = None
        absolute_gap = None
        relative_gap = None
        positive_ratio = None

    else:
        assigned_value = float(
            assigned_rating
        )

        strongest_value = float(
            strongest_rating
        )

        raw_gap = (
            strongest_value
            - assigned_value
        )

        absolute_gap = abs(
            raw_gap
        )

        relative_gap = (
            symmetric_relative_gap(
                assigned_rating=(
                    assigned_value
                ),
                strongest_rating=(
                    strongest_value
                ),
            )
        )

        if strongest_value > 0.0:
            positive_ratio = (
                assigned_value
                / strongest_value
            )

        else:
            positive_ratio = None

    if assigned_rank is None:
        reciprocal_rank = None
        strongest_indicator = None

    else:
        reciprocal_rank = (
            1.0
            / assigned_rank
        )

        strongest_indicator = (
            1.0
            if assigned_rank == 1
            else 0.0
        )

    return RoleSuitabilitySignals(
        assigned_role=(
            contribution.assigned_role
        ),
        strongest_role=(
            contribution.strongest_role
        ),

        assigned_role_rating=(
            assigned_rating
        ),
        strongest_role_rating=(
            strongest_rating
        ),

        assigned_role_rank=(
            assigned_rank
        ),

        raw_rating_gap=(
            raw_gap
        ),
        absolute_rating_gap=(
            absolute_gap
        ),
        symmetric_relative_gap=(
            relative_gap
        ),

        reciprocal_rank_score=(
            reciprocal_rank
        ),
        strongest_role_indicator=(
            strongest_indicator
        ),

        positive_scale_ratio=(
            positive_ratio
        ),

        signal_profile=(
            "role_suitability_diagnostics_v1"
        ),
        contextual_adjustment_applied=False,
    )
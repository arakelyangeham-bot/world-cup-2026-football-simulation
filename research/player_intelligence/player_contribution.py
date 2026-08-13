#player_contribution

from __future__ import annotations

from dataclasses import dataclass
import math

from research.player_intelligence.player_schema import (
    LineupAssignment,
    Player,
    RoleRatings,
)
from research.player_intelligence.role_projection import (
    project_attack,
    project_defense,
    project_goalkeeper,
    project_midfield,
)

from research.player_intelligence.context_realization import (
    ContextRealizationPolicy,
    context_multiplier,
    symmetric_relative_gap,
)

ROLE_NAMES = (
    "GK",
    "CB",
    "FB",
    "DM",
    "CM",
    "AM",
    "WM",
    "W",
    "ST",
)


@dataclass(frozen=True)
class PlayerContribution:
    """
    Auditable player contribution within a lineup assignment.

    Study 097A introduces no contextual strength adjustment.

    The four projected dimensions must reproduce the existing
    context-free role-projection functions exactly.
    """

    player: Player
    assignment: LineupAssignment

    attack: float
    midfield: float
    defense: float
    goalkeeper: float

    intrinsic_attack: float
    intrinsic_midfield: float
    intrinsic_defense: float
    intrinsic_goalkeeper: float

    assigned_role: str
    assigned_role_rating: float | None

    strongest_role: str | None
    strongest_role_rating: float | None

    assigned_role_rank: int | None
    role_rating_difference: float | None
    role_fit_ratio: float | None

    context_gap: float | None
    context_multiplier: float
    adjustment_strength: float

    projection_profile: str = (
        "legacy_context_free_v1"
    )

    contextual_adjustment_applied: bool = False

    def __post_init__(self) -> None:
        if self.assignment.player != self.player:
            raise ValueError(
                "PlayerContribution player does not match "
                "the assignment player."
            )

        if (
            self.assigned_role
            != self.assignment.tactical_role
        ):
            raise ValueError(
                "PlayerContribution assigned role does not "
                "match the lineup assignment."
            )

        projected_values = (
            self.attack,
            self.midfield,
            self.defense,
            self.goalkeeper,
        )

        if not all(
            math.isfinite(value)
            for value in projected_values
        ):
            raise ValueError(
                "PlayerContribution contains a non-finite "
                "projected dimension."
            )

        if (
            self.assigned_role_rank is not None
            and self.assigned_role_rank < 1
        ):
            raise ValueError(
                "Assigned role rank must be at least one."
            )

        intrinsic_values = (
            self.intrinsic_attack,
            self.intrinsic_midfield,
            self.intrinsic_defense,
            self.intrinsic_goalkeeper,
        )

        if not all(
            math.isfinite(value)
            for value in intrinsic_values
        ):
            raise ValueError(
                "PlayerContribution contains a non-finite "
                "intrinsic dimension."
            )

        if not math.isfinite(
            self.context_multiplier
        ):
            raise ValueError(
                "Context multiplier must be finite."
            )

        if self.context_multiplier < 0.0:
            raise ValueError(
                "Context multiplier cannot be negative."
            )


def finite_role_ratings(
    role_ratings: RoleRatings | None,
) -> dict[str, float]:
    """
    Return the player's finite role-rating population.
    """

    if role_ratings is None:
        return {}

    values: dict[str, float] = {}

    for role in ROLE_NAMES:
        value = getattr(
            role_ratings,
            role,
        )

        if value is None:
            continue

        numeric_value = float(value)

        if not math.isfinite(
            numeric_value
        ):
            continue

        values[role] = numeric_value

    return values


def assigned_role_rank(
    role_values: dict[str, float],
    assigned_role: str,
) -> int | None:
    """
    Return the one-indexed rank of the assigned role.

    Ties are ordered by the stable ROLE_NAMES vocabulary.
    """

    if assigned_role not in role_values:
        return None

    role_order = {
        role: index
        for index, role in enumerate(
            ROLE_NAMES
        )
    }

    ranked = sorted(
        role_values.items(),
        key=lambda item: (
            -item[1],
            role_order[item[0]],
        ),
    )

    for index, (
        role,
        _,
    ) in enumerate(
        ranked,
        start=1,
    ):
        if role == assigned_role:
            return index

    raise AssertionError(
        "Assigned role was present but no rank "
        "was produced."
    )


def build_player_contribution(
    assignment: LineupAssignment,
    *,
    realization_policy:
        ContextRealizationPolicy | None = None,
) -> PlayerContribution:
    """
    Build a zero-adjustment contribution from one assignment.

    This function preserves Version 2 projection behavior exactly.
    """

    player = assignment.player
    role_values = finite_role_ratings(
        player.role_ratings
    )

    assigned_role = (
        assignment.tactical_role
    )

    assigned_rating = role_values.get(
        assigned_role
    )

    policy = (
        realization_policy
        or ContextRealizationPolicy()
    )

    if role_values:
        strongest_role = max(
            role_values,
            key=lambda role: (
                role_values[role],
                -ROLE_NAMES.index(role),
            ),
        )

        strongest_rating = role_values[
            strongest_role
        ]

    else:
        strongest_role = None
        strongest_rating = None

    rank = assigned_role_rank(
        role_values,
        assigned_role,
    )

    if (
        assigned_rating is None
        or strongest_rating is None
    ):
        difference = None
        ratio = None

    else:
        difference = (
            strongest_rating
            - assigned_rating
        )

        if strongest_rating > 0.0:
            ratio = (
                assigned_rating
                / strongest_rating
            )
        else:
            ratio = None

    if (
        assigned_rating is None
        or strongest_rating is None
    ):
        gap = None
    else:
        gap = symmetric_relative_gap(
            assigned_rating=assigned_rating,
            strongest_rating=strongest_rating,
        )

    multiplier = context_multiplier(
        gap=gap,
        policy=policy,
    )

    intrinsic_attack = project_attack(
        player.role_ratings
    )

    intrinsic_midfield = project_midfield(
        player.role_ratings
    )

    intrinsic_defense = project_defense(
        player.role_ratings
    )

    intrinsic_goalkeeper = project_goalkeeper(
        player.role_ratings
    )

    realized_attack = (
        intrinsic_attack
        * multiplier
    )

    realized_midfield = (
        intrinsic_midfield
        * multiplier
    )

    realized_defense = (
        intrinsic_defense
        * multiplier
    )

    realized_goalkeeper = (
        intrinsic_goalkeeper
    )

    return PlayerContribution(
        player=player,
        assignment=assignment,

        intrinsic_attack=intrinsic_attack,
        intrinsic_midfield=intrinsic_midfield,
        intrinsic_defense=intrinsic_defense,
        intrinsic_goalkeeper=intrinsic_goalkeeper,

        attack=realized_attack,
        midfield=realized_midfield,
        defense=realized_defense,
        goalkeeper=realized_goalkeeper,

        assigned_role=assigned_role,
        assigned_role_rating=assigned_rating,

        strongest_role=strongest_role,
        strongest_role_rating=strongest_rating,

        assigned_role_rank=rank,
        role_rating_difference=difference,
        role_fit_ratio=ratio,

        context_gap=gap,
        context_multiplier=multiplier,
        adjustment_strength=(
            policy.adjustment_strength
        ),

        projection_profile=(
            "context_realization_v1"
            if policy.adjustment_strength > 0.0
            else "legacy_context_free_v1"
        ),

        contextual_adjustment_applied=(
            policy.adjustment_strength > 0.0
            and gap is not None
            and gap > 0.0
        ),
    )

def build_starting_xi_contributions(
    assignments: tuple[
        LineupAssignment,
        ...,
    ],
    *,
    realization_policy:
        ContextRealizationPolicy | None = None,
) -> tuple[
    PlayerContribution,
    ...,
]:
    """
    Build one contribution per preserved lineup assignment.
    """

    contributions = tuple(
        build_player_contribution(
            assignment,
            realization_policy=(
                realization_policy
            ),
        )
        for assignment in assignments
    )

    player_ids = tuple(
        contribution
        .player
        .identity
        .player_id
        for contribution
        in contributions
    )

    if len(player_ids) != len(
        set(player_ids)
    ):
        raise ValueError(
            "Contribution population contains duplicate "
            "players."
        )

    slots = tuple(
        contribution
        .assignment
        .slot
        for contribution
        in contributions
    )

    if len(slots) != len(
        set(slots)
    ):
        raise ValueError(
            "Contribution population contains duplicate "
            "lineup slots."
        )

    return contributions
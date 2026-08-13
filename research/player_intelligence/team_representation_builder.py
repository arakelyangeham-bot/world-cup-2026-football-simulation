from __future__ import annotations

from dataclasses import dataclass

from research.player_intelligence.aggregation_adapter import (
    AggregationSpecificationLike,
    aggregate_dimension_values,
    aggregation_profile_name,
)
from research.player_intelligence.player_schema import (
    Player,
    Squad,
    StartingXI,
)
from research.player_intelligence.role_projection import (
    project_attack,
    project_defense,
    project_goalkeeper,
    project_midfield,
)

from research.player_intelligence.player_contribution import (
    PlayerContribution,
    build_starting_xi_contributions,
)

from research.player_intelligence.context_realization import (
    ContextRealizationPolicy,
)

@dataclass(frozen=True)
class TeamRepresentation:
    national_team: str

    representation_type: str
    aggregation_profile: str

    attack: float
    midfield: float
    defense: float
    goalkeeper: float

    attack_depth: float
    midfield_depth: float
    defense_depth: float

    squad_quality: float
    evidence_score: float

    player_count: int
    available_player_count: int


def _mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def _top_n_mean(
    values: list[float],
    n: int,
) -> float:
    if not values:
        return 0.0

    return _mean(
        sorted(
            values,
            reverse=True,
        )[:n]
    )


def _aggregate_primary_dimensions(
    *,
    attack_values: list[float],
    midfield_values: list[float],
    defense_values: list[float],
    aggregation_profile: str,
    aggregation_specification:
        AggregationSpecificationLike | None,
) -> tuple[
    float,
    float,
    float,
    str,
]:
    """
    Aggregate the three primary outfield dimensions.

    Legacy behavior remains the default. When an explicit aggregation
    specification is supplied, all three dimensions are delegated to
    the reusable aggregation adapter.

    Goalkeeper and depth fields are intentionally outside this helper.
    """

    if aggregation_specification is None:
        return (
            _top_n_mean(
                attack_values,
                5,
            ),
            _top_n_mean(
                midfield_values,
                5,
            ),
            _top_n_mean(
                defense_values,
                5,
            ),
            aggregation_profile,
        )

    attack = aggregate_dimension_values(
        attack_values,
        specification=aggregation_specification,
    )

    midfield = aggregate_dimension_values(
        midfield_values,
        specification=aggregation_specification,
    )

    defense = aggregate_dimension_values(
        defense_values,
        specification=aggregation_specification,
    )

    resolved_profile = aggregation_profile_name(
        aggregation_specification
    )

    return (
        attack,
        midfield,
        defense,
        resolved_profile,
    )


def build_team_representation_from_players(
    national_team: str,
    players: tuple[Player, ...],
    representation_type: str = "full_squad",
    aggregation_profile: str = "legacy_top_5",
    aggregation_specification:
        AggregationSpecificationLike | None = None,
) -> TeamRepresentation:
    attack_values = [
        project_attack(
            player.role_ratings
        )
        for player in players
    ]

    midfield_values = [
        project_midfield(
            player.role_ratings
        )
        for player in players
    ]

    defense_values = [
        project_defense(
            player.role_ratings
        )
        for player in players
    ]

    goalkeeper_values = [
        project_goalkeeper(
            player.role_ratings
        )
        for player in players
    ]

    overall_values = [
        player.ratings.overall
        for player in players
    ]

    available_players = [
        player
        for player in players
        if player.availability.available
    ]

    evidence_values = [
        1.0
        for player in players
        if player.ratings.overall > 0
    ]

    (
        attack,
        midfield,
        defense,
        resolved_aggregation_profile,
    ) = _aggregate_primary_dimensions(
        attack_values=attack_values,
        midfield_values=midfield_values,
        defense_values=defense_values,
        aggregation_profile=aggregation_profile,
        aggregation_specification=(
            aggregation_specification
        ),
    )

    return TeamRepresentation(
        national_team=national_team,
        representation_type=representation_type,
        aggregation_profile=(
            resolved_aggregation_profile
        ),

        attack=attack,
        midfield=midfield,
        defense=defense,

        # Goalkeeper behavior remains unchanged in Study 091A.
        goalkeeper=(
            max(goalkeeper_values)
            if goalkeeper_values
            else 0.0
        ),

        # Existing depth definitions remain unchanged.
        attack_depth=_mean(
            attack_values
        ),
        midfield_depth=_mean(
            midfield_values
        ),
        defense_depth=_mean(
            defense_values
        ),

        squad_quality=_mean(
            overall_values
        ),

        evidence_score=(
            len(evidence_values)
            / len(players)
            if players
            else 0.0
        ),

        player_count=len(players),
        available_player_count=len(
            available_players
        ),
    )

def build_team_representation_from_contributions(
    *,
    national_team: str,
    contributions: tuple[
        PlayerContribution,
        ...,
    ],
    representation_type: str = (
        "expected_starting_xi_contributions"
    ),
    aggregation_profile: str = "legacy_top_5",
    aggregation_specification:
        AggregationSpecificationLike | None = None,
) -> TeamRepresentation:
    """
    Build a team representation from explicit player contributions.

    Study 097B applies no contextual adjustment. When contributions
    were built with ``legacy_context_free_v1``, this path must reproduce
    the established player-based builder exactly.
    """

    players = tuple(
        contribution.player
        for contribution in contributions
    )

    player_ids = tuple(
        str(
            player.identity.player_id
        )
        for player in players
    )

    if len(player_ids) != len(
        set(player_ids)
    ):
        raise ValueError(
            "Contribution population contains duplicate players."
        )

    assignment_slots = tuple(
        contribution.assignment.slot
        for contribution in contributions
    )

    if len(assignment_slots) != len(
        set(assignment_slots)
    ):
        raise ValueError(
            "Contribution population contains duplicate lineup slots."
        )

    mismatched_teams = [
        player.identity.national_team
        for player in players
        if (
            player.identity.national_team
            != national_team
        )
    ]

    if mismatched_teams:
        raise ValueError(
            "Contribution population contains players from a "
            "different national team."
        )

    attack_values = [
        contribution.attack
        for contribution in contributions
    ]

    midfield_values = [
        contribution.midfield
        for contribution in contributions
    ]

    defense_values = [
        contribution.defense
        for contribution in contributions
    ]

    goalkeeper_values = [
        contribution.goalkeeper
        for contribution in contributions
    ]

    overall_values = [
        contribution.player.ratings.overall
        for contribution in contributions
    ]

    available_players = [
        contribution.player
        for contribution in contributions
        if contribution.player.availability.available
    ]

    evidence_values = [
        1.0
        for contribution in contributions
        if contribution.player.ratings.overall > 0
    ]

    (
        attack,
        midfield,
        defense,
        resolved_aggregation_profile,
    ) = _aggregate_primary_dimensions(
        attack_values=attack_values,
        midfield_values=midfield_values,
        defense_values=defense_values,
        aggregation_profile=aggregation_profile,
        aggregation_specification=(
            aggregation_specification
        ),
    )

    return TeamRepresentation(
        national_team=national_team,
        representation_type=representation_type,
        aggregation_profile=(
            resolved_aggregation_profile
        ),

        attack=attack,
        midfield=midfield,
        defense=defense,

        goalkeeper=(
            max(goalkeeper_values)
            if goalkeeper_values
            else 0.0
        ),

        attack_depth=_mean(
            attack_values
        ),
        midfield_depth=_mean(
            midfield_values
        ),
        defense_depth=_mean(
            defense_values
        ),

        squad_quality=_mean(
            overall_values
        ),

        evidence_score=(
            len(evidence_values)
            / len(contributions)
            if contributions
            else 0.0
        ),

        player_count=len(
            contributions
        ),
        available_player_count=len(
            available_players
        ),
    )

def build_team_representation_from_squad(
    squad: Squad,
    *,
    aggregation_specification:
        AggregationSpecificationLike | None = None,
) -> TeamRepresentation:
    return build_team_representation_from_players(
        national_team=squad.national_team,
        players=squad.players,
        representation_type="full_squad",
        aggregation_profile="legacy_top_5",
        aggregation_specification=(
            aggregation_specification
        ),
    )


def build_team_representation_from_starting_xi(
    starting_xi: StartingXI,
    *,
    aggregation_specification:
        AggregationSpecificationLike | None = None,
) -> TeamRepresentation:
    return build_team_representation_from_players(
        national_team=starting_xi.national_team,
        players=starting_xi.players,
        representation_type=(
            "expected_starting_xi"
        ),
        aggregation_profile="legacy_top_5",
        aggregation_specification=(
            aggregation_specification
        ),
    )

def build_team_representation_from_starting_xi_contributions(
    starting_xi: StartingXI,
    *,
    aggregation_specification:
        AggregationSpecificationLike | None = None,
) -> TeamRepresentation:
    """
    Build the expected-XI representation through PlayerContribution.

    This is an additive Study 097B path. The established
    ``build_team_representation_from_starting_xi`` function remains
    unchanged as the Version 2 control.
    """

    if not starting_xi.assignments:
        raise ValueError(
            "Contribution-based starting-XI representation requires "
            "preserved lineup assignments."
        )

    contributions = (
        build_starting_xi_contributions(
            starting_xi.assignments
        )
    )

    return build_team_representation_from_contributions(
        national_team=starting_xi.national_team,
        contributions=contributions,
        representation_type=(
            "expected_starting_xi"
        ),
        aggregation_profile="legacy_top_5",
        aggregation_specification=(
            aggregation_specification
        ),
    )

def build_team_representation_from_starting_xi_contextual(
    starting_xi: StartingXI,
    *,
    realization_policy:
        ContextRealizationPolicy,
    aggregation_specification:
        AggregationSpecificationLike | None = None,
) -> TeamRepresentation:
    """
    Build an expected-XI representation from context-realized
    player contributions.
    """

    if not starting_xi.assignments:
        raise ValueError(
            "Context-aware starting-XI representation "
            "requires preserved lineup assignments."
        )

    contributions = (
        build_starting_xi_contributions(
            starting_xi.assignments,
            realization_policy=(
                realization_policy
            ),
        )
    )

    return build_team_representation_from_contributions(
        national_team=starting_xi.national_team,
        contributions=contributions,
        representation_type=(
            "expected_starting_xi_contextual"
        ),
        aggregation_profile="legacy_top_5",
        aggregation_specification=(
            aggregation_specification
        ),
    )
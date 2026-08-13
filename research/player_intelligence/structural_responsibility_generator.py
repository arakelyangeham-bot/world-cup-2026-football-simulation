#structural_responsibility_generator

from __future__ import annotations
from enum import Enum

from dataclasses import dataclass

from research.player_intelligence.football_responsibility import (
    ResponsibilityType,
    responsibility_definition,
)
from research.player_intelligence.formation_geometry import (
    FormationGeometry,
)
from research.player_intelligence.player_schema import (
    StartingXI,
)
from research.player_intelligence.positional_responsibility_generator import (
    PositionalResponsibilitySet,
)


STRUCTURAL_RESPONSIBILITY_TYPES = (
    ResponsibilityType.SUPPORT,
    ResponsibilityType.PROTECTION,
    ResponsibilityType.COVERAGE,
    ResponsibilityType.CONNECTION,
)

class StructuralHypothesisStatus(str, Enum):
    """
    Research lifecycle state for an expert-authored structural
    football hypothesis.
    """

    ACTIVE_DIAGNOSTIC = "active_diagnostic"
    REVISION_REQUIRED = "revision_required"
    DEFERRED = "deferred"

class StructuralUnitType(str, Enum):
    """
    Broad tactical unit described by a structural hypothesis.
    """

    DEFENSIVE = "defensive"
    MIDFIELD = "midfield"
    ATTACKING = "attacking"
    CROSS_UNIT = "cross_unit"

@dataclass(frozen=True)
class StructuralResponsibility:
    """
    One directional tactical responsibility inferred from an
    explicit structural rule.

    No strength, weight, probability, or causal effect is attached.
    """

    source_slot: str
    target_slot: str

    source_role: str
    target_role: str

    responsibility_type: ResponsibilityType

    rule_id: str
    hypothesis_status: StructuralHypothesisStatus

    supporting_positional_types: tuple[
        ResponsibilityType,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.source_slot.strip():
            raise ValueError(
                "Structural responsibility source slot "
                "must not be empty."
            )

        if not self.target_slot.strip():
            raise ValueError(
                "Structural responsibility target slot "
                "must not be empty."
            )

        if self.source_slot == self.target_slot:
            raise ValueError(
                "Structural responsibility cannot reference "
                "the same slot twice."
            )

        if not self.source_role.strip():
            raise ValueError(
                "Source role must not be empty."
            )

        if not self.target_role.strip():
            raise ValueError(
                "Target role must not be empty."
            )

        definition = responsibility_definition(
            self.responsibility_type
        )

        if (
            self.responsibility_type
            not in STRUCTURAL_RESPONSIBILITY_TYPES
        ):
            raise ValueError(
                "Study 103A supports structural "
                "responsibilities only."
            )

        if not definition.generation_enabled:
            raise ValueError(
                "Responsibility type is not enabled."
            )

        if not definition.directional:
            raise ValueError(
                "Study 103A structural responsibilities "
                "must be directional."
            )

        if not self.rule_id.strip():
            raise ValueError(
                "Structural responsibility must declare "
                "a rule ID."
            )

        if not self.supporting_positional_types:
            raise ValueError(
                "Structural responsibility must cite at least "
                "one positional relationship."
            )

        if not isinstance(
            self.hypothesis_status,
            StructuralHypothesisStatus,
        ):
            raise TypeError(
                "Hypothesis status must be a "
                "StructuralHypothesisStatus."
            )
        allowed_evidence = {
            ResponsibilityType.SAME_LINE,
            ResponsibilityType.ADJACENT_LINE,
            ResponsibilityType.SAME_CORRIDOR,
        }

        if not set(
            self.supporting_positional_types
        ).issubset(
            allowed_evidence
        ):
            raise ValueError(
                "Structural responsibility contains unsupported "
                "positional evidence."
            )

    @property
    def canonical_key(
        self,
    ) -> tuple[str, str, str, str, str]:
        return (
            self.source_slot,
            self.target_slot,
            self.responsibility_type.value,
            self.rule_id,
            self.hypothesis_status.value,
        )


@dataclass(frozen=True)
class StructuralResponsibilitySet:
    national_team: str
    formation: str

    responsibilities: tuple[
        StructuralResponsibility,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.national_team.strip():
            raise ValueError(
                "National team must not be empty."
            )

        if not self.formation.strip():
            raise ValueError(
                "Formation must not be empty."
            )

        keys = tuple(
            responsibility.canonical_key
            for responsibility
            in self.responsibilities
        )

        if len(keys) != len(set(keys)):
            raise ValueError(
                "Duplicate structural responsibilities found."
            )

        if keys != tuple(sorted(keys)):
            raise ValueError(
                "Structural responsibilities must use "
                "deterministic ordering."
            )

    def relationships_of_type(
        self,
        responsibility_type: ResponsibilityType,
    ) -> tuple[
        StructuralResponsibility,
        ...,
    ]:
        return tuple(
            responsibility
            for responsibility
            in self.responsibilities
            if (
                responsibility.responsibility_type
                == responsibility_type
            )
        )

@dataclass(frozen=True)
class StructuralHypothesisScope:
    """
    Declared applicability boundary for one structural hypothesis.

    Scope metadata does not establish empirical validity. It records
    where an expert-authored hypothesis is intended to be evaluated.
    """

    supported_formations: tuple[str, ...]
    supported_source_role_counts: tuple[int, ...]
    supported_target_role_counts: tuple[int, ...]
    unit_type: StructuralUnitType

    formation_general: bool = False
    requires_single_source_role: bool = False
    requires_multiple_source_roles: bool = False

    def __post_init__(self) -> None:
        if not self.supported_formations:
            raise ValueError(
                "Hypothesis scope must declare at least one "
                "supported formation."
            )

        if len(self.supported_formations) != len(
            set(self.supported_formations)
        ):
            raise ValueError(
                "Hypothesis scope contains duplicate formations."
            )

        if tuple(
            sorted(self.supported_formations)
        ) != self.supported_formations:
            raise ValueError(
                "Supported formations must use deterministic order."
            )

        if not self.supported_source_role_counts:
            raise ValueError(
                "Hypothesis scope must declare source-role counts."
            )

        if not self.supported_target_role_counts:
            raise ValueError(
                "Hypothesis scope must declare target-role counts."
            )

        if any(
            count <= 0
            for count in self.supported_source_role_counts
        ):
            raise ValueError(
                "Supported source-role counts must be positive."
            )

        if any(
            count <= 0
            for count in self.supported_target_role_counts
        ):
            raise ValueError(
                "Supported target-role counts must be positive."
            )

        if (
            self.requires_single_source_role
            and self.requires_multiple_source_roles
        ):
            raise ValueError(
                "A hypothesis cannot require both a single and "
                "multiple source-role population."
            )

        if (
            self.requires_single_source_role
            and self.supported_source_role_counts != (1,)
        ):
            raise ValueError(
                "Single-source hypotheses must support exactly "
                "one source role."
            )

        if (
            self.requires_multiple_source_roles
            and not all(
                count >= 2
                for count
                in self.supported_source_role_counts
            )
        ):
            raise ValueError(
                "Multiple-source hypotheses must require at least "
                "two source roles."
            )

        if (
            self.formation_general
            and len(self.supported_formations) < 2
        ):
            raise ValueError(
                "A formation-general hypothesis must declare "
                "multiple supported formations."
            )

@dataclass(frozen=True)
class StructuralRule:
    """
    One explicit expert-authored structural football hypothesis.

    A rule's status describes research maturity. It does not alter
    whether the rule can be generated diagnostically unless the
    caller explicitly filters by status.
    """

    rule_id: str

    source_roles: tuple[str, ...]
    target_roles: tuple[str, ...]

    responsibility_type: ResponsibilityType

    required_positional_types: tuple[
        ResponsibilityType,
        ...,
    ]

    status: StructuralHypothesisStatus
    scope: StructuralHypothesisScope

    source_must_be_deeper: bool = False
    source_must_be_more_advanced: bool = False

    same_broad_corridor_required: bool = False

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError(
                "Structural rule ID must not be empty."
            )

        if not self.source_roles:
            raise ValueError(
                "Structural rule must define source roles."
            )

        if not self.target_roles:
            raise ValueError(
                "Structural rule must define target roles."
            )

        if not self.required_positional_types:
            raise ValueError(
                "Structural rule must require positional evidence."
            )

        if (
            self.source_must_be_deeper
            and self.source_must_be_more_advanced
        ):
            raise ValueError(
                "A structural rule cannot require the source "
                "to be both deeper and more advanced."
            )

        if (
            self.responsibility_type
            not in STRUCTURAL_RESPONSIBILITY_TYPES
        ):
            raise ValueError(
                "Structural rule must produce a supported "
                "structural responsibility type."
            )

        if not isinstance(
            self.status,
            StructuralHypothesisStatus,
        ):
            raise TypeError(
                "Structural rule status must be a "
                "StructuralHypothesisStatus."
            )

        allowed_evidence = {
            ResponsibilityType.SAME_LINE,
            ResponsibilityType.ADJACENT_LINE,
            ResponsibilityType.SAME_CORRIDOR,
        }

        if not set(
            self.required_positional_types
        ).issubset(
            allowed_evidence
        ):
            raise ValueError(
                "Structural rule requires unsupported positional "
                "evidence."
            )

        if (
            self.same_broad_corridor_required
            and ResponsibilityType.SAME_CORRIDOR
            not in self.required_positional_types
        ):
            raise ValueError(
                "A broad-corridor requirement must cite "
                "same-corridor positional evidence."
            )

        if not isinstance(
            self.scope,
            StructuralHypothesisScope,
        ):
            raise TypeError(
                "Structural rule scope must be a "
                "StructuralHypothesisScope."
            )
    
INITIAL_STRUCTURAL_HYPOTHESES = (
    StructuralRule(
        rule_id="dm_protects_cb_v1",
        source_roles=("DM",),
        target_roles=("CB",),
        responsibility_type=(
            ResponsibilityType.PROTECTION
        ),
        required_positional_types=(
            ResponsibilityType.ADJACENT_LINE,
        ),
        status=(
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC
        ),
        scope=StructuralHypothesisScope(
            supported_formations=(
                "4-3-3",
            ),
            supported_source_role_counts=(
                1,
            ),
            supported_target_role_counts=(
                2,
            ),
            unit_type=(
                StructuralUnitType.DEFENSIVE
            ),
            formation_general=False,
            requires_single_source_role=True,
        ),
        source_must_be_more_advanced=True,
    ),

    StructuralRule(
        rule_id="dm_supports_cm_v1",
        source_roles=("DM",),
        target_roles=("CM",),
        responsibility_type=(
            ResponsibilityType.SUPPORT
        ),
        required_positional_types=(
            ResponsibilityType.ADJACENT_LINE,
        ),
        status=(
            StructuralHypothesisStatus.REVISION_REQUIRED
        ),
        scope=StructuralHypothesisScope(
            supported_formations=(
                "4-3-3",
            ),
            supported_source_role_counts=(
                1,
            ),
            supported_target_role_counts=(
                2,
            ),
            unit_type=(
                StructuralUnitType.MIDFIELD
            ),
            formation_general=False,
            requires_single_source_role=True,
        ),
        source_must_be_deeper=True,
    ),

    StructuralRule(
        rule_id="cb_covers_fb_v1",
        source_roles=("CB",),
        target_roles=("FB",),
        responsibility_type=(
            ResponsibilityType.COVERAGE
        ),
        required_positional_types=(
            ResponsibilityType.SAME_LINE,
            ResponsibilityType.SAME_CORRIDOR,
        ),
        status=(
            StructuralHypothesisStatus.ACTIVE_DIAGNOSTIC
        ),
        scope=StructuralHypothesisScope(
            supported_formations=(
                "4-2-3-1",
                "4-3-3",
            ),
            supported_source_role_counts=(
                2,
            ),
            supported_target_role_counts=(
                2,
            ),
            unit_type=(
                StructuralUnitType.DEFENSIVE
            ),
            formation_general=True,
        ),
        same_broad_corridor_required=True,
    ),

    StructuralRule(
        rule_id="cm_supports_w_v1",
        source_roles=("CM",),
        target_roles=("W",),
        responsibility_type=(
            ResponsibilityType.SUPPORT
        ),
        required_positional_types=(
            ResponsibilityType.ADJACENT_LINE,
            ResponsibilityType.SAME_CORRIDOR,
        ),
        status=(
            StructuralHypothesisStatus.REVISION_REQUIRED
        ),
        scope=StructuralHypothesisScope(
            supported_formations=(
                "4-3-3",
            ),
            supported_source_role_counts=(
                2,
            ),
            supported_target_role_counts=(
                2,
            ),
            unit_type=(
                StructuralUnitType.CROSS_UNIT
            ),
            formation_general=False,
        ),
        source_must_be_deeper=True,
        same_broad_corridor_required=True,
    ),

    StructuralRule(
        rule_id="dm_connects_cb_cm_v1",
        source_roles=("DM",),
        target_roles=("CM",),
        responsibility_type=(
            ResponsibilityType.CONNECTION
        ),
        required_positional_types=(
            ResponsibilityType.ADJACENT_LINE,
        ),
        status=(
            StructuralHypothesisStatus.DEFERRED
        ),
        scope=StructuralHypothesisScope(
            supported_formations=(
                "4-3-3",
            ),
            supported_source_role_counts=(
                1,
            ),
            supported_target_role_counts=(
                2,
            ),
            unit_type=(
                StructuralUnitType.CROSS_UNIT
            ),
            formation_general=False,
            requires_single_source_role=True,
        ),
        source_must_be_deeper=True,
    ),
)


STRUCTURAL_RULES = INITIAL_STRUCTURAL_HYPOTHESES

def role_population_by_type(
    starting_xi: StartingXI,
) -> dict[str, int]:
    population: dict[str, int] = {}

    for assignment in starting_xi.assignments:
        role = assignment.tactical_role

        population[role] = (
            population.get(
                role,
                0,
            )
            + 1
        )

    return population

def hypothesis_scope_matches(
    *,
    rule: StructuralRule,
    starting_xi: StartingXI,
) -> bool:
    """
    Determine whether a structural hypothesis is declared applicable
    to the current formation and role population.

    This checks scope metadata only. It does not evaluate positional
    evidence or create relationships.
    """

    if (
        starting_xi.formation
        not in rule.scope.supported_formations
    ):
        return False

    role_population = role_population_by_type(
        starting_xi
    )

    source_count = sum(
        role_population.get(
            role,
            0,
        )
        for role in rule.source_roles
    )

    target_count = sum(
        role_population.get(
            role,
            0,
        )
        for role in rule.target_roles
    )

    if source_count not in (
        rule.scope
        .supported_source_role_counts
    ):
        return False

    if target_count not in (
        rule.scope
        .supported_target_role_counts
    ):
        return False

    if (
        rule.scope.requires_single_source_role
        and source_count != 1
    ):
        return False

    if (
        rule.scope.requires_multiple_source_roles
        and source_count < 2
    ):
        return False

    return True

def positional_types_by_pair(
    positional_set: PositionalResponsibilitySet,
) -> dict[
    frozenset[str],
    set[ResponsibilityType],
]:
    mapping: dict[
        frozenset[str],
        set[ResponsibilityType],
    ] = {}

    for relationship in (
        positional_set.responsibilities
    ):
        key = frozenset(
            {
                relationship.source_slot,
                relationship.target_slot,
            }
        )

        mapping.setdefault(
            key,
            set(),
        ).add(
            relationship.responsibility_type
        )

    return mapping

def rule_matches(
    *,
    rule: StructuralRule,
    source_slot: str,
    source_role: str,
    source_y: float,
    target_slot: str,
    target_role: str,
    target_y: float,
    positional_types: set[
        ResponsibilityType
    ],
) -> bool:
    if source_role not in rule.source_roles:
        return False

    if target_role not in rule.target_roles:
        return False

    if not set(
        rule.required_positional_types
    ).issubset(
        positional_types
    ):
        return False

    if (
        rule.source_must_be_deeper
        and not source_y < target_y
    ):
        return False

    if (
        rule.source_must_be_more_advanced
        and not source_y > target_y
    ):
        return False

    return True

def generate_structural_responsibilities(
    *,
    starting_xi: StartingXI,
    geometry: FormationGeometry,
    positional_set: PositionalResponsibilitySet,
    included_statuses: tuple[
        StructuralHypothesisStatus,
        ...,
    ] | None = None,
    enforce_hypothesis_scope: bool = False,
) -> StructuralResponsibilitySet:

    if included_statuses is None:
        allowed_statuses = set(
            StructuralHypothesisStatus
        )
    else:
        allowed_statuses = set(
            included_statuses
        )

        invalid_statuses = [
            status
            for status in allowed_statuses
            if not isinstance(
                status,
                StructuralHypothesisStatus,
            )
        ]

        if invalid_statuses:
            raise TypeError(
                "included_statuses must contain only "
                "StructuralHypothesisStatus values."
            )

    if not isinstance(
        enforce_hypothesis_scope,
        bool,
    ):
        raise TypeError(
            "enforce_hypothesis_scope must be a boolean."
        )
    
    if not starting_xi.assignments:
        raise ValueError(
            "Structural generation requires preserved "
            "lineup assignments."
        )

    if (
        starting_xi.formation
        != geometry.formation
    ):
        raise ValueError(
            "Starting-XI formation does not match geometry."
        )

    if (
        positional_set.formation
        != starting_xi.formation
    ):
        raise ValueError(
            "Positional responsibility formation does not "
            "match the starting XI."
        )

    if (
        positional_set.national_team
        != starting_xi.national_team
    ):
        raise ValueError(
            "Positional responsibility team does not match "
            "the starting XI."
        )

    assignments_by_slot = {
        assignment.slot: assignment
        for assignment
        in starting_xi.assignments
    }

    positions_by_slot = {
        position.slot: position
        for position in geometry.positions
    }

    pair_evidence = positional_types_by_pair(
        positional_set
    )

    responsibilities: list[
        StructuralResponsibility
    ] = []

    for source_slot, source_assignment in (
        assignments_by_slot.items()
    ):
        source_position = positions_by_slot[
            source_slot
        ]

        for target_slot, target_assignment in (
            assignments_by_slot.items()
        ):
            if source_slot == target_slot:
                continue

            target_position = positions_by_slot[
                target_slot
            ]

            pair_key = frozenset(
                {
                    source_slot,
                    target_slot,
                }
            )

            positional_types = pair_evidence.get(
                pair_key,
                set(),
            )

            for rule in INITIAL_STRUCTURAL_HYPOTHESES:
                if rule.status not in allowed_statuses:
                    continue

                if (
                    enforce_hypothesis_scope
                    and not hypothesis_scope_matches(
                        rule=rule,
                        starting_xi=starting_xi,
                    )
                ):
                    continue

                if not rule_matches(
                    rule=rule,
                    source_slot=source_slot,
                    source_role=(
                        source_assignment
                        .tactical_role
                    ),
                    source_y=source_position.y,
                    target_slot=target_slot,
                    target_role=(
                        target_assignment
                        .tactical_role
                    ),
                    target_y=target_position.y,
                    positional_types=positional_types,
                ):
                    continue

                responsibilities.append(
                    StructuralResponsibility(
                        source_slot=source_slot,
                        target_slot=target_slot,
                        source_role=(
                            source_assignment
                            .tactical_role
                        ),
                        target_role=(
                            target_assignment
                            .tactical_role
                        ),
                        responsibility_type=(
                            rule.responsibility_type
                        ),
                        rule_id=rule.rule_id,
                        hypothesis_status=rule.status,
                        supporting_positional_types=tuple(
                            sorted(
                                rule.required_positional_types,
                                key=lambda item:
                                    item.value,
                            )
                        ),
                    )
                )

    ordered = tuple(
        sorted(
            responsibilities,
            key=lambda item:
                item.canonical_key,
        )
    )

    return StructuralResponsibilitySet(
        national_team=starting_xi.national_team,
        formation=starting_xi.formation,
        responsibilities=ordered,
    )


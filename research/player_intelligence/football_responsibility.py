#football_responsibility

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResponsibilityFamily(str, Enum):
    """
    High-level family of a football relationship.
    """

    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    POSITIONAL = "positional"
    EMERGENT = "emergent"


class ResponsibilityType(str, Enum):
    """
    Stable vocabulary for tactical relationships.

    Study 102B defines semantics only. It does not infer or
    instantiate relationships between lineup assignments.
    """

    # Structural responsibilities
    SUPPORT = "support"
    PROTECTION = "protection"
    COVERAGE = "coverage"
    CONNECTION = "connection"

    # Functional responsibilities
    PROGRESSION = "progression"
    CREATION = "creation"
    FINISHING_SUPPORT = "finishing_support"
    PRESSING_SUPPORT = "pressing_support"

    # Positional descriptions
    SAME_CORRIDOR = "same_corridor"
    SAME_LINE = "same_line"
    ADJACENT_LINE = "adjacent_line"

    # Emergent relationships are acknowledged but not generated.
    CHEMISTRY = "chemistry"
    FAMILIARITY = "familiarity"
    LEADERSHIP = "leadership"
    COMMUNICATION = "communication"
    TRUST = "trust"
    SYNCHRONIZATION = "synchronization"


@dataclass(frozen=True)
class ResponsibilityDefinition:
    """
    Semantic definition of one football relationship type.

    This object describes what a relationship means. It does not
    assert that the relationship exists between any two players.
    """

    responsibility_type: ResponsibilityType
    family: ResponsibilityFamily

    definition: str

    directional: bool
    symmetric: bool

    structural: bool
    style_dependent: bool
    directly_observable: bool
    stable_across_matches: bool

    generation_enabled: bool

    def __post_init__(self) -> None:
        if not self.definition.strip():
            raise ValueError(
                "Responsibility definition must not be empty."
            )

        if self.directional and self.symmetric:
            raise ValueError(
                "A directional responsibility cannot also be "
                "declared symmetric."
            )

        if (
            self.family == ResponsibilityFamily.EMERGENT
            and self.generation_enabled
        ):
            raise ValueError(
                "Emergent responsibilities must remain disabled "
                "until observable proxies are defined."
            )


RESPONSIBILITY_DEFINITIONS = (
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.SUPPORT,
        family=ResponsibilityFamily.STRUCTURAL,
        definition=(
            "One tactical position provides an option that enables "
            "another position to perform its role more effectively."
        ),
        directional=True,
        symmetric=False,
        structural=True,
        style_dependent=False,
        directly_observable=False,
        stable_across_matches=True,
        generation_enabled=True,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.PROTECTION,
        family=ResponsibilityFamily.STRUCTURAL,
        definition=(
            "One tactical position reduces defensive exposure for "
            "another position."
        ),
        directional=True,
        symmetric=False,
        structural=True,
        style_dependent=False,
        directly_observable=False,
        stable_across_matches=True,
        generation_enabled=True,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.COVERAGE,
        family=ResponsibilityFamily.STRUCTURAL,
        definition=(
            "One tactical position can occupy or defend space that "
            "may be vacated by another position."
        ),
        directional=True,
        symmetric=False,
        structural=True,
        style_dependent=False,
        directly_observable=False,
        stable_across_matches=True,
        generation_enabled=True,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.CONNECTION,
        family=ResponsibilityFamily.STRUCTURAL,
        definition=(
            "One tactical position links two neighboring tactical "
            "units or lines."
        ),
        directional=True,
        symmetric=False,
        structural=True,
        style_dependent=False,
        directly_observable=False,
        stable_across_matches=True,
        generation_enabled=True,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.PROGRESSION,
        family=ResponsibilityFamily.FUNCTIONAL,
        definition=(
            "One player or position helps advance possession toward "
            "a more dangerous tactical area."
        ),
        directional=True,
        symmetric=False,
        structural=False,
        style_dependent=True,
        directly_observable=True,
        stable_across_matches=False,
        generation_enabled=False,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.CREATION,
        family=ResponsibilityFamily.FUNCTIONAL,
        definition=(
            "One player increases another player's opportunity to "
            "create or finish an attacking action."
        ),
        directional=True,
        symmetric=False,
        structural=False,
        style_dependent=True,
        directly_observable=True,
        stable_across_matches=False,
        generation_enabled=False,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.FINISHING_SUPPORT,
        family=ResponsibilityFamily.FUNCTIONAL,
        definition=(
            "One player supplies, occupies space for, or otherwise "
            "supports another player's finishing opportunities."
        ),
        directional=True,
        symmetric=False,
        structural=False,
        style_dependent=True,
        directly_observable=True,
        stable_across_matches=False,
        generation_enabled=False,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.PRESSING_SUPPORT,
        family=ResponsibilityFamily.FUNCTIONAL,
        definition=(
            "One player coordinates pressure with another player "
            "during defensive pressing."
        ),
        directional=False,
        symmetric=True,
        structural=False,
        style_dependent=True,
        directly_observable=True,
        stable_across_matches=False,
        generation_enabled=False,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.SAME_CORRIDOR,
        family=ResponsibilityFamily.POSITIONAL,
        definition=(
            "Two tactical positions occupy the same broad vertical "
            "corridor in the formation geometry."
        ),
        directional=False,
        symmetric=True,
        structural=True,
        style_dependent=False,
        directly_observable=True,
        stable_across_matches=True,
        generation_enabled=True,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.SAME_LINE,
        family=ResponsibilityFamily.POSITIONAL,
        definition=(
            "Two tactical positions belong to the same tactical line."
        ),
        directional=False,
        symmetric=True,
        structural=True,
        style_dependent=False,
        directly_observable=True,
        stable_across_matches=True,
        generation_enabled=True,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.ADJACENT_LINE,
        family=ResponsibilityFamily.POSITIONAL,
        definition=(
            "Two tactical positions belong to neighboring tactical "
            "lines."
        ),
        directional=False,
        symmetric=True,
        structural=True,
        style_dependent=False,
        directly_observable=True,
        stable_across_matches=True,
        generation_enabled=True,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.CHEMISTRY,
        family=ResponsibilityFamily.EMERGENT,
        definition=(
            "A latent relationship reflecting how effectively two "
            "players function together beyond their measured roles."
        ),
        directional=False,
        symmetric=True,
        structural=False,
        style_dependent=True,
        directly_observable=False,
        stable_across_matches=False,
        generation_enabled=False,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.FAMILIARITY,
        family=ResponsibilityFamily.EMERGENT,
        definition=(
            "A relationship associated with repeated shared playing "
            "experience."
        ),
        directional=False,
        symmetric=True,
        structural=False,
        style_dependent=False,
        directly_observable=False,
        stable_across_matches=True,
        generation_enabled=False,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.LEADERSHIP,
        family=ResponsibilityFamily.EMERGENT,
        definition=(
            "A latent directional influence associated with guidance, "
            "organization, or decision-making."
        ),
        directional=True,
        symmetric=False,
        structural=False,
        style_dependent=True,
        directly_observable=False,
        stable_across_matches=False,
        generation_enabled=False,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.COMMUNICATION,
        family=ResponsibilityFamily.EMERGENT,
        definition=(
            "A latent relationship reflecting information exchange "
            "between players."
        ),
        directional=False,
        symmetric=True,
        structural=False,
        style_dependent=False,
        directly_observable=False,
        stable_across_matches=False,
        generation_enabled=False,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.TRUST,
        family=ResponsibilityFamily.EMERGENT,
        definition=(
            "A latent relationship affecting how confidently players "
            "rely upon one another."
        ),
        directional=False,
        symmetric=True,
        structural=False,
        style_dependent=False,
        directly_observable=False,
        stable_across_matches=False,
        generation_enabled=False,
    ),
    ResponsibilityDefinition(
        responsibility_type=ResponsibilityType.SYNCHRONIZATION,
        family=ResponsibilityFamily.EMERGENT,
        definition=(
            "A latent relationship reflecting coordinated timing and "
            "movement."
        ),
        directional=False,
        symmetric=True,
        structural=False,
        style_dependent=True,
        directly_observable=False,
        stable_across_matches=False,
        generation_enabled=False,
    ),
)


def responsibility_definition(
    responsibility_type: ResponsibilityType,
) -> ResponsibilityDefinition:
    for definition in RESPONSIBILITY_DEFINITIONS:
        if (
            definition.responsibility_type
            == responsibility_type
        ):
            return definition

    raise KeyError(
        "No definition registered for responsibility type "
        f"{responsibility_type.value!r}."
    )


def enabled_responsibility_types() -> tuple[
    ResponsibilityType,
    ...,
]:
    return tuple(
        definition.responsibility_type
        for definition in RESPONSIBILITY_DEFINITIONS
        if definition.generation_enabled
    )
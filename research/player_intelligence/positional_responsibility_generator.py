#positional_responsibility_generator

from __future__ import annotations

from dataclasses import dataclass

from research.player_intelligence.football_responsibility import (
    ResponsibilityType,
    responsibility_definition,
)
from research.player_intelligence.formation_geometry import (
    FormationGeometry,
    FormationPosition,
)
from research.player_intelligence.player_schema import (
    StartingXI,
)


POSITIONAL_RESPONSIBILITY_TYPES = (
    ResponsibilityType.SAME_LINE,
    ResponsibilityType.ADJACENT_LINE,
    ResponsibilityType.SAME_CORRIDOR,
)


@dataclass(frozen=True)
class FootballResponsibility:
    """
    One instantiated relationship between two lineup assignments.

    Study 102C supports symmetric positional descriptions only.
    The canonical source/target ordering prevents duplicate
    undirected relationships.
    """

    source_slot: str
    target_slot: str
    responsibility_type: ResponsibilityType

    def __post_init__(self) -> None:
        if not self.source_slot.strip():
            raise ValueError(
                "Responsibility source slot must not be empty."
            )

        if not self.target_slot.strip():
            raise ValueError(
                "Responsibility target slot must not be empty."
            )

        if self.source_slot == self.target_slot:
            raise ValueError(
                "A positional responsibility cannot connect "
                "a slot to itself."
            )

        definition = responsibility_definition(
            self.responsibility_type
        )

        if not definition.generation_enabled:
            raise ValueError(
                "Responsibility type is not enabled for "
                f"generation: {self.responsibility_type.value!r}."
            )

        if not definition.symmetric:
            raise ValueError(
                "Study 102C generates symmetric positional "
                "responsibilities only."
            )

        if self.source_slot > self.target_slot:
            raise ValueError(
                "Symmetric responsibilities must use canonical "
                "lexicographic slot ordering."
            )

    @property
    def canonical_key(
        self,
    ) -> tuple[str, str, str]:
        return (
            self.source_slot,
            self.target_slot,
            self.responsibility_type.value,
        )


@dataclass(frozen=True)
class PositionalResponsibilitySet:
    """
    Deterministic positional relationships for one starting XI.

    This is not yet a FootballGraph. It contains no node objects,
    edge weights, interaction scores, or strength adjustments.
    """

    national_team: str
    formation: str

    lineup_slots: tuple[str, ...]

    responsibilities: tuple[
        FootballResponsibility,
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

        if not self.lineup_slots:
            raise ValueError(
                "Responsibility set must contain lineup slots."
            )

        if len(self.lineup_slots) != len(
            set(self.lineup_slots)
        ):
            raise ValueError(
                "Responsibility set contains duplicate lineup slots."
            )

        valid_slots = set(
            self.lineup_slots
        )

        keys: list[
            tuple[str, str, str]
        ] = []

        for responsibility in self.responsibilities:
            if (
                responsibility.source_slot
                not in valid_slots
            ):
                raise ValueError(
                    "Responsibility references an unknown "
                    f"source slot: "
                    f"{responsibility.source_slot!r}."
                )

            if (
                responsibility.target_slot
                not in valid_slots
            ):
                raise ValueError(
                    "Responsibility references an unknown "
                    f"target slot: "
                    f"{responsibility.target_slot!r}."
                )

            keys.append(
                responsibility.canonical_key
            )

        if len(keys) != len(set(keys)):
            raise ValueError(
                "Responsibility set contains duplicate "
                "relationships."
            )

        if tuple(keys) != tuple(
            sorted(keys)
        ):
            raise ValueError(
                "Responsibilities must be stored in "
                "deterministic canonical order."
            )

    def relationships_of_type(
        self,
        responsibility_type: ResponsibilityType,
    ) -> tuple[
        FootballResponsibility,
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


def broad_corridor(
    position: FormationPosition,
) -> str:
    """
    Map the geometry's detailed side label into one broad corridor.

    This preserves the current geometry vocabulary while treating
    left and left-center as one broad corridor, and likewise on
    the right.
    """

    normalized = (
        position.side
        .strip()
        .lower()
    )

    mapping = {
        "left": "left",
        "left_center": "left",
        "center": "center",
        "right_center": "right",
        "right": "right",
    }

    try:
        return mapping[
            normalized
        ]

    except KeyError as exc:
        raise ValueError(
            "Unsupported formation side for corridor "
            f"generation: {position.side!r}."
        ) from exc


def canonical_slot_pair(
    slot_a: str,
    slot_b: str,
) -> tuple[str, str]:
    if slot_a == slot_b:
        raise ValueError(
            "Cannot create a relationship from one slot "
            "to itself."
        )

    return tuple(
        sorted(
            (
                slot_a,
                slot_b,
            )
        )
    )


def positional_relationship_types(
    position_a: FormationPosition,
    position_b: FormationPosition,
) -> tuple[
    ResponsibilityType,
    ...,
]:
    """
    Return every positional description implied by two positions.

    Multiple descriptions may apply to the same pair. For example,
    two center-backs can be both in the same tactical line and in
    different broad corridors.
    """

    relationship_types: list[
        ResponsibilityType
    ] = []

    if (
        position_a.tactical_line
        == position_b.tactical_line
    ):
        relationship_types.append(
            ResponsibilityType.SAME_LINE
        )

    if (
        abs(
            position_a.tactical_line
            - position_b.tactical_line
        )
        == 1
    ):
        relationship_types.append(
            ResponsibilityType.ADJACENT_LINE
        )

    if (
        broad_corridor(position_a)
        == broad_corridor(position_b)
    ):
        relationship_types.append(
            ResponsibilityType.SAME_CORRIDOR
        )

    return tuple(
        relationship_types
    )


def validate_lineup_against_geometry(
    *,
    starting_xi: StartingXI,
    geometry: FormationGeometry,
) -> None:
    if not starting_xi.assignments:
        raise ValueError(
            "Positional responsibility generation requires "
            "preserved lineup assignments."
        )

    if (
        starting_xi.formation
        != geometry.formation
    ):
        raise ValueError(
            "Starting-XI formation does not match geometry. "
            f"Lineup={starting_xi.formation!r}, "
            f"geometry={geometry.formation!r}."
        )

    lineup_mapping = {
        assignment.slot:
            assignment.tactical_role
        for assignment in starting_xi.assignments
    }

    geometry_mapping = {
        position.slot:
            position.role
        for position in geometry.positions
    }

    if set(lineup_mapping) != set(
        geometry_mapping
    ):
        missing_geometry = sorted(
            set(lineup_mapping)
            - set(geometry_mapping)
        )

        unused_geometry = sorted(
            set(geometry_mapping)
            - set(lineup_mapping)
        )

        raise AssertionError(
            "Starting XI and geometry have different slot "
            "populations. "
            f"Missing geometry={missing_geometry}, "
            f"unused geometry={unused_geometry}"
        )

    role_mismatches = [
        {
            "slot": slot,
            "lineup_role":
                lineup_mapping[slot],
            "geometry_role":
                geometry_mapping[slot],
        }
        for slot in lineup_mapping
        if (
            lineup_mapping[slot]
            != geometry_mapping[slot]
        )
    ]

    if role_mismatches:
        raise AssertionError(
            "Starting-XI tactical roles differ from "
            "geometry roles: "
            f"{role_mismatches}"
        )


def generate_positional_responsibilities(
    *,
    starting_xi: StartingXI,
    geometry: FormationGeometry,
) -> PositionalResponsibilitySet:
    """
    Generate deterministic positional relationships.

    This function uses only formation assignment and geometry.
    Player identity, ability, contribution values, club, and
    competition do not affect the output.
    """

    validate_lineup_against_geometry(
        starting_xi=starting_xi,
        geometry=geometry,
    )

    positions_by_slot = {
        position.slot: position
        for position in geometry.positions
    }

    lineup_slots = tuple(
        assignment.slot
        for assignment
        in starting_xi.assignments
    )

    responsibilities: list[
        FootballResponsibility
    ] = []

    for first_index, slot_a in enumerate(
        lineup_slots
    ):
        for slot_b in lineup_slots[
            first_index + 1:
        ]:
            position_a = positions_by_slot[
                slot_a
            ]

            position_b = positions_by_slot[
                slot_b
            ]

            source_slot, target_slot = (
                canonical_slot_pair(
                    slot_a,
                    slot_b,
                )
            )

            for responsibility_type in (
                positional_relationship_types(
                    position_a,
                    position_b,
                )
            ):
                responsibilities.append(
                    FootballResponsibility(
                        source_slot=source_slot,
                        target_slot=target_slot,
                        responsibility_type=(
                            responsibility_type
                        ),
                    )
                )

    ordered = tuple(
        sorted(
            responsibilities,
            key=lambda relationship:
                relationship.canonical_key,
        )
    )

    return PositionalResponsibilitySet(
        national_team=(
            starting_xi.national_team
        ),
        formation=starting_xi.formation,
        lineup_slots=lineup_slots,
        responsibilities=ordered,
    )
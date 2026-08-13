#test_football_responsibility

from __future__ import annotations

import pytest

from research.player_intelligence.football_responsibility import (
    RESPONSIBILITY_DEFINITIONS,
    ResponsibilityDefinition,
    ResponsibilityFamily,
    ResponsibilityType,
    enabled_responsibility_types,
    responsibility_definition,
)


def test_every_responsibility_type_has_one_definition() -> None:
    registered = [
        definition.responsibility_type
        for definition in RESPONSIBILITY_DEFINITIONS
    ]

    assert len(registered) == len(
        set(registered)
    )

    assert set(registered) == set(
        ResponsibilityType
    )


def test_directional_relationship_cannot_be_symmetric() -> None:
    with pytest.raises(
        ValueError,
        match="cannot also be declared symmetric",
    ):
        ResponsibilityDefinition(
            responsibility_type=(
                ResponsibilityType.SUPPORT
            ),
            family=(
                ResponsibilityFamily.STRUCTURAL
            ),
            definition="Invalid test relationship.",
            directional=True,
            symmetric=True,
            structural=True,
            style_dependent=False,
            directly_observable=False,
            stable_across_matches=True,
            generation_enabled=True,
        )


def test_emergent_relationship_cannot_be_enabled() -> None:
    with pytest.raises(
        ValueError,
        match="Emergent responsibilities",
    ):
        ResponsibilityDefinition(
            responsibility_type=(
                ResponsibilityType.CHEMISTRY
            ),
            family=(
                ResponsibilityFamily.EMERGENT
            ),
            definition="Invalid generated chemistry.",
            directional=False,
            symmetric=True,
            structural=False,
            style_dependent=True,
            directly_observable=False,
            stable_across_matches=False,
            generation_enabled=True,
        )


def test_initial_enabled_vocabulary_is_structural_or_positional() -> None:
    enabled = enabled_responsibility_types()

    assert enabled

    for responsibility_type in enabled:
        definition = responsibility_definition(
            responsibility_type
        )

        assert definition.family in {
            ResponsibilityFamily.STRUCTURAL,
            ResponsibilityFamily.POSITIONAL,
        }


def test_human_factors_are_defined_but_disabled() -> None:
    for responsibility_type in (
        ResponsibilityType.CHEMISTRY,
        ResponsibilityType.FAMILIARITY,
        ResponsibilityType.LEADERSHIP,
        ResponsibilityType.COMMUNICATION,
        ResponsibilityType.TRUST,
        ResponsibilityType.SYNCHRONIZATION,
    ):
        definition = responsibility_definition(
            responsibility_type
        )

        assert (
            definition.family
            == ResponsibilityFamily.EMERGENT
        )

        assert (
            definition.generation_enabled
            is False
        )


def test_support_is_directional() -> None:
    definition = responsibility_definition(
        ResponsibilityType.SUPPORT
    )

    assert definition.directional is True
    assert definition.symmetric is False


def test_same_line_is_symmetric() -> None:
    definition = responsibility_definition(
        ResponsibilityType.SAME_LINE
    )

    assert definition.directional is False
    assert definition.symmetric is True
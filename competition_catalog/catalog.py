#catalog.py

from __future__ import annotations

from competition_catalog.competition_definition import CompetitionDefinition


class CompetitionCatalog:
    """
    Registry for static football competition definitions.
    """

    def __init__(self) -> None:
        self._definitions: dict[str, CompetitionDefinition] = {}

    def register(self, definition: CompetitionDefinition) -> None:
        self._definitions[definition.name] = definition

    def get(self, name: str) -> CompetitionDefinition:
        if name not in self._definitions:
            raise KeyError(f"Unknown competition definition: {name}")

        return self._definitions[name]

    def names(self) -> list[str]:
        return sorted(self._definitions.keys())
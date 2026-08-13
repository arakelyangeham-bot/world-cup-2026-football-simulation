#competition_builder.py

from __future__ import annotations

from competition_catalog.competition_definition import CompetitionDefinition
from simulation.competition import Competition, Stage, StageType


class CompetitionBuilder:
    """
    Converts static catalog definitions into executable Competition objects.

    Initial supported structure:
    - league stages

    More complex structures will be added only when required by real
    competition definitions.
    """

    def build(
        self,
        definition: CompetitionDefinition,
        participants: list[str],
    ) -> Competition:
        self._validate_participants(definition, participants)

        stages = [
            self._build_stage(
                stage_definition=stage_definition,
                participants=participants,
            )
            for stage_definition in definition.stages
        ]

        return Competition(
            name=definition.name,
            participants=participants,
            stages=stages,
            #metadata={
            #    "competition_type": definition.competition_type,
            #   "region": definition.region,
            #    "governing_body": definition.governing_body,
            #    "catalog_definition": True,
            #    **definition.metadata,
            #},
        )

    def _build_stage(
        self,
        stage_definition,
        participants: list[str],
    ) -> Stage:
        if stage_definition.stage_type == "league":
            stage_type = StageType.LEAGUE
        else:
            raise NotImplementedError(
                "CompetitionBuilder does not yet support stage type "
                f"{stage_definition.stage_type!r}."
            )

        return Stage(
            name=stage_definition.name,
            stage_type=stage_type,
            participants=participants,
            matches=[],
            metadata={
                "competition_format": stage_definition.competition_format,
                **stage_definition.metadata,
            },
        )

    @staticmethod
    def _validate_participants(
        definition: CompetitionDefinition,
        participants: list[str],
    ) -> None:
        if not participants:
            raise ValueError(
                f"{definition.name} cannot be built without participants."
            )

        if len(participants) != len(set(participants)):
            raise ValueError(
                f"{definition.name} contains duplicate participants."
            )

        expected_count = definition.participant_count

        if expected_count is not None and len(participants) != expected_count:
            raise ValueError(
                f"{definition.name} expects {expected_count} participants, "
                f"but received {len(participants)}."
            )
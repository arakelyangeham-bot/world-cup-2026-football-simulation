from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProductionRepositoryConfig:
    """
    Configuration for one production club-repository build.
    """

    competition_id: int
    competition_name: str
    season_id: str

    repository_version: str
    repository_scope: str

    output_path: Path

    representation_type: str = "full_squad"
    expected_club_count: int | None = None

    def validate(self) -> None:
        if self.competition_id <= 0:
            raise ValueError(
                "competition_id must be positive."
            )

        if not self.competition_name.strip():
            raise ValueError(
                "competition_name must not be empty."
            )

        if not self.season_id.strip():
            raise ValueError(
                "season_id must not be empty."
            )

        if not self.repository_version.strip():
            raise ValueError(
                "repository_version must not be empty."
            )

        if not self.repository_scope.strip():
            raise ValueError(
                "repository_scope must not be empty."
            )

        if not self.representation_type.strip():
            raise ValueError(
                "representation_type must not be empty."
            )

        if (
            self.expected_club_count is not None
            and self.expected_club_count <= 0
        ):
            raise ValueError(
                "expected_club_count must be positive when "
                "provided."
            )
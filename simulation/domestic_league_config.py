#domestic_league_config

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DomesticLeagueSimulationConfig:
    """
    Configuration required to run a domestic double-round-robin
    league through the shared simulation platform.
    """

    key: str
    competition_name: str
    season: str

    participant_count: int
    matches_per_team: int
    matchday_count: int
    fixture_count: int

    top_four_count: int
    top_six_count: int
    relegation_count: int

    repository_source: str
    rating_prediction_date: date

    fixture_path: Path
    repository_path: Path
    goal_model_path: Path
    output_directory: Path

    @property
    def display_name(self) -> str:
        return (
            f"{self.competition_name} "
            f"{self.season}"
        )

    def validate(self) -> None:
        if self.participant_count < 2:
            raise ValueError(
                "participant_count must be at least 2."
            )

        if self.matches_per_team != (
            2 * (self.participant_count - 1)
        ):
            raise ValueError(
                "matches_per_team is inconsistent "
                "with a double round robin."
            )

        if self.matchday_count != (
            self.matches_per_team
        ):
            raise ValueError(
                "matchday_count must equal "
                "matches_per_team for this "
                "double-round-robin configuration."
            )

        expected_fixture_count = (
            self.participant_count
            * self.matches_per_team
            // 2
        )

        if (
            self.fixture_count
            != expected_fixture_count
        ):
            raise ValueError(
                "fixture_count is inconsistent "
                "with participant_count and "
                "matches_per_team."
            )

        if not (
            1
            <= self.top_four_count
            <= self.participant_count
        ):
            raise ValueError(
                "Invalid top_four_count."
            )

        if not (
            self.top_four_count
            <= self.top_six_count
            <= self.participant_count
        ):
            raise ValueError(
                "Invalid top_six_count."
            )

        if not (
            1
            <= self.relegation_count
            < self.participant_count
        ):
            raise ValueError(
                "Invalid relegation_count."
            )
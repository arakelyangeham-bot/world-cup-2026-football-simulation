#competition_team_repository

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.player_intelligence.competition_roster_builder import (
    CompetitionRosterBuilder,
    CompetitionSquadContext,
)
from research.player_intelligence.starting_xi_builder import (
    StartingXIBuilder,
)
from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
    build_team_representation_from_squad,
    build_team_representation_from_starting_xi,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FORMATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "formation_manifest.csv"
)


class CompetitionTeamRepository:
    """
    Produces team representations within a competition-season context.

    This repository coordinates existing football-intelligence components:

        CompetitionRosterBuilder
            -> Squad
            -> StartingXIBuilder
            -> StartingXI
            -> TeamRepresentation

    It does not calculate player ratings, resolve memberships, fit prediction
    models, or generate scorelines.
    """

    def __init__(
        self,
        roster_builder: CompetitionRosterBuilder | None = None,
        formation_path: Path = DEFAULT_FORMATION_PATH,
    ) -> None:
        self.roster_builder = (
            roster_builder
            or CompetitionRosterBuilder()
        )
        self.formation_path = formation_path
        self._formation_manifest: pd.DataFrame | None = None

    def load_formation_manifest(self) -> pd.DataFrame:
        if self._formation_manifest is not None:
            return self._formation_manifest

        if not self.formation_path.exists():
            raise FileNotFoundError(
                "Formation manifest does not exist: "
                f"{self.formation_path}"
            )

        formation_manifest = pd.read_csv(
            self.formation_path
        )

        required_columns = {
            "slot",
            "role",
        }

        missing = (
            required_columns
            - set(formation_manifest.columns)
        )

        if missing:
            raise ValueError(
                "Formation manifest is missing required columns: "
                f"{sorted(missing)}"
            )

        if formation_manifest.empty:
            raise ValueError(
                "Formation manifest is empty."
            )

        self._formation_manifest = formation_manifest
        return formation_manifest

    def get_formation(
        self,
        formation: str,
    ) -> pd.DataFrame:
        manifest = self.load_formation_manifest()

        if "formation" in manifest.columns:
            selected = manifest[
                manifest["formation"]
                .astype(str)
                .eq(formation)
            ].copy()
        else:
            selected = manifest.copy()

        if selected.empty:
            raise KeyError(
                f"No formation rows found for {formation!r}."
            )

        if selected["slot"].duplicated().any():
            duplicate_slots = (
                selected.loc[
                    selected["slot"].duplicated(
                        keep=False
                    ),
                    "slot",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Formation {formation!r} contains "
                f"duplicate slots: {duplicate_slots}"
            )

        return selected.reset_index(
            drop=True
        )

    def get_context(
        self,
        competition_id: int,
        season_id: int,
        team_id: int,
    ) -> CompetitionSquadContext:
        return self.roster_builder.get_context(
            competition_id=competition_id,
            season_id=season_id,
            team_id=team_id,
        )

    def get_full_squad_representation(
        self,
        competition_id: int,
        season_id: int,
        team_id: int,
    ) -> TeamRepresentation:
        squad = self.roster_builder.get_squad(
            competition_id=competition_id,
            season_id=season_id,
            team_id=team_id,
            require_complete_join=True,
        )

        return build_team_representation_from_squad(
            squad
        )

    def get_starting_xi_representation(
        self,
        competition_id: int,
        season_id: int,
        team_id: int,
        *,
        formation: str = "4-3-3",
    ) -> TeamRepresentation:
        squad = self.roster_builder.get_squad(
            competition_id=competition_id,
            season_id=season_id,
            team_id=team_id,
            require_complete_join=True,
        )

        formation_df = self.get_formation(
            formation
        )

        lineup_builder = StartingXIBuilder(
            formation=formation
        )

        starting_xi = (
            lineup_builder.build_for_squad(
                squad=squad,
                formation_df=formation_df,
            )
        )

        return build_team_representation_from_starting_xi(
            starting_xi
        )

    def get_team_representation(
        self,
        competition_id: int,
        season_id: int,
        team_id: int,
        *,
        representation_type: str = "expected_starting_xi",
        formation: str = "4-3-3",
    ) -> TeamRepresentation:
        if representation_type == "full_squad":
            return self.get_full_squad_representation(
                competition_id=competition_id,
                season_id=season_id,
                team_id=team_id,
            )

        if representation_type == "expected_starting_xi":
            return self.get_starting_xi_representation(
                competition_id=competition_id,
                season_id=season_id,
                team_id=team_id,
                formation=formation,
            )

        raise ValueError(
            "Unknown representation_type: "
            f"{representation_type!r}. "
            "Expected 'full_squad' or "
            "'expected_starting_xi'."
        )
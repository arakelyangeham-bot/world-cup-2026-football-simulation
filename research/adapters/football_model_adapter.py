from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)
from scripts.club_team_repository_loader import (
    load_club_team_repository,
    load_legacy_club_team_repository,
)
from scripts.team_strength_loader import (
    REPOSITORY_PATHS,
    load_team_repository,
)
from simulation.integrated_club_goal_predictor import (
    IntegratedClubGoalPredictor,
    DEFAULT_CLUB_GOAL_MODEL_ARTIFACT,
)
from simulation.live_match_observation_builder import (
    DEFAULT_CLUB_REPOSITORY_PATH,
    DEFAULT_CLUBELO_NAME_OVERRIDES,
    LiveMatchObservationBuilder,
    ProductionClubRepository,
)
from simulation.match_engine_adapter import (
    simulate_match_score,
    simulate_scoreline_from_lambdas,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLUBELO_CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "clubelo_histories"
)


LEGACY_CLUB_REPOSITORY_SOURCES = {
    "premier_league_validation",
}


@dataclass
class FootballModel:
    """
    Executable football model used by experiments and
    competition simulations.

    Legacy repositories continue to use the historical match
    engine route. The production club repository uses the
    Integrated Club Goal Model v1.
    """

    repository_source: str
    match_engine: str
    team_repository: dict[str, dict]
    metadata: dict[str, Any]

    live_observation_builder: (
        LiveMatchObservationBuilder | None
    ) = None

    club_goal_predictor: (
        IntegratedClubGoalPredictor | None
    ) = None

    def simulate_match(
        self,
        team1: str,
        team2: str,
        prediction_date: (
            str | date | datetime | None
        ) = None,
    ) -> tuple[int, int]:
        if self.match_engine == (
            "integrated_club_goal_model_v1"
        ):
            return self._simulate_production_club_match(
                home_team=team1,
                away_team=team2,
                prediction_date=prediction_date,
            )

        return simulate_match_score(
            self.team_repository[team1],
            self.team_repository[team2],
        )

    def _simulate_production_club_match(
        self,
        home_team: str,
        away_team: str,
        prediction_date: (
            str | date | datetime | None
        ),
    ) -> tuple[int, int]:
        if prediction_date is None:
            raise ValueError(
                "Production club match simulation requires "
                "a prediction date."
            )

        if self.live_observation_builder is None:
            raise RuntimeError(
                "Production club football model has no "
                "LiveMatchObservationBuilder."
            )

        if self.club_goal_predictor is None:
            raise RuntimeError(
                "Production club football model has no "
                "IntegratedClubGoalPredictor."
            )

        observation = (
            self.live_observation_builder.build(
                home_team=home_team,
                away_team=away_team,
                prediction_date=prediction_date,
            )
        )

        prediction = (
            self.club_goal_predictor
            .predict_features(
                feature_values=(
                    observation
                    .to_feature_mapping()
                ),
                prediction_date=(
                    observation.prediction_date
                ),
            )
        )

        return simulate_scoreline_from_lambdas(
            lambda_home=prediction.lambda_home,
            lambda_away=prediction.lambda_away,
        )


class FootballModelAdapter:
    """
    Resolve an ExperimentCondition into an executable
    football model.
    """

    def from_condition(
        self,
        condition,
    ) -> FootballModel:
        repository_source = (
            condition.repository_source
        )

        if repository_source == "premier_league_production_v1":
            return self._build_production_club_model(
                condition,
                repository_path=DEFAULT_CLUB_REPOSITORY_PATH,
                artifact_path=DEFAULT_CLUB_GOAL_MODEL_ARTIFACT,
                repository_source_name="premier_league_production_v1",
            )

        if repository_source == "premier_league_robust_candidate_v1":
            return self._build_production_club_model(
                condition,
                repository_path=(
                    PROJECT_ROOT
                    / "outputs"
                    / "study_096_premier_league_robust_candidate"
                    / "study_096c_runtime_repository"
                    / "premier_league_club_repository_robust_zscore.csv"
                ),
                artifact_path=(
                    PROJECT_ROOT
                    / "outputs"
                    / "study_096_premier_league_robust_candidate"
                    / "study_096d_goal_model"
                    / "integrated_club_goal_model_robust_candidate.json"
                ),
                repository_source_name="premier_league_robust_candidate_v1",
            )

        if repository_source not in REPOSITORY_PATHS:
            raise ValueError(
                "Unknown repository source for football "
                f"model: {repository_source}"
            )

        repository_path: Path = (
            REPOSITORY_PATHS[
                repository_source
            ]
        )

        if (
            repository_source
            in LEGACY_CLUB_REPOSITORY_SOURCES
        ):
            team_repository = (
                load_legacy_club_team_repository(
                    path=repository_path,
                )
            )
        else:
            team_repository = (
                load_team_repository(
                    path=repository_path,
                )
            )

        return FootballModel(
            repository_source=(
                repository_source
            ),
            match_engine=(
                condition.match_engine
            ),
            team_repository=(
                team_repository
            ),
            metadata={
                "condition": condition.name,
                "competition_format":
                    condition.competition_format,
                "simulation_count":
                    condition.simulation_count,
                "random_seed":
                    condition.random_seed,
                "repository_path":
                    str(repository_path),
                **condition.parameters,
            },
        )

    def _build_production_club_model(
        self,
        condition,
        *,
        repository_path: Path,
        artifact_path: Path,
        repository_source_name: str,
    ) -> FootballModel:
        production_repository = ProductionClubRepository(
            repository_path
        )

        clubelo_repository = (
            ClubEloRepository(
                cache_directory=(
                    CLUBELO_CACHE_DIRECTORY
                )
            )
        )

        observation_builder = (
            LiveMatchObservationBuilder(
                club_repository=production_repository,
                clubelo_repository=clubelo_repository,
                clubelo_name_overrides=(
                    DEFAULT_CLUBELO_NAME_OVERRIDES
                ),
           )
        )


        predictor = IntegratedClubGoalPredictor(
            artifact_path=artifact_path
        )

        # Preserve a dictionary-like repository for code
        # that inspects the model's available team names.
        team_repository = {
            club: {
                "team": club,
            }
            for club in (
                production_repository
                .list_clubs()
            )
        }

        return FootballModel(
            repository_source=repository_source_name,
            match_engine=(
                "integrated_club_goal_model_v1"
            ),
            team_repository=(
                team_repository
            ),
            live_observation_builder=(
                observation_builder
            ),
            club_goal_predictor=predictor,
            metadata={
                "condition": condition.name,
                "competition_format":
                    condition.competition_format,
                "simulation_count":
                    condition.simulation_count,
                "random_seed":
                    condition.random_seed,
                "clubelo_cache_directory": str(
                    CLUBELO_CACHE_DIRECTORY
                ),
                "repository_path": str(repository_path),
                "production_artifact": str(artifact_path),
                "production_baseline_version": (
                    predictor.model
                    .baseline_version
                ),
                **condition.parameters,
            },
        )
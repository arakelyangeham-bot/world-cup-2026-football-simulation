#production_prediction_pipeline_factory

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from simulation.production_goal_model import (
    ProductionGoalModel,
)
from research.production.production_prediction_pipeline import (
    ProductionPredictionPipeline,
)
from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)

# Adjust only if this module lives elsewhere.
from simulation.live_match_observation_builder import (
    LiveMatchObservationBuilder,
    ProductionClubRepository,
)


def build_production_prediction_pipeline(
    *,
    club_repository_path: Path,
    clubelo_cache_directory: Path,
    goal_model_artifact_path: Path,
    clubelo_name_overrides: (
        Mapping[str, str] | None
    ) = None,
) -> ProductionPredictionPipeline:
    """
    Construct the complete deterministic production prediction
    pipeline from persisted runtime artifacts.
    """

    club_repository = ProductionClubRepository(
        repository_path=club_repository_path
    )

    clubelo_repository = ClubEloRepository(
        cache_directory=clubelo_cache_directory
    )

    observation_builder = LiveMatchObservationBuilder(
        club_repository=club_repository,
        clubelo_repository=clubelo_repository,
        clubelo_name_overrides=(
            clubelo_name_overrides
        ),
    )

    goal_model = ProductionGoalModel.from_path(
        goal_model_artifact_path
    )

    return ProductionPredictionPipeline(
        observation_builder=observation_builder,
        goal_model=goal_model,
    )
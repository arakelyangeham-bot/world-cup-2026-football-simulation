#test_club_team_repository_loader

from pathlib import Path

from scripts.club_team_repository_loader import (
    load_legacy_club_team_repository,
)

from research import ExperimentCondition
from research.adapters import (
    FootballModelAdapter,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


def test_load_legacy_premier_league_repository() -> None:
    path = (
        PROJECT_ROOT
        / "data"
        / "team_repositories"
        / "premier_league_validation_repository.csv"
    )

    repository = (
        load_legacy_club_team_repository(
            path
        )
    )

    assert len(repository) == 20

    arsenal = repository[
        "Arsenal"
    ]

    assert arsenal[
        "poisson_attack"
    ] > 0.0

    assert arsenal[
        "poisson_defense"
    ] > 0.0

    assert arsenal[
        "rating_prior"
    ] > 0.0

    assert (
        arsenal[
            "rating_prior_source"
        ]
        == "opta_power_rating"
    )


def test_adapter_builds_legacy_premier_league_model() -> None:
    condition = ExperimentCondition(
        name="Legacy club adapter test",
        competition_format=(
            "double_round_robin"
        ),
        repository_source=(
            "premier_league_validation"
        ),
        match_engine=(
            "production_scoreline_first"
        ),
        simulation_count=1,
        random_seed=1,
        parameters={},
    )

    model = (
        FootballModelAdapter()
        .from_condition(condition)
    )

    assert len(
        model.team_repository
    ) == 20

    assert (
        model.repository_source
        == "premier_league_validation"
    )

    assert (
        model.live_observation_builder
        is None
    )

    assert (
        model.club_goal_predictor
        is None
    )
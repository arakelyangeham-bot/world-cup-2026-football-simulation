#test_football_model_adapter

from __future__ import annotations

from research.adapters.football_model_adapter import (
    FootballModelAdapter,
)
from research.experiment_condition import (
    ExperimentCondition,
)
from simulation.live_match_observation_builder import (
    DEFAULT_CLUBELO_NAME_OVERRIDES,
)


def _condition(
    *,
    parameters: dict | None = None,
) -> ExperimentCondition:
    return ExperimentCondition(
        name="test",
        competition_format="league",
        repository_source="test",
        match_engine="integrated_club_goal_model_v1",
        simulation_count=1,
        random_seed=1,
        parameters=parameters or {},
    )


def test_clubelo_overrides_use_defaults_when_custom_mapping_absent():
    adapter = FootballModelAdapter()

    overrides = (
        adapter._resolve_clubelo_name_overrides(
            _condition()
        )
    )

    assert overrides == dict(
        DEFAULT_CLUBELO_NAME_OVERRIDES
    )


def test_clubelo_overrides_merge_custom_mapping_with_defaults():
    adapter = FootballModelAdapter()

    overrides = (
        adapter._resolve_clubelo_name_overrides(
            _condition(
                parameters={
                    "clubelo_name_overrides": {
                        "FC Barcelona": "Barcelona",
                        "FC Bayern München": "Bayern",
                    }
                }
            )
        )
    )

    assert overrides["FC Barcelona"] == "Barcelona"
    assert (
        overrides["FC Bayern München"]
        == "Bayern"
    )

    for club, lookup in (
        DEFAULT_CLUBELO_NAME_OVERRIDES.items()
    ):
        assert overrides[club] == lookup
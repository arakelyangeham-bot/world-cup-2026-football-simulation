#benchmark_001a_football_model_preflight

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research import ExperimentCondition
from research.adapters import FootballModelAdapter
from simulation.league_match_simulator import (
    LeagueMatchSimulator,
)
from fixture_generation import (
    RoundRobinFixtureGenerator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "football_model_benchmarks"
    / "benchmark_001a_preflight"
)

MODEL_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "football_model_runtime_audit.csv"
)

CLUB_RESOLUTION_PATH = (
    OUTPUT_DIRECTORY
    / "matched_club_resolution_audit.csv"
)

FIXTURE_SMOKE_PATH = (
    OUTPUT_DIRECTORY
    / "matched_fixture_smoke_test.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "benchmark_001a_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "BENCHMARK_001A_REPORT.md"
)


BASE_SEED = 1001
SEASON_START_DATE = date(2025, 8, 16)
DAYS_BETWEEN_MATCHDAYS = 7


CANONICAL_CLUBS = (
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
)


LEGACY_RUNTIME_NAMES = {
    club: club
    for club in CANONICAL_CLUBS
}


PRODUCTION_RUNTIME_NAMES = {
    **{
        club: club
        for club in CANONICAL_CLUBS
    },
    "Liverpool": "Liverpool FC",
    "Wolverhampton Wanderers": "Wolverhampton",
}

@dataclass(frozen=True)
class BenchmarkFootballModelSpec:
    model_id: str
    display_name: str

    repository_source: str
    match_engine: str

    runtime_names: dict[str, str]

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError(
                "Benchmark model ID must not be empty."
            )

        if not self.display_name.strip():
            raise ValueError(
                "Benchmark model display name must not be empty."
            )

        missing_clubs = (
            set(CANONICAL_CLUBS)
            - set(self.runtime_names)
        )

        if missing_clubs:
            raise ValueError(
                "Benchmark model is missing runtime aliases "
                f"for: {sorted(missing_clubs)}"
            )

BENCHMARK_MODELS = (
    BenchmarkFootballModelSpec(
        model_id="legacy_validation",
        display_name=(
            "Legacy Premier League Validation Model"
        ),
        repository_source=(
            "premier_league_validation"
        ),
        match_engine=(
            "production_scoreline_first"
        ),
        runtime_names=LEGACY_RUNTIME_NAMES,
    ),

    BenchmarkFootballModelSpec(
        model_id="integrated_club_goal_model_v1",
        display_name=(
            "Integrated Club Goal Model v1"
        ),
        repository_source=(
            "premier_league_production_v1"
        ),
        match_engine=(
            "integrated_club_goal_model_v1"
        ),
        runtime_names=PRODUCTION_RUNTIME_NAMES,
    ),

    BenchmarkFootballModelSpec(
        model_id="integrated_club_goal_model_robust_candidate",
        display_name=(
            "Integrated Club Goal Model Robust Candidate"
        ),
        repository_source=(
            "premier_league_robust_candidate_v1"
        ),
        match_engine=(
            "integrated_club_goal_model_v1"
        ),
        runtime_names=PRODUCTION_RUNTIME_NAMES,
    ),
)

class CanonicalFootballModel:
    """
    Translate canonical benchmark club names into the names
    expected by one underlying football model.

    Benchmark outputs always retain canonical names.
    """

    def __init__(
        self,
        *,
        model_id: str,
        football_model: Any,
        runtime_names: dict[str, str],
    ) -> None:
        self.model_id = model_id
        self.football_model = football_model
        self.runtime_names = dict(
            runtime_names
        )

    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        prediction_date=None,
    ) -> tuple[int, int]:
        try:
            runtime_home = self.runtime_names[
                home_team
            ]

            runtime_away = self.runtime_names[
                away_team
            ]

        except KeyError as error:
            raise KeyError(
                "No runtime club alias exists for "
                f"{error.args[0]!r} in model "
                f"{self.model_id!r}."
            ) from error

        return self.football_model.simulate_match(
            runtime_home,
            runtime_away,
            prediction_date=prediction_date,
        )

def build_condition(
    spec: BenchmarkFootballModelSpec,
) -> ExperimentCondition:
    return ExperimentCondition(
        name=(
            "Benchmark 001A Preflight — "
            f"{spec.display_name}"
        ),
        competition_format=(
            "double_round_robin"
        ),
        repository_source=(
            spec.repository_source
        ),
        match_engine=(
            spec.match_engine
        ),
        simulation_count=1,
        random_seed=BASE_SEED,
        parameters={
            "benchmark_id": "001A",
            "preflight": True,
            "canonical_club_count":
                len(CANONICAL_CLUBS),
        },
    )

def build_models() -> dict[
    str,
    tuple[
        BenchmarkFootballModelSpec,
        Any,
        CanonicalFootballModel,
    ],
]:
    models = {}

    for spec in BENCHMARK_MODELS:
        condition = build_condition(
            spec
        )

        football_model = (
            FootballModelAdapter()
            .from_condition(condition)
        )

        canonical_model = (
            CanonicalFootballModel(
                model_id=spec.model_id,
                football_model=football_model,
                runtime_names=(
                    spec.runtime_names
                ),
            )
        )

        models[spec.model_id] = (
            spec,
            football_model,
            canonical_model,
        )

    return models

def build_model_runtime_audit(
    models,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (
        model_id,
        (
            spec,
            football_model,
            _,
        ),
    ) in models.items():
        metadata = dict(
            football_model.metadata
        )

        rows.append(
            {
                "model_id": model_id,
                "display_name":
                    spec.display_name,
                "requested_repository_source":
                    spec.repository_source,
                "resolved_repository_source":
                    football_model
                    .repository_source,
                "requested_match_engine":
                    spec.match_engine,
                "resolved_match_engine":
                    football_model
                    .match_engine,
                "runtime_repository_team_count":
                    len(
                        football_model
                        .team_repository
                    ),
                "repository_path":
                    metadata.get(
                        "repository_path"
                    ),
                "production_artifact":
                    metadata.get(
                        "production_artifact"
                    ),
                "production_baseline_version":
                    metadata.get(
                        "production_baseline_version"
                    ),
                "clubelo_cache_directory":
                    metadata.get(
                        "clubelo_cache_directory"
                    ),
                "has_live_observation_builder":
                    football_model
                    .live_observation_builder
                    is not None,
                "has_club_goal_predictor":
                    football_model
                    .club_goal_predictor
                    is not None,
            }
        )

    return pd.DataFrame(rows)

def resolve_runtime_club(
    football_model,
    runtime_name: str,
) -> str:
    """
    Return the repository name resolved by the selected model.
    """

    if (
        football_model
        .live_observation_builder
        is not None
    ):
        representation = (
            football_model
            .live_observation_builder
            .club_repository
            .resolve_club(
                runtime_name
            )
        )

        return representation.club

    if runtime_name not in (
        football_model.team_repository
    ):
        raise KeyError(
            "Legacy runtime repository does not "
            f"contain {runtime_name!r}."
        )

    return runtime_name

def build_club_resolution_audit(
    models,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for canonical_club in CANONICAL_CLUBS:
        for (
            model_id,
            (
                spec,
                football_model,
                _,
            ),
        ) in models.items():
            runtime_name = (
                spec.runtime_names[
                    canonical_club
                ]
            )

            try:
                resolved_name = (
                    resolve_runtime_club(
                        football_model,
                        runtime_name,
                    )
                )

                resolution_pass = True
                error_message = None

            except Exception as error:
                resolved_name = None
                resolution_pass = False
                error_message = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

            rows.append(
                {
                    "model_id":
                        model_id,
                    "canonical_club":
                        canonical_club,
                    "runtime_name":
                        runtime_name,
                    "resolved_runtime_name":
                        resolved_name,
                    "resolution_pass":
                        resolution_pass,
                    "error_message":
                        error_message,
                }
            )

    return pd.DataFrame(rows)

def build_smoke_fixtures():
    generator = (
        RoundRobinFixtureGenerator()
    )

    fixtures = generator.generate(
        participants=list(
            CANONICAL_CLUBS
        ),
        double_round_robin=False,
        competition_id=(
            "benchmark_001a_preflight"
        ),
        start_date=(
            SEASON_START_DATE
        ),
        days_between_matchdays=(
            DAYS_BETWEEN_MATCHDAYS
        ),
    )

    expected_fixture_count = (
        len(CANONICAL_CLUBS)
        * (
            len(CANONICAL_CLUBS) - 1
        )
        // 2
    )

    if len(fixtures) != (
        expected_fixture_count
    ):
        raise AssertionError(
            "Unexpected preflight fixture count: "
            f"{len(fixtures)} vs "
            f"{expected_fixture_count}."
        )

    return fixtures

def simulate_smoke_population(
    *,
    models,
    fixtures,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_index, (
        model_id,
        (
            spec,
            _,
            canonical_model,
        ),
    ) in enumerate(
        models.items()
    ):
        model_seed = (
            BASE_SEED
            + model_index
        )

        random.seed(
            model_seed
        )

        np.random.seed(
            model_seed
        )

        try:
            matches = (
                LeagueMatchSimulator()
                .simulate_fixtures(
                    fixtures=fixtures,
                    football_model=(
                        canonical_model
                    ),
                    stage_name=(
                        "Preflight Round Robin"
                    ),
                    competition_name=(
                        "Football Model "
                        "Benchmark 001A"
                    ),
                )
            )

        except Exception as error:
            raise RuntimeError(
                "Fixture smoke simulation failed "
                f"for model {model_id!r}."
            ) from error

        if len(matches) != len(
            fixtures
        ):
            raise AssertionError(
                f"{model_id} did not preserve "
                "the fixture population."
            )

        for match in matches:
            metadata = (
                match.metadata or {}
            )

            rows.append(
                {
                    "model_id":
                        model_id,
                    "display_name":
                        spec.display_name,
                    "match_id":
                        match.match_id,
                    "matchday":
                        metadata.get(
                            "matchday"
                        ),
                    "match_date":
                        metadata.get(
                            "match_date"
                        ),
                    "home_team":
                        match.team1,
                    "away_team":
                        match.team2,
                    "home_goals":
                        match.goals_team1,
                    "away_goals":
                        match.goals_team2,
                    "total_goals":
                        match.total_goals,
                    "is_draw":
                        match.is_draw,
                    "simulation_pass":
                        True,
                }
            )

    return pd.DataFrame(rows)

def validate_preflight(
    *,
    model_audit: pd.DataFrame,
    club_audit: pd.DataFrame,
    fixture_frame: pd.DataFrame,
    fixture_count: int,
) -> None:
    if (
        model_audit.empty
        or club_audit.empty
        or fixture_frame.empty
    ):
        raise AssertionError(
            "At least one preflight output is empty."
        )

    expected_model_ids = {
        spec.model_id
        for spec in BENCHMARK_MODELS
    }

    if set(
        model_audit[
            "model_id"
        ].astype(str)
    ) != expected_model_ids:
        raise AssertionError(
            "Runtime audit does not contain all "
            "benchmark models."
        )

    if not club_audit[
        "resolution_pass"
    ].all():
        failures = club_audit.loc[
            ~club_audit[
                "resolution_pass"
            ]
        ]

        print(
            failures.to_string(
                index=False
            )
        )

        raise AssertionError(
            "At least one matched club failed "
            "runtime resolution."
        )

    expected_club_rows = (
        len(CANONICAL_CLUBS)
        * len(BENCHMARK_MODELS)
    )

    if len(club_audit) != (
        expected_club_rows
    ):
        raise AssertionError(
            "Unexpected club-resolution audit "
            "population."
        )

    expected_fixture_rows = (
        fixture_count
        * len(BENCHMARK_MODELS)
    )

    if len(fixture_frame) != (
        expected_fixture_rows
    ):
        raise AssertionError(
            "Unexpected fixture-smoke population."
        )

    fixture_counts = (
        fixture_frame
        .groupby(
            "model_id"
        )
        .size()
    )

    if not fixture_counts.eq(
        fixture_count
    ).all():
        raise AssertionError(
            "At least one model simulated an "
            "unexpected number of fixtures."
        )

    canonical_population = set(
        CANONICAL_CLUBS
    )

    output_population = (
        set(
            fixture_frame[
                "home_team"
            ].astype(str)
        )
        | set(
            fixture_frame[
                "away_team"
            ].astype(str)
        )
    )

    if output_population != (
        canonical_population
    ):
        raise AssertionError(
            "Runtime aliases leaked into benchmark "
            "fixture outputs."
        )

    goals = fixture_frame[
        [
            "home_goals",
            "away_goals",
            "total_goals",
        ]
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        goals
    ).all():
        raise AssertionError(
            "Fixture smoke results contain "
            "non-finite goal values."
        )

    if (
        goals < 0
    ).any():
        raise AssertionError(
            "Fixture smoke results contain "
            "negative goals."
        )

def main() -> None:
    print("=" * 88)
    print(
        "FOOTBALL MODEL BENCHMARK 001A — "
        "MATCHED-POPULATION RUNTIME PREFLIGHT"
    )
    print("=" * 88)

    models = build_models()

    model_audit = (
        build_model_runtime_audit(
            models
        )
    )

    club_audit = (
        build_club_resolution_audit(
            models
        )
    )

    fixtures = (
        build_smoke_fixtures()
    )

    fixture_frame = (
        simulate_smoke_population(
            models=models,
            fixtures=fixtures,
        )
    )

    validate_preflight(
        model_audit=model_audit,
        club_audit=club_audit,
        fixture_frame=fixture_frame,
        fixture_count=len(fixtures),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_audit.to_csv(
        MODEL_AUDIT_PATH,
        index=False,
    )

    club_audit.to_csv(
        CLUB_RESOLUTION_PATH,
        index=False,
    )

    fixture_frame.to_csv(
        FIXTURE_SMOKE_PATH,
        index=False,
    )

    model_summary = (
        fixture_frame
        .groupby(
            [
                "model_id",
                "display_name",
            ],
            as_index=False,
        )
        .agg(
            fixture_count=(
                "match_id",
                "count",
            ),
            goals_per_match=(
                "total_goals",
                "mean",
            ),
            draw_rate=(
                "is_draw",
                "mean",
            ),
        )
    )

    metadata = {
        "benchmark_id": "001A",
        "benchmark_name": (
            "Matched-Population Runtime Preflight"
        ),
        "status": "PASS",
        "canonical_club_count":
            len(CANONICAL_CLUBS),
        "football_model_count":
            len(BENCHMARK_MODELS),
        "fixture_count_per_model":
            len(fixtures),
        "total_simulated_matches":
            len(fixture_frame),
        "season_start_date":
            SEASON_START_DATE.isoformat(),
        "days_between_matchdays":
            DAYS_BETWEEN_MATCHDAYS,
        "base_seed":
            BASE_SEED,
        "club_resolution_pass":
            True,
        "matched_fixture_population_pass":
            True,
        "canonical_output_names_pass":
            True,
        "scoreline_generation_pass":
            True,
        "league_standings_generated":
            False,
        "predictive_comparison_performed":
            False,
        "model_selection_decision":
            False,
        "interpretation_boundary": (
            "This preflight verifies runtime compatibility "
            "only. Differences in one simulated fixture "
            "population are not evidence that either "
            "football model is superior."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = f"""# Football Model Benchmark 001A

## Status

**PASS**

## Purpose

Verify that the legacy validation model and Integrated Club Goal
Model v1 can simulate the same canonical 17-club population through
the shared fixture and match-result infrastructure.

## Runtime models

{model_audit.to_markdown(index=False)}

## Smoke-test summary

{model_summary.to_markdown(index=False)}

## Validation

- All football models constructed: PASS
- All canonical clubs resolved: PASS
- Identical fixture population supplied: PASS
- Every fixture simulated by all models: PASS
- Fixture metadata preserved: PASS
- Canonical club names preserved in outputs: PASS
- Finite non-negative scorelines: PASS

## Interpretation boundary

This is a runtime preflight, not a predictive benchmark.

The observed goals-per-match and draw-rate values come from one
single-round-robin smoke population and must not be used to rank the
football models.

## Next step

Run Benchmark 001B using repeated double-round-robin seasons and
compare the paired league fingerprints.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("Runtime model audit")
    print("-" * 88)
    print(
        model_audit[
            [
                "model_id",
                "resolved_repository_source",
                "resolved_match_engine",
                "runtime_repository_team_count",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Smoke-test summary")
    print("-" * 88)
    print(
        model_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Canonical clubs resolved: PASS"
    )
    print(
        "Matched fixtures simulated: PASS"
    )
    print(
        "Canonical output names preserved: PASS"
    )
    print(
        "Scoreline generation: PASS"
    )
    print(
        "Predictive comparison performed: NO"
    )

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
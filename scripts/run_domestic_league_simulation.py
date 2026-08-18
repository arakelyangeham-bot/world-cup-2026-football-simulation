#run_premier_league_2026_27_simulation

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from competition_catalog import (
    CompetitionDefinition,
    StageDefinition,
)
from fixture_generation.canonical_fixture_loader import (
    load_canonical_fixtures,
)

from simulation.domestic_league_configs import (
    DOMESTIC_LEAGUE_CONFIGS,
)
from research import ExperimentCondition
from research.adapters import FootballModelAdapter
from simulation.league_monte_carlo import (
    LeagueMonteCarloRunner,
)




PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FIXTURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "premier_league_2026_27_bootstrap"
    / "premier_league_2026_27_fixtures.csv"
)

DEFAULT_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "premier_league_2026_27_bootstrap"
    / "premier_league_2026_27_club_repository.csv"
)

DEFAULT_GOAL_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_069_production_club_goal_model_v1"
    / "integrated_club_goal_model_v1.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "premier_league_2026_27_simulation"
)

DEFAULT_SEED = 202627


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Monte Carlo simulations of a "
            "configured domestic league season."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "structural",
            "production",
        ],
        default="structural",
        help=(
            "Football model used for match simulation. "
            "Structural mode is infrastructure-only and "
            "must not be interpreted as a forecast."
        ),
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help=(
            "Number of complete league seasons. "
            "Default: 100."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Base random seed. "
            f"Default: {DEFAULT_SEED}."
        ),
    )

    parser.add_argument(
        "--fixture-path",
        type=None,
        default=None,
    )

    parser.add_argument(
        "--repository-path",
        type=None,
        default=None,
    )

    parser.add_argument(
        "--goal-model-path",
        type=None,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=None,
        default=None,
    )

    parser.add_argument(
        "--competition",
        choices=sorted(
            DOMESTIC_LEAGUE_CONFIGS
        ),
        default="premier_league",
        help=(
            "Domestic league configuration to run. "
            "Default: premier_league."
        ),
    )

    return parser.parse_args()


class StructuralFootballModel:
    """
    Infrastructure-only scoreline model.

    This model intentionally contains no team-strength
    information and must never be interpreted as a forecast.
    """

    def simulate_match(
        self,
        home_team,
        away_team,
        prediction_date=None,
    ):
        return (
            int(
                np.random.poisson(
                    1.45
                )
            ),
            int(
                np.random.poisson(
                    1.20
                )
            ),
        )


def build_production_model(
    *,
    config,
    repository_path: Path,
    goal_model_path: Path,
    simulation_count: int,
    seed: int,
):
    condition = ExperimentCondition(
        name=(
            f"{config.display_name} "
            "Production Simulation"
        ),
        competition_format=(
            "double_round_robin"
        ),
        repository_source=(
            config.repository_source
        ),
        match_engine=(
            "integrated_club_goal_model_v1"
        ),
        simulation_count=simulation_count,
        random_seed=seed,
        parameters={
            "competition":
                config.competition_name,
            "season":
                config.season,
            "repository_path":
                str(repository_path),
            "production_artifact":
                str(goal_model_path),
            "rating_prediction_date":
                config.rating_prediction_date.isoformat(),
        },
    )

    return (
        FootballModelAdapter()
        .from_condition(condition)
    )


def build_competition_definition(
    config,
) -> CompetitionDefinition:
    return CompetitionDefinition(
        name=config.display_name,
        competition_type="domestic_league",
        region=None,
        governing_body=None,
        participant_count=(
            config.participant_count
        ),
        stages=[
            StageDefinition(
                name=config.competition_name,
                stage_type="league",
                participant_count=(
                    config.participant_count
                ),
                competition_format=(
                    "double_round_robin"
                ),
                metadata={
                    "matches_per_team":
                        config.matches_per_team,
                    "points_system": "3-1-0",
                },
            )
        ],
        metadata={
            "competition":
                config.competition_name,
            "season":
                config.season,
            "official_fixture_calendar":
                True,
        },
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            f"Cannot write empty CSV: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def build_position_rows(
    club_rows: list[dict[str, Any]],
    participant_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for club_row in club_rows:
        team = club_row["team"]

        for position in range(
            1,
            participant_count + 1,
        ):
            rows.append(
                {
                    "team": team,
                    "position": position,
                    "probability": club_row[
                        f"position_{position}_probability"
                    ],
                }
            )

    return rows


def validate_result(
    club_rows: list[dict[str, Any]],
    participant_count: int,
) -> None:
    if len(club_rows) != participant_count:
        raise AssertionError(
            "Monte Carlo output club count "
            "does not match participant count."
        )

    title_probability_sum = sum(
        float(
            row[
                "champion_probability"
            ]
        )
        for row in club_rows
    )

    if abs(
        title_probability_sum - 1.0
    ) > 1e-12:
        raise AssertionError(
            "Champion probabilities do not "
            "sum to 1."
        )

    for row in club_rows:
        position_sum = sum(
            float(
                row[
                    f"position_{position}_probability"
                ]
            )
            for position in range(
                1,
                participant_count + 1,
            )
        )

        if abs(
            position_sum - 1.0
        ) > 1e-12:
            raise AssertionError(
                "Finishing-position probabilities "
                f"do not sum to 1 for "
                f"{row['team']}."
            )


def build_metadata(
    *,
    competition_name: str,
    season: str,
    model_mode: str,
    simulation_count: int,
    seed: int,
    fixtures,
    participants: list[str],
    fixture_path: Path,
    repository_path: Path,
    goal_model_path: Path,
) -> dict[str, Any]:
    return {
        "competition": competition_name,
        "season": season,
        "model_mode":
            model_mode,
        "simulation_count":
            simulation_count,
        "base_seed":
            seed,
        "club_count":
            len(participants),
        "fixture_count":
            len(fixtures),
        "matchday_count":
            len(
                {
                    fixture.matchday
                    for fixture in fixtures
                }
            ),
        "season_start_date":
            min(
                fixture.match_date
                for fixture in fixtures
            ).isoformat(),
        "season_end_date":
            max(
                fixture.match_date
                for fixture in fixtures
            ).isoformat(),
        "fixture_path":
            str(fixture_path),
        "repository_path":
            (
                str(repository_path)
                if model_mode == "production"
                else None
            ),
        "goal_model_path":
            (
                str(goal_model_path)
                if model_mode == "production"
                else None
            ),
        "official_fixture_calendar":
            True,
        "forecast_interpretation_allowed":
            model_mode == "production",
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


def print_summary(
    rows: list[dict[str, Any]],
    *,
    competition_name: str,
    season: str,
    model_mode: str,
    simulation_count: int,
) -> None:
    print()
    print(
        f"{competition_name} {season} "
        "Monte Carlo Simulation"
    )
    print("=" * 84)

    print(
        f"Model: {model_mode}"
    )
    print(
        f"Simulations: "
        f"{simulation_count}"
    )

    print()
    print(
        f"{'Team':<24}"
        f"{'Title':>9}"
        f"{'Top 4':>9}"
        f"{'Top 6':>9}"
        f"{'Releg.':>9}"
        f"{'Avg Pos':>10}"
        f"{'Avg Pts':>10}"
    )

    for row in rows:
        print(
            f"{row['team']:<24}"
            f"{row['champion_probability']:>9.3f}"
            f"{row['top_four_probability']:>9.3f}"
            f"{row['top_six_probability']:>9.3f}"
            f"{row['relegation_probability']:>9.3f}"
            f"{row['average_position']:>10.2f}"
            f"{row['average_points']:>10.2f}"
        )

    print()

    if model_mode == "structural":
        print(
            "STRUCTURAL MODE: results validate "
            "infrastructure only."
        )
        print(
            "These probabilities are NOT forecasts."
        )
    else:
        print(
            "PRODUCTION MODE: results use the "
            "configured production football model."
        )


def main() -> None:
    arguments = parse_arguments()

    config = DOMESTIC_LEAGUE_CONFIGS[
        arguments.competition
    ]

    config.validate()

    fixture_path = (
        arguments.fixture_path
        if arguments.fixture_path is not None
        else config.fixture_path
    )

    repository_path = (
        arguments.repository_path
        if arguments.repository_path is not None
        else config.repository_path
    )

    goal_model_path = (
        arguments.goal_model_path
        if arguments.goal_model_path is not None
        else config.goal_model_path
    )

    output_directory = (
        arguments.output_dir
        if arguments.output_dir is not None
        else config.output_directory
    )

    if arguments.simulations < 1:
        raise SystemExit(
            "--simulations must be at least 1."
        )

    fixtures = load_canonical_fixtures(
        fixture_path
    )

    participants = sorted(
        {
            team
            for fixture in fixtures
            for team in (
                fixture.home_team,
                fixture.away_team,
            )
        }
    )

    if len(fixtures) != config.fixture_count:
        raise RuntimeError(
            f"Expected {config.fixture_count} "
            f"fixtures for {config.display_name}, "
            f"found {len(fixtures)}."
        )

    if (
        len(participants)
        != config.participant_count
    ):
        raise RuntimeError(
            f"Expected "
            f"{config.participant_count} clubs "
            f"for {config.display_name}, "
            f"found {len(participants)}."
        )

    if arguments.model == "structural":
        football_model = (
            StructuralFootballModel()
        )
    else:
        football_model = build_production_model(
            config=config,
            repository_path=repository_path,
            goal_model_path=goal_model_path,
            simulation_count=arguments.simulations,
            seed=arguments.seed,
        )

        model_participants = set(
            football_model.team_repository
        )

        if model_participants != set(
            participants
        ):
            raise RuntimeError(
                "Production repository participant "
                "population does not match the "
                "official fixture calendar."
            )

    definition = (
        build_competition_definition(
            config
        )
    )

    runner = LeagueMonteCarloRunner(
        definition=definition,
        participants=participants,
        fixtures=fixtures,
        football_model=football_model,
        top_four_count=(
            config.top_four_count
        ),
        top_six_count=(
            config.top_six_count
        ),
        relegation_count=(
            config.relegation_count
        ),
    )

    result = runner.run(
        simulation_count=(
            arguments.simulations
        ),
        base_seed=arguments.seed,
    )

    validate_result(
        club_rows=result.club_rows,
        participant_count=len(
            participants
        ),
    )

    position_rows = (
        build_position_rows(
            club_rows=result.club_rows,
            participant_count=(
                len(participants)
            ),
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_directory
        / "club_probability_summary.csv",
        result.club_rows,
    )

    write_csv(
        output_directory
        / "finishing_position_probabilities.csv",
        position_rows,
    )

    metadata = build_metadata(
        competition_name=config.competition_name,
        season=config.season,
        model_mode=arguments.model,
        simulation_count=(
            arguments.simulations
        ),
        seed=arguments.seed,
        fixtures=fixtures,
        participants=participants,
        fixture_path=(
            fixture_path
        ),
        repository_path=(
            repository_path
        ),
        goal_model_path=(
            goal_model_path
        ),
    )

    (
        output_directory
        / "simulation_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_summary(
        result.club_rows,
        competition_name=config.competition_name,
        season=config.season,
        model_mode=arguments.model,
        simulation_count=(
            arguments.simulations
        ),
    )

    print()
    print(
        f"Outputs: {output_directory}"
    )


if __name__ == "__main__":
    main()
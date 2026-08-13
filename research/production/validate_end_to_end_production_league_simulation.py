#validate_end_to_end_production_league_simulation

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from competition_catalog import CompetitionBuilder
from competition_catalog import (
    CompetitionBuilder,
    CompetitionDefinition,
    StageDefinition,
)
from fixture_generation import (
    RoundRobinFixtureGenerator,
)
from research import ExperimentCondition
from research.adapters import (
    FootballModelAdapter,
)
from simulation.competition import (
    CompetitionEngine,
)
from simulation.league_match_simulator import (
    LeagueMatchSimulator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_074_end_to_end_production_league_simulation"
)

FIXTURE_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "league_fixture_audit.csv"
)

MATCH_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "league_match_results.csv"
)

STANDINGS_PATH = (
    OUTPUT_DIRECTORY
    / "league_standings.csv"
)

CLUBELO_COVERAGE_PATH = (
    OUTPUT_DIRECTORY
    / "clubelo_cache_coverage.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


SEASON_START_DATE = date(2025, 8, 16)
DAYS_BETWEEN_MATCHDAYS = 7
RANDOM_SEED = 74001

def build_validation_competition_definition(
    participant_count: int,
) -> CompetitionDefinition:
    """
    Build a generic league definition matching the current
    production repository population.

    This is an integration-validation competition, not the
    official 20-club Premier League definition.
    """

    if participant_count < 2:
        raise ValueError(
            "Validation competition requires at least "
            "two participants."
        )

    return CompetitionDefinition(
        name=(
            "Production Club League v1 "
            f"({participant_count} clubs)"
        ),
        competition_type=(
            "production_integration_validation"
        ),
        region="England",
        governing_body=None,
        participant_count=participant_count,
        stages=[
            StageDefinition(
                name="League Season",
                stage_type="league",
                participant_count=(
                    participant_count
                ),
                competition_format=(
                    "double_round_robin"
                ),
                metadata={
                    "matches_per_team": (
                        2
                        * (
                            participant_count
                            - 1
                        )
                    ),
                    "points_system": "3-1-0",
                    "integration_validation":
                        True,
                },
            )
        ],
        metadata={
            "study": "074",
            "official_competition": False,
            "repository_scope": (
                "premier_league_production_v1"
            ),
        },
    )

def build_football_model():
    condition = ExperimentCondition(
        name=(
            "Study 074 End-to-End Production "
            "League Simulation"
        ),
        competition_format=(
            "double_round_robin"
        ),
        repository_source=(
            "premier_league_production_v1"
        ),
        match_engine=(
            "integrated_club_goal_model_v1"
        ),
        simulation_count=1,
        random_seed=RANDOM_SEED,
        parameters={
            "study": "074",
            "season_start_date":
                SEASON_START_DATE.isoformat(),
            "days_between_matchdays":
                DAYS_BETWEEN_MATCHDAYS,
        },
    )

    return (
        FootballModelAdapter()
        .from_condition(condition)
    )


def get_participants(
    football_model,
) -> list[str]:
    participants = sorted(
        football_model.team_repository
    )

    if len(participants) < 2:
        raise AssertionError(
            "Production repository contains fewer "
            "than two clubs."
        )

    if len(participants) != len(
        set(participants)
    ):
        raise AssertionError(
            "Production participants are not unique."
        )

    return participants


def validate_clubelo_cache_coverage(
    football_model,
    participants: list[str],
) -> pd.DataFrame:
    builder = (
        football_model
        .live_observation_builder
    )

    if builder is None:
        raise RuntimeError(
            "Production football model has no live "
            "observation builder."
        )

    clubelo_repository = (
        builder.clubelo_repository
    )

    records: list[dict[str, object]] = []

    for club in participants:
        representation = (
            builder.club_repository
            .resolve_club(club)
        )

        lookup_name = (
            builder._clubelo_lookup_name(
                representation.club
            )
        )

        cache_path = (
            clubelo_repository.cache_path(
                lookup_name
            )
        )

        records.append(
            {
                "club": club,
                "clubelo_lookup_name":
                    lookup_name,
                "cache_path": str(cache_path),
                "cache_exists":
                    cache_path.exists(),
            }
        )

    audit = pd.DataFrame(records)

    if not audit["cache_exists"].all():
        missing = audit.loc[
            ~audit["cache_exists"],
            [
                "club",
                "clubelo_lookup_name",
                "cache_path",
            ],
        ]

        print(
            missing.to_string(index=False)
        )

        raise FileNotFoundError(
            "One or more production clubs have no "
            "matching cached ClubElo history."
        )

    return audit


def build_fixtures(
    participants: list[str],
):
    generator = (
        RoundRobinFixtureGenerator()
    )

    fixtures = generator.generate(
        participants=participants,
        double_round_robin=True,
        competition_id=(
            "production_league_v1"
        ),
        start_date=SEASON_START_DATE,
        days_between_matchdays=(
            DAYS_BETWEEN_MATCHDAYS
        ),
    )

    expected_fixture_count = (
        len(participants)
        * (len(participants) - 1)
    )

    if len(fixtures) != expected_fixture_count:
        raise AssertionError(
            "Unexpected double-round-robin fixture "
            f"count: {len(fixtures)} vs "
            f"{expected_fixture_count}."
        )

    if any(
        fixture.match_date is None
        for fixture in fixtures
    ):
        raise AssertionError(
            "One or more production fixtures have "
            "no match date."
        )

    return fixtures


def simulate_league(
    football_model,
    participants: list[str],
    fixtures,
):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    competition_definition = (
        build_validation_competition_definition(
            participant_count=len(
                participants
            )
        )
    )

    competition = (
        CompetitionBuilder().build(
            definition=(
                competition_definition
            ),
            participants=participants,
        )
    )

    league_stage = competition.stages[0]

    league_stage.matches = (
        LeagueMatchSimulator()
        .simulate_fixtures(
            fixtures=fixtures,
            football_model=football_model,
            stage_name=league_stage.name,
            competition_name=(
                competition.name
            ),
        )
    )

    if len(league_stage.matches) != len(
        fixtures
    ):
        raise AssertionError(
            "League simulation did not preserve "
            "the complete fixture population."
        )

    result = (
        CompetitionEngine().resolve(
            competition
        )
    )

    if not result.stage_results:
        raise AssertionError(
            "Competition produced no stage results."
        )

    stage_result = result.stage_results[0]

    if stage_result.standings is None:
        raise AssertionError(
            "League stage produced no standings."
        )

    standings = (
        stage_result.standings.as_rows()
    )

    return (
        competition,
        league_stage.matches,
        standings,
    )


def build_fixture_audit(
    fixtures,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fixture_id":
                    fixture.fixture_id,
                "matchday":
                    fixture.matchday,
                "match_date":
                    fixture.match_date.isoformat(),
                "home_team":
                    fixture.home_team,
                "away_team":
                    fixture.away_team,
                "leg":
                    fixture.leg,
            }
            for fixture in fixtures
        ]
    )


def build_match_audit(
    matches,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for match in matches:
        metadata = match.metadata or {}

        records.append(
            {
                "match_id":
                    match.match_id,
                "team1":
                    match.team1,
                "team2":
                    match.team2,
                "goals_team1":
                    match.goals_team1,
                "goals_team2":
                    match.goals_team2,
                "total_goals":
                    (
                        match.goals_team1
                        + match.goals_team2
                    ),
                "stage":
                    match.stage,
                "matchday":
                    metadata.get(
                        "matchday"
                    ),
                "match_date":
                    metadata.get(
                        "match_date"
                    ),
                "calendar_aware_fixture":
                    metadata.get(
                        "calendar_aware_fixture"
                    ),
            }
        )

    return pd.DataFrame(records)


def validate_results(
    participants: list[str],
    fixtures,
    match_audit: pd.DataFrame,
    standings: list[dict[str, Any]],
) -> pd.DataFrame:
    club_count = len(participants)

    expected_matches_per_club = (
        2 * (club_count - 1)
    )

    if len(match_audit) != len(fixtures):
        raise AssertionError(
            "Match-result population differs from "
            "the fixture population."
        )

    if match_audit["match_id"].duplicated().any():
        raise AssertionError(
            "Simulated match IDs are not unique."
        )

    if not match_audit[
        "calendar_aware_fixture"
    ].eq(True).all():
        raise AssertionError(
            "One or more simulated matches lost "
            "calendar-awareness metadata."
        )

    if match_audit[
        "match_date"
    ].isna().any():
        raise AssertionError(
            "One or more simulated matches lost "
            "its fixture date."
        )

    goal_columns = [
        "goals_team1",
        "goals_team2",
        "total_goals",
    ]

    if not np.isfinite(
        match_audit[
            goal_columns
        ].to_numpy(dtype=float)
    ).all():
        raise AssertionError(
            "Match results contain non-finite "
            "goal values."
        )

    if (
        match_audit[
            [
                "goals_team1",
                "goals_team2",
            ]
        ] < 0
    ).any().any():
        raise AssertionError(
            "Match results contain negative goals."
        )

    standings_dataframe = (
        pd.DataFrame(standings)
    )

    if len(standings_dataframe) != club_count:
        raise AssertionError(
            "Standings row count differs from the "
            "participant count."
        )

    if set(
        standings_dataframe["team"]
    ) != set(participants):
        raise AssertionError(
            "Standings team population differs from "
            "the participant population."
        )

    if not standings_dataframe[
        "matches_played"
    ].eq(
        expected_matches_per_club
    ).all():
        raise AssertionError(
            "One or more clubs played an unexpected "
            "number of matches."
        )

    expected_team_match_total = (
        2 * len(fixtures)
    )

    if int(
        standings_dataframe[
            "matches_played"
        ].sum()
    ) != expected_team_match_total:
        raise AssertionError(
            "Standings match totals do not reconcile "
            "with the fixture population."
        )

    if not (
        standings_dataframe[
            "wins"
        ]
        + standings_dataframe[
            "draws"
        ]
        + standings_dataframe[
            "losses"
        ]
    ).eq(
        standings_dataframe[
            "matches_played"
        ]
    ).all():
        raise AssertionError(
            "Standings win/draw/loss arithmetic "
            "failed."
        )

    if not (
        3 * standings_dataframe[
            "wins"
        ]
        + standings_dataframe[
            "draws"
        ]
    ).eq(
        standings_dataframe[
            "points"
        ]
    ).all():
        raise AssertionError(
            "Standings points arithmetic failed."
        )

    return standings_dataframe


def build_metadata(
    football_model,
    participants: list[str],
    fixtures,
    match_audit: pd.DataFrame,
    standings: pd.DataFrame,
) -> dict[str, object]:
    champion = (
        standings
        .sort_values("rank")
        .iloc[0]
    )

    return {
        "study_id": "074",
        "study_name": (
            "End-to-End Production League "
            "Simulation"
        ),
        "repository_source":
            football_model.repository_source,
        "match_engine":
            football_model.match_engine,
        "production_baseline_version":
            football_model.metadata[
                "production_baseline_version"
            ],
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
        "matches_per_club":
            int(
                standings[
                    "matches_played"
                ].iloc[0]
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
        "total_goals":
            int(
                match_audit[
                    "total_goals"
                ].sum()
            ),
        "goals_per_match":
            float(
                match_audit[
                    "total_goals"
                ].mean()
            ),
        "champion":
            str(champion["team"]),
        "champion_points":
            int(champion["points"]),
        "clubelo_cache_coverage_pass": True,
        "calendar_fixture_generation_pass": True,
        "fixture_date_propagation_pass": True,
        "production_prediction_route_pass": True,
        "scoreline_generation_pass": True,
        "fixture_population_preservation_pass": True,
        "standings_resolution_pass": True,
        "standings_arithmetic_pass": True,
        "overall_result": "PASS",
    }


def write_report(
    metadata: dict[str, object],
) -> None:
    report = f"""# Study 074 — End-to-End Production League Simulation

## Purpose

Validate one complete calendar-aware league season through
the Integrated Club Goal Model v1 production route.

## Runtime configuration

- Repository:
  `{metadata["repository_source"]}`
- Match engine:
  `{metadata["match_engine"]}`
- Baseline version:
  `{metadata["production_baseline_version"]}`

## Competition population

- Clubs: {metadata["club_count"]}
- Fixtures: {metadata["fixture_count"]}
- Matchdays: {metadata["matchday_count"]}
- Matches per club:
  {metadata["matches_per_club"]}
- Season start:
  {metadata["season_start_date"]}
- Season end:
  {metadata["season_end_date"]}

The production repository currently contains 17 clubs.
This study therefore validates a 17-club double round robin,
not a complete 20-club Premier League season.

## Season result

- Champion: `{metadata["champion"]}`
- Champion points:
  {metadata["champion_points"]}
- Total goals:
  {metadata["total_goals"]}
- Goals per match:
  {metadata["goals_per_match"]:.4f}

The single simulated season is an integration validation,
not a predictive claim about a real competition.

## Validation

- Complete ClubElo cache coverage: PASS
- Calendar-aware fixture generation: PASS
- Fixture-date propagation: PASS
- Live observation assembly: PASS
- Integrated Club Goal Model v1: PASS
- Existing Dixon–Coles sampler: PASS
- Fixture population preservation: PASS
- Non-negative scorelines: PASS
- Competition resolution: PASS
- Complete standings population: PASS
- Matches-played reconciliation: PASS
- Win/draw/loss arithmetic: PASS
- Points arithmetic: PASS

## Result

**OVERALL RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    football_model = (
        build_football_model()
    )

    participants = get_participants(
        football_model
    )

    clubelo_coverage = (
        validate_clubelo_cache_coverage(
            football_model=football_model,
            participants=participants,
        )
    )

    fixtures = build_fixtures(
        participants
    )

    (
        competition,
        matches,
        standings,
    ) = simulate_league(
        football_model=football_model,
        participants=participants,
        fixtures=fixtures,
    )

    fixture_audit = (
        build_fixture_audit(
            fixtures
        )
    )

    match_audit = (
        build_match_audit(
            matches
        )
    )

    standings_dataframe = validate_results(
        participants=participants,
        fixtures=fixtures,
        match_audit=match_audit,
        standings=standings,
    )

    metadata = build_metadata(
        football_model=football_model,
        participants=participants,
        fixtures=fixtures,
        match_audit=match_audit,
        standings=standings_dataframe,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    clubelo_coverage.to_csv(
        CLUBELO_COVERAGE_PATH,
        index=False,
    )

    fixture_audit.to_csv(
        FIXTURE_AUDIT_PATH,
        index=False,
    )

    match_audit.to_csv(
        MATCH_AUDIT_PATH,
        index=False,
    )

    standings_dataframe.to_csv(
        STANDINGS_PATH,
        index=False,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(metadata)

    print(
        "Study 074 — End-to-End Production "
        "League Simulation"
    )
    print("=" * 76)
    print()
    print(
        f"Clubs: {metadata['club_count']}"
    )
    print(
        f"Fixtures: "
        f"{metadata['fixture_count']}"
    )
    print(
        f"Matchdays: "
        f"{metadata['matchday_count']}"
    )
    print(
        f"Matches per club: "
        f"{metadata['matches_per_club']}"
    )
    print(
        "Season: "
        f"{metadata['season_start_date']} "
        f"through "
        f"{metadata['season_end_date']}"
    )
    print()
    print("Season Summary")
    print("-" * 76)
    print(
        f"Champion: "
        f"{metadata['champion']}"
    )
    print(
        f"Champion points: "
        f"{metadata['champion_points']}"
    )
    print(
        f"Total goals: "
        f"{metadata['total_goals']}"
    )
    print(
        f"Goals per match: "
        f"{metadata['goals_per_match']:.4f}"
    )
    print()
    print("ClubElo cache coverage: PASS")
    print("Calendar fixture generation: PASS")
    print("Fixture-date propagation: PASS")
    print("Production prediction route: PASS")
    print("Scoreline generation: PASS")
    print("Fixture population preservation: PASS")
    print("Standings resolution: PASS")
    print("Standings arithmetic: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
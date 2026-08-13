#validate_premier_league_simulation

from __future__ import annotations

import random

import numpy as np

from competition_catalog import CompetitionBuilder
from competition_catalog.competitions import PREMIER_LEAGUE
from fixture_generation import RoundRobinFixtureGenerator
from research import ExperimentCondition
from research.adapters import FootballModelAdapter
from simulation.competition import CompetitionEngine
from simulation.league_match_simulator import LeagueMatchSimulator


PREMIER_LEAGUE_TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]

BASE_SEED = 41001
REPOSITORY_SOURCE = "premier_league_validation"


def build_football_model():
    condition = ExperimentCondition(
        name="Premier League validation condition",
        competition_format="double_round_robin",
        repository_source=REPOSITORY_SOURCE,
        match_engine="production_scoreline_first",
        simulation_count=1,
        random_seed=BASE_SEED,
        parameters={
            "competition": "Premier League",
            "validation": True,
        },
    )

    return FootballModelAdapter().from_condition(condition)


def main() -> None:
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)

    builder = CompetitionBuilder()
    fixture_generator = RoundRobinFixtureGenerator()
    match_simulator = LeagueMatchSimulator()
    competition_engine = CompetitionEngine()

    competition = builder.build(
        definition=PREMIER_LEAGUE,
        participants=PREMIER_LEAGUE_TEAMS,
    )

    fixtures = fixture_generator.generate(
        participants=competition.participants,
        double_round_robin=True,
        competition_id="premier_league",
    )

    football_model = build_football_model()

    missing_teams = [
        team
        for team in competition.participants
        if team not in football_model.team_repository
    ]

    if missing_teams:
        raise KeyError(
            "The selected football model repository does not contain these "
            f"Premier League clubs: {missing_teams}"
        )

    league_stage = competition.stages[0]

    league_stage.matches = match_simulator.simulate_fixtures(
        fixtures=fixtures,
        football_model=football_model,
        stage_name=league_stage.name,
        competition_name=competition.name,
    )

    competition_result = competition_engine.resolve(competition)

    if not competition_result.stage_results:
        raise RuntimeError("Competition produced no stage results.")

    stage_result = competition_result.stage_results[0]

    if stage_result.standings is None:
        raise RuntimeError("League stage produced no standings table.")

    standings_rows = stage_result.standings.as_rows()

    assert len(fixtures) == 380
    assert len(league_stage.matches) == 380
    assert len(standings_rows) == 20
    assert all(row["matches_played"] == 38 for row in standings_rows)

    print("Premier League Simulation Validation")
    print("====================================")
    print(f"Competition: {competition_result.competition_name}")
    print(f"Teams: {len(competition.participants)}")
    print(f"Fixtures simulated: {len(league_stage.matches)}")
    print(f"Champion: {competition_result.champion}")
    print(f"Runner-up: {competition_result.runner_up}")
    print()

    print("Final Standings")
    print("---------------")
    print(
        f"{'Pos':>3}  "
        f"{'Team':<28} "
        f"{'P':>3} "
        f"{'W':>3} "
        f"{'D':>3} "
        f"{'L':>3} "
        f"{'GF':>4} "
        f"{'GA':>4} "
        f"{'GD':>4} "
        f"{'Pts':>4}"
    )

    for row in standings_rows:
        print(
            f"{row['rank']:>3}  "
            f"{row['team']:<28} "
            f"{row['matches_played']:>3} "
            f"{row['wins']:>3} "
            f"{row['draws']:>3} "
            f"{row['losses']:>3} "
            f"{row['goals_for']:>4} "
            f"{row['goals_against']:>4} "
            f"{row['goal_difference']:>4} "
            f"{row['points']:>4}"
        )

    print()
    print("Top Four")
    print("--------")
    for row in standings_rows[:4]:
        print(f"{row['rank']}. {row['team']} — {row['points']} points")

    print()
    print("Bottom Three")
    print("------------")
    for row in standings_rows[-3:]:
        print(f"{row['rank']}. {row['team']} — {row['points']} points")

    print()
    print("All structural checks passed.")


if __name__ == "__main__":
    main()
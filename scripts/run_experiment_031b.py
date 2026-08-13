#run_experiment_031b.py

import random

from research import Experiment
from research.experiment_runner import ExperimentRunner
from research.experiments import (
    ExperimentResult,
)
from research.metrics import (
    AverageChampionStrengthMetric,
    ChampionVarianceMetric,
    StrongestTeamChampionshipRateMetric,
    UpsetRateMetric,
)
from simulation.competition import (
    BracketBuilder,
    Competition,
    CompetitionEngine,
    MatchResult,
    Stage,
    StageType,
)
from research.adapters import CompetitionAdapter

TEAM_STRENGTHS = {
    "Team A": 100,
    "Team B": 95,
    "Team C": 90,
    "Team D": 85,
    "Team E": 80,
    "Team F": 75,
    "Team G": 70,
    "Team H": 65,
}

SIMULATION_COUNT = 1000
BASE_SEED = 31031


def win_probability(team1: str, team2: str) -> float:
    strength1 = TEAM_STRENGTHS[team1]
    strength2 = TEAM_STRENGTHS[team2]
    return strength1 / (strength1 + strength2)


def simulate_decisive_match(
    team1: str,
    team2: str,
    rng: random.Random,
    stage: str,
    match_id: str,
) -> MatchResult:
    p_team1 = win_probability(team1, team2)

    if rng.random() < p_team1:
        winner = team1
        loser = team2
    else:
        winner = team2
        loser = team1

    return MatchResult(
        team1=winner,
        team2=loser,
        goals_team1=2,
        goals_team2=1,
        stage=stage,
        match_id=match_id,
        metadata={
            "synthetic_result": True,
            "original_team1": team1,
            "original_team2": team2,
        },
    )


def build_experiment_definition(metrics) -> Experiment:
    return Experiment(
        experiment_id="031B",
        title="League vs Knockout",
        research_question=(
            "Does a league competition identify the strongest team more "
            "reliably than a knockout competition?"
        ),
        hypothesis=(
            "League competitions will identify the strongest team more "
            "frequently because repeated matches reduce the influence of "
            "single-match variance."
        ),
        fixed_variables={
            "team_set": "synthetic_8_team_strength_ladder",
            "team_strengths": TEAM_STRENGTHS,
            "simulation_count": SIMULATION_COUNT,
            "random_seed_policy": f"base_seed_{BASE_SEED}_plus_run_id",
            "match_generation": "strength_weighted_decisive_match",
        },
        independent_variable="competition_format",
        dependent_variables=[
            "average_champion_strength",
            "strongest_team_championship_rate",
            "champion_variance",
            "upset_rate",
        ],
        metrics=metrics,
        metadata={
            "version": "3",
            "program": "competition_research",
            "status": "experiment",
        },
    )


def build_league_competition(run_id: int, rng: random.Random) -> Competition:
    teams = list(TEAM_STRENGTHS.keys())
    match_results: list[MatchResult] = []

    match_number = 1

    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i >= j:
                continue

            match_results.append(
                simulate_decisive_match(
                    team1=team1,
                    team2=team2,
                    rng=rng,
                    stage="League Stage",
                    match_id=f"league_run_{run_id}_match_{match_number}",
                )
            )
            match_number += 1

    league_stage = Stage(
        name="League Stage",
        stage_type=StageType.LEAGUE,
        participants=teams,
        matches=match_results,
        metadata={
            "experiment": "031B",
            "format": "single_round_robin_league",
            "run_id": run_id,
        },
    )

    return Competition(
        name="Experiment 031B League",
        participants=teams,
        stages=[league_stage],
        metadata={
            "experiment": "031B",
            "format": "single_round_robin_league",
            "run_id": run_id,
        },
    )


def build_knockout_competition(run_id: int, rng: random.Random) -> Competition:
    teams = list(TEAM_STRENGTHS.keys())
    builder = BracketBuilder()

    stages: list[Stage] = []
    current_teams = teams
    round_names = [
        "Quarterfinals",
        "Semifinals",
        "Final",
    ]

    for round_name in round_names:
        bracket = builder.build_high_low_bracket(
            teams=current_teams,
            name=f"{round_name} Bracket",
        )

        for tie_index, tie in enumerate(bracket.ties, start=1):
            tie.add_match_result(
                simulate_decisive_match(
                    team1=tie.team1,
                    team2=tie.team2,
                    rng=rng,
                    stage=round_name,
                    match_id=f"knockout_run_{run_id}_{round_name}_{tie_index}",
                )
            )

        stage_type = StageType.FINAL if round_name == "Final" else StageType.KNOCKOUT

        stage = Stage(
            name=round_name,
            stage_type=stage_type,
            participants=current_teams,
            matches=bracket.as_stage_matches(),
            metadata={
                "experiment": "031B",
                "format": "seeded_knockout",
                "run_id": run_id,
            },
        )

        temporary_competition = Competition(
            name=f"Experiment 031B Knockout {round_name}",
            participants=current_teams,
            stages=[stage],
        )

        temporary_result = CompetitionEngine().resolve(temporary_competition)
        stage_result = temporary_result.stage_results[0]
        current_teams = stage_result.qualifiers

        stages.append(stage)

    return Competition(
        name="Experiment 031B Knockout",
        participants=teams,
        stages=stages,
        metadata={
            "experiment": "031B",
            "format": "seeded_knockout",
            "run_id": run_id,
        },
    )

def run_condition(format_name: str) -> ExperimentResult:
    strongest_team = max(
        TEAM_STRENGTHS,
        key=TEAM_STRENGTHS.get,
    )

    experiment_result = ExperimentResult(
        experiment_name="League vs Knockout",
        format_name=format_name,
        strongest_team=strongest_team,
        team_strengths=TEAM_STRENGTHS,
        metadata={
            "simulation_count": SIMULATION_COUNT,
        },
    )

    engine = CompetitionEngine()
    adapter = CompetitionAdapter(team_strengths=TEAM_STRENGTHS)

    for run_id in range(1, SIMULATION_COUNT + 1):
        rng = random.Random(BASE_SEED + run_id)

        if format_name == "single_round_robin_league":
            competition = build_league_competition(run_id, rng)
        elif format_name == "seeded_knockout":
            competition = build_knockout_competition(run_id, rng)
        else:
            raise ValueError(f"Unknown format: {format_name}")

        competition_result = engine.resolve(competition)

        experiment_result.add_run(
            adapter.to_experiment_run_result(
                competition_result=competition_result,
                experiment_name="League vs Knockout",
                format_name=format_name,
                run_id=run_id,
            )
        )

    return experiment_result


def print_report(report) -> None:
    print(f"Format: {report.format_name}")
    print(f"Runs: {report.simulation_count}")
    print()

    for row in report.as_rows():
        print(
            f"{row['metric_name']}: "
            f"{row['metric_value']}"
        )

    print()


def main() -> None:
    metrics = [
        AverageChampionStrengthMetric(),
        StrongestTeamChampionshipRateMetric(),
        ChampionVarianceMetric(),
        UpsetRateMetric(),
    ]

    experiment = build_experiment_definition(metrics)
    runner = ExperimentRunner(metrics=metrics)

    print(f"Experiment {experiment.experiment_id}: {experiment.title}")
    print()
    print(f"Research question: {experiment.research_question}")
    print()
    print(f"Hypothesis: {experiment.hypothesis}")
    print()

    league_result = run_condition("single_round_robin_league")
    knockout_result = run_condition("seeded_knockout")

    league_report = runner.evaluate(league_result)
    knockout_report = runner.evaluate(knockout_result)

    print("Metric Report")
    print("=============")
    print()

    print_report(league_report)
    print_report(knockout_report)


if __name__ == "__main__":
    main()
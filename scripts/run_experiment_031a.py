#run_experiment_031a.py

from research import Experiment
from research.experiment_runner import ExperimentRunner
from research.experiments import (
    ExperimentResult,
    ExperimentRunResult,
)
from research.metrics import (
    AverageChampionStrengthMetric,
    ChampionVarianceMetric,
    StrongestTeamChampionshipRateMetric,
    UpsetRateMetric,
)
from simulation.competition import (
    Competition,
    CompetitionEngine,
    MatchResult,
    Stage,
    StageType,
)


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


def build_single_round_robin_results(teams: list[str]) -> list[MatchResult]:
    results: list[MatchResult] = []

    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i >= j:
                continue

            goals_team1 = (i + j) % 4
            goals_team2 = (j - i) % 3

            results.append(
                MatchResult(
                    team1=team1,
                    team2=team2,
                    goals_team1=goals_team1,
                    goals_team2=goals_team2,
                    stage="League Stage",
                    match_id=f"{team1}_vs_{team2}",
                    metadata={
                        "experiment": "031A",
                        "placeholder_result": True,
                    },
                )
            )

    return results


def build_experiment_definition(metrics) -> Experiment:
    return Experiment(
        experiment_id="031A",
        title="Research Framework Validation",
        research_question=(
            "Can the Version 3 research framework produce a valid experiment "
            "report from a completed synthetic competition?"
        ),
        hypothesis=(
            "The Version 3 research framework can convert completed synthetic "
            "competition results into a reusable metric report."
        ),
        fixed_variables={
            "team_set": "synthetic_8_team_strength_ladder",
            "competition_format": "single_round_robin_league",
            "match_results": "deterministic_placeholder_results",
            "simulation_count": 1,
        },
        independent_variable="none",
        dependent_variables=[
            "average_champion_strength",
            "strongest_team_championship_rate",
            "champion_variance",
            "upset_rate",
        ],
        metrics=metrics,
        metadata={
            "version": "3",
            "program": "research_framework_validation",
            "status": "validation",
        },
    )


def build_competition() -> Competition:
    teams = list(TEAM_STRENGTHS.keys())

    league_stage = Stage(
        name="Synthetic League Stage",
        stage_type=StageType.LEAGUE,
        participants=teams,
        matches=build_single_round_robin_results(teams),
        metadata={
            "experiment": "031A",
            "format": "single_round_robin",
        },
    )

    return Competition(
        name="Experiment 031A Synthetic League",
        participants=teams,
        stages=[league_stage],
        metadata={
            "experiment": "031A",
        },
    )


def collect_match_results(competition_result) -> list[MatchResult]:
    match_results: list[MatchResult] = []

    for stage_result in competition_result.stage_results:
        if stage_result.match_results:
            match_results.extend(stage_result.match_results)

    return match_results


def main() -> None:
    metrics = [
        AverageChampionStrengthMetric(),
        StrongestTeamChampionshipRateMetric(),
        ChampionVarianceMetric(),
        UpsetRateMetric(),
    ]

    experiment = build_experiment_definition(metrics)
    competition = build_competition()

    competition_engine = CompetitionEngine()
    competition_result = competition_engine.resolve(competition)

    champion = competition_result.champion
    champion_strength = (
        TEAM_STRENGTHS.get(champion)
        if champion is not None
        else None
    )

    experiment_result = ExperimentResult(
        experiment_name=experiment.title,
        format_name="single_round_robin_league",
        strongest_team="Team A",
        team_strengths=TEAM_STRENGTHS,
    )

    experiment_result.add_run(
        ExperimentRunResult(
            experiment_name=experiment.title,
            format_name="single_round_robin_league",
            run_id=1,
            champion=champion,
            champion_strength=champion_strength,
            match_results=collect_match_results(competition_result),
            metadata={
                "competition_name": competition_result.competition_name,
            },
        )
    )

    runner = ExperimentRunner(metrics=metrics)
    report = runner.evaluate(experiment_result)

    print(f"Experiment {experiment.experiment_id}: {experiment.title}")
    print()
    print(f"Research question: {experiment.research_question}")
    print()
    print(f"Hypothesis: {experiment.hypothesis}")
    print()
    print(f"Format: {report.format_name}")
    print(f"Runs: {report.simulation_count}")
    print(f"Champion: {champion}")
    print(f"Champion strength: {champion_strength}")
    print()

    print("Metric Report")
    print("-------------")
    for row in report.as_rows():
        print(
            f"{row['metric_name']}: "
            f"{row['metric_value']}"
        )


if __name__ == "__main__":
    main()
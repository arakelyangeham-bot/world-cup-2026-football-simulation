#validate_experiment_runner.py

from research.experiment_runner import ExperimentRunner
from research.experiments import (
    ExperimentResult,
    ExperimentRunResult,
)
from research.metrics import (
    AverageChampionStrengthMetric,
    ChampionDistributionMetric,
    ChampionVarianceMetric,
    StrongestTeamChampionshipRateMetric,
    UpsetRateMetric,
)
from simulation.competition import MatchResult


def build_fake_experiment_result() -> ExperimentResult:
    team_strengths = {
        "Team A": 100,
        "Team B": 95,
        "Team C": 90,
        "Team D": 85,
    }

    result = ExperimentResult(
        experiment_name="Experiment Runner Validation",
        format_name="test_format",
        strongest_team="Team A",
        team_strengths=team_strengths,
    )

    result.add_run(
        ExperimentRunResult(
            experiment_name="Experiment Runner Validation",
            format_name="test_format",
            run_id=1,
            champion="Team A",
            champion_strength=100,
            match_results=[
                MatchResult("Team A", "Team B", 2, 1),
                MatchResult("Team C", "Team D", 0, 1),
            ],
        )
    )

    result.add_run(
        ExperimentRunResult(
            experiment_name="Experiment Runner Validation",
            format_name="test_format",
            run_id=2,
            champion="Team B",
            champion_strength=95,
            match_results=[
                MatchResult("Team B", "Team C", 1, 0),
                MatchResult("Team D", "Team A", 2, 1),
            ],
        )
    )

    return result


def main() -> None:
    metrics = [
        AverageChampionStrengthMetric(),
        StrongestTeamChampionshipRateMetric(),
        ChampionVarianceMetric(),
        ChampionDistributionMetric(),
        UpsetRateMetric(),
    ]

    runner = ExperimentRunner(metrics=metrics)
    experiment_result = build_fake_experiment_result()
    report = runner.evaluate(experiment_result)

    print(f"Experiment: {report.experiment_name}")
    print(f"Format: {report.format_name}")
    print(f"Runs: {report.simulation_count}")
    print()

    for row in report.as_rows():
        print(
            f"{row['metric_name']}: "
            f"{row['metric_value']}"
        )


if __name__ == "__main__":
    main()
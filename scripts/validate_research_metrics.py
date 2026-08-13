#validate_research_metrics.py

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


def main() -> None:
    team_strengths = {
        "Team A": 100,
        "Team B": 95,
        "Team C": 90,
        "Team D": 85,
    }

    result = ExperimentResult(
        experiment_name="Metric Validation",
        format_name="test_format",
        strongest_team="Team A",
        team_strengths=team_strengths,
    )

    result.add_run(
        ExperimentRunResult(
            experiment_name="Metric Validation",
            format_name="test_format",
            run_id=1,
            champion="Team A",
            champion_strength=100,
            match_results=[
                MatchResult("Team A", "Team B", 2, 1),
                MatchResult("Team C", "Team D", 0, 1),  # upset
            ],
        )
    )

    result.add_run(
        ExperimentRunResult(
            experiment_name="Metric Validation",
            format_name="test_format",
            run_id=2,
            champion="Team B",
            champion_strength=95,
            match_results=[
                MatchResult("Team B", "Team C", 1, 0),
                MatchResult("Team D", "Team A", 2, 1),  # upset
            ],
        )
    )

    metrics = [
        AverageChampionStrengthMetric(),
        StrongestTeamChampionshipRateMetric(),
        ChampionVarianceMetric(),
        ChampionDistributionMetric(),
        UpsetRateMetric(),
    ]

    for metric in metrics:
        value = metric.compute(result)
        print(f"{metric.name}: {value}")


if __name__ == "__main__":
    main()
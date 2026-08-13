#validate_experiment_definition.py

from research import Experiment
from research.metrics import (
    AverageChampionStrengthMetric,
    ChampionVarianceMetric,
    StrongestTeamChampionshipRateMetric,
    UpsetRateMetric,
)


def main() -> None:
    experiment = Experiment(
        experiment_id="031",
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
            "team_strengths": [100, 95, 90, 85, 80, 75, 70, 65],
            "match_engine": "controlled_placeholder_v1",
            "simulation_count": "to_be_determined",
            "random_seed_policy": "fixed_seed_per_condition",
        },
        independent_variable="competition_format",
        dependent_variables=[
            "strongest_team_championship_rate",
            "average_champion_strength",
            "champion_variance",
            "upset_rate",
        ],
        metrics=[
            AverageChampionStrengthMetric(),
            StrongestTeamChampionshipRateMetric(),
            ChampionVarianceMetric(),
            UpsetRateMetric(),
        ],
        metadata={
            "version": "3",
            "program": "competition_research",
            "status": "planned",
        },
    )

    print(f"Experiment {experiment.experiment_id}: {experiment.title}")
    print()
    print(f"Research question: {experiment.research_question}")
    print()
    print(f"Hypothesis: {experiment.hypothesis}")
    print()
    print(f"Independent variable: {experiment.independent_variable}")
    print()
    print("Dependent variables:")
    for variable in experiment.dependent_variables:
        print(f"- {variable}")

    print()
    print("Metrics:")
    for metric_name in experiment.metric_names():
        print(f"- {metric_name}")


if __name__ == "__main__":
    main()
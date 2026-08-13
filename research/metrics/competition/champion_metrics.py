#champion_metrics.py

from __future__ import annotations

from collections import Counter
from typing import Any

from research.metrics.base_metric import Metric


class AverageChampionStrengthMetric(Metric):
    name = "average_champion_strength"
    description = "Average pre-competition strength of tournament champions."

    def compute(self, experiment_result: Any) -> float:
        champion_strengths = [
            run.champion_strength
            for run in experiment_result.runs
            if run.champion_strength is not None
        ]

        if not champion_strengths:
            return 0.0

        return sum(champion_strengths) / len(champion_strengths)


class StrongestTeamChampionshipRateMetric(Metric):
    name = "strongest_team_championship_rate"
    description = "Share of simulations won by the strongest team."

    def compute(self, experiment_result: Any) -> float:
        if not experiment_result.runs:
            return 0.0

        strongest_team = experiment_result.strongest_team

        wins = sum(
            1
            for run in experiment_result.runs
            if run.champion == strongest_team
        )

        return wins / len(experiment_result.runs)


class ChampionVarianceMetric(Metric):
    name = "champion_variance"
    description = "Number of distinct teams that won the competition."

    def compute(self, experiment_result: Any) -> int:
        champions = [
            run.champion
            for run in experiment_result.runs
            if run.champion is not None
        ]

        return len(set(champions))


class ChampionDistributionMetric(Metric):
    name = "champion_distribution"
    description = "Champion counts and probabilities by team."

    def compute(self, experiment_result: Any) -> list[dict]:
        if not experiment_result.runs:
            return []

        counter = Counter(
            run.champion
            for run in experiment_result.runs
            if run.champion is not None
        )

        total = sum(counter.values())

        return [
            {
                "team": team,
                "count": count,
                "probability": count / total,
            }
            for team, count in counter.most_common()
        ]
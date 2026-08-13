#upset_metrics.py

from __future__ import annotations

from typing import Any

from research.metrics.base_metric import Metric


class UpsetRateMetric(Metric):
    name = "upset_rate"
    description = "Share of decisive matches won by the lower-strength team."

    def compute(self, experiment_result: Any) -> float:
        upset_count = 0
        decisive_match_count = 0

        team_strengths = experiment_result.team_strengths

        for run in experiment_result.runs:
            for match in run.match_results:
                winner = match.winner
                loser = match.loser

                if winner is None or loser is None:
                    continue

                winner_strength = team_strengths.get(winner)
                loser_strength = team_strengths.get(loser)

                if winner_strength is None or loser_strength is None:
                    continue

                decisive_match_count += 1

                if winner_strength < loser_strength:
                    upset_count += 1

        if decisive_match_count == 0:
            return 0.0

        return upset_count / decisive_match_count
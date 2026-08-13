#draw_calibrated_sampler.py

"""
Draw-calibrated goal sampler.

Version 4.1 experiment:
- Preserve hierarchical stochastic lambda sampling.
- Add an optional draw-tempering layer after score generation.
- draw_strength=0.0 must behave exactly like the base sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random



@dataclass(frozen=True)
class DrawCalibratedSampler:
    base_sampler: Any
    draw_strength: float = 0.0
    max_draw_goal: int = 4

    def sample_score(self, home_lambda: float, away_lambda: float) -> tuple[int, int]:
        home_goals, away_goals = self.base_sampler.sample_score(
            home_lambda=home_lambda,
            away_lambda=away_lambda,
        )

        if self.draw_strength <= 0.0:
            return home_goals, away_goals

        if home_goals == away_goals:
            return home_goals, away_goals

        goal_gap = abs(home_goals - away_goals)

        # Only nudge close matches. Blowouts should stay blowouts.
        if goal_gap != 1:
            return home_goals, away_goals

        lower_goal = min(home_goals, away_goals)

        if lower_goal > self.max_draw_goal:
            return home_goals, away_goals

        if random.random() >= self.draw_strength:
            return home_goals, away_goals

        # Convert 1-goal margin into the nearby draw.
        return lower_goal, lower_goal
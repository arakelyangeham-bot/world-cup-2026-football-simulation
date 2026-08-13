#football_profile.py

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research.football_observatory.observatory_schema import MatchObservation
from research.football_observatory.uncertainty import wilson_interval


@dataclass(frozen=True)
class FootballProfile:
    label: str
    matches: int

    avg_total_goals: float
    avg_home_goals: float
    avg_away_goals: float
    avg_goal_difference: float
    avg_abs_goal_difference: float

    draw_rate: float
    home_win_rate: float
    away_win_rate: float
    one_goal_rate: float
    clean_sheet_rate: float
    both_teams_scored_rate: float
    high_scoring_rate: float
    blowout_rate: float

    zero_zero_rate: float
    one_one_rate: float
    one_zero_rate: float
    two_one_rate: float
    one_two_rate: float


def _rate(values: list[bool]) -> float:
    if not values:
        return float("nan")

    return sum(values) / len(values)


def build_football_profile(
    observations: list[MatchObservation],
    label: str,
) -> FootballProfile:
    matches = len(observations)

    if matches == 0:
        return FootballProfile(
            label=label,
            matches=0,
            avg_total_goals=float("nan"),
            avg_home_goals=float("nan"),
            avg_away_goals=float("nan"),
            avg_goal_difference=float("nan"),
            avg_abs_goal_difference=float("nan"),
            draw_rate=float("nan"),
            home_win_rate=float("nan"),
            away_win_rate=float("nan"),
            one_goal_rate=float("nan"),
            clean_sheet_rate=float("nan"),
            both_teams_scored_rate=float("nan"),
            high_scoring_rate=float("nan"),
            blowout_rate=float("nan"),
            zero_zero_rate=float("nan"),
            one_one_rate=float("nan"),
            one_zero_rate=float("nan"),
            two_one_rate=float("nan"),
            one_two_rate=float("nan"),
        )

    outcomes = [observation.outcome for observation in observations]

    total_goals = [outcome.total_goals for outcome in outcomes]
    home_goals = [outcome.home_score for outcome in outcomes]
    away_goals = [outcome.away_score for outcome in outcomes]
    goal_differences = [outcome.goal_difference for outcome in outcomes]
    abs_goal_differences = [
        outcome.absolute_goal_difference
        for outcome in outcomes
    ]

    return FootballProfile(
        label=label,
        matches=matches,
        avg_total_goals=sum(total_goals) / matches,
        avg_home_goals=sum(home_goals) / matches,
        avg_away_goals=sum(away_goals) / matches,
        avg_goal_difference=sum(goal_differences) / matches,
        avg_abs_goal_difference=sum(abs_goal_differences) / matches,

        draw_rate=_rate([outcome.is_draw for outcome in outcomes]),
        home_win_rate=_rate([outcome.is_home_win for outcome in outcomes]),
        away_win_rate=_rate([outcome.is_away_win for outcome in outcomes]),
        one_goal_rate=_rate([outcome.is_one_goal_match for outcome in outcomes]),
        clean_sheet_rate=_rate([outcome.is_clean_sheet for outcome in outcomes]),
        both_teams_scored_rate=_rate(
            [outcome.both_teams_scored for outcome in outcomes]
        ),
        high_scoring_rate=_rate([outcome.is_high_scoring for outcome in outcomes]),
        blowout_rate=_rate([outcome.is_blowout for outcome in outcomes]),

        zero_zero_rate=_rate(
            [
                outcome.home_score == 0 and outcome.away_score == 0
                for outcome in outcomes
            ]
        ),
        one_one_rate=_rate(
            [
                outcome.home_score == 1 and outcome.away_score == 1
                for outcome in outcomes
            ]
        ),
        one_zero_rate=_rate(
            [
                outcome.home_score == 1 and outcome.away_score == 0
                for outcome in outcomes
            ]
        ),
        two_one_rate=_rate(
            [
                outcome.home_score == 2 and outcome.away_score == 1
                for outcome in outcomes
            ]
        ),
        one_two_rate=_rate(
            [
                outcome.home_score == 1 and outcome.away_score == 2
                for outcome in outcomes
            ]
        ),
    )


def profile_to_dataframe(profile: FootballProfile) -> pd.DataFrame:
    return pd.DataFrame([profile.__dict__])


def compare_profiles(
    baseline: FootballProfile,
    subset: FootballProfile,
) -> pd.DataFrame:
    rows = []

    for metric, baseline_value in baseline.__dict__.items():
        if metric in {"label", "matches"}:
            continue

        subset_value = getattr(subset, metric)

        rows.append(
            {
                "metric": metric,
                "baseline_label": baseline.label,
                "subset_label": subset.label,
                "baseline_value": baseline_value,
                "subset_value": subset_value,
                "difference": subset_value - baseline_value,
            }
        )

    return pd.DataFrame(rows)
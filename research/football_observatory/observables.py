#observables.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from research.football_observatory.observatory_schema import MatchObservation


Predicate = Callable[[MatchObservation], bool]


@dataclass(frozen=True)
class FootballObservable:
    name: str
    description: str
    predicate: Predicate

    def evaluate(self, observation: MatchObservation) -> bool:
        return bool(self.predicate(observation))


CORE_OBSERVABLES: list[FootballObservable] = [
    FootballObservable(
        name="draw",
        description="Match ended level after regulation/full time.",
        predicate=lambda obs: obs.outcome.is_draw,
    ),
    FootballObservable(
        name="home_win",
        description="Home/team1 side won the match.",
        predicate=lambda obs: obs.outcome.is_home_win,
    ),
    FootballObservable(
        name="away_win",
        description="Away/team2 side won the match.",
        predicate=lambda obs: obs.outcome.is_away_win,
    ),
    FootballObservable(
        name="one_goal_match",
        description="Match was decided by exactly one goal.",
        predicate=lambda obs: obs.outcome.is_one_goal_match,
    ),
    FootballObservable(
        name="clean_sheet",
        description="At least one team failed to concede.",
        predicate=lambda obs: obs.outcome.is_clean_sheet,
    ),
    FootballObservable(
        name="both_teams_scored",
        description="Both teams scored at least once.",
        predicate=lambda obs: obs.outcome.both_teams_scored,
    ),
    FootballObservable(
        name="high_scoring",
        description="Match had five or more total goals.",
        predicate=lambda obs: obs.outcome.is_high_scoring,
    ),
    FootballObservable(
        name="blowout",
        description="Match was decided by three or more goals.",
        predicate=lambda obs: obs.outcome.is_blowout,
    ),
    FootballObservable(
        name="zero_zero",
        description="Match ended 0-0.",
        predicate=lambda obs: obs.outcome.home_score == 0
        and obs.outcome.away_score == 0,
    ),
    FootballObservable(
        name="one_one",
        description="Match ended 1-1.",
        predicate=lambda obs: obs.outcome.home_score == 1
        and obs.outcome.away_score == 1,
    ),
    FootballObservable(
        name="two_one",
        description="Match ended 2-1.",
        predicate=lambda obs: obs.outcome.home_score == 2
        and obs.outcome.away_score == 1,
    ),
    FootballObservable(
        name="one_two",
        description="Match ended 1-2.",
        predicate=lambda obs: obs.outcome.home_score == 1
        and obs.outcome.away_score == 2,
    ),
]
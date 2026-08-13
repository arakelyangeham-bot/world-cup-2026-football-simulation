#league_match_simulator

from __future__ import annotations

from typing import Any

from fixture_generation import ScheduledFixture
from simulation.competition import MatchResult


class LeagueMatchSimulator:
    """
    Converts scheduled league fixtures into simulated MatchResult objects.

    This class does not generate fixtures, calculate standings, or resolve
    competitions. It only asks the supplied football model to simulate each
    scheduled match.
    """

    def simulate_fixtures(
        self,
        fixtures: list[ScheduledFixture],
        football_model: Any,
        stage_name: str,
        competition_name: str | None = None,
    ) -> list[MatchResult]:
        match_results: list[MatchResult] = []

        for fixture in fixtures:
            home_goals, away_goals = football_model.simulate_match(
                fixture.home_team,
                fixture.away_team,
                prediction_date=fixture.match_date,
            )

            match_results.append(
                MatchResult(
                    team1=fixture.home_team,
                    team2=fixture.away_team,
                    goals_team1=home_goals,
                    goals_team2=away_goals,
                    stage=stage_name,
                    match_id=fixture.fixture_id,
                    metadata={
                        "competition": competition_name,
                        "matchday": fixture.matchday,
                        "leg": fixture.leg,
                        "match_date": (
                            fixture.match_date.isoformat()
                            if fixture.match_date is not None
                            else None
                        ),
                        "home_team": fixture.home_team,
                        "away_team": fixture.away_team,
                        "scheduled_fixture": True,
                        "calendar_aware_fixture": (
                            fixture.match_date is not None
                        ),
                    },
                )
            )

        return match_results
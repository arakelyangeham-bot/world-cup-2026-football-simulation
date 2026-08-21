#test_champions_league_league_phase

from __future__ import annotations

import pytest

from simulation.competition.champions_league_league_phase import (
    build_synthetic_champions_league_league_phase_schedule,
)

from simulation.competition.champions_league_advancement import (
    resolve_champions_league_league_phase,
)
from simulation.competition.match_result import MatchResult
from simulation.competition.stage import Stage, StageType
from simulation.competition.stage_resolver import StageResolver


def _teams() -> list[str]:
    return [
        f"Team {index:02d}"
        for index in range(1, 37)
    ]


def test_synthetic_league_phase_has_144_fixtures():
    fixtures = (
        build_synthetic_champions_league_league_phase_schedule(
            _teams()
        )
    )

    assert len(fixtures) == 144


def test_synthetic_league_phase_has_eight_matchdays():
    fixtures = (
        build_synthetic_champions_league_league_phase_schedule(
            _teams()
        )
    )

    matchdays = {
        fixture.matchday
        for fixture in fixtures
    }

    assert matchdays == set(range(1, 9))


def test_each_matchday_has_18_matches():
    fixtures = (
        build_synthetic_champions_league_league_phase_schedule(
            _teams()
        )
    )

    for matchday in range(1, 9):
        matchday_fixtures = [
            fixture
            for fixture in fixtures
            if fixture.matchday == matchday
        ]

        assert len(matchday_fixtures) == 18


def test_every_team_plays_once_per_matchday():
    fixtures = (
        build_synthetic_champions_league_league_phase_schedule(
            _teams()
        )
    )

    for matchday in range(1, 9):
        appearances: list[str] = []

        for fixture in fixtures:
            if fixture.matchday != matchday:
                continue

            appearances.extend(
                [
                    fixture.home_team,
                    fixture.away_team,
                ]
            )

        assert len(appearances) == 36
        assert len(set(appearances)) == 36


def test_every_team_plays_eight_matches():
    fixtures = (
        build_synthetic_champions_league_league_phase_schedule(
            _teams()
        )
    )

    appearances = {
        team: 0
        for team in _teams()
    }

    for fixture in fixtures:
        appearances[fixture.home_team] += 1
        appearances[fixture.away_team] += 1

    assert set(appearances.values()) == {8}


def test_every_team_has_four_home_and_four_away_matches():
    fixtures = (
        build_synthetic_champions_league_league_phase_schedule(
            _teams()
        )
    )

    home_counts = {
        team: 0
        for team in _teams()
    }

    away_counts = {
        team: 0
        for team in _teams()
    }

    for fixture in fixtures:
        home_counts[fixture.home_team] += 1
        away_counts[fixture.away_team] += 1

    assert set(home_counts.values()) == {4}
    assert set(away_counts.values()) == {4}


def test_synthetic_league_phase_has_no_self_matches():
    fixtures = (
        build_synthetic_champions_league_league_phase_schedule(
            _teams()
        )
    )

    assert all(
        fixture.home_team != fixture.away_team
        for fixture in fixtures
    )


def test_synthetic_league_phase_has_no_repeated_opponents():
    fixtures = (
        build_synthetic_champions_league_league_phase_schedule(
            _teams()
        )
    )

    pairings = [
        frozenset(
            {
                fixture.home_team,
                fixture.away_team,
            }
        )
        for fixture in fixtures
    ]

    assert len(pairings) == 144
    assert len(set(pairings)) == 144


def test_synthetic_league_phase_rejects_wrong_team_count():
    with pytest.raises(
        ValueError,
        match="exactly 36 teams",
    ):
        build_synthetic_champions_league_league_phase_schedule(
            _teams()[:-1]
        )


def test_synthetic_league_phase_rejects_duplicate_teams():
    teams = _teams()
    teams[-1] = teams[0]

    with pytest.raises(
        ValueError,
        match="duplicate teams",
    ):
        build_synthetic_champions_league_league_phase_schedule(
            teams
        )

def test_synthetic_league_phase_resolves_full_36_team_table():
    teams = _teams()

    fixtures = (
        build_synthetic_champions_league_league_phase_schedule(
            teams
        )
    )

    match_results = []

    for fixture in fixtures:
        home_number = int(
            fixture.home_team.split()[-1]
        )
        away_number = int(
            fixture.away_team.split()[-1]
        )

        # Deterministic synthetic scoreline.
        if home_number < away_number:
            home_goals = 2
            away_goals = 1
        else:
            home_goals = 0
            away_goals = 1

        match_results.append(
            MatchResult(
                team1=fixture.home_team,
                team2=fixture.away_team,
                goals_team1=home_goals,
                goals_team2=away_goals,
                stage="League Phase",
                metadata={
                    "matchday": fixture.matchday,
                },
            )
        )

    stage = Stage(
        name="League Phase",
        stage_type=StageType.SWISS,
        participants=teams,
        matches=match_results,
    )

    result = StageResolver().resolve(stage)

    assert result.standings is not None

    ranked_rows = result.standings.ranked_rows()

    assert len(ranked_rows) == 36

    assert {
        row.matches_played
        for row in ranked_rows
    } == {8}

    assert sum(
        row.matches_played
        for row in ranked_rows
    ) == 36 * 8

    advancement = (
        resolve_champions_league_league_phase(
            ranked_rows
        )
    )

    assert len(
        advancement.direct_round_of_16
    ) == 8

    assert len(
        advancement.knockout_playoff
    ) == 16

    assert len(
        advancement.eliminated
    ) == 12
#test_knockout_engine

from __future__ import annotations

import pytest

from simulation.competition.knockout_engine import KnockoutEngine
from simulation.competition.match_result import MatchResult
from simulation.competition.stage import Stage, StageType
from simulation.competition.tie import Tie


def test_single_match_knockout_behavior_remains_supported():
    stage = Stage(
        name="Final",
        stage_type=StageType.FINAL,
        participants=["Team A", "Team B"],
    )

    tie = Tie(
        team1="Team A",
        team2="Team B",
        match_results=[
            MatchResult(
                team1="Team A",
                team2="Team B",
                goals_team1=2,
                goals_team2=1,
            )
        ],
    )

    result = KnockoutEngine().resolve(
        stage=stage,
        match_results=[tie],
    )

    assert result.qualifiers == ["Team A"]
    assert result.eliminated == ["Team B"]
    assert result.winner == "Team A"
    assert result.runner_up == "Team B"

    tie_result = result.match_results[0]

    assert tie_result.winner == "Team A"
    assert tie_result.loser == "Team B"


def test_two_leg_knockout_resolves_aggregate_winner():
    stage = Stage(
        name="Round of 16",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=["Team A", "Team B"],
    )

    tie = Tie(
        team1="Team A",
        team2="Team B",
        match_results=[
            MatchResult(
                team1="Team A",
                team2="Team B",
                goals_team1=2,
                goals_team2=1,
            ),
            MatchResult(
                team1="Team B",
                team2="Team A",
                goals_team1=1,
                goals_team2=1,
            ),
        ],
    )

    result = KnockoutEngine().resolve(
        stage=stage,
        match_results=[tie],
    )

    assert result.qualifiers == ["Team A"]
    assert result.eliminated == ["Team B"]

    tie_result = result.match_results[0]

    assert tie_result.winner == "Team A"
    assert tie_result.loser == "Team B"

    assert tie_result.metadata["aggregate_team1"] == 3
    assert tie_result.metadata["aggregate_team2"] == 2


def test_two_leg_knockout_does_not_apply_away_goals_rule():
    stage = Stage(
        name="Quarterfinal",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=["Team A", "Team B"],
    )

    tie = Tie(
        team1="Team A",
        team2="Team B",
        match_results=[
            MatchResult(
                team1="Team A",
                team2="Team B",
                goals_team1=1,
                goals_team2=0,
            ),
            MatchResult(
                team1="Team B",
                team2="Team A",
                goals_team1=2,
                goals_team2=1,
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="level on aggregate",
    ):
        KnockoutEngine().resolve(
            stage=stage,
            match_results=[tie],
        )

def test_two_leg_knockout_allows_drawn_leg():
    stage = Stage(
        name="Semifinal",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=["Team A", "Team B"],
    )

    tie = Tie(
        team1="Team A",
        team2="Team B",
        match_results=[
            MatchResult(
                team1="Team A",
                team2="Team B",
                goals_team1=1,
                goals_team2=1,
            ),
            MatchResult(
                team1="Team B",
                team2="Team A",
                goals_team1=0,
                goals_team2=2,
            ),
        ],
    )

    result = KnockoutEngine().resolve(
        stage=stage,
        match_results=[tie],
    )

    tie_result = result.match_results[0]

    assert tie_result.winner == "Team A"
    assert tie_result.loser == "Team B"
    assert tie_result.metadata["aggregate_team1"] == 3
    assert tie_result.metadata["aggregate_team2"] == 1

def test_two_leg_knockout_rejects_unexpected_participant():
    stage = Stage(
        name="Round of 16",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=["Team A", "Team B"],
    )

    tie = Tie(
        team1="Team A",
        team2="Team B",
        match_results=[
            MatchResult(
                team1="Team A",
                team2="Team B",
                goals_team1=2,
                goals_team2=0,
            ),
            MatchResult(
                team1="Team C",
                team2="Team A",
                goals_team1=1,
                goals_team2=1,
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="unexpected participants",
    ):
        KnockoutEngine().resolve(
            stage=stage,
            match_results=[tie],
        )

def test_two_leg_knockout_preserves_tie_metadata():
    stage = Stage(
        name="Playoff",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=["Team A", "Team B"],
    )

    tie = Tie(
        team1="Team A",
        team2="Team B",
        match_results=[
            MatchResult(
                team1="Team A",
                team2="Team B",
                goals_team1=2,
                goals_team2=0,
            ),
            MatchResult(
                team1="Team B",
                team2="Team A",
                goals_team1=1,
                goals_team2=1,
            ),
        ],
        metadata={
            "band_id": "I",
        },
    )

    result = KnockoutEngine().resolve(
        stage=stage,
        match_results=[tie],
    )

    tie_result = result.match_results[0]

    assert tie_result.metadata["band_id"] == "I"
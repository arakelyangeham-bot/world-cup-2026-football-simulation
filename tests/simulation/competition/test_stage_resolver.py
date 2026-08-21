#test_stage_resolver

from simulation.competition.match_result import MatchResult
from simulation.competition.stage import Stage, StageType
from simulation.competition.stage_resolver import StageResolver
from simulation.competition.tie import Tie


def test_stage_resolver_dispatches_two_leg_knockout():
    stage = Stage(
        name="Quarterfinal",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=["Team A", "Team B"],
        matches=[
            Tie(
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
            )
        ],
    )

    result = StageResolver().resolve(stage)

    assert result.qualifiers == ["Team A"]
    assert result.eliminated == ["Team B"]
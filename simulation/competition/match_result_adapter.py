#match_result_adapter.py

from __future__ import annotations

from typing import Any

from simulation.competition.match_result import MatchResult


def to_match_result(
    raw_match: Any,
    stage: str | None = None,
) -> MatchResult:
    """
    Convert an existing simulator match result object into a generic
    competition-layer MatchResult.

    This allows competition engines and observers to use a stable match
    interface without depending on the internal format of the match engine.
    """

    team1 = get_team_name(raw_match.team1)
    team2 = get_team_name(raw_match.team2)

    return MatchResult(
        team1=team1,
        team2=team2,
        goals_team1=raw_match.goals_team1,
        goals_team2=raw_match.goals_team2,
        stage=stage or getattr(raw_match, "stage", None),
        match_id=getattr(raw_match, "match_id", None),
        metadata={
            "went_to_extra_time": getattr(raw_match, "went_to_extra_time", False),
            "went_to_penalties": getattr(raw_match, "went_to_penalties", False),
        },
    )


def get_team_name(team_or_entry: Any) -> str:
    """
    Extract a team name from either a plain string or an object with a .team field.
    """

    if hasattr(team_or_entry, "team"):
        return team_or_entry.team

    return str(team_or_entry)
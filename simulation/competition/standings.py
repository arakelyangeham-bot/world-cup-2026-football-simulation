#standings.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StandingRow:
    team: str
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


class StandingsTable:
    """
    Generic football standings table.

    This is intended to support groups, domestic leagues, league phases,
    and other table-based competition stages.
    """

    def __init__(
        self,
        teams: list[str],
        points_for_win: int = 3,
        points_for_draw: int = 1,
        points_for_loss: int = 0,
    ) -> None:
        self.points_for_win = points_for_win
        self.points_for_draw = points_for_draw
        self.points_for_loss = points_for_loss

        self.rows: dict[str, StandingRow] = {
            team: StandingRow(team=team)
            for team in teams
        }

    def record_match(
        self,
        team1: str,
        team2: str,
        goals_team1: int,
        goals_team2: int,
    ) -> None:
        if team1 not in self.rows:
            self.rows[team1] = StandingRow(team=team1)

        if team2 not in self.rows:
            self.rows[team2] = StandingRow(team=team2)

        row1 = self.rows[team1]
        row2 = self.rows[team2]

        row1.matches_played += 1
        row2.matches_played += 1

        row1.goals_for += goals_team1
        row1.goals_against += goals_team2

        row2.goals_for += goals_team2
        row2.goals_against += goals_team1

        if goals_team1 > goals_team2:
            row1.wins += 1
            row2.losses += 1
            row1.points += self.points_for_win
            row2.points += self.points_for_loss

        elif goals_team2 > goals_team1:
            row2.wins += 1
            row1.losses += 1
            row2.points += self.points_for_win
            row1.points += self.points_for_loss

        else:
            row1.draws += 1
            row2.draws += 1
            row1.points += self.points_for_draw
            row2.points += self.points_for_draw

    def ranked_rows(self) -> list[StandingRow]:
        return sorted(
            self.rows.values(),
            key=lambda row: (
                row.points,
                row.goal_difference,
                row.goals_for,
                row.wins,
            ),
            reverse=True,
        )

    def as_rows(self) -> list[dict]:
        return [
            {
                "rank": rank,
                "team": row.team,
                "matches_played": row.matches_played,
                "wins": row.wins,
                "draws": row.draws,
                "losses": row.losses,
                "goals_for": row.goals_for,
                "goals_against": row.goals_against,
                "goal_difference": row.goal_difference,
                "points": row.points,
            }
            for rank, row in enumerate(self.ranked_rows(), start=1)
        ]
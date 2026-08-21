#champions_league_league_phase

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChampionsLeagueLeaguePhaseFixture:
    matchday: int
    home_team: str
    away_team: str


def build_synthetic_champions_league_league_phase_schedule(
    teams: list[str],
) -> tuple[ChampionsLeagueLeaguePhaseFixture, ...]:
    if len(teams) != 36:
        raise ValueError(
            "Synthetic Champions League league phase "
            "requires exactly 36 teams."
        )

    if len(set(teams)) != 36:
        raise ValueError(
            "Synthetic Champions League league phase "
            "contains duplicate teams."
        )

    first_group = teams[:18]
    second_group = teams[18:]

    fixtures: list[
        ChampionsLeagueLeaguePhaseFixture
    ] = []

    for matchday_index in range(8):
        matchday = matchday_index + 1

        for team_index, first_team in enumerate(
            first_group
        ):
            second_team = second_group[
                (
                    team_index
                    + matchday_index
                )
                % 18
            ]

            if matchday_index % 2 == 0:
                home_team = first_team
                away_team = second_team
            else:
                home_team = second_team
                away_team = first_team

            fixtures.append(
                ChampionsLeagueLeaguePhaseFixture(
                    matchday=matchday,
                    home_team=home_team,
                    away_team=away_team,
                )
            )

    return tuple(fixtures)
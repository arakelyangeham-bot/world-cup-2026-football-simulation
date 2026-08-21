#champions_league_advancement

from __future__ import annotations

from dataclasses import dataclass

from simulation.competition.standings import StandingRow


@dataclass(frozen=True)
class ChampionsLeagueLeaguePhaseAdvancement:
    direct_round_of_16: tuple[str, ...]
    knockout_playoff: tuple[str, ...]
    eliminated: tuple[str, ...]


def resolve_champions_league_league_phase(
    ranked_rows: list[StandingRow],
) -> ChampionsLeagueLeaguePhaseAdvancement:
    if len(ranked_rows) != 36:
        raise ValueError(
            "Champions League league phase requires exactly "
            "36 ranked teams."
        )

    teams = [row.team for row in ranked_rows]

    if len(set(teams)) != 36:
        raise ValueError(
            "Champions League league-phase standings contain "
            "duplicate teams."
        )

    return ChampionsLeagueLeaguePhaseAdvancement(
        direct_round_of_16=tuple(teams[:8]),
        knockout_playoff=tuple(teams[8:24]),
        eliminated=tuple(teams[24:]),
    )
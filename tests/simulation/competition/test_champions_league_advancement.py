#test_champions_league_advancement

from __future__ import annotations

import pytest

from simulation.competition.champions_league_advancement import (
    resolve_champions_league_league_phase,
)
from simulation.competition.standings import StandingRow


def _ranked_rows(count: int) -> list[StandingRow]:
    return [
        StandingRow(
            team=f"Team {index:02d}",
            points=100 - index,
        )
        for index in range(1, count + 1)
    ]


def test_league_phase_partitions_36_teams_correctly():
    result = resolve_champions_league_league_phase(
        _ranked_rows(36)
    )

    assert len(result.direct_round_of_16) == 8
    assert len(result.knockout_playoff) == 16
    assert len(result.eliminated) == 12

    assert result.direct_round_of_16 == tuple(
        f"Team {index:02d}"
        for index in range(1, 9)
    )

    assert result.knockout_playoff == tuple(
        f"Team {index:02d}"
        for index in range(9, 25)
    )

    assert result.eliminated == tuple(
        f"Team {index:02d}"
        for index in range(25, 37)
    )


def test_league_phase_rejects_wrong_team_count():
    with pytest.raises(
        ValueError,
        match="exactly 36 ranked teams",
    ):
        resolve_champions_league_league_phase(
            _ranked_rows(35)
        )


def test_league_phase_rejects_duplicate_team():
    rows = _ranked_rows(36)
    rows[-1].team = rows[0].team

    with pytest.raises(
        ValueError,
        match="duplicate teams",
    ):
        resolve_champions_league_league_phase(rows)
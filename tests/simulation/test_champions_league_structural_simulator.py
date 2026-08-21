#test_champions_league_structural_simulator

from __future__ import annotations

from simulation.champions_league_structural_simulator import (
    simulate_champions_league_structural,
)


def _teams() -> list[str]:
    return [
        f"Team {index:02d}"
        for index in range(1, 37)
    ]


def test_structural_simulator_completes_full_champions_league():
    result = simulate_champions_league_structural(
        teams=_teams(),
        seed=202627,
    )

    assert len(result.league_phase_rankings) == 36

    assert len(result.direct_round_of_16) == 8
    assert len(result.knockout_playoff) == 16
    assert len(result.league_phase_eliminated) == 12

    assert len(result.playoff_winners) == 8
    assert len(result.round_of_16_winners) == 8
    assert len(result.quarterfinal_winners) == 4
    assert len(result.semifinal_winners) == 2

    assert result.champion in _teams()
    assert result.runner_up in _teams()
    assert result.champion != result.runner_up


def test_structural_simulator_is_reproducible_from_seed():
    first = simulate_champions_league_structural(
        teams=_teams(),
        seed=202627,
    )

    second = simulate_champions_league_structural(
        teams=_teams(),
        seed=202627,
    )

    assert first == second
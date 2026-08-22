#test_champions_league_match_simulator

from __future__ import annotations

from datetime import date

from simulation.champions_league_match_simulator import (
    build_football_model_match_simulator,
)


class FakeFootballModel:
    def __init__(self) -> None:
        self.calls = []

    def simulate_match(
        self,
        team1: str,
        team2: str,
        prediction_date=None,
    ) -> tuple[int, int]:
        self.calls.append(
            (
                team1,
                team2,
                prediction_date,
            )
        )

        return (3, 1)


def test_football_model_match_simulator_adapts_contract():
    football_model = FakeFootballModel()

    simulator = build_football_model_match_simulator(
        football_model=football_model,
        prediction_date=date(2026, 9, 15),
    )

    result = simulator(
        "Club A",
        "Club B",
        "League Phase",
    )

    assert result == (3, 1)

    assert football_model.calls == [
        (
            "Club A",
            "Club B",
            date(2026, 9, 15),
        )
    ]

from simulation.champions_league_structural_simulator import (
    simulate_champions_league_structural,
)


def _teams() -> list[str]:
    return [
        f"Team {index:02d}"
        for index in range(1, 37)
    ]


def test_football_model_adapter_runs_full_structural_simulation():
    football_model = FakeFootballModel()

    simulator = build_football_model_match_simulator(
        football_model=football_model,
        prediction_date=date(2026, 9, 15),
    )

    result = simulate_champions_league_structural(
        teams=_teams(),
        seed=202627,
        match_simulator=simulator,
    )

    assert result.champion in _teams()
    assert result.runner_up in _teams()

    assert len(football_model.calls) > 144
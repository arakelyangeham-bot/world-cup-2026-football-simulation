#champions_league_match_simulator

from __future__ import annotations

from datetime import date

from research.adapters.football_model_adapter import FootballModel


def build_football_model_match_simulator(
    *,
    football_model: FootballModel,
    prediction_date: date,
):
    def simulate(
        home_team: str,
        away_team: str,
        stage: str,
    ) -> tuple[int, int]:
        del stage

        return football_model.simulate_match(
            home_team,
            away_team,
            prediction_date=prediction_date,
        )

    return simulate
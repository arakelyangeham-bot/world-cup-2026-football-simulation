#domestic_league_configs

from __future__ import annotations

from datetime import date
from pathlib import Path

from simulation.domestic_league_config import (
    DomesticLeagueSimulationConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PREMIER_LEAGUE_2026_27 = (
    DomesticLeagueSimulationConfig(
        key="premier_league",
        competition_name="Premier League",
        season="2026-27",
        participant_count=20,
        matches_per_team=38,
        matchday_count=38,
        fixture_count=380,
        top_four_count=4,
        top_six_count=6,
        relegation_count=3,
        repository_source=(
            "premier_league_production_v1"
        ),
        rating_prediction_date=date(
            2026,
            8,
            15,
        ),
        fixture_path=(
            PROJECT_ROOT
            / "outputs"
            / "premier_league_2026_27_bootstrap"
            / "premier_league_2026_27_fixtures.csv"
        ),
        repository_path=(
            PROJECT_ROOT
            / "outputs"
            / "premier_league_2026_27_bootstrap"
            / "premier_league_2026_27_club_repository.csv"
        ),
        goal_model_path=(
            PROJECT_ROOT
            / "outputs"
            / "study_069_production_club_goal_model_v1"
            / "integrated_club_goal_model_v1.json"
        ),
        output_directory=(
            PROJECT_ROOT
            / "outputs"
            / "premier_league_2026_27_simulation"
        ),
    )
)


LA_LIGA_2026_27 = (
    DomesticLeagueSimulationConfig(
        key="la_liga",
        competition_name="La Liga",
        season="2026-27",
        participant_count=20,
        matches_per_team=38,
        matchday_count=38,
        fixture_count=380,
        top_four_count=4,
        top_six_count=6,
        relegation_count=3,
        repository_source=(
            "la_liga_production_v1"
        ),
        rating_prediction_date=date(
            2026,
            8,
            15,
        ),
        fixture_path=(
            PROJECT_ROOT
            / "outputs"
            / "la_liga_2026_27_bootstrap"
            / "la_liga_2026_27_fixtures.csv"
        ),
        repository_path=(
            PROJECT_ROOT
            / "outputs"
            / "la_liga_2026_27_bootstrap"
            / "la_liga_2026_27_club_repository.csv"
        ),
        goal_model_path=(
            PROJECT_ROOT
            / "outputs"
            / "study_069_production_club_goal_model_v1"
            / "integrated_club_goal_model_v1.json"
        ),
        output_directory=(
            PROJECT_ROOT
            / "outputs"
            / "la_liga_2026_27_simulation"
        ),
    )
)

BUNDESLIGA_2026_27 = (
    DomesticLeagueSimulationConfig(
        key="bundesliga",
        competition_name="Bundesliga",
        season="2026-27",
        participant_count=18,
        matches_per_team=34,
        matchday_count=34,
        fixture_count=306,
        top_four_count=4,
        top_six_count=6,
        relegation_count=3,
        repository_source=(
            "bundesliga_production_v1"
        ),
        rating_prediction_date=date(
            2026,
            8,
            15,
        ),
        fixture_path=(
            PROJECT_ROOT
            / "outputs"
            / "bundesliga_2026_27_bootstrap"
            / "bundesliga_2026_27_fixtures.csv"
        ),
        repository_path=(
            PROJECT_ROOT
            / "outputs"
            / "bundesliga_2026_27_bootstrap"
            / "bundesliga_2026_27_club_repository.csv"
        ),
        goal_model_path=(
            PROJECT_ROOT
            / "outputs"
            / "study_069_production_club_goal_model_v1"
            / "integrated_club_goal_model_v1.json"
        ),
        output_directory=(
            PROJECT_ROOT
            / "outputs"
            / "bundesliga_2026_27_simulation"
        ),
    )
)

SERIE_A_2026_27 = DomesticLeagueSimulationConfig(
    key="serie_a",
    competition_name="Serie A",
    season="2026-27",
    participant_count=20,
    matches_per_team=38,
    matchday_count=38,
    fixture_count=380,
    top_four_count=4,
    top_six_count=6,
    relegation_count=3,
    repository_source=(
        "serie_a_production_v1"
    ),
    rating_prediction_date=date(
        2026,
        8,
        15,
    ),
    fixture_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_a_2026_27_bootstrap"
        / "serie_a_2026_27_fixtures.csv"
    ),
    repository_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_a_2026_27_bootstrap"
        / "serie_a_2026_27_club_repository.csv"
    ),
    goal_model_path=(
        PROJECT_ROOT
        / "outputs"
        / "study_069_production_club_goal_model_v1"
        / "integrated_club_goal_model_v1.json"
    ),
    output_directory=(
        PROJECT_ROOT
        / "outputs"
        / "serie_a_2026_27_simulation"
    ),
)

DOMESTIC_LEAGUE_CONFIGS = {
    PREMIER_LEAGUE_2026_27.key:
        PREMIER_LEAGUE_2026_27,
    LA_LIGA_2026_27.key:
        LA_LIGA_2026_27,
    BUNDESLIGA_2026_27.key:
        BUNDESLIGA_2026_27,
    SERIE_A_2026_27.key:
        SERIE_A_2026_27,
}

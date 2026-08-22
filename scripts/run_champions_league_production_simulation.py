#run_champions_league_production_simulation

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from research.adapters.football_model_adapter import (
    FootballModelAdapter,
)
from research.experiment_condition import ExperimentCondition
from research.production.domestic_clubelo_identity import (
    LA_LIGA_CLUBELO_NAME_OVERRIDES,
)
from research.production.preload_domestic_league_clubelo import (
    BUNDESLIGA_2026_27_CLUBELO_NAME_OVERRIDES,
)
from simulation.champions_league_match_simulator import (
    build_football_model_match_simulator,
)
from simulation.champions_league_structural_simulator import (
    simulate_champions_league_structural,
)
from simulation.domestic_league_configs import (
    LA_LIGA_2026_27,
)

CHAMPIONS_LEAGUE_REPOSITORY_PATH = Path(
    "outputs/champions_league_2026_27_bootstrap/"
    "champions_league_2026_27_club_repository.csv"
)

CHAMPIONS_LEAGUE_PARTICIPANTS_PATH = Path(
    "outputs/champions_league_2026_27_bootstrap/"
    "champions_league_2026_27_structural_participants.csv"
)

def load_structural_participants(
    repository_path: Path,
    participants_path: Path,
) -> list[str]:
    repository_frame = pd.read_csv(
        repository_path,
        low_memory=False,
    )

    participants_frame = pd.read_csv(
        participants_path,
        low_memory=False,
    )

    if "club" not in repository_frame.columns:
        raise ValueError(
            "Production repository must contain a "
            "'club' column."
        )

    if "club" not in participants_frame.columns:
        raise ValueError(
            "Participant artifact must contain a "
            "'club' column."
        )

    repository_clubs = set(
        repository_frame["club"]
        .dropna()
        .astype(str)
        .tolist()
    )

    participants = (
        participants_frame["club"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if len(participants) != 36:
        raise ValueError(
            "Champions League participant artifact "
            "must contain exactly 36 clubs."
        )

    if len(set(participants)) != 36:
        raise ValueError(
            "Champions League participant artifact "
            "contains duplicate clubs."
        )

    unknown_clubs = [
        club
        for club in participants
        if club not in repository_clubs
    ]

    if unknown_clubs:
        raise ValueError(
            "Champions League participant artifact "
            "contains clubs not present in production "
            f"repository: {unknown_clubs}"
        )

    return participants

def build_champions_league_football_model():
    prediction_date = (
        LA_LIGA_2026_27.rating_prediction_date
    )

    condition = ExperimentCondition(
        name=(
            "Champions League Production Simulation"
        ),
        competition_format="league_phase",
        repository_source="la_liga_production_v1",
        match_engine="integrated_club_goal_model_v1",
        simulation_count=1,
        random_seed=202627,
        parameters={
            "repository_path": str(
                CHAMPIONS_LEAGUE_REPOSITORY_PATH
            ),
            "production_artifact": str(
                LA_LIGA_2026_27.goal_model_path
            ),
            "rating_prediction_date": (
                prediction_date.isoformat()
            ),
            "clubelo_name_overrides": {
                **LA_LIGA_CLUBELO_NAME_OVERRIDES,
                **BUNDESLIGA_2026_27_CLUBELO_NAME_OVERRIDES,
            },
        },
    )

    model = (
        FootballModelAdapter()
        .from_condition(condition)
    )

    return model, prediction_date

def main() -> None:
    participants = load_structural_participants(
        CHAMPIONS_LEAGUE_REPOSITORY_PATH,
        CHAMPIONS_LEAGUE_PARTICIPANTS_PATH,
    )

    football_model, prediction_date = (
        build_champions_league_football_model()
    )

    match_simulator = (
        build_football_model_match_simulator(
            football_model=football_model,
            prediction_date=prediction_date,
        )
    )

    with patch(
        "research.rating_priors.clubelo_repository.urlopen",
        side_effect=RuntimeError(
            "Network access is disabled during "
            "production Champions League simulation."
        ),
    ):
        result = (
            simulate_champions_league_structural(
                teams=participants,
                seed=202627,
                match_simulator=match_simulator,
            )
        )

    print()
    print("=" * 72)
    print("CHAMPIONS LEAGUE PRODUCTION SIMULATION")
    print("=" * 72)

    print()
    print(
        f"Participants: {len(participants)}"
    )

    print()
    print(
        "Direct Round-of-16 qualifiers: "
        f"{len(result.direct_round_of_16)}"
    )
    print(
        "Knockout-playoff participants: "
        f"{len(result.knockout_playoff)}"
    )
    print(
        "League-phase eliminated: "
        f"{len(result.league_phase_eliminated)}"
    )

    print()
    print(
        f"Runner-up: {result.runner_up}"
    )
    print(
        f"Champion: {result.champion}"
    )


if __name__ == "__main__":
    main()
#test_sofascore_feature_engineering

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_feature_engineering_preserves_stable_schema(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    #
    # Deliberately omit:
    # - expectedGoals
    # - expectedAssists
    # - goalsPrevented
    #
    # The production feature schema must still contain
    # their required downstream features as NA.
    #
    source = pd.DataFrame(
        [
            {
                "competition": "Test League",
                "competition_type": "club_league",
                "competition_id": 999,
                "season_id": 12345,
                "season_year": "25/26",
                "player_id": "1001",
                "player": "Test Player",
                "team_id": "5001",
                "team": "Test FC",
                "minutesPlayed": 900,
                "goals": 10,
                "assists": 5,
                "shotsOnTarget": 20,
                "accurateCrosses": 15,
                "successfulDribbles": 30,
                "ballRecovery": 40,
                "cleanSheet": 2,
                "clearances": 12,
                "interceptions": 18,
                "keyPasses": 25,
                "saves": 0,
                "tackles": 22,
                "rating": 7.1,
            }
        ]
    )

    source.to_csv(
        input_path,
        index=False,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.sofascore_feature_engineering",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        completed.stdout
        + "\n"
        + completed.stderr
    )

    result = pd.read_csv(
        output_path,
        low_memory=False,
    )

    expected_missing_features = [
        "expectedAssists_per90",
        "expectedGoals_per90",
        "goalsPrevented",
    ]

    for feature in expected_missing_features:
        assert feature in result.columns
        assert result[feature].notna().sum() == 0
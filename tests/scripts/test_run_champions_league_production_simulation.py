#test_run_champions_league_production_simulation

from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_champions_league_production_simulation import (
    load_structural_participants,
)

def test_load_structural_participants_uses_explicit_participant_file(
    tmp_path,
):
    repository_path = tmp_path / "repository.csv"
    participants_path = tmp_path / "participants.csv"

    repository_clubs = [
        f"Club {index:02d}"
        for index in range(1, 39)
    ]

    selected_clubs = repository_clubs[2:38]

    pd.DataFrame(
        {
            "club": repository_clubs,
        }
    ).to_csv(
        repository_path,
        index=False,
    )

    pd.DataFrame(
        {
            "club": selected_clubs,
        }
    ).to_csv(
        participants_path,
        index=False,
    )

    participants = load_structural_participants(
        repository_path,
        participants_path,
    )

    assert participants == selected_clubs
    assert len(participants) == 36

def test_load_structural_participants_requires_exactly_36(
    tmp_path,
):
    repository_path = tmp_path / "repository.csv"
    participants_path = tmp_path / "participants.csv"

    clubs = [
        f"Club {index:02d}"
        for index in range(1, 39)
    ]

    pd.DataFrame(
        {
            "club": clubs,
        }
    ).to_csv(
        repository_path,
        index=False,
    )

    pd.DataFrame(
        {
            "club": clubs[:35],
        }
    ).to_csv(
        participants_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="exactly 36",
    ):
        load_structural_participants(
            repository_path,
            participants_path,
        )

def test_load_structural_participants_rejects_unknown_club(
    tmp_path,
):
    repository_path = tmp_path / "repository.csv"
    participants_path = tmp_path / "participants.csv"

    repository_clubs = [
        f"Club {index:02d}"
        for index in range(1, 39)
    ]

    selected_clubs = (
        repository_clubs[:35]
        + ["Unknown Club"]
    )

    pd.DataFrame(
        {
            "club": repository_clubs,
        }
    ).to_csv(
        repository_path,
        index=False,
    )

    pd.DataFrame(
        {
            "club": selected_clubs,
        }
    ).to_csv(
        participants_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="not present in production repository",
    ):
        load_structural_participants(
            repository_path,
            participants_path,
        )

def test_load_structural_participants_rejects_duplicates(
    tmp_path,
):
    repository_path = tmp_path / "repository.csv"
    participants_path = tmp_path / "participants.csv"

    clubs = [
        f"Club {index:02d}"
        for index in range(1, 39)
    ]

    participants = (
        clubs[:35]
        + [clubs[0]]
    )

    pd.DataFrame(
        {"club": clubs}
    ).to_csv(
        repository_path,
        index=False,
    )

    pd.DataFrame(
        {"club": participants}
    ).to_csv(
        participants_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        load_structural_participants(
            repository_path,
            participants_path,
        )
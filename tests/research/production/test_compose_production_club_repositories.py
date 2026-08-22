#test_compose_production_club_repositories

from __future__ import annotations

import pandas as pd

from research.production.compose_production_club_repositories import (
    compose_production_club_repositories,
)


def _repository_frame(
    *,
    clubs: list[str],
    repository_scope: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "club": clubs,
            "attack": [1.0] * len(clubs),
            "midfield": [1.0] * len(clubs),
            "defense": [1.0] * len(clubs),
            "goalkeeper": [1.0] * len(clubs),
            "attack_depth": [1.0] * len(clubs),
            "midfield_depth": [1.0] * len(clubs),
            "defense_depth": [1.0] * len(clubs),
            "squad_quality": [1.0] * len(clubs),
            "evidence_score": [1.0] * len(clubs),
            "representation_type": ["full_squad"] * len(clubs),
            "aggregation_profile": ["legacy_top_5"] * len(clubs),
            "player_count": [25] * len(clubs),
            "available_player_count": [25] * len(clubs),
            "repository_version": ["2026.27-v1"] * len(clubs),
            "repository_scope": [repository_scope] * len(clubs),
            "representation_season_id": ["2026-27"] * len(clubs),
        }
    )


def test_composer_combines_compatible_repositories(
    tmp_path,
):
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    output_path = tmp_path / "combined.csv"

    _repository_frame(
        clubs=["Club A", "Club B"],
        repository_scope="league_a",
    ).to_csv(
        first_path,
        index=False,
    )

    _repository_frame(
        clubs=["Club C", "Club D"],
        repository_scope="league_b",
    ).to_csv(
        second_path,
        index=False,
    )

    result = compose_production_club_repositories(
        source_paths=[
            first_path,
            second_path,
        ],
        repository_scope=(
            "champions_league_structural_v1"
        ),
        output_path=output_path,
    )

    assert len(result) == 4

    assert result["club"].tolist() == [
        "Club A",
        "Club B",
        "Club C",
        "Club D",
    ]

    assert (
        result["repository_scope"]
        .unique()
        .tolist()
        == ["champions_league_structural_v1"]
    )

    assert output_path.exists()

import pytest


def test_composer_rejects_different_repository_versions(
    tmp_path,
):
    first = _repository_frame(
        clubs=["Club A"],
        repository_scope="league_a",
    )

    second = _repository_frame(
        clubs=["Club B"],
        repository_scope="league_b",
    )

    second["repository_version"] = "different-version"

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)

    with pytest.raises(
        ValueError,
        match="repository_version",
    ):
        compose_production_club_repositories(
            source_paths=[
                first_path,
                second_path,
            ],
            repository_scope="combined",
            output_path=tmp_path / "combined.csv",
        )

def test_composer_rejects_duplicate_clubs_across_sources(
    tmp_path,
):
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    _repository_frame(
        clubs=["Club A", "Club B"],
        repository_scope="league_a",
    ).to_csv(first_path, index=False)

    _repository_frame(
        clubs=["Club B", "Club C"],
        repository_scope="league_b",
    ).to_csv(second_path, index=False)

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        compose_production_club_repositories(
            source_paths=[
                first_path,
                second_path,
            ],
            repository_scope="combined",
            output_path=tmp_path / "combined.csv",
        )

def test_composer_rejects_mismatched_schemas(
    tmp_path,
):
    first = _repository_frame(
        clubs=["Club A"],
        repository_scope="league_a",
    )

    second = _repository_frame(
        clubs=["Club B"],
        repository_scope="league_b",
    ).drop(
        columns=["goalkeeper"]
    )

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)

    with pytest.raises(
        ValueError,
        match="schemas do not match",
    ):
        compose_production_club_repositories(
            source_paths=[
                first_path,
                second_path,
            ],
            repository_scope="combined",
            output_path=tmp_path / "combined.csv",
        )

from research.adapters.football_model_adapter import (
    ProductionClubRepository,
)


def test_composed_repository_reloads_through_production_loader(
    tmp_path,
):
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    output_path = tmp_path / "combined.csv"

    _repository_frame(
        clubs=["Club A", "Club B"],
        repository_scope="league_a",
    ).to_csv(first_path, index=False)

    _repository_frame(
        clubs=["Club C", "Club D"],
        repository_scope="league_b",
    ).to_csv(second_path, index=False)

    compose_production_club_repositories(
        source_paths=[
            first_path,
            second_path,
        ],
        repository_scope="combined",
        output_path=output_path,
    )

    repository = ProductionClubRepository(
        output_path
    )

    assert set(repository.list_clubs()) == {
        "Club A",
        "Club B",
        "Club C",
        "Club D",
    }

    resolved = repository.resolve_club("Club C")

    assert resolved.club == "Club C"

def test_composer_allows_different_representation_season_ids(
    tmp_path,
):
    first = _repository_frame(
        clubs=["Club A"],
        repository_scope="league_a",
    )

    second = _repository_frame(
        clubs=["Club B"],
        repository_scope="league_b",
    )

    first[
        "representation_season_id"
    ] = "2026-27"

    second[
        "representation_season_id"
    ] = "97464"

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    output_path = tmp_path / "combined.csv"

    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)

    result = compose_production_club_repositories(
        source_paths=[
            first_path,
            second_path,
        ],
        repository_scope="combined",
        output_path=output_path,
    )

    assert result[
        "representation_season_id"
    ].astype(str).tolist() == [
        "2026-27",
        "97464",
    ]
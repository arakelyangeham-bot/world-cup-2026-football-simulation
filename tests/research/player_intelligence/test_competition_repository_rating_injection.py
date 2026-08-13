#test_competition_repository_rating_injection

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.player_intelligence.competition_player_repository import (
    CompetitionPlayerRepository,
)
from research.player_intelligence.competition_roster_builder import (
    CompetitionRosterBuilder,
)
from research.player_intelligence.competition_team_repository import (
    CompetitionTeamRepository,
)
from research.player_intelligence.player_repository import (
    DEFAULT_PLAYER_FEATURES_PATH,
    PlayerRepository,
)


def test_player_repository_uses_canonical_default_path() -> None:
    repository = PlayerRepository()

    assert (
        repository.player_features_path
        == DEFAULT_PLAYER_FEATURES_PATH
    )


def test_competition_repository_preserves_injected_player_repository(
    tmp_path: Path,
) -> None:
    ratings_path = (
        tmp_path
        / "alternative_player_ratings.csv"
    )

    player_repository = PlayerRepository(
        player_features_path=ratings_path
    )

    competition_repository = (
        CompetitionPlayerRepository(
            player_repository=player_repository
        )
    )

    assert (
        competition_repository.player_repository
        is player_repository
    )

    assert (
        competition_repository
        .player_repository
        .player_features_path
        == ratings_path
    )


def test_repository_dependency_chain_preserves_rating_path(
    tmp_path: Path,
) -> None:
    ratings_path = (
        tmp_path
        / "candidate_player_ratings.csv"
    )

    player_repository = PlayerRepository(
        player_features_path=ratings_path
    )

    competition_repository = (
        CompetitionPlayerRepository(
            player_repository=player_repository
        )
    )

    roster_builder = CompetitionRosterBuilder(
        repository=competition_repository
    )

    team_repository = CompetitionTeamRepository(
        roster_builder=roster_builder
    )

    resolved_path = (
        team_repository
        .roster_builder
        .repository
        .player_repository
        .player_features_path
    )

    assert resolved_path == ratings_path


def test_player_repository_loads_injected_file(
    tmp_path: Path,
) -> None:
    ratings_path = (
        tmp_path
        / "minimal_player_ratings.csv"
    )

    pd.DataFrame(
        [
            {
                "player_id": 123,
                "player": "Test Player",
                "country": "Test Country",
                "position": "M",
                "minutesPlayed": 900,
                "rating_AM": 0.75,
                "rating_CB": None,
                "rating_CM": 0.70,
                "rating_DM": None,
                "rating_FB": None,
                "rating_GK": None,
                "rating_ST": None,
                "rating_W": None,
                "rating_WM": None,
                "evidence_confidence": 1.0,
            }
        ]
    ).to_csv(
        ratings_path,
        index=False,
    )

    repository = PlayerRepository(
        player_features_path=ratings_path
    )

    players = repository.load_players()

    assert len(players) == 1

    player = players[0]

    assert (
        player.identity.player_id
        == "123"
    )

    assert (
        player.identity.name
        == "Test Player"
    )

    assert (
        player.identity.national_team
        == "Test Country"
    )

    assert (
        player.role_ratings.AM
        == pytest.approx(0.75)
    )

    assert (
        player.role_ratings.CM
        == pytest.approx(0.70)
    )
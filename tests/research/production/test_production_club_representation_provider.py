import pandas as pd
import pytest

from research.production.production_club_representation_provider import (
    ClubRepresentationProvider,
    build_club_representation,
    build_players_for_club,
    player_from_v4_rating_row,
)


def test_player_from_v4_rating_row_preserves_club_and_role_ratings():
    row = pd.Series(
        {
            "canonical_player_id": "101",
            "player_id": "101",
            "player": "Example Striker",
            "country": "England",
            "current_team": "Example FC",
            "position": "ST",
            "minutesPlayed": 1800.0,
            "total_weighted_evidence": 1500.0,
            "evidence_confidence": 0.9,
            "competition_count": 1,
            "season_count": 1,
            "rating_GK": None,
            "rating_CB": None,
            "rating_FB": None,
            "rating_DM": None,
            "rating_CM": None,
            "rating_AM": 0.4,
            "rating_WM": 0.5,
            "rating_W": 0.7,
            "rating_ST": 1.2,
        }
    )

    player = player_from_v4_rating_row(row)

    assert player.identity.player_id == "101"
    assert player.identity.name == "Example Striker"
    assert player.identity.club == "Example FC"
    assert player.identity.primary_position == "ST"

    assert player.role_ratings is not None
    assert player.role_ratings.ST == 1.2
    assert player.role_ratings.W == 0.7
    assert player.role_ratings.GK is None

    assert player.evidence.minutes_played == 1800.0
    assert player.evidence.total_weighted_evidence == 1500.0
    assert player.evidence.evidence_confidence == 0.9

    # Modern robust club representations deliberately preserve
    # the established zero-valued compatibility fields for
    # squad_quality and evidence_score.
    assert player.ratings.overall == 0.0

def test_build_players_for_club_uses_membership_ids():
    ratings = pd.DataFrame(
        [
            {
                "canonical_player_id": "101",
                "player_id": "101",
                "player": "Player One",
                "country": "England",
                "current_team": "Old Club",
                "position": "ST",
                "rating_ST": 1.2,
            },
            {
                "canonical_player_id": "102",
                "player_id": "102",
                "player": "Player Two",
                "country": "England",
                "current_team": "Another Club",
                "position": "CM",
                "rating_CM": 0.8,
            },
            {
                "canonical_player_id": "103",
                "player_id": "103",
                "player": "Player Three",
                "country": "England",
                "current_team": "Unrelated Club",
                "position": "CB",
                "rating_CB": 0.9,
            },
        ]
    )

    membership = pd.DataFrame(
        [
            {
                "player_id": "101",
                "team": "Example FC",
            },
            {
                "player_id": "102",
                "team": "Example FC",
            },
        ]
    )

    players = build_players_for_club(
        club="Example FC",
        membership=membership,
        ratings=ratings,
    )

    assert len(players) == 2

    assert {
        player.identity.player_id
        for player in players
    } == {"101", "102"}

    assert all(
        player.identity.club == "Example FC"
        for player in players
    )

def test_build_players_for_club_rejects_missing_rating():
    ratings = pd.DataFrame(
        [
            {
                "canonical_player_id": "101",
                "player_id": "101",
                "player": "Player One",
                "country": "England",
                "current_team": "Old Club",
                "position": "ST",
                "rating_ST": 1.2,
            },
        ]
    )

    membership = pd.DataFrame(
        [
            {
                "player_id": "101",
                "team": "Example FC",
            },
            {
                "player_id": "999",
                "team": "Example FC",
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing Player Ratings v4",
    ):
        build_players_for_club(
            club="Example FC",
            membership=membership,
            ratings=ratings,
        )

def test_build_players_for_club_rejects_duplicate_membership_ids():
    ratings = pd.DataFrame(
        [
            {
                "canonical_player_id": "101",
                "player_id": "101",
                "player": "Player One",
                "country": "England",
                "current_team": "Old Club",
                "position": "ST",
                "rating_ST": 1.2,
            },
        ]
    )

    membership = pd.DataFrame(
        [
            {
                "player_id": "101",
                "team": "Example FC",
            },
            {
                "player_id": "101",
                "team": "Example FC",
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicate player IDs",
    ):
        build_players_for_club(
            club="Example FC",
            membership=membership,
            ratings=ratings,
        )

def test_build_club_representation_returns_full_squad_representation():
    ratings = pd.DataFrame(
        [
            {
                "canonical_player_id": "101",
                "player_id": "101",
                "player": "Example Striker",
                "country": "England",
                "current_team": "Old Club",
                "position": "ST",
                "rating_ST": 1.2,
            },
            {
                "canonical_player_id": "102",
                "player_id": "102",
                "player": "Example Midfielder",
                "country": "England",
                "current_team": "Another Club",
                "position": "CM",
                "rating_CM": 0.8,
            },
            {
                "canonical_player_id": "103",
                "player_id": "103",
                "player": "Example Defender",
                "country": "England",
                "current_team": "Third Club",
                "position": "CB",
                "rating_CB": 0.9,
            },
        ]
    )

    membership = pd.DataFrame(
        [
            {"player_id": "101", "team": "Example FC"},
            {"player_id": "102", "team": "Example FC"},
            {"player_id": "103", "team": "Example FC"},
        ]
    )

    representation = build_club_representation(
        club="Example FC",
        membership=membership,
        ratings=ratings,
    )

    assert representation.national_team == "Example FC"
    assert representation.representation_type == "full_squad"
    assert representation.aggregation_profile == "legacy_top_5"

    assert representation.player_count == 3
    assert representation.available_player_count == 3

    assert representation.squad_quality == 0.0
    assert representation.evidence_score == 0.0

    assert representation.attack > 0.0
    assert representation.midfield > 0.0
    assert representation.defense > 0.0

def test_club_representation_provider_is_callable():
    ratings = pd.DataFrame(
        [
            {
                "canonical_player_id": "101",
                "player_id": "101",
                "player": "Player One",
                "country": "England",
                "current_team": "Old Club",
                "position": "ST",
                "rating_ST": 1.2,
            },
            {
                "canonical_player_id": "102",
                "player_id": "102",
                "player": "Player Two",
                "country": "England",
                "current_team": "Old Club",
                "position": "CM",
                "rating_CM": 0.8,
            },
        ]
    )

    membership = pd.DataFrame(
        [
            {
                "player_id": "101",
                "team": "Example FC",
            },
            {
                "player_id": "102",
                "team": "Example FC",
            },
            
        ]
    )

    provider = ClubRepresentationProvider(
        membership=membership,
        ratings=ratings,
    )

    representation = provider("Example FC")

    assert representation.national_team == "Example FC"
    assert representation.representation_type == "full_squad"
    assert representation.player_count == 2
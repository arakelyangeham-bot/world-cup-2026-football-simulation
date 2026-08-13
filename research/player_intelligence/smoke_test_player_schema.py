#smoke_test_player_schema.py

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.player_intelligence.player_schema import (
    Player,
    PlayerAvailability,
    PlayerDerivedTeamStrength,
    PlayerIdentity,
    PlayerRatings,
    Squad,
    StartingXI,
)


def main() -> None:
    goalkeeper = Player(
        identity=PlayerIdentity(
            player_id="test_gk_001",
            name="Test Goalkeeper",
            national_team="Testland",
            club="Test FC",
            primary_position="GK",
        ),
        ratings=PlayerRatings(
            overall=82.0,
            attack=10.0,
            midfield=20.0,
            defense=40.0,
            goalkeeper=88.0,
        ),
        availability=PlayerAvailability(
            available=True,
            expected_to_start=True,
        ),
    )

    striker = Player(
        identity=PlayerIdentity(
            player_id="test_st_001",
            name="Test Striker",
            national_team="Testland",
            club="Test FC",
            primary_position="ST",
        ),
        ratings=PlayerRatings(
            overall=84.0,
            attack=88.0,
            midfield=55.0,
            defense=25.0,
            goalkeeper=5.0,
        ),
    )

    squad = Squad(
        national_team="Testland",
        players=(goalkeeper, striker),
    )

    starting_xi = StartingXI(
        national_team="Testland",
        formation="4-3-3",
        players=(goalkeeper, striker),
    )

    team_strength = PlayerDerivedTeamStrength(
        national_team="Testland",
        attack=78.0,
        midfield=74.0,
        defense=72.0,
        goalkeeper=88.0,
        poisson_attack=1.45,
        poisson_defense=0.92,
        overall=77.5,
    )

    print("Player Intelligence Schema Smoke Test")
    print("-------------------------------------")
    print(f"Player: {goalkeeper.identity.name}")
    print(f"Squad: {squad.national_team}, players={len(squad.players)}")
    print(
        f"Starting XI: {starting_xi.national_team}, "
        f"formation={starting_xi.formation}, players={len(starting_xi.players)}"
    )
    print(
        f"Team strength: attack={team_strength.attack}, "
        f"gk={team_strength.goalkeeper}"
    )
    print()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
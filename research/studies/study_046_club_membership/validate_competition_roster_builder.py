#validate_competition_roster_builder

from __future__ import annotations

from research.player_intelligence.competition_roster_builder import (
    CompetitionRosterBuilder,
)


def main() -> None:
    builder = CompetitionRosterBuilder()

    competition_seasons = (
        builder.list_competition_seasons()
    )

    competition_seasons_processed = 0
    squads_processed = 0
    players_processed = 0

    identity_failures: list[str] = []
    player_count_failures: list[str] = []
    duplicate_player_failures: list[str] = []

    minimum_squad_size: int | None = None
    maximum_squad_size: int | None = None

    for season in competition_seasons.itertuples(
        index=False
    ):
        competition_id = int(
            season.competition_id
        )
        season_id = int(
            season.season_id
        )

        teams = builder.list_teams(
            competition_id=competition_id,
            season_id=season_id,
        )

        for team in teams.itertuples(
            index=False
        ):
            team_id = int(
                team.team_id
            )

            context = builder.get_context(
                competition_id=competition_id,
                season_id=season_id,
                team_id=team_id,
            )

            squad = builder.get_squad(
                competition_id=competition_id,
                season_id=season_id,
                team_id=team_id,
                require_complete_join=True,
            )

            membership_rows = (
                builder.repository
                .get_membership_rows(
                    competition_id=competition_id,
                    season_id=season_id,
                    team_id=team_id,
                )
            )

            expected_player_count = int(
                membership_rows["player_id"]
                .nunique()
            )

            if squad.national_team != context.team:
                identity_failures.append(
                    f"{context.competition} "
                    f"{context.season_year}: "
                    f"context={context.team!r}, "
                    f"squad={squad.national_team!r}"
                )

            if len(squad.players) != expected_player_count:
                player_count_failures.append(
                    f"{context.competition} "
                    f"{context.season_year}: "
                    f"{context.team}: "
                    f"expected={expected_player_count}, "
                    f"actual={len(squad.players)}"
                )

            player_ids = [
                str(
                    player.identity.player_id
                )
                for player in squad.players
            ]

            if len(player_ids) != len(set(player_ids)):
                duplicate_player_failures.append(
                    f"{context.competition} "
                    f"{context.season_year}: "
                    f"{context.team}"
                )

            squad_size = len(squad.players)

            minimum_squad_size = (
                squad_size
                if minimum_squad_size is None
                else min(
                    minimum_squad_size,
                    squad_size,
                )
            )

            maximum_squad_size = (
                squad_size
                if maximum_squad_size is None
                else max(
                    maximum_squad_size,
                    squad_size,
                )
            )

            squads_processed += 1
            players_processed += squad_size

        competition_seasons_processed += 1

    print("Competition Roster Builder Validation")
    print("=====================================")
    print()
    print(
        "Competition-seasons processed: "
        f"{competition_seasons_processed}"
    )
    print(
        f"Squads processed: {squads_processed}"
    )
    print(
        f"Players processed: {players_processed}"
    )
    print(
        "Minimum squad size: "
        f"{minimum_squad_size}"
    )
    print(
        "Maximum squad size: "
        f"{maximum_squad_size}"
    )
    print()
    print(
        "Identity failures: "
        f"{len(identity_failures)}"
    )
    print(
        "Player-count failures: "
        f"{len(player_count_failures)}"
    )
    print(
        "Duplicate-player failures: "
        f"{len(duplicate_player_failures)}"
    )

    if identity_failures:
        print()
        print("Identity failures")
        print("-----------------")

        for failure in identity_failures[:20]:
            print(failure)

    if player_count_failures:
        print()
        print("Player-count failures")
        print("---------------------")

        for failure in player_count_failures[:20]:
            print(failure)

    if duplicate_player_failures:
        print()
        print("Duplicate-player failures")
        print("-------------------------")

        for failure in duplicate_player_failures[:20]:
            print(failure)

    if (
        identity_failures
        or player_count_failures
        or duplicate_player_failures
    ):
        raise AssertionError(
            "Competition Roster Builder "
            "validation failed."
        )

    if squads_processed == 0:
        raise AssertionError(
            "No competition squads were built."
        )

    print()
    print("Context resolution: PASS")
    print("Squad construction: PASS")
    print("Player-count integrity: PASS")
    print("Player uniqueness: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
#validate_competition_player_repository

from __future__ import annotations

from research.player_intelligence.competition_player_repository import (
    CompetitionPlayerRepository,
)


def main() -> None:
    repository = CompetitionPlayerRepository()

    competition_seasons = (
        repository.list_competition_seasons()
    )

    teams_processed = 0
    memberships_processed = 0
    players_joined = 0

    empty_rosters: list[str] = []
    join_failures: list[str] = []

    for season in competition_seasons.itertuples(
        index=False
    ):
        teams = repository.list_teams(
            competition_id=int(
                season.competition_id
            ),
            season_id=int(
                season.season_id
            ),
        )

        for team in teams.itertuples(
            index=False
        ):
            membership_rows = (
                repository.get_membership_rows(
                    competition_id=int(
                        season.competition_id
                    ),
                    season_id=int(
                        season.season_id
                    ),
                    team_id=int(
                        team.team_id
                    ),
                )
            )

            if membership_rows.empty:
                empty_rosters.append(
                    f"{season.competition} "
                    f"{season.season_year}: "
                    f"{team.team}"
                )
                continue

            try:
                players = (
                    repository.get_players_for_team(
                        competition_id=int(
                            season.competition_id
                        ),
                        season_id=int(
                            season.season_id
                        ),
                        team_id=int(
                            team.team_id
                        ),
                        require_complete_join=True,
                    )
                )

            except ValueError as exc:
                join_failures.append(
                    f"{season.competition} "
                    f"{season.season_year}: "
                    f"{team.team}: {exc}"
                )
                continue

            membership_count = int(
                membership_rows["player_id"]
                .nunique()
            )

            if len(players) != membership_count:
                raise AssertionError(
                    "Joined player count does not match "
                    "unique membership count for "
                    f"{season.competition} "
                    f"{season.season_year}, "
                    f"{team.team}. "
                    f"Memberships={membership_count}, "
                    f"players={len(players)}."
                )

            joined_ids = {
                str(
                    player.identity.player_id
                )
                for player in players
            }

            membership_ids = set(
                membership_rows["player_id"]
                .astype(str)
            )

            if joined_ids != membership_ids:
                raise AssertionError(
                    "Joined Player IDs do not match "
                    "membership Player IDs for "
                    f"{season.competition} "
                    f"{season.season_year}, "
                    f"{team.team}."
                )

            teams_processed += 1
            memberships_processed += (
                membership_count
            )
            players_joined += len(players)

    print(
        "Competition Player Repository Validation"
    )
    print(
        "========================================"
    )
    print()
    print(
        "Competition-seasons: "
        f"{len(competition_seasons)}"
    )
    print(
        f"Teams processed: {teams_processed}"
    )
    print(
        "Unique memberships processed: "
        f"{memberships_processed}"
    )
    print(
        f"Player objects joined: {players_joined}"
    )
    print(
        f"Empty rosters: {len(empty_rosters)}"
    )
    print(
        f"Join failures: {len(join_failures)}"
    )

    if empty_rosters:
        print()
        print("Empty rosters")
        print("-------------")

        for item in empty_rosters[:20]:
            print(item)

    if join_failures:
        print()
        print("Join failures")
        print("-------------")

        for item in join_failures[:20]:
            print(item)

    if empty_rosters or join_failures:
        raise AssertionError(
            "Competition Player Repository "
            "validation failed."
        )

    print()
    print("Membership lookup: PASS")
    print("Player join integrity: PASS")
    print("Roster completeness: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
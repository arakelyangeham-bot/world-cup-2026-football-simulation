#validate_competition_team_repository

from __future__ import annotations

from math import isfinite

from research.player_intelligence.competition_team_repository import (
    CompetitionTeamRepository,
)


REPRESENTATION_FIELDS = (
    "attack",
    "midfield",
    "defense",
    "goalkeeper",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
    "squad_quality",
    "evidence_score",
)


def validate_numeric_fields(
    representation,
    label: str,
) -> None:
    for field in REPRESENTATION_FIELDS:
        value = getattr(
            representation,
            field,
        )

        if not isfinite(value):
            raise AssertionError(
                f"{label}: non-finite value for "
                f"{field}: {value}"
            )


def main() -> None:
    repository = CompetitionTeamRepository()

    competition_seasons = (
        repository
        .roster_builder
        .list_competition_seasons()
    )

    club_seasons = competition_seasons[
        competition_seasons[
            "competition_type"
        ]
        .astype(str)
        .str.contains(
            "club",
            case=False,
            na=False,
        )
    ]

    if club_seasons.empty:
        raise AssertionError(
            "No club competition-seasons were found."
        )

    teams_processed = 0
    full_squad_representations = 0
    starting_xi_representations = 0

    identity_failures: list[str] = []
    count_failures: list[str] = []
    representation_failures: list[str] = []

    for season in club_seasons.itertuples(
        index=False
    ):
        competition_id = int(
            season.competition_id
        )
        season_id = int(
            season.season_id
        )

        teams = (
            repository
            .roster_builder
            .list_teams(
                competition_id=competition_id,
                season_id=season_id,
            )
        )

        for team in teams.itertuples(
            index=False
        ):
            team_id = int(
                team.team_id
            )

            context = repository.get_context(
                competition_id=competition_id,
                season_id=season_id,
                team_id=team_id,
            )

            full_squad = (
                repository
                .get_team_representation(
                    competition_id=competition_id,
                    season_id=season_id,
                    team_id=team_id,
                    representation_type="full_squad",
                )
            )

            starting_xi = (
                repository
                .get_team_representation(
                    competition_id=competition_id,
                    season_id=season_id,
                    team_id=team_id,
                    representation_type=(
                        "expected_starting_xi"
                    ),
                    formation="4-3-3",
                )
            )

            validate_numeric_fields(
                full_squad,
                (
                    f"{context.competition} "
                    f"{context.season_year} "
                    f"{context.team} full_squad"
                ),
            )

            validate_numeric_fields(
                starting_xi,
                (
                    f"{context.competition} "
                    f"{context.season_year} "
                    f"{context.team} starting_xi"
                ),
            )

            if (
                full_squad.national_team
                != context.team
            ):
                identity_failures.append(
                    f"{context.team}: full-squad "
                    "identity mismatch."
                )

            if (
                starting_xi.national_team
                != context.team
            ):
                identity_failures.append(
                    f"{context.team}: starting-XI "
                    "identity mismatch."
                )

            if (
                full_squad.representation_type
                != "full_squad"
            ):
                representation_failures.append(
                    f"{context.team}: unexpected "
                    "full-squad representation type."
                )

            if (
                starting_xi.representation_type
                != "expected_starting_xi"
            ):
                representation_failures.append(
                    f"{context.team}: unexpected "
                    "starting-XI representation type."
                )

            if starting_xi.player_count != 11:
                count_failures.append(
                    f"{context.competition} "
                    f"{context.season_year}: "
                    f"{context.team}: expected XI "
                    f"contains "
                    f"{starting_xi.player_count} players."
                )

            if (
                full_squad.player_count
                < starting_xi.player_count
            ):
                count_failures.append(
                    f"{context.team}: full squad is "
                    "smaller than starting XI."
                )

            teams_processed += 1
            full_squad_representations += 1
            starting_xi_representations += 1

    print(
        "Competition Team Repository Validation"
    )
    print(
        "======================================"
    )
    print()
    print(
        "Club competition-seasons processed: "
        f"{len(club_seasons)}"
    )
    print(
        f"Teams processed: {teams_processed}"
    )
    print(
        "Full-squad representations: "
        f"{full_squad_representations}"
    )
    print(
        "Starting-XI representations: "
        f"{starting_xi_representations}"
    )
    print()
    print(
        "Identity failures: "
        f"{len(identity_failures)}"
    )
    print(
        "Player-count failures: "
        f"{len(count_failures)}"
    )
    print(
        "Representation failures: "
        f"{len(representation_failures)}"
    )

    if identity_failures:
        print()
        print("Identity failures")
        print("-----------------")

        for failure in identity_failures[:20]:
            print(failure)

    if count_failures:
        print()
        print("Player-count failures")
        print("---------------------")

        for failure in count_failures[:20]:
            print(failure)

    if representation_failures:
        print()
        print("Representation failures")
        print("-----------------------")

        for failure in representation_failures[:20]:
            print(failure)

    if (
        identity_failures
        or count_failures
        or representation_failures
    ):
        raise AssertionError(
            "Competition Team Repository "
            "validation failed."
        )

    if teams_processed == 0:
        raise AssertionError(
            "No club-team representations were built."
        )

    print()
    print("Context resolution: PASS")
    print("Full-squad representation: PASS")
    print("Starting-XI representation: PASS")
    print("Representation integrity: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
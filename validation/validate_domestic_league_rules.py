#validate_domestic_league_rules

from shared.domestic_league_rules import (
    DOMESTIC_LEAGUE_RULES,
)


EXPECTED_COMPLETED_MATCHES = {
    "premier_league": 380,
    "la_liga": 380,
    "serie_a": 380,
    "bundesliga": 306,
    "ligue_1": 306,
}


def main() -> None:
    print("Domestic League Rules Validation")
    print("===============================")

    for competition_key, expected_matches in (
        EXPECTED_COMPLETED_MATCHES.items()
    ):
        rules = DOMESTIC_LEAGUE_RULES[
            competition_key
        ]

        print()
        print(
            f"Competition: "
            f"{rules.competition_key}"
        )
        print(
            f"Teams: {rules.team_count}"
        )
        print(
            "Matches per team: "
            f"{rules.matches_per_team}"
        )
        print(
            "Home/away split: "
            f"{rules.home_matches_per_team}/"
            f"{rules.away_matches_per_team}"
        )
        print(
            "Completed matches: "
            f"{rules.completed_match_count}"
        )
        print(
            "Unique pairings: "
            f"{rules.unique_pairing_count}"
        )

        if (
            rules.completed_match_count
            != expected_matches
        ):
            raise ValueError(
                f"{competition_key} produced "
                f"{rules.completed_match_count} matches; "
                f"expected {expected_matches}."
            )

        if (
            rules.home_matches_per_team
            + rules.away_matches_per_team
            != rules.matches_per_team
        ):
            raise ValueError(
                f"{competition_key} has an invalid "
                "home-away split."
            )

    print()
    print(
        "All domestic league rule checks passed."
    )


if __name__ == "__main__":
    main()
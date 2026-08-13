#validate_domestic_competition_registry

from shared.competition_registry import get_competition


DOMESTIC_COMPETITION_KEYS = [
    "premier_league",
    "la_liga",
    "serie_a",
    "bundesliga",
    "ligue_1",
]


def main() -> None:
    print("Domestic Competition Registry Validation")
    print("========================================")

    for key in DOMESTIC_COMPETITION_KEYS:
        competition = get_competition(key)

        filename = competition.filename_pattern.format(
            year=2024,
        )

        print()
        print(f"Key: {competition.key}")
        print(f"Name: {competition.display_name}")
        print(f"Category: {competition.category}")
        print(f"Importance: {competition.importance}")
        print(f"Example filename: {filename}")

        if competition.category != "domestic_league":
            raise ValueError(
                f"{key} has incorrect category: "
                f"{competition.category}"
            )

    print()
    print("All domestic competition registry checks passed.")


if __name__ == "__main__":
    main()
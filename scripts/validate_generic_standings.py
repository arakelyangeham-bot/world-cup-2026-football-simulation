#validate_generic_standings.py

from simulation.competition import StandingsTable


def main() -> None:
    table = StandingsTable(
        teams=[
            "Argentina",
            "Japan",
            "Morocco",
            "Canada",
        ]
    )

    table.record_match("Argentina", "Japan", 2, 1)
    table.record_match("Morocco", "Canada", 1, 1)
    table.record_match("Argentina", "Morocco", 0, 0)
    table.record_match("Japan", "Canada", 3, 2)
    table.record_match("Argentina", "Canada", 4, 1)
    table.record_match("Japan", "Morocco", 1, 1)

    for row in table.as_rows():
        print(
            f"{row['rank']}. {row['team']:<12} "
            f"{row['points']} pts "
            f"GF {row['goals_for']} "
            f"GA {row['goals_against']} "
            f"GD {row['goal_difference']}"
        )


if __name__ == "__main__":
    main()
#audit_competition_registry.py

from shared.competition_registry import COMPETITIONS


def main() -> None:
    print(f"Registered competitions: {len(COMPETITIONS)}")
    print()

    print(
        f"{'Key':25}"
        f"{'Importance':>12}"
        f"{'Category':>35}"
        f"  Display Name"
    )
    print("-" * 95)

    for competition in COMPETITIONS:
        print(
            f"{competition.key:25}"
            f"{competition.importance:12.2f}"
            f"{competition.category:>35}"
            f"  {competition.display_name}"
        )


if __name__ == "__main__":
    main()
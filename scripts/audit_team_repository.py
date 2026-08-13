from scripts.team_strength_loader import load_team_repository


def compute_strength_score(team: dict) -> float:
    """
    Simple diagnostic score.

    This is NOT used by the simulator.
    It simply helps us inspect whether repository values
    are broadly sensible.
    """

    attack = team["attack"]

    # Lower defensive values are better.
    defense_component = -team["defense"]

    fifa_component = team.get("fifa_points_diff", 0.0)

    return attack + defense_component + fifa_component


def main():
    repository = load_team_repository()

    print(f"Loaded {len(repository)} teams.\n")

    rows = []

    for team_name, team in repository.items():
        rows.append(
            {
                "team": team_name,
                "fifa": team.get("fifa_points", 0),
                "attack": team["attack"],
                "defense": team["defense"],
                "fifa_points_diff": team.get("fifa_points_diff", 0.0),
                "strength_score": compute_strength_score(team),
            }
        )

    rows.sort(
        key=lambda r: r["fifa"],
        reverse=True,
    )

    print(
        f"{'Team':25}"
        f"{'FIFA':>8}"
        f"{'Attack':>12}"
        f"{'Defense':>12}"
        f"{'FIFA Δ':>12}"
        f"{'Score':>12}"
    )

    print("-" * 81)

    for row in rows:
        print(
            f"{row['team'][:25]:25}"
            f"{row['fifa']:8.1f}"
            f"{row['attack']:12.3f}"
            f"{row['defense']:12.3f}"
            f"{row['fifa_points_diff']:12.3f}"
            f"{row['strength_score']:12.3f}"
        )


if __name__ == "__main__":
    main()
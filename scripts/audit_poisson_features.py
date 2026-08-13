from statistics import mean, stdev

from scripts.team_strength_loader import load_team_repository


def correlation(xs: list[float], ys: list[float]) -> float:
    x_mean = mean(xs)
    y_mean = mean(ys)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    )

    denominator_x = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    denominator_y = sum((y - y_mean) ** 2 for y in ys) ** 0.5

    if denominator_x == 0 or denominator_y == 0:
        return 0.0

    return numerator / (denominator_x * denominator_y)


def print_rows(title: str, rows: list[dict], sort_key: str, reverse: bool = True) -> None:
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'Team':25}"
        f"{'Attack':>10}"
        f"{'Pois Att':>10}"
        f"{'Defense':>10}"
        f"{'Pois Def':>10}"
    )

    for row in sorted(rows, key=lambda r: r[sort_key], reverse=reverse)[:10]:
        print(
            f"{row['team'][:25]:25}"
            f"{row['attack']:10.3f}"
            f"{row['poisson_attack']:10.3f}"
            f"{row['defense']:10.3f}"
            f"{row['poisson_defense']:10.3f}"
        )


def main() -> None:
    repository = load_team_repository()

    rows = []

    for team_name, team in repository.items():
        rows.append(
            {
                "team": team_name,
                "attack": float(team["attack"]),
                "defense": float(team["defense"]),
                "poisson_attack": float(team["poisson_attack"]),
                "poisson_defense": float(team["poisson_defense"]),
            }
        )

    print(f"Loaded {len(rows)} teams.")

    print_rows("Top 10 by composite attack", rows, "attack")
    print_rows("Top 10 by poisson attack", rows, "poisson_attack")
    print_rows("Best 10 by composite defense", rows, "defense", reverse=False)
    print_rows("Best 10 by poisson defense", rows, "poisson_defense", reverse=False)

    attack_values = [row["attack"] for row in rows]
    poisson_attack_values = [row["poisson_attack"] for row in rows]
    defense_values = [row["defense"] for row in rows]
    poisson_defense_values = [row["poisson_defense"] for row in rows]

    print()
    print("Feature family correlations")
    print("---------------------------")
    print(
        "Composite attack vs poisson attack: "
        f"{correlation(attack_values, poisson_attack_values):.3f}"
    )
    print(
        "Composite defense vs poisson defense: "
        f"{correlation(defense_values, poisson_defense_values):.3f}"
    )

    print()
    print("Poisson feature distributions")
    print("-----------------------------")
    print(
        f"Poisson attack mean/std/min/max: "
        f"{mean(poisson_attack_values):.3f} / "
        f"{stdev(poisson_attack_values):.3f} / "
        f"{min(poisson_attack_values):.3f} / "
        f"{max(poisson_attack_values):.3f}"
    )
    print(
        f"Poisson defense mean/std/min/max: "
        f"{mean(poisson_defense_values):.3f} / "
        f"{stdev(poisson_defense_values):.3f} / "
        f"{min(poisson_defense_values):.3f} / "
        f"{max(poisson_defense_values):.3f}"
    )


if __name__ == "__main__":
    main()
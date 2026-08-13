#audit_team_strength_distribution.py

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


def print_distribution(name: str, rows: list[dict], key: str) -> None:
    values = [row[key] for row in rows]

    print()
    print(name)
    print("-" * len(name))
    print(f"Mean: {mean(values):.3f}")
    print(f"Std dev: {stdev(values):.3f}")
    print(f"Min: {min(values):.3f}")
    print(f"Max: {max(values):.3f}")

    print("\nTop 10:")
    for row in sorted(rows, key=lambda r: r[key], reverse=True)[:10]:
        print(f"  {row['team']:<25} {row[key]:>8.3f}")

    print("\nBottom 10:")
    for row in sorted(rows, key=lambda r: r[key])[:10]:
        print(f"  {row['team']:<25} {row[key]:>8.3f}")


def main() -> None:
    repository = load_team_repository()

    rows = []

    for team_name, team in repository.items():
        rows.append(
            {
                "team": team_name,
                "fifa": float(team.get("fifa_points", 0.0)),
                "attack": float(team["attack"]),
                "defense": float(team["defense"]),
            }
        )

    print(f"Loaded {len(rows)} teams.")

    print_distribution("Attack distribution", rows, "attack")
    print_distribution("Defense distribution", rows, "defense")

    fifa_values = [row["fifa"] for row in rows]
    attack_values = [row["attack"] for row in rows]
    defense_values = [row["defense"] for row in rows]

    print()
    print("Correlations")
    print("------------")
    print(f"FIFA vs attack:  {correlation(fifa_values, attack_values):.3f}")
    print(f"FIFA vs defense: {correlation(fifa_values, defense_values):.3f}")


if __name__ == "__main__":
    main()
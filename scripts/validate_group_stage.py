# scripts/validate_group_stage.py

from collections import Counter

from team_strength_loader import load_poisson_team_strengths
from wc2026_group_stage import simulate_group_stage, extract_qualifiers_from_standings


def main(n: int = 1000):
    team_strengths = load_poisson_team_strengths()

    total_matches = 0
    total_goals = 0
    points_counter = Counter()
    qualification_counter = Counter()

    for _ in range(n):
        standings = simulate_group_stage(team_strengths)
        qualifiers = extract_qualifiers_from_standings(standings)

        for rows in standings.values():
            for row in rows:
                points_counter[row.points] += 1
                total_goals += row.goals_for

            total_matches += 6

        for q in qualifiers:
            qualification_counter[q.team] += 1

    print(f"Simulations: {n}")
    print(f"Avg goals per match: {total_goals / total_matches:.3f}")

    print()
    print("Group points distribution:")
    for points, count in sorted(points_counter.items()):
        print(f"{points:2d} pts: {count}")

    print()
    print("Top 20 qualification rates:")
    for team, count in qualification_counter.most_common(20):
        print(f"{team}: {count / n:.3f}")


if __name__ == "__main__":
    main()
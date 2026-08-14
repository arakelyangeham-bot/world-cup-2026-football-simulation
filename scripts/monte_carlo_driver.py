# monte_carlo_driver.py

import argparse
from collections import Counter
import random
from pathlib import Path
import csv

import numpy as np

from scripts.team_strength_loader import load_team_repository
from scripts.wc2026_tournament_simulator import simulate_tournament
from simulation.observers import (
    ObserverManager,
    StatisticsObserver,
    SimulationStatistics,
    ExtremeEventsObserver,
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the WC 2026 Monte Carlo tournament simulator "
            "and write tournament probability outputs."
        )
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help=(
            "Number of tournament simulations to run. "
            "Default: 1000."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used for Python and NumPy. "
            "Default: 42."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/monte_carlo"),
        help=(
            "Directory for generated Monte Carlo CSV outputs. "
            "Default: outputs/monte_carlo."
        ),
    )

    return parser.parse_args()

def counter_to_prob_rows(counter: Counter, n: int) -> list[dict]:
    return [
        {
            "team": team,
            "count": count,
            "probability": count / n,
        }
        for team, count in counter.most_common()
    ]

def run_monte_carlo(
    n: int = 1000,
    seed: int | None = None,
    team_repository: dict[str, dict] | None = None,
) -> dict:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if team_repository is None:
        team_repository = load_team_repository()

    print(
            f"Loaded canonical team repository "
            f"({len(team_repository)} teams)"
        )

    champion_counter = Counter()
    runner_up_counter = Counter()
    semifinal_counter = Counter()
    quarterfinal_counter = Counter()
    round_of_16_counter = Counter()
    observer_manager = ObserverManager([
        StatisticsObserver(),
        ExtremeEventsObserver(),
    ])

    for simulation_id in range(n):
        result = simulate_tournament(team_repository)

        observer_manager.observe(
            result=result,
            simulation_id=simulation_id,
            team_repository=team_repository,
        )

        champion_counter[result.champion] += 1
        runner_up_counter[result.runner_up] += 1

        for team in result.semifinalists:
            semifinal_counter[team] += 1

        for team in result.quarterfinalists:
            quarterfinal_counter[team] += 1

        for team in result.round_of_16:
            round_of_16_counter[team] += 1
    
    observer_outputs = observer_manager.finalize()

    return {
        "champion": counter_to_prob_rows(champion_counter, n),
        "runner_up": counter_to_prob_rows(runner_up_counter, n),
        "semifinal": counter_to_prob_rows(semifinal_counter, n),
        "quarterfinal": counter_to_prob_rows(quarterfinal_counter, n),
        "round_of_16": counter_to_prob_rows(round_of_16_counter, n),
        "statistics": observer_outputs["StatisticsObserver"]["statistics"],
        "statistics_summary": observer_outputs["StatisticsObserver"]["summary"],
        "extreme_event_leaderboards": observer_outputs["ExtremeEventsObserver"]["rows"],
    }

def run_monte_carlo_repository(
    repository_path: Path | None,
    n: int,
    seed: int,
) -> dict:
    repository = load_team_repository(path=repository_path)

    return run_monte_carlo(
        n=n,
        seed=seed,
        team_repository=repository,
    )

def print_statistics(stats: SimulationStatistics) -> None:
    print()
    print("Tournament statistics")
    print("---------------------")
    print(f"Tournaments: {stats.tournaments}")
    print(f"Total matches: {stats.total_matches}")
    print(f"Total goals: {stats.total_goals}")
    print(f"Avg goals per tournament: {stats.total_goals / stats.tournaments:.3f}")
    print(f"Avg goals per match: {stats.total_goals / stats.total_matches:.3f}")
    print(f"Extra-time frequency: {stats.extra_time_matches / stats.total_matches:.3f}")
    print(f"Penalty-shootout frequency: {stats.penalty_shootouts / stats.total_matches:.3f}")

def print_table(title: str, rows: list[dict], limit: int = 20) -> None:
    print()
    print(title)
    print("-" * len(title))

    for row in rows[:limit]:
        print(
            f"{row['team']:<25} "
            f"{row['count']:>6} "
            f"{row['probability']:.3f}"
        )

def write_probability_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["team", "count", "probability"])
        writer.writeheader()
        writer.writerows(rows)

def write_statistics_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

def write_extreme_event_leaderboards_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({
        field
        for row in rows
        for field in row.keys()
    })

    preferred = [
        "event_name",
        "simulation_id",
        "value",
        "description",
    ]

    ordered_fieldnames = [
        field for field in preferred
        if field in fieldnames
    ] + [
        field for field in fieldnames
        if field not in preferred
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_outputs(results: dict, output_dir: str = "outputs/monte_carlo") -> None:
    output_path = Path(output_dir)

    write_probability_csv(
        output_path / "champion_probabilities.csv",
        results["champion"],
    )
    write_probability_csv(
        output_path / "runner_up_probabilities.csv",
        results["runner_up"],
    )
    write_probability_csv(
        output_path / "semifinal_probabilities.csv",
        results["semifinal"],
    )
    write_probability_csv(
        output_path / "quarterfinal_probabilities.csv",
        results["quarterfinal"],
    )
    write_probability_csv(
        output_path / "round_of_16_probabilities.csv",
        results["round_of_16"],
    )
    write_statistics_csv(
        output_path / "simulation_statistics.csv",
        results["statistics_summary"],
    )
    write_extreme_event_leaderboards_csv(
        output_path / "extreme_event_leaderboards.csv",
        results["extreme_event_leaderboards"],
    )
    print()
    print(f"Wrote Monte Carlo outputs to: {output_path}")

def main():
    arguments = parse_arguments()

    if arguments.simulations <= 0:
        raise SystemExit(
            "--simulations must be greater than zero."
        )

    n = arguments.simulations
    seed = arguments.seed

    results = run_monte_carlo(
        n=n,
        seed=seed,
    )

    print(f"Simulations: {n}")
    print(f"Seed: {seed}")

    print_table(
        "Champion probability",
        results["champion"],
    )
    print_table(
        "Runner-up probability",
        results["runner_up"],
    )
    print_table(
        "Semifinal appearances",
        results["semifinal"],
    )
    print_table(
        "Quarterfinal appearances",
        results["quarterfinal"],
    )
    print_table(
        "Round of 16 appearances",
        results["round_of_16"],
    )

    print_statistics(
        results["statistics"]
    )

    write_outputs(
        results,
        output_dir=arguments.output_dir,
    )


if __name__ == "__main__":
    main()
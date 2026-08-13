#benchmark_match_models.py

from dataclasses import dataclass
from math import log2
from collections import Counter

from scripts.monte_carlo_driver import run_monte_carlo
import simulation.simulation_config as simulation_config


@dataclass
class BenchmarkResult:
    model_name: str
    tournaments: int
    avg_goals_per_tournament: float
    avg_goals_per_match: float
    extra_time_rate: float
    penalty_rate: float
    unique_champions: int
    champion_entropy: float
    top_champion: str
    top_champion_probability: float


def entropy(rows: list[dict]) -> float:
    value = 0.0

    for row in rows:
        p = row["probability"]
        if p > 0:
            value -= p * log2(p)

    return value


def summarize_results(model_name: str, results: dict) -> BenchmarkResult:
    stats = results["statistics"]
    champions = results["champion"]

    return BenchmarkResult(
        model_name=model_name,
        tournaments=stats.tournaments,
        avg_goals_per_tournament=(
            stats.total_goals / stats.tournaments
        ),
        avg_goals_per_match=(
            stats.total_goals / stats.total_matches
        ),
        extra_time_rate=(
            stats.extra_time_matches / stats.total_matches
        ),
        penalty_rate=(
            stats.penalty_shootouts / stats.total_matches
        ),
        unique_champions=len(champions),
        champion_entropy=entropy(champions),
        top_champion=champions[0]["team"],
        top_champion_probability=champions[0]["probability"],
    )


def print_benchmark_table(rows: list[BenchmarkResult]) -> None:
    print()
    print("Model benchmark")
    print("---------------")

    print(
        f"{'Model':18}"
        f"{'Tourn':>8}"
        f"{'Goals/T':>10}"
        f"{'Goals/M':>10}"
        f"{'ET Rate':>10}"
        f"{'Pen Rate':>10}"
        f"{'Champs':>8}"
        f"{'Entropy':>10}"
        f"{'Top Champion':>18}"
        f"{'Top P':>8}"
    )

    for row in rows:
        print(
            f"{row.model_name:18}"
            f"{row.tournaments:8}"
            f"{row.avg_goals_per_tournament:10.3f}"
            f"{row.avg_goals_per_match:10.3f}"
            f"{row.extra_time_rate:10.3f}"
            f"{row.penalty_rate:10.3f}"
            f"{row.unique_champions:8}"
            f"{row.champion_entropy:10.3f}"
            f"{row.top_champion[:18]:>18}"
            f"{row.top_champion_probability:8.3f}"
        )


def run_benchmark_for_lambda_model(
    lambda_model: str,
    n: int,
    seed: int,
) -> BenchmarkResult:
    original_lambda_model = simulation_config.LAMBDA_MODEL

    try:
        simulation_config.LAMBDA_MODEL = lambda_model

        results = run_monte_carlo(
            n=n,
            seed=seed,
        )

        return summarize_results(
            lambda_model,
            results,
        )

    finally:
        simulation_config.LAMBDA_MODEL = original_lambda_model


def print_comparison(rows: list[BenchmarkResult]) -> None:
    if len(rows) != 2:
        return

    left, right = rows

    print()
    print("Model comparison")
    print("----------------")
    print(f"{'Metric':28}{left.model_name:>14}{right.model_name:>14}{'Delta':>14}")
    print("-" * 70)

    comparisons = [
        (
            "Goals / tournament",
            left.avg_goals_per_tournament,
            right.avg_goals_per_tournament,
        ),
        (
            "Goals / match",
            left.avg_goals_per_match,
            right.avg_goals_per_match,
        ),
        (
            "Extra-time rate",
            left.extra_time_rate,
            right.extra_time_rate,
        ),
        (
            "Penalty rate",
            left.penalty_rate,
            right.penalty_rate,
        ),
        (
            "Unique champions",
            left.unique_champions,
            right.unique_champions,
        ),
        (
            "Champion entropy",
            left.champion_entropy,
            right.champion_entropy,
        ),
        (
            "Top champion probability",
            left.top_champion_probability,
            right.top_champion_probability,
        ),
    ]

    for label, left_value, right_value in comparisons:
        delta = right_value - left_value

        print(
            f"{label:<28}"
            f"{left_value:>14.3f}"
            f"{right_value:>14.3f}"
            f"{delta:>14.3f}"
        )


def main() -> None:
    n = 1000
    seed = 42

    print("Running benchmark")
    print("-----------------")
    print(f"Tournaments per model: {n}")
    print(f"Seed: {seed}")

    rows = [
        run_benchmark_for_lambda_model(
            "heuristic",
            n=n,
            seed=seed,
        ),
        run_benchmark_for_lambda_model(
            "calibrated",
            n=n,
            seed=seed,
        ),
    ]

    print_benchmark_table(rows)
    print_comparison(rows)


if __name__ == "__main__":
    main()
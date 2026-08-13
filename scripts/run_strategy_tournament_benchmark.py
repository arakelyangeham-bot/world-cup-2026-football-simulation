#run_strategy_tournament_benchmark.py

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.monte_carlo_driver import (
    run_monte_carlo_repository,
    write_outputs,
    print_statistics,
    print_table,
)


REPOSITORY_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_011_team_representation_calibration"
    / "repositories"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "study_011_team_representation_calibration"
    / "tournament_benchmarks"
)

N_SIMULATIONS = 1000
SEED = 42


REPOSITORIES = {
    "legacy": None,
    "dimension_specific": REPOSITORY_DIR / "dimension_specific_team_repository.csv",
    "top_11_mean": REPOSITORY_DIR / "top_11_mean_team_repository.csv",
    "top_5_mean": REPOSITORY_DIR / "top_5_mean_team_repository.csv",
    "star_weighted": REPOSITORY_DIR / "star_weighted_team_repository.csv",
}


def main() -> None:
    for strategy, repository_path in REPOSITORIES.items():
        print()
        print("=" * 80)
        print(f"Running tournament benchmark: {strategy}")
        print("=" * 80)

        results = run_monte_carlo_repository(
            repository_path=repository_path,
            n=N_SIMULATIONS,
            seed=SEED,
        )

        print(f"Simulations: {N_SIMULATIONS}")
        print(f"Seed: {SEED}")

        print_table("Champion probability", results["champion"])
        print_table("Runner-up probability", results["runner_up"])
        print_table("Semifinal appearances", results["semifinal"])
        print_table("Quarterfinal appearances", results["quarterfinal"])
        print_table("Round of 16 appearances", results["round_of_16"])
        print_statistics(results["statistics"])

        output_dir = OUTPUT_ROOT / strategy
        write_outputs(
            results=results,
            output_dir=str(output_dir),
        )

    print()
    print("Strategy tournament benchmarks complete.")
    print(f"Wrote outputs under -> {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
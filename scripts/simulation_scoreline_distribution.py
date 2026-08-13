#simulation_scoreline_distribution.py

from collections import Counter
from pathlib import Path
import random

import numpy as np
import pandas as pd

from simulation_utils import canonical_scoreline
from team_strength_loader import load_poisson_team_strengths
from wc2026_tournament_simulator import simulate_tournament


OUTPUT_DIR = Path("outputs/monte_carlo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def iter_all_matches(result):

    # Group stage
    for match in result.group_stage_results:
        yield match

    # Knockout
    for stage in (
        result.r32_results,
        result.r16_results,
        result.qf_results,
        result.sf_results,
        result.third_place_results,
        result.final_results,
    ):
        for match in stage:
            yield match


def main(n: int = 1000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    strengths = load_poisson_team_strengths()

    counter = Counter()
    total_matches = 0

    for i in range(n):
        result = simulate_tournament(strengths)

        for match in iter_all_matches(result):
            scoreline = canonical_scoreline(
                match.goals_team1,
                match.goals_team2,
            )
            counter[scoreline] += 1
            total_matches += 1

        if (i + 1) % 100 == 0:
            print(f"Simulated tournaments: {i + 1}")

    distribution = (
        pd.DataFrame(
            [
                {
                    "scoreline": scoreline,
                    "count": count,
                    "probability": count / total_matches,
                }
                for scoreline, count in counter.items()
            ]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    out_file = OUTPUT_DIR / "simulation_scoreline_distribution.csv"
    distribution.to_csv(out_file, index=False)

    print()
    print(distribution.head(25).to_string(index=False))
    print()
    print(f"Total matches: {total_matches}")
    print(f"Probability sum: {distribution['probability'].sum():.6f}")
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
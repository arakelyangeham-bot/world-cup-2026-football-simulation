#audit_fast_predictor_equivalence.py

import math
from time import perf_counter

from scripts.team_strength_loader import load_team_repository
from inference.match_predictor import MatchPredictor


MATCHUPS = [
    ("Argentina", "France"),
    ("Spain", "Germany"),
    ("Brazil", "England"),
    ("Netherlands", "Portugal"),
    ("Canada", "Mexico"),
]

N = 1000


def main() -> None:
    repo = load_team_repository()
    predictor = MatchPredictor()

    print("Fast Predictor Equivalence Audit")
    print("--------------------------------")

    for home, away in MATCHUPS:
        home_data = repo[home]
        away_data = repo[away]

        slow = predictor.predict_match(home_data, away_data)
        fast = predictor.predict_match_fast(home_data, away_data)

        print()
        print(f"{home} vs {away}")
        print("-" * (len(home) + len(away) + 4))
        print("slow:", slow)
        print("fast:", fast)

        for key in slow:
            if not math.isclose(slow[key], fast[key], rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(
                    f"Mismatch for {home} vs {away}, {key}: "
                    f"slow={slow[key]}, fast={fast[key]}"
                )

    print()
    print("Prediction equivalence: PASS")

    first_home, first_away = MATCHUPS[0]
    home_data = repo[first_home]
    away_data = repo[first_away]

    start = perf_counter()
    for _ in range(N):
        predictor.predict_match(home_data, away_data)
    slow_time = perf_counter() - start

    start = perf_counter()
    for _ in range(N):
        predictor.predict_match_fast(home_data, away_data)
    fast_time = perf_counter() - start

    print()
    print("Timing")
    print("------")
    print(f"Iterations: {N}")
    print(f"Slow path total: {slow_time:.3f}s")
    print(f"Fast path total: {fast_time:.3f}s")
    print(f"Speedup: {slow_time / fast_time:.2f}x")


if __name__ == "__main__":
    main()
#audit_ml_predictor_runtime.py

from pathlib import Path
from time import perf_counter

import cProfile
import pstats
from io import StringIO

from scripts.team_strength_loader import load_team_repository
from inference.match_predictor import MatchPredictor


N = 500


MATCHUPS = [
    ("Argentina", "France"),
    ("Spain", "Germany"),
    ("Brazil", "England"),
    ("Netherlands", "Portugal"),
    ("Canada", "Mexico"),
]


def profile_predictor(predictor, home_team, away_team) -> None:
    profiler = cProfile.Profile()

    profiler.enable()

    for _ in range(N):
        predictor.predict_match(home_team, away_team)

    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(35)

    print(stream.getvalue())


def main() -> None:
    repo = load_team_repository()
    predictor = MatchPredictor()

    print("ML Predictor Runtime Audit")
    print("--------------------------")
    print(f"Iterations per matchup: {N}")

    for home, away in MATCHUPS:
        print()
        print(f"{home} vs {away}")
        print("-" * (len(home) + len(away) + 4))

        home_data = repo[home]
        away_data = repo[away]

        start = perf_counter()

        for _ in range(N):
            predictor.predict_match(home_data, away_data)

        elapsed = perf_counter() - start

        print(f"Total time: {elapsed:.3f}s")
        print(f"Per prediction: {elapsed / N:.6f}s")

    print()
    print("Detailed profile for first matchup")
    print("----------------------------------")

    first_home, first_away = MATCHUPS[0]
    profile_predictor(
        predictor,
        repo[first_home],
        repo[first_away],
    )


if __name__ == "__main__":
    main()
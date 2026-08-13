#profile_single_tournament.py

import cProfile
import pstats
from io import StringIO

from scripts.team_strength_loader import load_team_repository
from scripts.wc2026_tournament_simulator import simulate_tournament


def main() -> None:
    team_repository = load_team_repository()

    profiler = cProfile.Profile()

    profiler.enable()
    simulate_tournament(team_repository)
    profiler.disable()

    stream = StringIO()

    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(40)

    print("Single Tournament Profile")
    print("-------------------------")
    print(stream.getvalue())


if __name__ == "__main__":
    main()
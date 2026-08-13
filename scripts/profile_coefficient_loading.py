#profile_coefficient_loading.py

from time import perf_counter

from simulation.poisson_calibration import load_poisson_goal_coefficients


N = 100_000


def main() -> None:
    print("Poisson Coefficient Loading Profile")
    print("-----------------------------------")
    print(f"Iterations: {N:,}")
    print()

    #
    # First call
    #
    start = perf_counter()

    coefficients = load_poisson_goal_coefficients()

    elapsed = perf_counter() - start

    print(f"First load: {elapsed:.6f} seconds")
    print(f"Models loaded: {len(coefficients)}")

    #
    # Cached calls
    #
    start = perf_counter()

    for _ in range(N):
        load_poisson_goal_coefficients()

    elapsed = perf_counter() - start

    print()
    print(f"{N:,} cached loads: {elapsed:.6f} seconds")
    print(f"Average per cached call: {elapsed / N:.10f} seconds")

    #
    # Identity check
    #
    print()
    print("Cache verification")
    print("------------------")

    a = load_poisson_goal_coefficients()
    b = load_poisson_goal_coefficients()

    print(f"Same object: {a is b}")


if __name__ == "__main__":
    main()
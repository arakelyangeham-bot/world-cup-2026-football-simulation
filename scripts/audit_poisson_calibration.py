#audit_poisson_calibration.py

from simulation.poisson_calibration import load_poisson_goal_coefficients


def main() -> None:
    coefficients = load_poisson_goal_coefficients()

    for model_name, model_coefficients in coefficients.items():
        print()
        print(model_name)
        print("-" * len(model_name))

        for feature, value in model_coefficients.items():
            print(f"{feature:<30} {value:>12.6f}")


if __name__ == "__main__":
    main()
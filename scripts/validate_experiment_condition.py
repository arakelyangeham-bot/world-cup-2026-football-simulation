#validate_experiment_condition.py

from research import ExperimentCondition


def main() -> None:
    league = ExperimentCondition(
        name="League Baseline",
        competition_format="league",
        repository_source="synthetic_strength_ladder",
        match_engine="synthetic_match_model",
        simulation_count=1000,
        random_seed=31031,
    )

    knockout = ExperimentCondition(
        name="Knockout Baseline",
        competition_format="knockout",
        repository_source="synthetic_strength_ladder",
        match_engine="synthetic_match_model",
        simulation_count=1000,
        random_seed=31031,
    )

    print("Condition 1")
    print("-----------")
    for key, value in league.summary.items():
        print(f"{key}: {value}")

    print()

    print("Condition 2")
    print("-----------")
    for key, value in knockout.summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
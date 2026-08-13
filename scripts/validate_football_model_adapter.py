from research import ExperimentCondition
from research.adapters import FootballModelAdapter


def main() -> None:
    condition = ExperimentCondition(
        name="League Production Baseline",
        competition_format="league",
        repository_source="dimension_specific",
        match_engine="production_scoreline_first",
        simulation_count=1000,
        random_seed=31031,
        parameters={
            "team_count": 8,
            "experiment": "031C",
        },
    )

    adapter = FootballModelAdapter()
    model = adapter.from_condition(condition)

    print("Football Model")
    print("--------------")
    print(f"Repository source: {model.repository_source}")
    print(f"Match engine: {model.match_engine}")
    print(f"Teams loaded: {len(model.team_repository)}")
    print()

    print("Metadata")
    print("--------")
    for key, value in model.metadata.items():
        print(f"{key}: {value}")

    print()
    print("Sample match")
    print("------------")
    goals1, goals2 = model.simulate_match("Argentina", "France")
    print(f"Argentina {goals1} - {goals2} France")


if __name__ == "__main__":
    main()
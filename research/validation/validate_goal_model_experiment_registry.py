#validate_goal_model_experiment_registry

from __future__ import annotations

from research.experiments.experiment_registry import (
    get_goal_model_experiment,
    list_goal_model_experiments,
)
from research.experiments.goal_model_experiments import (
    CLUBELO_INCREMENTAL_INFORMATION,
)
from research.modeling.football_feature_registry import (
    get_club_goal_model_feature_spec,
)


EXPECTED_EXPERIMENT_NAME = (
    "clubelo_incremental_information"
)

EXPECTED_FEATURE_SPECIFICATIONS = (
    "attack_defense",
    "attack_defense_rating_prior",
    "attack_defense_attack_depth",
    (
        "attack_defense_attack_depth_"
        "rating_prior"
    ),
)

EXPECTED_OVERLAP_FEATURES = (
    "attack_diff",
    "defense_diff",
    "attack_depth_diff",
    "rating_prior_diff",
)


def validate_registry_lookup() -> None:
    names = list_goal_model_experiments()

    if EXPECTED_EXPERIMENT_NAME not in names:
        raise AssertionError(
            "ClubElo experiment is missing from "
            "the experiment registry."
        )

    retrieved = get_goal_model_experiment(
        EXPECTED_EXPERIMENT_NAME
    )

    if (
        retrieved
        is not CLUBELO_INCREMENTAL_INFORMATION
    ):
        raise AssertionError(
            "Experiment registry lookup did not return "
            "the registered experiment instance."
        )


def validate_feature_specifications() -> None:
    experiment = (
        CLUBELO_INCREMENTAL_INFORMATION
    )

    if (
        experiment.feature_specifications
        != EXPECTED_FEATURE_SPECIFICATIONS
    ):
        raise AssertionError(
            "ClubElo experiment contains unexpected "
            "feature specifications."
        )

    for specification_name in (
        experiment.feature_specifications
    ):
        specification = (
            get_club_goal_model_feature_spec(
                specification_name
            )
        )

        if not specification.home_features:
            raise AssertionError(
                f"{specification_name}: no home features."
            )

        if not specification.away_features:
            raise AssertionError(
                f"{specification_name}: no away features."
            )


def validate_rating_prior_isolation() -> None:
    experiment = (
        CLUBELO_INCREMENTAL_INFORMATION
    )

    for comparison in (
        experiment.paired_comparisons
    ):
        baseline = (
            get_club_goal_model_feature_spec(
                comparison.baseline_specification
            )
        )

        candidate = (
            get_club_goal_model_feature_spec(
                comparison.candidate_specification
            )
        )

        baseline_home = set(
            baseline.home_features
        )

        baseline_away = set(
            baseline.away_features
        )

        candidate_home = set(
            candidate.home_features
        )

        candidate_away = set(
            candidate.away_features
        )

        added_home = (
            candidate_home
            - baseline_home
        )

        added_away = (
            candidate_away
            - baseline_away
        )

        removed_home = (
            baseline_home
            - candidate_home
        )

        removed_away = (
            baseline_away
            - candidate_away
        )

        if added_home != {
            "rating_prior_diff"
        }:
            raise AssertionError(
                f"{comparison.name}: candidate does not "
                "add exactly rating_prior_diff to the "
                "home-goal model."
            )

        if added_away != {
            "rating_prior_diff"
        }:
            raise AssertionError(
                f"{comparison.name}: candidate does not "
                "add exactly rating_prior_diff to the "
                "away-goal model."
            )

        if removed_home or removed_away:
            raise AssertionError(
                f"{comparison.name}: candidate removes "
                "baseline features."
            )


def validate_overlap_protocol() -> None:
    experiment = (
        CLUBELO_INCREMENTAL_INFORMATION
    )

    if (
        experiment.overlap_features
        != EXPECTED_OVERLAP_FEATURES
    ):
        raise AssertionError(
            "Unexpected information-overlap features."
        )

    if (
        "rating_prior_diff"
        not in experiment.overlap_features
    ):
        raise AssertionError(
            "Information-overlap analysis does not "
            "contain rating_prior_diff."
        )


def validate_benchmark_adapter() -> None:
    experiment = (
        CLUBELO_INCREMENTAL_INFORMATION
    )

    benchmark_config = (
        experiment.to_benchmark_config()
    )

    if (
        benchmark_config.name
        != experiment.name
    ):
        raise AssertionError(
            "Benchmark adapter changed the "
            "experiment name."
        )

    if (
        benchmark_config.datasets
        != experiment.datasets
    ):
        raise AssertionError(
            "Benchmark adapter changed the datasets."
        )

    if (
        benchmark_config.feature_specifications
        != experiment.feature_specifications
    ):
        raise AssertionError(
            "Benchmark adapter changed the feature "
            "specifications."
        )

    if (
        benchmark_config.train_fractions
        != experiment.train_fractions
    ):
        raise AssertionError(
            "Benchmark adapter changed the training "
            "fractions."
        )

    if (
        benchmark_config.alpha_values
        != experiment.alpha_values
    ):
        raise AssertionError(
            "Benchmark adapter changed the alpha values."
        )


def validate_expected_run_count() -> None:
    experiment = (
        CLUBELO_INCREMENTAL_INFORMATION
    )

    expected = 60

    if (
        experiment.expected_benchmark_runs
        != expected
    ):
        raise AssertionError(
            "Unexpected benchmark run count: "
            f"{experiment.expected_benchmark_runs} "
            f"vs {expected}."
        )


def main() -> None:
    experiment = (
        CLUBELO_INCREMENTAL_INFORMATION
    )

    experiment.validate()

    validate_registry_lookup()
    validate_feature_specifications()
    validate_rating_prior_isolation()
    validate_overlap_protocol()
    validate_benchmark_adapter()
    validate_expected_run_count()

    benchmark_config = (
        experiment.to_benchmark_config()
    )

    print(
        "Goal Model Experiment Registry Validation"
    )
    print("=" * 76)
    print()
    print(
        f"Registered experiments: "
        f"{len(list_goal_model_experiments())}"
    )
    print(
        f"Experiment: {experiment.name}"
    )
    print(
        "Datasets: "
        f"{len(experiment.datasets)}"
    )
    print(
        "Feature specifications: "
        f"{len(experiment.feature_specifications)}"
    )
    print(
        "Paired comparisons: "
        f"{len(experiment.paired_comparisons)}"
    )
    print(
        "Training fractions: "
        f"{len(experiment.train_fractions)}"
    )
    print(
        "Alpha values: "
        f"{len(experiment.alpha_values)}"
    )
    print(
        "Expected benchmark runs: "
        f"{experiment.expected_benchmark_runs}"
    )
    print()

    print("Feature Specifications")
    print("-" * 76)

    for specification_name in (
        experiment.feature_specifications
    ):
        specification = (
            get_club_goal_model_feature_spec(
                specification_name
            )
        )

        print(
            f"{specification.name}: "
            f"{', '.join(specification.required_columns())}"
        )

    print()
    print("Registry lookup: PASS")
    print("Experiment validation: PASS")
    print("Feature-specification validation: PASS")
    print("Rating-prior isolation: PASS")
    print("Paired-comparison validation: PASS")
    print("Overlap-analysis protocol: PASS")
    print("Benchmark adapter: PASS")
    print("Expected run count: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
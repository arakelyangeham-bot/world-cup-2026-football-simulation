#goal_model_experiments

from __future__ import annotations

from pathlib import Path

from research.benchmarking.goal_model_benchmark import (
    GoalModelDatasetConfig,
)
from research.experiments.experiment_registry import (
    GoalModelExperiment,
    GoalModelPairedComparison,
    register_goal_model_experiment,
)
from research.baselines.club_goal_model import (
    CURRENT_CLUB_GOAL_MODEL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLUBELO_ENRICHED_OBSERVATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_060_clubelo_enriched_observations"
    / "full_squad_observations_with_clubelo.csv"
)


CLUBELO_INCREMENTAL_INFORMATION = (
    GoalModelExperiment(
        name="clubelo_incremental_information",
        description=(
            "Evaluate whether temporally valid historical "
            "ClubElo ratings contribute predictive "
            "information beyond player-derived attack, "
            "defense, and attacking-depth features."
        ),
        datasets=(
            GoalModelDatasetConfig(
                name="full_squad_clubelo",
                path=(
                    CLUBELO_ENRICHED_OBSERVATION_PATH
                ),
                representation_type="full_squad",
            ),
        ),
        feature_specifications=(
            "attack_defense",
            "attack_defense_rating_prior",
            "attack_defense_attack_depth",
            (
                "attack_defense_attack_depth_"
                "rating_prior"
            ),
        ),
        train_fractions=(
            0.70,
            0.75,
            0.80,
        ),
        alpha_values=(
            0.0,
            0.0001,
            0.001,
            0.005,
            0.01,
        ),
        paired_comparisons=(
            GoalModelPairedComparison(
                name=(
                    "rating_prior_increment_"
                    "attack_defense"
                ),
                baseline_specification=(
                    "attack_defense"
                ),
                candidate_specification=(
                    "attack_defense_rating_prior"
                ),
                description=(
                    "Measure the incremental contribution "
                    "of ClubElo beyond attack and defense."
                ),
            ),
            GoalModelPairedComparison(
                name=(
                    "rating_prior_increment_"
                    "attack_defense_attack_depth"
                ),
                baseline_specification=(
                    "attack_defense_attack_depth"
                ),
                candidate_specification=(
                    "attack_defense_attack_depth_"
                    "rating_prior"
                ),
                description=(
                    "Measure the incremental contribution "
                    "of ClubElo beyond attack, defense, "
                    "and attacking depth."
                ),
            ),
        ),
        overlap_features=(
            "attack_diff",
            "defense_diff",
            "attack_depth_diff",
            "rating_prior_diff",
        ),
    )
)

DYNAMIC_FORM_INCREMENTAL_INFORMATION = (
    GoalModelExperiment(
        name="dynamic_form_incremental_information",
        description=(
            "Evaluate whether leakage-safe recent "
            "attacking and defensive form contributes "
            "predictive information beyond the current "
            "integrated club goal-model baseline."
        ),
        datasets=(
            GoalModelDatasetConfig(
                name="full_squad_dynamic_form",
                path=(
                    PROJECT_ROOT
                    / "outputs"
                    / (
                        "study_067_dynamic_form_"
                        "observation_enrichment"
                    )
                    / (
                        "full_squad_observations_with_"
                        "complete_dynamic_form.csv"
                    )
                ),
                representation_type="full_squad",
            ),
        ),
        feature_specifications=(
            (
                CURRENT_CLUB_GOAL_MODEL
                .feature_specification
            ),
            (
                "attack_defense_attack_depth_"
                "rating_prior_dynamic_form"
            ),
        ),
        train_fractions=(
            0.60,
            0.70,
            0.75,
            0.80,
        ),
        alpha_values=(
            0.0,
            0.0001,
            0.001,
            0.005,
            0.01,
        ),
        paired_comparisons=(
            GoalModelPairedComparison(
                name="dynamic_form_increment",
                baseline_specification=(
                    CURRENT_CLUB_GOAL_MODEL
                    .feature_specification
                ),
                candidate_specification=(
                    "attack_defense_attack_depth_"
                    "rating_prior_dynamic_form"
                ),
                description=(
                    "Measure the incremental predictive "
                    "contribution of recent attacking and "
                    "defensive form beyond Version 1."
                ),
            ),
        ),
        overlap_features=(
            "attack_depth_diff",
            "rating_prior_diff",
            "attack_form_diff",
            "defense_form_diff",
        ),
    )
)


register_goal_model_experiment(
    DYNAMIC_FORM_INCREMENTAL_INFORMATION
)


register_goal_model_experiment(
    CLUBELO_INCREMENTAL_INFORMATION
)
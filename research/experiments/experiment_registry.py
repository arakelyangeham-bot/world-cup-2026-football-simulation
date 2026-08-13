#experiment_registry

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from research.benchmarking.goal_model_benchmark import (
    DEFAULT_IDENTITY_COLUMNS,
    DEFAULT_METRICS,
    DEFAULT_TARGET_COLUMNS,
    GoalModelBenchmarkConfig,
    GoalModelDatasetConfig,
)
from research.modeling.football_feature_registry import (
    get_club_goal_model_feature_spec,
)


@dataclass(frozen=True)
class GoalModelPairedComparison:
    """
    Defines one controlled feature-specification comparison.

    The candidate should differ from the baseline only by
    the football concept being tested.
    """

    name: str
    baseline_specification: str
    candidate_specification: str
    description: str = ""

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Paired-comparison name cannot be empty."
            )

        if (
            self.baseline_specification
            == self.candidate_specification
        ):
            raise ValueError(
                "A paired comparison must contain "
                "different baseline and candidate "
                "specifications."
            )

        get_club_goal_model_feature_spec(
            self.baseline_specification
        )

        get_club_goal_model_feature_spec(
            self.candidate_specification
        )


@dataclass(frozen=True)
class GoalModelExperiment:
    """
    Declarative research protocol for one goal-model study.

    This object describes what should be tested. It contains
    no fitting, prediction, or evaluation logic.
    """

    name: str
    description: str

    datasets: tuple[GoalModelDatasetConfig, ...]
    feature_specifications: tuple[str, ...]
    train_fractions: tuple[float, ...]
    alpha_values: tuple[float, ...]

    paired_comparisons: tuple[
        GoalModelPairedComparison,
        ...,
    ] = ()

    overlap_features: tuple[str, ...] = ()

    identity_columns: tuple[str, ...] = (
        DEFAULT_IDENTITY_COLUMNS
    )

    target_columns: tuple[str, ...] = (
        DEFAULT_TARGET_COLUMNS
    )

    ranking_metrics: tuple[str, ...] = (
        DEFAULT_METRICS
    )

    outcome_grid_max_goals: int = 15
    probability_floor: float = 1e-15

    require_matched_populations: bool = True
    capture_predictions: bool = True
    capture_coefficients: bool = True

    def validate(self) -> None:
        """
        Validate the complete experiment protocol before
        converting it into a benchmark configuration.
        """
        if not self.name.strip():
            raise ValueError(
                "Experiment name cannot be empty."
            )

        if not self.description.strip():
            raise ValueError(
                "Experiment description cannot be empty."
            )

        if not self.datasets:
            raise ValueError(
                "An experiment must contain at least "
                "one dataset."
            )

        dataset_names = tuple(
            dataset.name
            for dataset in self.datasets
        )

        if len(dataset_names) != len(
            set(dataset_names)
        ):
            raise ValueError(
                "Experiment dataset names must be unique."
            )

        if not self.feature_specifications:
            raise ValueError(
                "An experiment must contain at least one "
                "feature specification."
            )

        if len(
            self.feature_specifications
        ) != len(
            set(self.feature_specifications)
        ):
            raise ValueError(
                "Experiment feature specifications "
                "must be unique."
            )

        for specification_name in (
            self.feature_specifications
        ):
            get_club_goal_model_feature_spec(
                specification_name
            )

        if not self.train_fractions:
            raise ValueError(
                "An experiment must contain at least one "
                "training fraction."
            )

        if len(self.train_fractions) != len(
            set(self.train_fractions)
        ):
            raise ValueError(
                "Training fractions must be unique."
            )

        for fraction in self.train_fractions:
            if not 0.0 < fraction < 1.0:
                raise ValueError(
                    "Training fractions must lie between "
                    f"zero and one: {fraction}"
                )

        if not self.alpha_values:
            raise ValueError(
                "An experiment must contain at least one "
                "alpha value."
            )

        if len(self.alpha_values) != len(
            set(self.alpha_values)
        ):
            raise ValueError(
                "Alpha values must be unique."
            )

        for alpha in self.alpha_values:
            if alpha < 0:
                raise ValueError(
                    "Alpha values cannot be negative: "
                    f"{alpha}"
                )

        comparison_names = tuple(
            comparison.name
            for comparison in self.paired_comparisons
        )

        if len(comparison_names) != len(
            set(comparison_names)
        ):
            raise ValueError(
                "Paired-comparison names must be unique."
            )

        configured_specifications = set(
            self.feature_specifications
        )

        for comparison in self.paired_comparisons:
            comparison.validate()

            required = {
                comparison.baseline_specification,
                comparison.candidate_specification,
            }

            missing = (
                required
                - configured_specifications
            )

            if missing:
                raise ValueError(
                    "Paired comparison references feature "
                    "specifications not included in the "
                    f"experiment: {sorted(missing)}"
                )

        if len(self.overlap_features) != len(
            set(self.overlap_features)
        ):
            raise ValueError(
                "Overlap-analysis features must be unique."
            )

        if self.outcome_grid_max_goals < 1:
            raise ValueError(
                "Outcome-grid maximum goals must be "
                "positive."
            )

        if not 0.0 < self.probability_floor < 1.0:
            raise ValueError(
                "Probability floor must lie between "
                "zero and one."
            )

    @property
    def expected_benchmark_runs(self) -> int:
        return (
            len(self.datasets)
            * len(self.feature_specifications)
            * len(self.train_fractions)
            * len(self.alpha_values)
        )

    def required_feature_columns(self) -> tuple[
        str,
        ...,
    ]:
        columns: list[str] = []

        for specification_name in (
            self.feature_specifications
        ):
            specification = (
                get_club_goal_model_feature_spec(
                    specification_name
                )
            )

            columns.extend(
                specification.required_columns()
            )

        return _deduplicate_preserving_order(
            columns
        )

    def to_benchmark_config(
        self,
    ) -> GoalModelBenchmarkConfig:
        """
        Compile the experiment protocol into the existing
        validated benchmark-engine configuration.
        """
        self.validate()

        return GoalModelBenchmarkConfig(
            name=self.name,
            datasets=self.datasets,
            feature_specifications=(
                self.feature_specifications
            ),
            train_fractions=(
                self.train_fractions
            ),
            alpha_values=self.alpha_values,
            identity_columns=(
                self.identity_columns
            ),
            target_columns=(
                self.target_columns
            ),
            ranking_metrics=(
                self.ranking_metrics
            ),
            outcome_grid_max_goals=(
                self.outcome_grid_max_goals
            ),
            probability_floor=(
                self.probability_floor
            ),
            require_matched_populations=(
                self.require_matched_populations
            ),
            capture_predictions=(
                self.capture_predictions
            ),
            capture_coefficients=(
                self.capture_coefficients
            ),
        )


GOAL_MODEL_EXPERIMENTS: dict[
    str,
    GoalModelExperiment,
] = {}


def _deduplicate_preserving_order(
    values: Iterable[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        output.append(value)

    return tuple(output)


def register_goal_model_experiment(
    experiment: GoalModelExperiment,
) -> None:
    """
    Validate and register one named experiment protocol.
    """
    experiment.validate()

    if experiment.name in GOAL_MODEL_EXPERIMENTS:
        raise ValueError(
            "Goal-model experiment is already "
            f"registered: {experiment.name!r}"
        )

    GOAL_MODEL_EXPERIMENTS[
        experiment.name
    ] = experiment


def get_goal_model_experiment(
    name: str,
) -> GoalModelExperiment:
    try:
        return GOAL_MODEL_EXPERIMENTS[name]
    except KeyError as error:
        available = ", ".join(
            sorted(GOAL_MODEL_EXPERIMENTS)
        )

        raise KeyError(
            "Unknown goal-model experiment: "
            f"{name!r}. Available experiments: "
            f"{available}"
        ) from error


def list_goal_model_experiments() -> tuple[
    str,
    ...,
]:
    return tuple(
        sorted(GOAL_MODEL_EXPERIMENTS)
    )
#club_goal_model

from __future__ import annotations

from dataclasses import dataclass

from research.modeling.football_feature_registry import (
    FootballFeatureSpecification,
    get_club_goal_model_feature_spec,
)


@dataclass(frozen=True)
class ClubGoalModelBaseline:
    """
    Evidence-backed reference configuration for club
    goal-model research.

    The baseline records the project recommendation rather
    than a fitted model. Coefficients, datasets, and fitted
    estimators remain responsibilities of the benchmark and
    production-model layers.
    """

    name: str
    version: str
    feature_specification: str
    description: str
    supporting_studies: tuple[str, ...]
    status: str
    notes: str

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Baseline name cannot be empty."
            )

        if not self.version.strip():
            raise ValueError(
                "Baseline version cannot be empty."
            )

        if not self.feature_specification.strip():
            raise ValueError(
                "Baseline feature specification cannot "
                "be empty."
            )

        if not self.description.strip():
            raise ValueError(
                "Baseline description cannot be empty."
            )

        if not self.supporting_studies:
            raise ValueError(
                "A baseline must cite at least one "
                "supporting study."
            )

        if len(self.supporting_studies) != len(
            set(self.supporting_studies)
        ):
            raise ValueError(
                "Supporting study identifiers must be "
                "unique."
            )

        if self.status not in {
            "recommended",
            "deprecated",
            "experimental",
        }:
            raise ValueError(
                "Unsupported baseline status: "
                f"{self.status!r}"
            )

        get_club_goal_model_feature_spec(
            self.feature_specification
        )

    def get_feature_specification(
        self,
    ) -> FootballFeatureSpecification:
        """
        Resolve the baseline's registered football feature
        specification.
        """
        self.validate()

        return get_club_goal_model_feature_spec(
            self.feature_specification
        )


CLUB_GOAL_MODEL_V1 = ClubGoalModelBaseline(
    name="integrated_club_goal_model",
    version="1.0",
    feature_specification=(
        "attack_defense_attack_depth_rating_prior"
    ),
    description=(
        "Integrated club goal-model baseline combining "
        "player-derived attacking and defensive strength, "
        "relative attacking depth, and a temporally valid "
        "historical ClubElo rating prior."
    ),
    supporting_studies=(
        "052",
        "054",
        "060",
        "061",
    ),
    status="recommended",
    notes=(
        "Recommended reference specification for future "
        "club goal-model improvement experiments. New "
        "candidate specifications should be compared "
        "against this baseline using matched observations, "
        "chronological splits, and the validated benchmark "
        "engine."
    ),
)


CLUB_GOAL_MODEL_BASELINES: dict[
    str,
    ClubGoalModelBaseline,
] = {
    CLUB_GOAL_MODEL_V1.version:
        CLUB_GOAL_MODEL_V1,
}


CURRENT_CLUB_GOAL_MODEL = (
    CLUB_GOAL_MODEL_V1
)


def get_club_goal_model_baseline(
    version: str,
) -> ClubGoalModelBaseline:
    try:
        baseline = CLUB_GOAL_MODEL_BASELINES[
            version
        ]
    except KeyError as error:
        available = ", ".join(
            sorted(CLUB_GOAL_MODEL_BASELINES)
        )

        raise KeyError(
            "Unknown club goal-model baseline version: "
            f"{version!r}. Available versions: "
            f"{available}"
        ) from error

    baseline.validate()

    return baseline


def list_club_goal_model_baselines() -> tuple[
    str,
    ...,
]:
    return tuple(
        sorted(CLUB_GOAL_MODEL_BASELINES)
    )
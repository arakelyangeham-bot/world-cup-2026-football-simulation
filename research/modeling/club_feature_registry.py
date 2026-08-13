#club_feature_registry

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoalModelFeatureSpecification:
    """
    Defines the feature contract for one club goal-model experiment.

    The home model predicts home goals.
    The away model predicts away goals.
    """

    name: str
    description: str
    home_features: tuple[str, ...]
    away_features: tuple[str, ...]

    def required_columns(self) -> tuple[str, ...]:
        """
        Return every unique feature column required by this specification.
        """
        return tuple(
            sorted(
                set(self.home_features)
                | set(self.away_features)
            )
        )


CLUB_GOAL_MODEL_FEATURE_SPECS = {
    "attack_defense": GoalModelFeatureSpecification(
        name="attack_defense",
        description=(
            "Direct attacking strength and opposing defensive strength only."
        ),
        home_features=(
            "home_attack",
            "away_defense",
        ),
        away_features=(
            "away_attack",
            "home_defense",
        ),
    ),

    "attack_defense_midfield": GoalModelFeatureSpecification(
        name="attack_defense_midfield",
        description=(
            "Attack and defense features with relative midfield strength."
        ),
        home_features=(
            "home_attack",
            "away_defense",
            "midfield_diff",
        ),
        away_features=(
            "away_attack",
            "home_defense",
            "midfield_diff",
        ),
    ),

    "attack_defense_depth": GoalModelFeatureSpecification(
        name="attack_defense_depth",
        description=(
            "Attack and defense features with relative squad-depth strength."
        ),
        home_features=(
            "home_attack",
            "away_defense",
            "depth_diff",
        ),
        away_features=(
            "away_attack",
            "home_defense",
            "depth_diff",
        ),
    ),

    "attack_defense_goalkeeper": GoalModelFeatureSpecification(
        name="attack_defense_goalkeeper",
        description=(
            "Attack and defense features with goalkeeper-strength difference."
        ),
        home_features=(
            "home_attack",
            "away_defense",
            "goalkeeper_diff",
        ),
        away_features=(
            "away_attack",
            "home_defense",
            "goalkeeper_diff",
        ),
    ),

    "attack_defense_midfield_depth": GoalModelFeatureSpecification(
        name="attack_defense_midfield_depth",
        description=(
            "Attack and defense with midfield and squad-depth differences."
        ),
        home_features=(
            "home_attack",
            "away_defense",
            "midfield_diff",
            "depth_diff",
        ),
        away_features=(
            "away_attack",
            "home_defense",
            "midfield_diff",
            "depth_diff",
        ),
    ),

    "all_representation_features": GoalModelFeatureSpecification(
        name="all_representation_features",
        description=(
            "Attack, defense, midfield, depth, and goalkeeper information."
        ),
        home_features=(
            "home_attack",
            "away_defense",
            "midfield_diff",
            "depth_diff",
            "goalkeeper_diff",
        ),
        away_features=(
            "away_attack",
            "home_defense",
            "midfield_diff",
            "depth_diff",
            "goalkeeper_diff",
        ),
    ),
}


def get_club_goal_model_feature_spec(
    name: str,
) -> GoalModelFeatureSpecification:
    try:
        return CLUB_GOAL_MODEL_FEATURE_SPECS[name]
    except KeyError as error:
        available = ", ".join(
            sorted(CLUB_GOAL_MODEL_FEATURE_SPECS)
        )

        raise KeyError(
            f"Unknown club goal-model feature specification: {name!r}. "
            f"Available specifications: {available}"
        ) from error


def list_club_goal_model_feature_specs() -> tuple[str, ...]:
    return tuple(
        sorted(CLUB_GOAL_MODEL_FEATURE_SPECS)
    )
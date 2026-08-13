#football_feature_registry

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FootballFeatureGroup:
    """
    A reusable football concept represented by one or more
    observation-dataset columns.

    The home and away feature lists may differ because the
    home-goal and away-goal models view the match from
    different perspectives.
    """

    name: str
    description: str
    home_features: tuple[str, ...]
    away_features: tuple[str, ...]

    def required_columns(self) -> tuple[str, ...]:
        """
        Return every unique observation column required by
        this feature group.
        """
        return tuple(
            sorted(
                set(self.home_features)
                | set(self.away_features)
            )
        )


@dataclass(frozen=True)
class FootballFeatureSpecification:
    """
    A complete model feature specification assembled from
    one or more football feature groups.
    """

    name: str
    description: str
    group_names: tuple[str, ...]
    home_features: tuple[str, ...]
    away_features: tuple[str, ...]

    def required_columns(self) -> tuple[str, ...]:
        """
        Return every unique observation column required by
        this specification.
        """
        return tuple(
            sorted(
                set(self.home_features)
                | set(self.away_features)
            )
        )


FOOTBALL_FEATURE_GROUPS: dict[
    str,
    FootballFeatureGroup,
] = {
    "attack_defense": FootballFeatureGroup(
        name="attack_defense",
        description=(
            "Direct attacking strength and opposing "
            "defensive strength."
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

    "midfield": FootballFeatureGroup(
        name="midfield",
        description=(
            "Relative midfield-strength difference."
        ),
        home_features=(
            "midfield_diff",
        ),
        away_features=(
            "midfield_diff",
        ),
    ),

    "goalkeeper": FootballFeatureGroup(
        name="goalkeeper",
        description=(
            "Relative goalkeeper-strength difference."
        ),
        home_features=(
            "goalkeeper_diff",
        ),
        away_features=(
            "goalkeeper_diff",
        ),
    ),

    "attack_depth": FootballFeatureGroup(
        name="attack_depth",
        description=(
            "Relative attacking-depth difference."
        ),
        home_features=(
            "attack_depth_diff",
        ),
        away_features=(
            "attack_depth_diff",
        ),
    ),

    "midfield_depth": FootballFeatureGroup(
        name="midfield_depth",
        description=(
            "Relative midfield-depth difference."
        ),
        home_features=(
            "midfield_depth_diff",
        ),
        away_features=(
            "midfield_depth_diff",
        ),
    ),

    "defense_depth": FootballFeatureGroup(
        name="defense_depth",
        description=(
            "Relative defensive-depth difference."
        ),
        home_features=(
            "defense_depth_diff",
        ),
        away_features=(
            "defense_depth_diff",
        ),
    ),

    "squad_quality": FootballFeatureGroup(
        name="squad_quality",
        description=(
            "Relative aggregate squad-quality difference."
        ),
        home_features=(
            "squad_quality_diff",
        ),
        away_features=(
            "squad_quality_diff",
        ),
    ),

    "rating_prior": FootballFeatureGroup(
        name="rating_prior",
        description=(
            "Relative historical team-strength rating "
            "prior."
        ),
        home_features=(
            "rating_prior_diff",
        ),
        away_features=(
            "rating_prior_diff",
        ),
    ),

    "dynamic_form": FootballFeatureGroup(
        name="dynamic_form",
        description=(
            "Recent attacking and defensive performance "
            "relative to leakage-safe Version 1 "
            "expectations."
        ),
        home_features=(
            "home_attack_form",
            "away_defense_form",
        ),
        away_features=(
            "away_attack_form",
            "home_defense_form",
        ),
    ),
}


CLUB_GOAL_MODEL_FEATURE_SPECS: dict[
    str,
    FootballFeatureSpecification,
] = {}


def _deduplicate_preserving_order(
    values: Iterable[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        ordered.append(value)

    return tuple(ordered)


def build_feature_specification(
    name: str,
    description: str,
    group_names: Iterable[str],
) -> FootballFeatureSpecification:
    """
    Construct a model feature specification from registered
    football feature groups.
    """
    normalized_group_names = tuple(group_names)

    if not normalized_group_names:
        raise ValueError(
            "A feature specification must contain at "
            "least one feature group."
        )

    missing_groups = [
        group_name
        for group_name in normalized_group_names
        if group_name not in FOOTBALL_FEATURE_GROUPS
    ]

    if missing_groups:
        available = ", ".join(
            sorted(FOOTBALL_FEATURE_GROUPS)
        )

        raise KeyError(
            "Unknown football feature groups: "
            f"{missing_groups}. "
            f"Available groups: {available}"
        )

    home_features: list[str] = []
    away_features: list[str] = []

    for group_name in normalized_group_names:
        group = FOOTBALL_FEATURE_GROUPS[
            group_name
        ]

        home_features.extend(
            group.home_features
        )

        away_features.extend(
            group.away_features
        )

    return FootballFeatureSpecification(
        name=name,
        description=description,
        group_names=normalized_group_names,
        home_features=(
            _deduplicate_preserving_order(
                home_features
            )
        ),
        away_features=(
            _deduplicate_preserving_order(
                away_features
            )
        ),
    )


def register_feature_specification(
    specification: FootballFeatureSpecification,
) -> None:
    """
    Register one complete feature specification.
    """
    if specification.name in (
        CLUB_GOAL_MODEL_FEATURE_SPECS
    ):
        raise ValueError(
            "Feature specification is already "
            f"registered: {specification.name!r}"
        )

    CLUB_GOAL_MODEL_FEATURE_SPECS[
        specification.name
    ] = specification


register_feature_specification(
    build_feature_specification(
        name="attack_defense",
        description=(
            "Baseline using direct attacking strength "
            "and opposing defensive strength."
        ),
        group_names=(
            "attack_defense",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name="attack_defense_midfield",
        description=(
            "Baseline attack and defense with relative "
            "midfield strength."
        ),
        group_names=(
            "attack_defense",
            "midfield",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name="attack_defense_goalkeeper",
        description=(
            "Baseline attack and defense with relative "
            "goalkeeper strength."
        ),
        group_names=(
            "attack_defense",
            "goalkeeper",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name="attack_defense_attack_depth",
        description=(
            "Baseline attack and defense with relative "
            "attacking depth."
        ),
        group_names=(
            "attack_defense",
            "attack_depth",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name="attack_defense_midfield_depth",
        description=(
            "Baseline attack and defense with relative "
            "midfield depth."
        ),
        group_names=(
            "attack_defense",
            "midfield_depth",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name="attack_defense_defense_depth",
        description=(
            "Baseline attack and defense with relative "
            "defensive depth."
        ),
        group_names=(
            "attack_defense",
            "defense_depth",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name="attack_defense_squad_quality",
        description=(
            "Baseline attack and defense with relative "
            "aggregate squad quality."
        ),
        group_names=(
            "attack_defense",
            "squad_quality",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name=(
            "attack_defense_"
            "attack_depth_midfield"
        ),
        description=(
            "Baseline attack and defense with "
            "relative attacking depth and "
            "midfield strength."
        ),
        group_names=(
            "attack_defense",
            "attack_depth",
            "midfield",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name="attack_defense_rating_prior",
        description=(
            "Baseline attack and defense with a "
            "historical team-strength rating prior."
        ),
        group_names=(
            "attack_defense",
            "rating_prior",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name=(
            "attack_defense_attack_depth_"
            "rating_prior"
        ),
        description=(
            "Baseline attack and defense with relative "
            "attacking depth and a historical "
            "team-strength rating prior."
        ),
        group_names=(
            "attack_defense",
            "attack_depth",
            "rating_prior",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name="all_representation_features",
        description=(
            "Attack, defense, midfield, goalkeeper, "
            "positional depth, and aggregate squad quality."
        ),
        group_names=(
            "attack_defense",
            "midfield",
            "goalkeeper",
            "attack_depth",
            "midfield_depth",
            "defense_depth",
            "squad_quality",
        ),
    )
)

register_feature_specification(
    build_feature_specification(
        name=(
            "attack_defense_attack_depth_"
            "rating_prior_dynamic_form"
        ),
        description=(
            "Integrated club goal model combining "
            "player-derived attack and defense, attacking "
            "depth, historical rating prior, and recent "
            "dynamic attacking and defensive form."
        ),
        group_names=(
            "attack_defense",
            "attack_depth",
            "rating_prior",
            "dynamic_form",
        ),
    )
)

def get_football_feature_group(
    name: str,
) -> FootballFeatureGroup:
    try:
        return FOOTBALL_FEATURE_GROUPS[name]
    except KeyError as error:
        available = ", ".join(
            sorted(FOOTBALL_FEATURE_GROUPS)
        )

        raise KeyError(
            f"Unknown football feature group: {name!r}. "
            f"Available groups: {available}"
        ) from error


def get_club_goal_model_feature_spec(
    name: str,
) -> FootballFeatureSpecification:
    try:
        return CLUB_GOAL_MODEL_FEATURE_SPECS[
            name
        ]
    except KeyError as error:
        available = ", ".join(
            sorted(
                CLUB_GOAL_MODEL_FEATURE_SPECS
            )
        )

        raise KeyError(
            "Unknown club goal-model feature "
            f"specification: {name!r}. "
            f"Available specifications: {available}"
        ) from error


def list_football_feature_groups() -> tuple[
    str,
    ...,
]:
    return tuple(
        sorted(FOOTBALL_FEATURE_GROUPS)
    )


def list_club_goal_model_feature_specs() -> tuple[
    str,
    ...,
]:
    return tuple(
        sorted(
            CLUB_GOAL_MODEL_FEATURE_SPECS
        )
    )
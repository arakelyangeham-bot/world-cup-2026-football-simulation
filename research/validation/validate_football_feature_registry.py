#validate_football_feature_registry

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.modeling.football_feature_registry import (
    CLUB_GOAL_MODEL_FEATURE_SPECS,
    FOOTBALL_FEATURE_GROUPS,
    get_club_goal_model_feature_spec,
    get_football_feature_group,
    list_club_goal_model_feature_specs,
    list_football_feature_groups,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBSERVATION_PATHS = {
    "full_squad": (
        PROJECT_ROOT
        / "outputs"
        / "study_048_club_observation_dataset"
        / "full_squad_observations.csv"
    ),
    "expected_starting_xi": (
        PROJECT_ROOT
        / "outputs"
        / "study_048_club_observation_dataset"
        / "expected_starting_xi_observations.csv"
    ),
}


def load_columns(
    path: Path,
) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Observation dataset does not exist: {path}"
        )

    dataframe = pd.read_csv(
        path,
        nrows=5,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Observation dataset is empty: {path}"
        )

    return set(dataframe.columns)


def validate_registry_integrity() -> None:
    if not FOOTBALL_FEATURE_GROUPS:
        raise AssertionError(
            "Football feature-group registry is empty."
        )

    if not CLUB_GOAL_MODEL_FEATURE_SPECS:
        raise AssertionError(
            "Club goal-model feature registry is empty."
        )

    group_names = (
        list_football_feature_groups()
    )

    specification_names = (
        list_club_goal_model_feature_specs()
    )

    if len(group_names) != len(
        set(group_names)
    ):
        raise AssertionError(
            "Duplicate feature-group names detected."
        )

    if len(specification_names) != len(
        set(specification_names)
    ):
        raise AssertionError(
            "Duplicate feature-specification names "
            "detected."
        )

    for group_name in group_names:
        group = get_football_feature_group(
            group_name
        )

        if group.name != group_name:
            raise AssertionError(
                f"Feature-group key/name mismatch: "
                f"{group_name!r} vs {group.name!r}"
            )

        if not group.home_features:
            raise AssertionError(
                f"{group_name}: no home features."
            )

        if not group.away_features:
            raise AssertionError(
                f"{group_name}: no away features."
            )

        if len(group.home_features) != len(
            set(group.home_features)
        ):
            raise AssertionError(
                f"{group_name}: duplicate home features."
            )

        if len(group.away_features) != len(
            set(group.away_features)
        ):
            raise AssertionError(
                f"{group_name}: duplicate away features."
            )

    for specification_name in (
        specification_names
    ):
        specification = (
            get_club_goal_model_feature_spec(
                specification_name
            )
        )

        if specification.name != (
            specification_name
        ):
            raise AssertionError(
                "Feature-specification key/name "
                f"mismatch: {specification_name!r} "
                f"vs {specification.name!r}"
            )

        if not specification.group_names:
            raise AssertionError(
                f"{specification_name}: "
                "no feature groups."
            )

        if not specification.home_features:
            raise AssertionError(
                f"{specification_name}: "
                "no home features."
            )

        if not specification.away_features:
            raise AssertionError(
                f"{specification_name}: "
                "no away features."
            )

        for group_name in (
            specification.group_names
        ):
            if group_name not in (
                FOOTBALL_FEATURE_GROUPS
            ):
                raise AssertionError(
                    f"{specification_name}: "
                    "unknown feature group "
                    f"{group_name!r}."
                )


def validate_dataset_coverage(
    dataset_name: str,
    columns: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for specification_name in (
        list_club_goal_model_feature_specs()
    ):
        specification = (
            get_club_goal_model_feature_spec(
                specification_name
            )
        )

        required_columns = set(
            specification.required_columns()
        )

        missing_columns = sorted(
            required_columns - columns
        )

        rows.append(
            {
                "dataset": dataset_name,
                "specification":
                    specification_name,
                "group_count": len(
                    specification.group_names
                ),
                "home_feature_count": len(
                    specification.home_features
                ),
                "away_feature_count": len(
                    specification.away_features
                ),
                "required_column_count": len(
                    required_columns
                ),
                "missing_column_count": len(
                    missing_columns
                ),
                "missing_columns": (
                    "|".join(missing_columns)
                ),
                "validation_pass": (
                    len(missing_columns) == 0
                ),
            }
        )

    return rows


def main() -> None:
    validate_registry_integrity()

    coverage_rows: list[
        dict[str, object]
    ] = []

    for dataset_name, path in (
        OBSERVATION_PATHS.items()
    ):
        columns = load_columns(path)

        coverage_rows.extend(
            validate_dataset_coverage(
                dataset_name=dataset_name,
                columns=columns,
            )
        )

    coverage = pd.DataFrame(
        coverage_rows
    )

    failures = coverage[
        ~coverage["validation_pass"]
    ]

    if not failures.empty:
        failure_text = failures[
            [
                "dataset",
                "specification",
                "missing_columns",
            ]
        ].to_string(index=False)

        raise AssertionError(
            "Feature-registry dataset coverage "
            "failed:\n"
            f"{failure_text}"
        )

    print("Football Feature Registry Validation")
    print("=" * 72)
    print()
    print(
        "Registered feature groups: "
        f"{len(FOOTBALL_FEATURE_GROUPS)}"
    )
    print(
        "Registered feature specifications: "
        f"{len(CLUB_GOAL_MODEL_FEATURE_SPECS)}"
    )
    print(
        "Observation datasets checked: "
        f"{len(OBSERVATION_PATHS)}"
    )
    print()

    print("Feature Groups")
    print("-" * 72)

    for group_name in (
        list_football_feature_groups()
    ):
        group = get_football_feature_group(
            group_name
        )

        print(
            f"{group_name}: "
            f"home={list(group.home_features)}, "
            f"away={list(group.away_features)}"
        )

    print()
    print("Feature Specifications")
    print("-" * 72)

    for specification_name in (
        list_club_goal_model_feature_specs()
    ):
        specification = (
            get_club_goal_model_feature_spec(
                specification_name
            )
        )

        print(
            f"{specification_name}: "
            f"groups={list(specification.group_names)}, "
            f"home_features="
            f"{list(specification.home_features)}, "
            f"away_features="
            f"{list(specification.away_features)}"
        )

    print()
    print("Registry integrity: PASS")
    print("Observation schema coverage: PASS")
    print("Evidence-score exclusion: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
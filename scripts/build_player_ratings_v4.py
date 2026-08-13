# build_player_ratings_v4.py

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ATTRIBUTES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_attribute_scores.csv"
)

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_registry.csv"
)

ROLE_ATTRIBUTE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "role_attribute_manifest.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_ratings.csv"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build player role ratings from a configurable "
            "player-attribute input file."
        )
    )

    parser.add_argument(
        "--attribute-path",
        type=Path,
        default=DEFAULT_ATTRIBUTES_PATH,
        help=(
            "Player attribute CSV used to construct role ratings. "
            "Defaults to the canonical production artifact."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Destination player-rating CSV. Defaults to the "
            "canonical production artifact."
        ),
    )

    parser.add_argument(
        "--registry-path",
        type=Path,
        default=REGISTRY_PATH,
        help=(
            "Player registry used for role eligibility. "
            "Defaults to the canonical registry."
        ),
    )

    parser.add_argument(
        "--role-attribute-path",
        type=Path,
        default=ROLE_ATTRIBUTE_PATH,
        help=(
            "Role-attribute weighting manifest. "
            "Defaults to the canonical manifest."
        ),
    )

    return parser.parse_args()


def parse_roles(
    value: object,
) -> list[str]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [
            str(role)
            for role in value
        ]

    try:
        parsed = ast.literal_eval(
            str(value)
        )
    except (
        ValueError,
        SyntaxError,
    ):
        return []

    if not isinstance(
        parsed,
        list,
    ):
        return []

    return [
        str(role)
        for role in parsed
    ]


def load_inputs(
    *,
    attribute_path: Path,
    registry_path: Path,
    role_attribute_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if not attribute_path.exists():
        raise FileNotFoundError(
            "Player attribute file does not exist: "
            f"{attribute_path}"
        )

    if not registry_path.exists():
        raise FileNotFoundError(
            "Player registry does not exist: "
            f"{REGISTRY_PATH}"
        )

    if not role_attribute_path.exists():
        raise FileNotFoundError(
            "Role-attribute manifest does not exist: "
            f"{ROLE_ATTRIBUTE_PATH}"
        )

    attributes = pd.read_csv(
        attribute_path,
        low_memory=False,
    )

    registry = pd.read_csv(
        registry_path,
        low_memory=False,
    )

    role_manifest = pd.read_csv(
        role_attribute_path,
        low_memory=False,
    )

    if attributes.empty:
        raise ValueError(
            "Player attribute input is empty."
        )

    if registry.empty:
        raise ValueError(
            "Player registry is empty."
        )

    if role_manifest.empty:
        raise ValueError(
            "Role-attribute manifest is empty."
        )

    required_attribute_columns = {
        "canonical_player_id",
        "player_id",
        "player",
        "minutesPlayed",
        "total_weighted_evidence",
        "evidence_confidence",
        "source_competitions",
        "competition_count",
        "season_count",
    }

    missing_attribute_columns = (
        required_attribute_columns
        - set(attributes.columns)
    )

    if missing_attribute_columns:
        raise ValueError(
            "Player attribute input is missing required columns: "
            f"{sorted(missing_attribute_columns)}"
        )

    required_registry_columns = {
        "canonical_player_id",
        "eligible_roles",
        "position",
        "positions_detailed",
        "country",
        "current_team",
    }

    missing_registry_columns = (
        required_registry_columns
        - set(registry.columns)
    )

    if missing_registry_columns:
        raise ValueError(
            "Player registry is missing required columns: "
            f"{sorted(missing_registry_columns)}"
        )

    required_manifest_columns = {
        "role",
        "attribute",
        "weight",
    }

    missing_manifest_columns = (
        required_manifest_columns
        - set(role_manifest.columns)
    )

    if missing_manifest_columns:
        raise ValueError(
            "Role-attribute manifest is missing required columns: "
            f"{sorted(missing_manifest_columns)}"
        )

    if attributes[
        "canonical_player_id"
    ].duplicated().any():
        raise ValueError(
            "Player attribute input contains duplicate "
            "canonical player IDs."
        )

    registry_subset = registry[
        [
            "canonical_player_id",
            "eligible_roles",
            "position",
            "positions_detailed",
            "country",
            "current_team",
        ]
    ].drop_duplicates(
        subset=[
            "canonical_player_id",
        ]
    )

    return (
        attributes,
        registry_subset,
        role_manifest,
    )


def build_player_ratings(
    *,
    attribute_path: Path,
    registry_path: Path,
    role_attribute_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    (
        attributes,
        registry,
        role_manifest,
    ) = load_inputs(
        attribute_path=attribute_path,
        registry_path=registry_path,
        role_attribute_path=role_attribute_path,
    )

    dataframe = attributes.merge(
        registry,
        on="canonical_player_id",
        how="left",
        validate="one_to_one",
    )

    if len(dataframe) != len(
        attributes
    ):
        raise AssertionError(
            "Registry merge changed the player population."
        )

    dataframe[
        "eligible_roles"
    ] = dataframe[
        "eligible_roles"
    ].fillna("[]")

    dataframe[
        "eligible_roles_list"
    ] = dataframe[
        "eligible_roles"
    ].apply(
        parse_roles
    )

    role_manifest = role_manifest.copy()

    role_manifest[
        "weight"
    ] = pd.to_numeric(
        role_manifest[
            "weight"
        ],
        errors="coerce",
    ).fillna(1.0)

    roles = sorted(
        role_manifest[
            "role"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    if not roles:
        raise ValueError(
            "Role-attribute manifest contains no roles."
        )

    for role in roles:
        role_rows = role_manifest.loc[
            role_manifest[
                "role"
            ].eq(role)
        ]

        raw_scores: list[
            float,
        ] = []

        for _, player in dataframe.iterrows():
            eligible_roles = player[
                "eligible_roles_list"
            ]

            if role not in eligible_roles:
                raw_scores.append(
                    np.nan
                )
                continue

            score_parts: list[
                float,
            ] = []

            used_weights: list[
                float,
            ] = []

            for _, attribute_row in (
                role_rows.iterrows()
            ):
                attribute = str(
                    attribute_row[
                        "attribute"
                    ]
                )

                weight = float(
                    attribute_row[
                        "weight"
                    ]
                )

                attribute_column = (
                    f"attribute_{attribute}"
                )

                if (
                    attribute_column
                    not in dataframe.columns
                ):
                    continue

                value = player[
                    attribute_column
                ]

                if pd.isna(value):
                    continue

                score_parts.append(
                    float(value)
                    * weight
                )

                used_weights.append(
                    weight
                )

            if score_parts:
                raw_scores.append(
                    sum(score_parts)
                    / sum(used_weights)
                )
            else:
                raw_scores.append(
                    np.nan
                )

        dataframe[
            f"raw_rating_{role}"
        ] = raw_scores

        dataframe[
            f"rating_{role}"
        ] = (
            dataframe[
                f"raw_rating_{role}"
            ]
            * dataframe[
                "evidence_confidence"
            ]
        )

    rating_columns = [
        f"rating_{role}"
        for role in roles
    ]

    dataframe[
        "best_rating"
    ] = dataframe[
        rating_columns
    ].max(
        axis=1,
        skipna=True,
    )

    dataframe[
        "best_role"
    ] = pd.NA

    has_rating = dataframe[
        "best_rating"
    ].notna()

    dataframe.loc[
        has_rating,
        "best_role",
    ] = (
        dataframe.loc[
            has_rating,
            rating_columns,
        ]
        .idxmax(axis=1)
        .str.replace(
            "rating_",
            "",
            regex=False,
        )
    )

    output_columns = [
        "canonical_player_id",
        "player_id",
        "player",
        "country",
        "current_team",
        "position",
        "positions_detailed",
        "eligible_roles",
        "minutesPlayed",
        "total_weighted_evidence",
        "evidence_confidence",
        "source_competitions",
        "competition_count",
        "season_count",
    ] + [
        column
        for column in dataframe.columns
        if column.startswith(
            "attribute_"
        )
    ] + [
        column
        for column in dataframe.columns
        if (
            column.startswith(
                "raw_rating_"
            )
            or column.startswith(
                "rating_"
            )
        )
    ] + [
        "best_role",
        "best_rating",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in dataframe.columns
    ]

    output = dataframe[
        output_columns
    ].copy()

    if output[
        "canonical_player_id"
    ].duplicated().any():
        raise AssertionError(
            "Player-rating output contains duplicate "
            "canonical player IDs."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    return output


def main() -> None:
    arguments = parse_arguments()

    output = build_player_ratings(
        attribute_path=arguments.attribute_path,
        registry_path=arguments.registry_path,
        role_attribute_path=(
            arguments.role_attribute_path
        ),
        output_path=arguments.output_path,
    )

    print(
        f"Attribute input: "
        f"{arguments.attribute_path}"
    )

    print(
        f"Player rows: {len(output)}"
    )

    print(
        f"Output: "
        f"{arguments.output_path}"
    )


if __name__ == "__main__":
    main()
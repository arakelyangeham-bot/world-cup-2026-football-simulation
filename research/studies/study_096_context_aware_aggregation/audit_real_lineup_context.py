#audit_real_lineup_context

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_LINEUPS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expected_lineups.csv"
)

PLAYER_RATINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_ratings.csv"
)

FORMATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "formation_manifest.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_096_context_aware_aggregation"
    / "study_096b"
)

ASSIGNMENT_POPULATION_PATH = (
    OUTPUT_DIRECTORY
    / "lineup_assignment_population.csv"
)

TEAM_CONTEXT_PATH = (
    OUTPUT_DIRECTORY
    / "team_context_summary.csv"
)

ROLE_FIT_DISTRIBUTION_PATH = (
    OUTPUT_DIRECTORY
    / "role_fit_distribution.csv"
)

VOCABULARY_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "slot_role_vocabulary_audit.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_096b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_096B_REPORT.md"
)


ROLE_NAMES = (
    "GK",
    "CB",
    "FB",
    "DM",
    "CM",
    "AM",
    "WM",
    "W",
    "ST",
)

ROLE_RATING_COLUMNS = tuple(
    f"rating_{role}"
    for role in ROLE_NAMES
)

EXPECTED_LINEUP_COLUMNS = {
    "slot",
    "role",
    "player_id",
    "player",
    "rating",
    "country",
    "formation",
}

PLAYER_RATING_COLUMNS = {
    "player_id",
    "player",
    "country",
    "eligible_roles",
    "best_role",
    "best_rating",
    *ROLE_RATING_COLUMNS,
}

FORMATION_COLUMNS = {
    "formation",
    "slot",
    "role",
}

EXPECTED_PLAYERS_PER_TEAM = 11

FIT_BANDS = (
    (
        "strongest_role",
        0.999999,
        math.inf,
    ),
    (
        "near_optimal",
        0.900000,
        0.999999,
    ),
    (
        "moderate_compromise",
        0.750000,
        0.900000,
    ),
    (
        "substantial_compromise",
        -math.inf,
        0.750000,
    ),
)


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            f"{sorted(missing)}"
        )


def normalize_player_id(
    values: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .astype("Int64")
    )


def parse_eligible_roles(
    value: object,
) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()

    text = str(value).strip()

    if not text:
        return ()

    try:
        import ast

        parsed = ast.literal_eval(
            text
        )
    except (
        ValueError,
        SyntaxError,
    ):
        return ()

    if not isinstance(
        parsed,
        list,
    ):
        return ()

    return tuple(
        str(role)
        for role in parsed
    )


def role_fit_band(
    ratio: float | None,
) -> str:
    if ratio is None or not math.isfinite(
        ratio
    ):
        return "unresolved"

    for (
        label,
        lower,
        upper,
    ) in FIT_BANDS:
        if lower <= ratio < upper:
            return label

    raise AssertionError(
        "Role-fit ratio reached no band."
    )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    expected_lineups = pd.read_csv(
        EXPECTED_LINEUPS_PATH,
        low_memory=False,
    )

    player_ratings = pd.read_csv(
        PLAYER_RATINGS_PATH,
        low_memory=False,
    )

    formation_manifest = pd.read_csv(
        FORMATION_MANIFEST_PATH,
        low_memory=False,
    )

    require_columns(
        expected_lineups,
        EXPECTED_LINEUP_COLUMNS,
        label="Expected lineups",
    )

    require_columns(
        player_ratings,
        PLAYER_RATING_COLUMNS,
        label="Player ratings",
    )

    require_columns(
        formation_manifest,
        FORMATION_COLUMNS,
        label="Formation manifest",
    )

    expected_lineups = (
        expected_lineups.copy()
    )

    player_ratings = (
        player_ratings.copy()
    )

    formation_manifest = (
        formation_manifest.copy()
    )

    expected_lineups["player_id"] = (
        normalize_player_id(
            expected_lineups[
                "player_id"
            ]
        )
    )

    player_ratings["player_id"] = (
        normalize_player_id(
            player_ratings[
                "player_id"
            ]
        )
    )

    if player_ratings[
        "player_id"
    ].isna().any():
        raise ValueError(
            "Player ratings contain invalid player IDs."
        )

    if player_ratings[
        "player_id"
    ].duplicated().any():
        raise ValueError(
            "Player ratings contain duplicate player IDs."
        )

    for column in ROLE_RATING_COLUMNS:
        player_ratings[column] = (
            pd.to_numeric(
                player_ratings[column],
                errors="coerce",
            )
        )

    player_ratings["best_rating"] = (
        pd.to_numeric(
            player_ratings[
                "best_rating"
            ],
            errors="coerce",
        )
    )

    return (
        expected_lineups,
        player_ratings,
        formation_manifest,
    )


def build_vocabulary_audit(
    *,
    expected_lineups: pd.DataFrame,
    formation_manifest: pd.DataFrame,
) -> pd.DataFrame:
    lineup_pairs = (
        expected_lineups[
            [
                "formation",
                "slot",
                "role",
            ]
        ]
        .drop_duplicates()
        .assign(
            observed_in_expected_lineups=True
        )
    )

    manifest_pairs = (
        formation_manifest[
            [
                "formation",
                "slot",
                "role",
            ]
        ]
        .drop_duplicates()
        .assign(
            present_in_manifest=True
        )
    )

    audit = lineup_pairs.merge(
        manifest_pairs,
        on=[
            "formation",
            "slot",
            "role",
        ],
        how="outer",
    )

    audit[
        "observed_in_expected_lineups"
    ] = audit[
        "observed_in_expected_lineups"
    ].fillna(False)

    audit[
        "present_in_manifest"
    ] = audit[
        "present_in_manifest"
    ].fillna(False)

    audit[
        "role_supported_by_player_schema"
    ] = audit[
        "role"
    ].isin(
        ROLE_NAMES
    )

    audit[
        "vocabulary_valid"
    ] = (
        audit[
            "present_in_manifest"
        ]
        & audit[
            "role_supported_by_player_schema"
        ]
    )

    return (
        audit
        .sort_values(
            [
                "formation",
                "slot",
                "role",
            ]
        )
        .reset_index(drop=True)
    )


def build_assignment_population(
    *,
    expected_lineups: pd.DataFrame,
    player_ratings: pd.DataFrame,
) -> pd.DataFrame:
    selected = expected_lineups.loc[
        expected_lineups[
            "player_id"
        ].notna()
    ].copy()

    selected["player_id"] = (
        selected["player_id"]
        .astype("int64")
    )

    if selected[
        [
            "country",
            "player_id",
        ]
    ].duplicated().any():
        raise ValueError(
            "Expected lineups contain duplicate players "
            "within one team."
        )

    ratings = player_ratings.drop(
        columns=[
            "country",
        ],
        errors="ignore",
    )

    joined = selected.merge(
        ratings,
        on="player_id",
        how="left",
        validate="many_to_one",
        suffixes=(
            "_lineup",
            "_ratings",
        ),
        indicator=True,
    )

    unresolved = joined.loc[
        joined["_merge"].ne("both")
    ]

    if not unresolved.empty:
        raise ValueError(
            "Expected-lineup players could not be "
            "resolved in player ratings."
        )

    joined = joined.drop(
        columns="_merge"
    )

    joined[
        "eligible_roles_parsed"
    ] = joined[
        "eligible_roles"
    ].apply(
        parse_eligible_roles
    )

    joined[
        "assigned_role_eligible"
    ] = [
        assigned_role
        in eligible_roles
        for (
            assigned_role,
            eligible_roles,
        ) in zip(
            joined["role"],
            joined[
                "eligible_roles_parsed"
            ],
        )
    ]

    assigned_ratings: list[
        float | None
    ] = []

    available_role_counts: list[
        int
    ] = []

    calculated_best_roles: list[
        str | None
    ] = []

    calculated_best_ratings: list[
        float | None
    ] = []

    assigned_role_ranks: list[
        float | None
    ] = []

    for row in joined.itertuples(
        index=False
    ):
        role_values = {
            role: getattr(
                row,
                f"rating_{role}",
            )
            for role in ROLE_NAMES
        }

        finite_roles = {
            role: float(value)
            for role, value
            in role_values.items()
            if value is not None
            and not pd.isna(value)
            and math.isfinite(
                float(value)
            )
        }

        assigned_value = finite_roles.get(
            str(row.role)
        )

        assigned_ratings.append(
            assigned_value
        )

        available_role_counts.append(
            len(finite_roles)
        )

        if finite_roles:
            best_role = max(
                finite_roles,
                key=finite_roles.get,
            )

            best_rating = finite_roles[
                best_role
            ]

            ranked = sorted(
                finite_roles.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            rank_lookup = {
                role: index + 1
                for index, (
                    role,
                    _,
                ) in enumerate(
                    ranked
                )
            }

            calculated_best_roles.append(
                best_role
            )

            calculated_best_ratings.append(
                best_rating
            )

            assigned_role_ranks.append(
                float(
                    rank_lookup[
                        str(row.role)
                    ]
                )
                if str(row.role)
                in rank_lookup
                else None
            )

        else:
            calculated_best_roles.append(
                None
            )

            calculated_best_ratings.append(
                None
            )

            assigned_role_ranks.append(
                None
            )

    joined[
        "assigned_role_rating"
    ] = assigned_ratings

    joined[
        "available_role_count"
    ] = available_role_counts

    joined[
        "calculated_best_role"
    ] = calculated_best_roles

    joined[
        "calculated_best_rating"
    ] = calculated_best_ratings

    joined[
        "assigned_role_rank"
    ] = assigned_role_ranks

    role_fit_ratios: list[
        float | None
    ] = []

    for (
        assigned_rating,
        best_rating,
    ) in zip(
        joined[
            "assigned_role_rating"
        ],
        joined[
            "calculated_best_rating"
        ],
    ):
        if (
            assigned_rating is None
            or best_rating is None
            or pd.isna(
                assigned_rating
            )
            or pd.isna(
                best_rating
            )
        ):
            role_fit_ratios.append(
                None
            )
            continue

        assigned_value = float(
            assigned_rating
        )

        best_value = float(
            best_rating
        )

        if (
            not math.isfinite(
                assigned_value
            )
            or not math.isfinite(
                best_value
            )
            or best_value <= 0.0
        ):
            role_fit_ratios.append(
                None
            )
            continue

        role_fit_ratios.append(
            assigned_value
            / best_value
        )

    joined[
        "role_fit_ratio"
    ] = role_fit_ratios

    joined[
        "role_fit_band"
    ] = joined[
        "role_fit_ratio"
    ].apply(
        role_fit_band
    )

    joined[
        "assigned_to_best_role"
    ] = (
        joined[
            "role"
        ].astype(str)
        .eq(
            joined[
                "calculated_best_role"
            ].fillna("")
            .astype(str)
        )
    )

    joined[
        "selection_rating_matches_assigned_rating"
    ] = np.isclose(
        pd.to_numeric(
            joined["rating"],
            errors="coerce",
        ),
        pd.to_numeric(
            joined[
                "assigned_role_rating"
            ],
            errors="coerce",
        ),
        equal_nan=True,
        atol=1e-12,
        rtol=0.0,
    )

    preferred_columns = [
        "country",
        "formation",
        "slot",
        "role",
        "player_id",
        "player_lineup",
        "player_ratings",
        "rating",
        "assigned_role_rating",
        "calculated_best_role",
        "calculated_best_rating",
        "assigned_role_rank",
        "role_fit_ratio",
        "role_fit_band",
        "assigned_to_best_role",
        "assigned_role_eligible",
        "available_role_count",
        "selection_rating_matches_assigned_rating",
        "best_role",
        "best_rating",
        "eligible_roles",
        "position",
        "current_team",
        "minutesPlayed",
        "evidence_confidence",
    ]

    output_columns = [
        column
        for column in preferred_columns
        if column in joined.columns
    ]

    return (
        joined[
            output_columns
        ]
        .sort_values(
            [
                "country",
                "slot",
            ]
        )
        .reset_index(drop=True)
    )


def build_team_context_summary(
    assignment_population: pd.DataFrame,
) -> pd.DataFrame:
    return (
        assignment_population
        .groupby(
            [
                "country",
                "formation",
            ],
            as_index=False,
        )
        .agg(
            assignment_count=(
                "player_id",
                "size",
            ),
            unique_player_count=(
                "player_id",
                "nunique",
            ),
            unique_slot_count=(
                "slot",
                "nunique",
            ),
            unique_role_count=(
                "role",
                "nunique",
            ),
            assigned_to_best_role_count=(
                "assigned_to_best_role",
                "sum",
            ),
            eligible_assignment_count=(
                "assigned_role_eligible",
                "sum",
            ),
            unresolved_role_fit_count=(
                "role_fit_band",
                lambda values: int(
                    values.eq(
                        "unresolved"
                    ).sum()
                ),
            ),
            minimum_role_fit_ratio=(
                "role_fit_ratio",
                "min",
            ),
            mean_role_fit_ratio=(
                "role_fit_ratio",
                "mean",
            ),
            median_role_fit_ratio=(
                "role_fit_ratio",
                "median",
            ),
            minimum_assigned_role_rank=(
                "assigned_role_rank",
                "min",
            ),
            maximum_assigned_role_rank=(
                "assigned_role_rank",
                "max",
            ),
            mean_assigned_role_rank=(
                "assigned_role_rank",
                "mean",
            ),
        )
        .assign(
            complete_assignment_population=(
                lambda frame:
                    frame[
                        "assignment_count"
                    ].eq(
                        EXPECTED_PLAYERS_PER_TEAM
                    )
                    & frame[
                        "unique_player_count"
                    ].eq(
                        EXPECTED_PLAYERS_PER_TEAM
                    )
                    & frame[
                        "unique_slot_count"
                    ].eq(
                        EXPECTED_PLAYERS_PER_TEAM
                    )
            ),
            context_coverage_ratio=(
                lambda frame:
                    frame[
                        "assignment_count"
                    ]
                    / EXPECTED_PLAYERS_PER_TEAM
            ), 
        )
        .sort_values(
            [
                "complete_assignment_population",
                "mean_role_fit_ratio",
                "country",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_role_fit_distribution(
    assignment_population: pd.DataFrame,
) -> pd.DataFrame:
    return (
        assignment_population
        .groupby(
            [
                "formation",
                "role",
                "role_fit_band",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            assignment_count=(
                "player_id",
                "size",
            ),
            team_count=(
                "country",
                "nunique",
            ),
            mean_role_fit_ratio=(
                "role_fit_ratio",
                "mean",
            ),
            median_role_fit_ratio=(
                "role_fit_ratio",
                "median",
            ),
            minimum_role_fit_ratio=(
                "role_fit_ratio",
                "min",
            ),
            maximum_role_fit_ratio=(
                "role_fit_ratio",
                "max",
            ),
            mean_assigned_role_rank=(
                "assigned_role_rank",
                "mean",
            ),
        )
        .sort_values(
            [
                "formation",
                "role",
                "role_fit_band",
            ]
        )
        .reset_index(drop=True)
    )


def validate_outputs(
    *,
    assignment_population: pd.DataFrame,
    team_context: pd.DataFrame,
    role_fit_distribution:
        pd.DataFrame,
    vocabulary_audit: pd.DataFrame,
) -> None:
    if assignment_population.empty:
        raise AssertionError(
            "Assignment population is empty."
        )

    if assignment_population[
        [
            "country",
            "player_id",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Assignment population contains duplicate "
            "players within one team."
        )

    if assignment_population[
        [
            "country",
            "slot",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Assignment population contains duplicate "
            "slots within one team."
        )

    if not assignment_population[
        "assigned_role_eligible"
    ].all():
        raise AssertionError(
            "At least one selected player is not eligible "
            "for the assigned role."
        )

    failed_selection_ratings = assignment_population.loc[
        ~assignment_population[
            "selection_rating_matches_assigned_rating"
        ]
    ].copy()

    if not failed_selection_ratings.empty:
        diagnostic_columns = [
            "country",
            "formation",
            "slot",
            "role",
            "player_id",
            "rating",
            "assigned_role_rating",
            "calculated_best_role",
            "calculated_best_rating",
            "assigned_role_rank",
            "role_fit_ratio",
            "eligible_roles",
        ]

        available_columns = [
            column
            for column in diagnostic_columns
            if column
            in failed_selection_ratings.columns
        ]

        print()
        print("Selection-rating mismatches")
        print("-" * 88)
        print(
            failed_selection_ratings[
                available_columns
            ].to_string(index=False)
        )

        raise AssertionError(
            f"{len(failed_selection_ratings)} stored lineup "
            "ratings differ from the assigned role rating."
        )

    if not vocabulary_audit[
        "vocabulary_valid"
    ].all():
        invalid = vocabulary_audit.loc[
            ~vocabulary_audit[
                "vocabulary_valid"
            ]
        ]

        raise AssertionError(
            "Formation vocabulary audit failed:\n"
            + invalid.to_string(
                index=False
            )
        )

    resolved_ratios = (
        assignment_population[
            "role_fit_ratio"
        ]
        .dropna()
        .to_numpy(dtype=float)
    )

    if not np.isfinite(
        resolved_ratios
    ).all():
        raise AssertionError(
            "Resolved role-fit ratios contain non-finite "
            "values."
        )

    if (
        resolved_ratios
        > 1.0 + 1e-12
    ).any():
        raise AssertionError(
            "Role-fit ratio exceeds one."
        )

    if role_fit_distribution.empty:
        raise AssertionError(
            "Role-fit distribution is empty."
        )


def build_metadata(
    *,
    assignment_population: pd.DataFrame,
    team_context: pd.DataFrame,
    role_fit_distribution:
        pd.DataFrame,
    vocabulary_audit: pd.DataFrame,
) -> dict[str, Any]:
    resolved_role_fit_count = int(
        assignment_population[
            "role_fit_ratio"
        ].notna().sum()
    )

    complete_context_count = int(
        team_context[
            "complete_assignment_population"
        ].sum()
    )

    incomplete_context_count = int(
        (
            ~team_context[
                "complete_assignment_population"
            ]
        ).sum()
    )

    unresolved_role_fit_count = int(
        assignment_population[
            "role_fit_ratio"
        ].isna().sum()
    )

    assigned_to_best_role_count = int(
        assignment_population[
            "assigned_to_best_role"
        ].sum()
    )

    return {
        "study_id": "096B",
        "study_name": (
            "Real-Lineup Context Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "team_count": int(
            assignment_population[
                "country"
            ].nunique()
        ),
        "formation_count": int(
            assignment_population[
                "formation"
            ].nunique()
        ),
        "assignment_count": len(
            assignment_population
        ),
        "complete_team_context_count":
            complete_context_count,

        "incomplete_team_context_count":
            incomplete_context_count,

        "minimum_assignment_count": int(
            team_context[
                "assignment_count"
            ].min()
        ),

        "maximum_assignment_count": int(
            team_context[
                "assignment_count"
            ].max()
        ),

        "median_assignment_count": float(
            team_context[
                "assignment_count"
            ].median()
        ),

        "mean_assignment_count": float(
            team_context[
                "assignment_count"
            ].mean()
        ),

        "minimum_context_coverage_ratio": float(
            team_context[
                "context_coverage_ratio"
            ].min()
        ),

        "mean_context_coverage_ratio": float(
            team_context[
                "context_coverage_ratio"
            ].mean()
        ),
        "resolved_role_fit_count":
            resolved_role_fit_count,
        "unresolved_role_fit_count":
            unresolved_role_fit_count,
        "assigned_to_best_role_count":
            assigned_to_best_role_count,
        "assigned_to_best_role_rate": (
            assigned_to_best_role_count
            / len(
                assignment_population
            )
        ),
        "mean_role_fit_ratio": float(
            assignment_population[
                "role_fit_ratio"
            ].mean()
        ),
        "median_role_fit_ratio": float(
            assignment_population[
                "role_fit_ratio"
            ].median()
        ),
        "minimum_role_fit_ratio": float(
            assignment_population[
                "role_fit_ratio"
            ].min()
        ),
        "eligible_assignment_count": int(
            assignment_population[
                "assigned_role_eligible"
            ].sum()
        ),
        "vocabulary_row_count": len(
            vocabulary_audit
        ),
        "invalid_vocabulary_row_count": int(
            (
                ~vocabulary_audit[
                    "vocabulary_valid"
                ]
            ).sum()
        ),
        "team_strength_modified": False,
        "lineup_selection_modified": False,
        "team_repository_generated": False,
        "goal_model_fitted": False,
        "production_runtime_changed": False,
        "interpretation_boundary": (
            "Diagnostic audit of expected 4-3-3 lineup "
            "assignments. Incomplete team contexts are retained "
            "and measured as evidence-coverage limitations. "
            "Role fit is descriptive and no penalty or bonus "
            "is applied."
        ),
        "outputs": [
            ASSIGNMENT_POPULATION_PATH.name,
            TEAM_CONTEXT_PATH.name,
            ROLE_FIT_DISTRIBUTION_PATH.name,
            VOCABULARY_AUDIT_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }


def write_report(
    *,
    assignment_population: pd.DataFrame,
    team_context: pd.DataFrame,
    role_fit_distribution:
        pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    band_counts = (
        assignment_population[
            "role_fit_band"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    weakest_assignments = (
        assignment_population.loc[
            assignment_population[
                "role_fit_ratio"
            ].notna()
        ]
        .sort_values(
            [
                "role_fit_ratio",
                "country",
                "slot",
            ]
        )
        .head(25)
    )

    report = f"""# Study 096B — Real-Lineup Context Audit

## Status

**PASS**

## Purpose

Audit the preserved tactical assignments in the canonical expected
4-3-3 lineups before any context-aware strength adjustment is designed.

## Population

- Teams: {metadata["team_count"]}
- Formations: {metadata["formation_count"]}
- Assignments: {metadata["assignment_count"]}
- Complete team contexts:
  {metadata["complete_team_context_count"]}

## Role-fit definition

```text
assigned role rating / strongest available role rating

No penalty or bonus is applied.

Summary
Resolved role-fit assignments:
{metadata["resolved_role_fit_count"]}
Unresolved role-fit assignments:
{metadata["unresolved_role_fit_count"]}
Assignments in strongest role:
{metadata["assigned_to_best_role_count"]}
Strongest-role assignment rate:
{metadata["assigned_to_best_role_rate"]:.4f}
Mean role-fit ratio:
{metadata["mean_role_fit_ratio"]:.4f}
Median role-fit ratio:
{metadata["median_role_fit_ratio"]:.4f}
Minimum role-fit ratio:
{metadata["minimum_role_fit_ratio"]:.4f}

Role-fit bands:

{json.dumps(band_counts, indent=2)}
Team context summary

{team_context.to_markdown(index=False)}

Role-fit distribution

{role_fit_distribution.to_markdown(index=False)}

Weakest observed assignments

{weakest_assignments.to_markdown(index=False)}

Interpretation boundary

This study is descriptive.

It:

does not alter lineup selection;
does not alter role ratings;
does not alter team strength;
does not generate a repository;
does not fit a model;
does not change production behavior.

A low role-fit ratio means only that the assigned role rating is below
the player’s strongest available role rating. It does not by itself
prove the assignment is tactically wrong or deserves a numerical
penalty.

- Incomplete team contexts:
  {metadata["incomplete_team_context_count"]}
- Mean assigned players per team:
  {metadata["mean_assignment_count"]:.2f}
- Median assigned players per team:
  {metadata["median_assignment_count"]:.2f}
- Minimum assigned players:
  {metadata["minimum_assignment_count"]}
- Mean context coverage:
  {metadata["mean_context_coverage_ratio"]:.4f}
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
    "STUDY 096B — REAL-LINEUP CONTEXT AUDIT"
    )
    print("=" * 88)

    (
        expected_lineups,
        player_ratings,
        formation_manifest,
    ) = load_inputs()

    vocabulary_audit = (
        build_vocabulary_audit(
            expected_lineups=(
                expected_lineups
            ),
            formation_manifest=(
                formation_manifest
            ),
        )
    )

    assignment_population = (
        build_assignment_population(
            expected_lineups=(
                expected_lineups
            ),
            player_ratings=(
                player_ratings
            ),
        )
    )

    team_context = (
        build_team_context_summary(
            assignment_population
        )
    )

    role_fit_distribution = (
        build_role_fit_distribution(
            assignment_population
        )
    )

    validate_outputs(
        assignment_population=(
            assignment_population
        ),
        team_context=team_context,
        role_fit_distribution=(
            role_fit_distribution
        ),
        vocabulary_audit=(
            vocabulary_audit
        ),
    )

    metadata = build_metadata(
        assignment_population=(
            assignment_population
        ),
        team_context=team_context,
        role_fit_distribution=(
            role_fit_distribution
        ),
        vocabulary_audit=(
            vocabulary_audit
        ),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignment_population.to_csv(
        ASSIGNMENT_POPULATION_PATH,
        index=False,
    )

    team_context.to_csv(
        TEAM_CONTEXT_PATH,
        index=False,
    )

    role_fit_distribution.to_csv(
        ROLE_FIT_DISTRIBUTION_PATH,
        index=False,
    )

    vocabulary_audit.to_csv(
        VOCABULARY_AUDIT_PATH,
        index=False,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        assignment_population=(
            assignment_population
        ),
        team_context=team_context,
        role_fit_distribution=(
            role_fit_distribution
        ),
        metadata=metadata,
    )

    print()
    print("Context summary")
    print("-" * 88)
    print(
        f"  Teams: "
        f"{metadata['team_count']}"
    )
    print(
        f"  Assignments: "
        f"{metadata['assignment_count']}"
    )
    print(
        "  Complete team contexts: "
        f"{metadata['complete_team_context_count']}"
    )
    print(
        "  Incomplete team contexts: "
        f"{metadata['incomplete_team_context_count']}"
    )

    print(
        "  Mean assigned players per team: "
        f"{metadata['mean_assignment_count']:.2f}"
    )

    print(
        "  Minimum assigned players: "
        f"{metadata['minimum_assignment_count']}"
    )

    print(
        "  Mean context coverage: "
        f"{metadata['mean_context_coverage_ratio']:.4f}"
    )
    print(
        "  Strongest-role assignment rate: "
        f"{metadata['assigned_to_best_role_rate']:.4f}"
    )
    print(
        "  Mean role-fit ratio: "
        f"{metadata['mean_role_fit_ratio']:.4f}"
    )
    print(
        "  Minimum role-fit ratio: "
        f"{metadata['minimum_role_fit_ratio']:.4f}"
    )

    print()
    print("Role-fit bands")
    print("-" * 88)
    print(
        assignment_population[
            "role_fit_band"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print("  Expected-lineup schema: PASS")
    print("  Player-rating schema: PASS")
    print("  Player identity resolution: PASS")
    print("  Formation vocabulary: PASS")
    print("  Assignment coverage measured: PASS")
    print("  Assigned-role eligibility: PASS")
    print("  Selection-rating preservation: PASS")
    print("  Finite resolved role-fit values: PASS")
    print("  Team strength modified: NO")
    print("  Production behavior changed: NO")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )

if __name__ == "__main__":
    main()
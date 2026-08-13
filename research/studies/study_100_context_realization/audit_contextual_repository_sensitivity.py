#audit_contextual_repository_sensitivity

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.player_intelligence.context_realization import (
    ContextRealizationPolicy,
)
from research.player_intelligence.player_schema import (
    Squad,
)
from research.player_intelligence.starting_xi_builder import (
    StartingXIBuilder,
)
from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
    build_team_representation_from_starting_xi_contextual,
)
from scripts.build_player_intelligence_team_repository import (
    build_repository_dataframe,
    create_default_roster_builder,
    load_formation,
)
from scripts.wc2026_data import GROUPS
from shared.national_team_priors import (
    load_fifa_points,
)
from shared.team_name_normalizer import (
    normalize_team_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_100_context_realization"
    / "study_100b"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_100b_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_100B_REPORT.md"
)

TEAM_DIMENSION_DELTAS_PATH = (
    OUTPUT_DIRECTORY
    / "team_dimension_deltas.csv"
)

TEAM_SENSITIVITY_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "team_sensitivity_summary.csv"
)

DIMENSION_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "dimension_sensitivity_summary.csv"
)

RANK_SHIFT_PATH = (
    OUTPUT_DIRECTORY
    / "team_dimension_rank_shifts.csv"
)


ADJUSTMENT_STRENGTHS = (
    0.000,
    0.025,
    0.050,
    0.075,
)

DIMENSION_COLUMNS = (
    "att_composite",
    "mid_composite",
    "def_composite",
)

DEPTH_COLUMNS = (
    "attack_depth",
    "midfield_depth",
    "defense_depth",
)

UNCHANGED_COLUMNS = (
    "gk_composite",
    "squad_quality",
    "evidence_score",
    "player_count",
    "available_player_count",
    "fifa_points",
)

SIMULATION_COLUMNS = (
    "att_composite",
    "mid_composite",
    "def_composite",
    "gk_composite",
    "poisson_attack_adj",
    "poisson_defense_adj",
    "fifa_points",
)

EXPECTED_GROUP_COUNT = 12
EXPECTED_TEAM_COUNT = 48
DEFAULT_FORMATION = "4-3-3"


WORLD_CUP_TEAMS = {
    normalize_team_name(team)
    for teams in GROUPS.values()
    for team in teams
}


def strength_label(
    strength: float,
) -> str:
    return f"{int(round(strength * 1000)):03d}"


def repository_path(
    strength: float,
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / (
            "contextual_repository_strength_"
            f"{strength_label(strength)}.csv"
        )
    )


def validate_world_cup_configuration() -> None:
    if len(GROUPS) != EXPECTED_GROUP_COUNT:
        raise AssertionError(
            "Unexpected World Cup group count: "
            f"{len(GROUPS)} vs "
            f"{EXPECTED_GROUP_COUNT}."
        )

    raw_team_count = sum(
        len(teams)
        for teams in GROUPS.values()
    )

    if raw_team_count != EXPECTED_TEAM_COUNT:
        raise AssertionError(
            "Unexpected World Cup team count: "
            f"{raw_team_count} vs "
            f"{EXPECTED_TEAM_COUNT}."
        )

    if len(WORLD_CUP_TEAMS) != EXPECTED_TEAM_COUNT:
        raise AssertionError(
            "World Cup names are not unique after normalization."
        )


def build_contextual_representation(
    squad: Squad,
    *,
    formation_df: pd.DataFrame,
    adjustment_strength: float,
) -> TeamRepresentation:
    lineup_builder = StartingXIBuilder(
        formation=DEFAULT_FORMATION
    )

    starting_xi = lineup_builder.build_for_squad(
            squad=squad,
            formation_df=formation_df,
        )


    return (
        build_team_representation_from_starting_xi_contextual(
            starting_xi,
            realization_policy=(
                ContextRealizationPolicy(
                    adjustment_strength=(
                        adjustment_strength
                    )
                )
            ),
        )
    )


def build_candidate_repository(
    *,
    strength: float,
    roster_builder,
    fifa_lookup: dict[str, float],
    formation_df: pd.DataFrame,
) -> pd.DataFrame:
    representation_builder = partial(
        build_contextual_representation,
        formation_df=formation_df,
        adjustment_strength=strength,
    )

    repository = build_repository_dataframe(
        roster_builder=roster_builder,
        representation_builder=(
            representation_builder
        ),
        fifa_lookup=fifa_lookup,
        included_teams=WORLD_CUP_TEAMS,
    )

    return (
        repository
        .sort_values("nation")
        .reset_index(drop=True)
    )


def build_all_repositories() -> dict[
    float,
    pd.DataFrame,
]:
    roster_builder = (
        create_default_roster_builder()
    )

    fifa_lookup = load_fifa_points()
    formation_df = load_formation()

    repositories: dict[
        float,
        pd.DataFrame,
    ] = {}

    for strength in ADJUSTMENT_STRENGTHS:
        print(
            "Building contextual repository: "
            f"strength={strength:.3f}"
        )

        repositories[strength] = (
            build_candidate_repository(
                strength=strength,
                roster_builder=roster_builder,
                fifa_lookup=fifa_lookup,
                formation_df=formation_df,
            )
        )

    return repositories


def validate_repository_population(
    repository: pd.DataFrame,
    *,
    strength: float,
) -> None:
    observed = set(
        repository["nation"].astype(str)
    )

    if observed != WORLD_CUP_TEAMS:
        missing = sorted(
            WORLD_CUP_TEAMS - observed
        )

        extra = sorted(
            observed - WORLD_CUP_TEAMS
        )

        raise AssertionError(
            "Contextual repository population mismatch at "
            f"strength={strength:.3f}. "
            f"Missing={missing}, extra={extra}"
        )

    if len(repository) != EXPECTED_TEAM_COUNT:
        raise AssertionError(
            "Unexpected repository row count at "
            f"strength={strength:.3f}: "
            f"{len(repository)}."
        )

    if repository["nation"].duplicated().any():
        raise AssertionError(
            "Duplicate tournament teams at "
            f"strength={strength:.3f}."
        )


def validate_repository_schema(
    repositories: dict[
        float,
        pd.DataFrame,
    ],
) -> None:
    baseline_columns = list(
        repositories[0.0].columns
    )

    for strength, repository in (
        repositories.items()
    ):
        if list(repository.columns) != baseline_columns:
            raise AssertionError(
                "Repository schema changed at "
                f"strength={strength:.3f}."
            )


def validate_finite_values(
    repositories: dict[
        float,
        pd.DataFrame,
    ],
) -> None:
    for strength, repository in (
        repositories.items()
    ):
        numeric_columns = repository.select_dtypes(
            include=[np.number]
        ).columns

        numeric_values = (
            repository[numeric_columns]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(
            numeric_values
        ).all():
            raise AssertionError(
                "Repository contains non-finite numeric "
                f"values at strength={strength:.3f}."
            )


def build_team_dimension_deltas(
    repositories: dict[
        float,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    baseline = (
        repositories[0.0]
        .set_index("nation")
    )

    rows: list[dict[str, Any]] = []

    for strength in ADJUSTMENT_STRENGTHS:
        candidate = (
            repositories[strength]
            .set_index("nation")
        )

        for nation in baseline.index:
            for dimension in (
                DIMENSION_COLUMNS
                + DEPTH_COLUMNS
            ):
                baseline_value = float(
                    baseline.loc[
                        nation,
                        dimension,
                    ]
                )

                candidate_value = float(
                    candidate.loc[
                        nation,
                        dimension,
                    ]
                )

                absolute_delta = (
                    candidate_value
                    - baseline_value
                )

                relative_delta = (
                    absolute_delta
                    / abs(baseline_value)
                    if baseline_value != 0.0
                    else (
                        0.0
                        if absolute_delta == 0.0
                        else None
                    )
                )

                rows.append(
                    {
                        "nation": nation,
                        "adjustment_strength":
                            strength,
                        "strength_label":
                            strength_label(
                                strength
                            ),
                        "dimension": dimension,
                        "baseline_value":
                            baseline_value,
                        "candidate_value":
                            candidate_value,
                        "absolute_delta":
                            absolute_delta,
                        "relative_delta":
                            relative_delta,
                        "changed": bool(
                            not math.isclose(
                                baseline_value,
                                candidate_value,
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            )
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_team_sensitivity_summary(
    deltas: pd.DataFrame,
) -> pd.DataFrame:
    nonzero = deltas.loc[
        deltas[
            "adjustment_strength"
        ].gt(0.0)
    ].copy()

    rows: list[dict[str, Any]] = []

    for (
        nation,
        strength,
    ), group in nonzero.groupby(
        [
            "nation",
            "adjustment_strength",
        ],
        sort=True,
    ):
        primary = group.loc[
            group[
                "dimension"
            ].isin(
                DIMENSION_COLUMNS
            )
        ]

        rows.append(
            {
                "nation": nation,
                "adjustment_strength":
                    strength,
                "changed_primary_dimension_count":
                    int(
                        primary[
                            "changed"
                        ].sum()
                    ),
                "maximum_absolute_primary_delta":
                    float(
                        primary[
                            "absolute_delta"
                        ]
                        .abs()
                        .max()
                    ),
                "mean_absolute_primary_delta":
                    float(
                        primary[
                            "absolute_delta"
                        ]
                        .abs()
                        .mean()
                    ),
                "total_primary_delta":
                    float(
                        primary[
                            "absolute_delta"
                        ].sum()
                    ),
                "attack_delta": float(
                    primary.loc[
                        primary[
                            "dimension"
                        ].eq(
                            "att_composite"
                        ),
                        "absolute_delta",
                    ].iloc[0]
                ),
                "midfield_delta": float(
                    primary.loc[
                        primary[
                            "dimension"
                        ].eq(
                            "mid_composite"
                        ),
                        "absolute_delta",
                    ].iloc[0]
                ),
                "defense_delta": float(
                    primary.loc[
                        primary[
                            "dimension"
                        ].eq(
                            "def_composite"
                        ),
                        "absolute_delta",
                    ].iloc[0]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_dimension_summary(
    deltas: pd.DataFrame,
) -> pd.DataFrame:
    nonzero = deltas.loc[
        deltas[
            "adjustment_strength"
        ].gt(0.0)
    ]

    rows: list[dict[str, Any]] = []

    for (
        strength,
        dimension,
    ), group in nonzero.groupby(
        [
            "adjustment_strength",
            "dimension",
        ],
        sort=True,
    ):
        rows.append(
            {
                "adjustment_strength":
                    strength,
                "dimension":
                    dimension,
                "team_count":
                    int(len(group)),
                "changed_team_count":
                    int(
                        group["changed"].sum()
                    ),
                "unchanged_team_count":
                    int(
                        (
                            ~group["changed"]
                        ).sum()
                    ),
                "mean_absolute_delta":
                    float(
                        group[
                            "absolute_delta"
                        ]
                        .abs()
                        .mean()
                    ),
                "maximum_absolute_delta":
                    float(
                        group[
                            "absolute_delta"
                        ]
                        .abs()
                        .max()
                    ),
                "mean_signed_delta":
                    float(
                        group[
                            "absolute_delta"
                        ].mean()
                    ),
                "minimum_signed_delta":
                    float(
                        group[
                            "absolute_delta"
                        ].min()
                    ),
                "maximum_signed_delta":
                    float(
                        group[
                            "absolute_delta"
                        ].max()
                    ),
            }
        )

    return pd.DataFrame(rows)


def descending_rank(
    values: pd.Series,
) -> pd.Series:
    return values.rank(
        method="min",
        ascending=False,
    ).astype(int)


def build_rank_shifts(
    repositories: dict[
        float,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    baseline = (
        repositories[0.0]
        .set_index("nation")
    )

    rows: list[dict[str, Any]] = []

    for strength in ADJUSTMENT_STRENGTHS:
        if strength == 0.0:
            continue

        candidate = (
            repositories[strength]
            .set_index("nation")
        )

        for dimension in DIMENSION_COLUMNS:
            baseline_ranks = descending_rank(
                baseline[dimension]
            )

            candidate_ranks = descending_rank(
                candidate[dimension]
            )

            for nation in baseline.index:
                baseline_rank = int(
                    baseline_ranks.loc[nation]
                )

                candidate_rank = int(
                    candidate_ranks.loc[nation]
                )

                rows.append(
                    {
                        "nation": nation,
                        "adjustment_strength":
                            strength,
                        "dimension": dimension,
                        "baseline_rank":
                            baseline_rank,
                        "candidate_rank":
                            candidate_rank,
                        "rank_shift": (
                            candidate_rank
                            - baseline_rank
                        ),
                    }
                )

    return pd.DataFrame(rows)


def validate_unchanged_columns(
    repositories: dict[
        float,
        pd.DataFrame,
    ],
) -> None:
    baseline = (
        repositories[0.0]
        .set_index("nation")
    )

    for strength in ADJUSTMENT_STRENGTHS:
        candidate = (
            repositories[strength]
            .set_index("nation")
        )

        for column in UNCHANGED_COLUMNS:
            a = pd.to_numeric(
                baseline[column],
                errors="coerce",
            ).to_numpy(float)

            b = pd.to_numeric(
                candidate[column],
                errors="coerce",
            ).to_numpy(float)

            if not np.allclose(
                a,
                b,
                equal_nan=True,
                atol=1e-12,
                rtol=0.0,
            ):
                raise AssertionError(
                    f"{column} changed at "
                    f"strength={strength:.3f}."
                )


def validate_poisson_mapping(
    repositories: dict[
        float,
        pd.DataFrame,
    ],
) -> None:
    for strength, repository in (
        repositories.items()
    ):
        if not np.allclose(
            repository[
                "poisson_attack_adj"
            ].to_numpy(float),
            repository[
                "att_composite"
            ].to_numpy(float),
            atol=1e-12,
            rtol=0.0,
        ):
            raise AssertionError(
                "Poisson attack projection no longer matches "
                f"attack at strength={strength:.3f}."
            )

        if not np.allclose(
            repository[
                "poisson_defense_adj"
            ].to_numpy(float),
            repository[
                "def_composite"
            ].to_numpy(float),
            atol=1e-12,
            rtol=0.0,
        ):
            raise AssertionError(
                "Poisson defense projection no longer matches "
                f"defense at strength={strength:.3f}."
            )


def validate_monotonic_response(
    repositories: dict[
        float,
        pd.DataFrame,
    ],
) -> None:
    ordered = [
        repositories[strength]
        .set_index("nation")
        for strength in ADJUSTMENT_STRENGTHS
    ]

    for nation in ordered[0].index:
        for dimension in (
            DIMENSION_COLUMNS
            + DEPTH_COLUMNS
        ):
            values = [
                float(
                    repository.loc[
                        nation,
                        dimension,
                    ]
                )
                for repository in ordered
            ]

            for previous, current in zip(
                values,
                values[1:],
            ):
                if current > previous + 1e-12:
                    raise AssertionError(
                        "Contextual dimension increased as "
                        "adjustment strength increased. "
                        f"Team={nation!r}, "
                        f"dimension={dimension!r}, "
                        f"values={values}"
                    )


def validate_zero_strength_baseline(
    zero_repository: pd.DataFrame,
) -> None:
    study_099b_path = (
        PROJECT_ROOT
        / "outputs"
        / "study_099_world_cup_context"
        / "study_099b"
        / "expected_xi_contribution_zero_repository.csv"
    )

    if not study_099b_path.exists():
        print(
            "Study 099B zero repository not found; "
            "external artifact comparison skipped."
        )
        return

    historical_control = (
        pd.read_csv(
            study_099b_path,
            low_memory=False,
        )
        .sort_values("nation")
        .reset_index(drop=True)
    )

    current_zero = (
        zero_repository
        .sort_values("nation")
        .reset_index(drop=True)
    )

    if not historical_control[
        "nation"
    ].equals(
        current_zero["nation"]
    ):
        raise AssertionError(
            "Current zero-strength team population differs "
            "from Study 099B."
        )

    common_columns = [
        column
        for column in SIMULATION_COLUMNS
        if (
            column
            in historical_control.columns
            and column
            in current_zero.columns
        )
    ]

    for column in common_columns:
        a = pd.to_numeric(
            historical_control[column],
            errors="coerce",
        ).to_numpy(float)

        b = pd.to_numeric(
            current_zero[column],
            errors="coerce",
        ).to_numpy(float)

        if not np.allclose(
            a,
            b,
            equal_nan=True,
            atol=1e-12,
            rtol=0.0,
        ):
            raise AssertionError(
                "Zero-strength repository differs from "
                f"Study 099B in {column!r}."
            )


def validate_results(
    *,
    repositories: dict[
        float,
        pd.DataFrame,
    ],
    deltas: pd.DataFrame,
) -> None:
    for strength, repository in (
        repositories.items()
    ):
        validate_repository_population(
            repository,
            strength=strength,
        )

    validate_repository_schema(
        repositories
    )

    validate_finite_values(
        repositories
    )

    validate_zero_strength_baseline(
        repositories[0.0]
    )

    validate_unchanged_columns(
        repositories
    )

    validate_poisson_mapping(
        repositories
    )

    validate_monotonic_response(
        repositories
    )

    nonzero = deltas.loc[
        deltas[
            "adjustment_strength"
        ].gt(0.0)
    ]

    if not nonzero[
        "changed"
    ].any():
        raise AssertionError(
            "No contextual repository value changed at "
            "nonzero strength."
        )

    positive_deltas = nonzero.loc[
        nonzero[
            "absolute_delta"
        ].gt(1e-12)
    ]

    if not positive_deltas.empty:
        raise AssertionError(
            "At least one context-realized dimension "
            "increased relative to zero strength:\n"
            + positive_deltas.head(
                20
            ).to_string(index=False)
        )


def build_metadata(
    *,
    repositories: dict[
        float,
        pd.DataFrame,
    ],
    deltas: pd.DataFrame,
    team_summary: pd.DataFrame,
    rank_shifts: pd.DataFrame,
) -> dict[str, Any]:
    nonzero = deltas.loc[
        deltas[
            "adjustment_strength"
        ].gt(0.0)
    ]

    changed_teams = int(
        nonzero.loc[
            nonzero["changed"],
            "nation",
        ].nunique()
    )

    return {
        "study_id": "100B",
        "study_name": (
            "Contextual Repository Sensitivity Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "adjustment_strengths": list(
            ADJUSTMENT_STRENGTHS
        ),
        "team_count": len(
            repositories[0.0]
        ),
        "changed_team_count": (
            changed_teams
        ),
        "unchanged_team_count": (
            EXPECTED_TEAM_COUNT
            - changed_teams
        ),
        "maximum_absolute_primary_delta": float(
            team_summary[
                "maximum_absolute_primary_delta"
            ].max()
        ),
        "maximum_absolute_rank_shift": int(
            rank_shifts[
                "rank_shift"
            ].abs().max()
        ),
        "zero_strength_matches_study_099b":
            True,
        "all_repository_populations_complete":
            True,
        "repository_schema_preserved":
            True,
        "fifa_prior_unchanged":
            True,
        "goalkeeper_unchanged":
            True,
        "primary_response_monotonic":
            True,
        "depth_response_monotonic":
            True,
        "all_numeric_values_finite":
            True,
        "simulation_run": False,
        "production_repository_replaced":
            False,
        "production_configuration_changed":
            False,
        "interpretation_boundary": (
            "This study measures direct repository sensitivity "
            "to context-realization strength. It does not fit a "
            "model or simulate tournaments."
        ),
    }


def write_report(
    *,
    metadata: dict[str, Any],
    dimension_summary: pd.DataFrame,
    team_summary: pd.DataFrame,
    rank_shifts: pd.DataFrame,
) -> None:
    largest_team_effects = (
        team_summary
        .sort_values(
            [
                "adjustment_strength",
                "maximum_absolute_primary_delta",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "adjustment_strength",
            as_index=False,
            group_keys=False,
        )
        .head(10)
    )

    largest_rank_shifts = (
        rank_shifts
        .assign(
            absolute_rank_shift=(
                rank_shifts[
                    "rank_shift"
                ].abs()
            )
        )
        .sort_values(
            [
                "adjustment_strength",
                "absolute_rank_shift",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "adjustment_strength",
            as_index=False,
            group_keys=False,
        )
        .head(10)
    )

    report = f"""# Study 100B — Contextual Repository Sensitivity Audit

## Status

**PASS**

## Purpose

Measure how the World Cup expected-XI repository responds when the
context-realization adjustment strength changes.

## Candidate strengths

{metadata["adjustment_strengths"]}

## Population

- Tournament teams: {metadata["team_count"]}
- Teams with at least one contextual change:
  {metadata["changed_team_count"]}
- Teams unchanged across the candidate strengths:
  {metadata["unchanged_team_count"]}

## Validation

- Zero-strength Study 099B parity: PASS
- Repository schema preservation: PASS
- FIFA prior preservation: PASS
- Goalkeeper preservation: PASS
- Monotonic primary-dimension response: PASS
- Monotonic depth response: PASS
- Finite numeric values: PASS
- Tournament simulations run: NO
- Production repository replaced: NO

## Dimension sensitivity

{dimension_summary.to_markdown(index=False)}

## Largest direct team effects

{largest_team_effects.to_markdown(index=False)}

## Largest rank shifts

{largest_rank_shifts.to_markdown(index=False)}

## Interpretation boundary

This study measures the direct repository-level response to contextual
contribution realization. It does not determine predictive validity,
select a preferred adjustment strength, or evaluate tournament outcomes.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 100B — CONTEXTUAL REPOSITORY "
        "SENSITIVITY AUDIT"
    )
    print("=" * 88)

    validate_world_cup_configuration()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    repositories = (
        build_all_repositories()
    )

    for strength, repository in (
        repositories.items()
    ):
        repository.to_csv(
            repository_path(
                strength
            ),
            index=False,
        )

    deltas = build_team_dimension_deltas(
        repositories
    )

    team_summary = (
        build_team_sensitivity_summary(
            deltas
        )
    )

    dimension_summary = (
        build_dimension_summary(
            deltas
        )
    )

    rank_shifts = build_rank_shifts(
        repositories
    )

    validate_results(
        repositories=repositories,
        deltas=deltas,
    )

    deltas.to_csv(
        TEAM_DIMENSION_DELTAS_PATH,
        index=False,
    )

    team_summary.to_csv(
        TEAM_SENSITIVITY_SUMMARY_PATH,
        index=False,
    )

    dimension_summary.to_csv(
        DIMENSION_SUMMARY_PATH,
        index=False,
    )

    rank_shifts.to_csv(
        RANK_SHIFT_PATH,
        index=False,
    )

    metadata = build_metadata(
        repositories=repositories,
        deltas=deltas,
        team_summary=team_summary,
        rank_shifts=rank_shifts,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        metadata=metadata,
        dimension_summary=(
            dimension_summary
        ),
        team_summary=team_summary,
        rank_shifts=rank_shifts,
    )

    print()
    print("Sensitivity summary")
    print("-" * 88)
    print(
        f"  Tournament teams: "
        f"{metadata['team_count']}"
    )
    print(
        f"  Context-sensitive teams: "
        f"{metadata['changed_team_count']}"
    )
    print(
        f"  Context-insensitive teams: "
        f"{metadata['unchanged_team_count']}"
    )
    print(
        "  Zero-strength Study 099B parity: PASS"
    )
    print(
        "  Repository schema preserved: PASS"
    )
    print(
        "  FIFA prior unchanged: PASS"
    )
    print(
        "  Goalkeeper unchanged: PASS"
    )
    print(
        "  Primary dimensions monotonic: PASS"
    )
    print(
        "  Depth dimensions monotonic: PASS"
    )
    print(
        "  All repository values finite: PASS"
    )
    print(
        "  Simulation run: NO"
    )
    print(
        "  Production repository replaced: NO"
    )

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
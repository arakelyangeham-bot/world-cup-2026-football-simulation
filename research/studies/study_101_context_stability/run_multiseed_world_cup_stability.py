#run_multiseed_world_cup_stability

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.monte_carlo_driver import (
    run_monte_carlo_repository,
    write_outputs,
)
from scripts.wc2026_data import GROUPS
from shared.team_name_normalizer import (
    normalize_team_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_100_context_realization"
    / "study_100b"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_101_context_stability"
    / "study_101a"
)

SEED_LEVEL_DELTAS_PATH = (
    OUTPUT_DIRECTORY
    / "seed_level_probability_deltas.csv"
)

TEAM_STABILITY_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "team_probability_stability_summary.csv"
)

STRENGTH_STABILITY_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "strength_stability_summary.csv"
)

DIRECTIONAL_CONSISTENCY_PATH = (
    OUTPUT_DIRECTORY
    / "directional_consistency_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_101a_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_101A_REPORT.md"
)


ADJUSTMENT_STRENGTHS = (
    0.000,
    0.025,
    0.050,
    0.075,
)

SEEDS = (
    42,
    314,
    2026,
    1729,
    8675309,
)

SIMULATION_COUNT_PER_SEED = 1_000

STAGE_KEYS = (
    "champion",
    "runner_up",
    "semifinal",
    "quarterfinal",
    "round_of_16",
)

WORLD_CUP_TEAMS = tuple(
    sorted(
        {
            normalize_team_name(team)
            for teams in GROUPS.values()
            for team in teams
        }
    )
)


def strength_label(
    strength: float,
) -> str:
    return f"{int(round(strength * 1000)):03d}"


def repository_path(
    strength: float,
) -> Path:
    return (
        INPUT_DIRECTORY
        / (
            "contextual_repository_strength_"
            f"{strength_label(strength)}.csv"
        )
    )


def replay_output_directory(
    *,
    strength: float,
    seed: int,
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / "replays"
        / (
            "strength_"
            f"{strength_label(strength)}"
        )
        / f"seed_{seed}"
    )


def validate_inputs() -> None:
    if len(WORLD_CUP_TEAMS) != 48:
        raise AssertionError(
            "Expected 48 unique World Cup teams."
        )

    missing_paths = [
        repository_path(strength)
        for strength in ADJUSTMENT_STRENGTHS
        if not repository_path(
            strength
        ).exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing Study 100B repositories:\n"
            + "\n".join(
                str(path)
                for path in missing_paths
            )
        )

    schemas: list[list[str]] = []
    populations: list[tuple[str, ...]] = []

    for strength in ADJUSTMENT_STRENGTHS:
        repository = (
            pd.read_csv(
                repository_path(
                    strength
                ),
                low_memory=False,
            )
            .sort_values("nation")
            .reset_index(drop=True)
        )

        if len(repository) != 48:
            raise AssertionError(
                "Unexpected repository size at "
                f"strength={strength:.3f}: "
                f"{len(repository)}."
            )

        schemas.append(
            list(repository.columns)
        )

        populations.append(
            tuple(
                repository[
                    "nation"
                ].astype(str)
            )
        )

    if any(
        schema != schemas[0]
        for schema in schemas[1:]
    ):
        raise AssertionError(
            "Repository schemas differ."
        )

    if any(
        population != populations[0]
        for population in populations[1:]
    ):
        raise AssertionError(
            "Repository populations differ."
        )


def complete_probability_frame(
    *,
    result: dict[str, Any],
    stage: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        result[stage]
    ).copy()

    required_columns = {
        "team",
        "count",
        "probability",
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise AssertionError(
            f"{stage!r} output missing columns: "
            f"{sorted(missing)}"
        )

    frame["team"] = (
        frame["team"]
        .astype(str)
        .map(normalize_team_name)
    )

    if frame[
        "team"
    ].duplicated().any():
        raise AssertionError(
            f"{stage!r} output contains duplicate teams."
        )

    unexpected = sorted(
        set(frame["team"])
        - set(WORLD_CUP_TEAMS)
    )

    if unexpected:
        raise AssertionError(
            f"{stage!r} output contains unexpected teams: "
            f"{unexpected}"
        )

    completed = (
        frame
        .set_index("team")
        .reindex(WORLD_CUP_TEAMS)
    )

    completed["count"] = (
        pd.to_numeric(
            completed["count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    completed["probability"] = (
        pd.to_numeric(
            completed["probability"],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

    expected_probability = (
        completed["count"]
        / SIMULATION_COUNT_PER_SEED
    )

    if not np.allclose(
        completed[
            "probability"
        ].to_numpy(float),
        expected_probability.to_numpy(float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            f"{stage!r} probabilities do not match "
            "count / simulation count."
        )

    return completed


def run_all_replays() -> dict[
    tuple[float, int],
    dict[str, Any],
]:
    results: dict[
        tuple[float, int],
        dict[str, Any],
    ] = {}

    for seed in SEEDS:
        for strength in ADJUSTMENT_STRENGTHS:
            print()
            print(
                "Running replay: "
                f"strength={strength:.3f}, "
                f"seed={seed}, "
                f"n={SIMULATION_COUNT_PER_SEED}"
            )

            result = (
                run_monte_carlo_repository(
                    repository_path=(
                        repository_path(
                            strength
                        )
                    ),
                    n=(
                        SIMULATION_COUNT_PER_SEED
                    ),
                    seed=seed,
                )
            )

            results[
                (
                    strength,
                    seed,
                )
            ] = result

            write_outputs(
                result,
                output_dir=str(
                    replay_output_directory(
                        strength=strength,
                        seed=seed,
                    )
                ),
            )

    return results


def build_seed_level_deltas(
    results: dict[
        tuple[float, int],
        dict[str, Any],
    ],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        baseline_result = results[
            (
                0.0,
                seed,
            )
        ]

        baseline_frames = {
            stage: (
                complete_probability_frame(
                    result=baseline_result,
                    stage=stage,
                )
            )
            for stage in STAGE_KEYS
        }

        for strength in ADJUSTMENT_STRENGTHS:
            candidate_result = results[
                (
                    strength,
                    seed,
                )
            ]

            for stage in STAGE_KEYS:
                baseline = baseline_frames[
                    stage
                ]

                candidate = (
                    complete_probability_frame(
                        result=candidate_result,
                        stage=stage,
                    )
                )

                for team in WORLD_CUP_TEAMS:
                    baseline_count = int(
                        baseline.loc[
                            team,
                            "count",
                        ]
                    )

                    candidate_count = int(
                        candidate.loc[
                            team,
                            "count",
                        ]
                    )

                    baseline_probability = float(
                        baseline.loc[
                            team,
                            "probability",
                        ]
                    )

                    candidate_probability = float(
                        candidate.loc[
                            team,
                            "probability",
                        ]
                    )

                    delta = (
                        candidate_probability
                        - baseline_probability
                    )

                    rows.append(
                        {
                            "seed": seed,
                            "team": team,
                            "stage": stage,
                            "adjustment_strength":
                                strength,
                            "baseline_count":
                                baseline_count,
                            "candidate_count":
                                candidate_count,
                            "count_delta": (
                                candidate_count
                                - baseline_count
                            ),
                            "baseline_probability":
                                baseline_probability,
                            "candidate_probability":
                                candidate_probability,
                            "probability_delta":
                                delta,
                            "absolute_probability_delta":
                                abs(delta),
                        }
                    )

    return pd.DataFrame(rows)


def sign_value(
    value: float,
    *,
    tolerance: float = 1e-12,
) -> int:
    if value > tolerance:
        return 1

    if value < -tolerance:
        return -1

    return 0


def build_team_stability_summary(
    seed_deltas: pd.DataFrame,
) -> pd.DataFrame:
    nonzero = seed_deltas.loc[
        seed_deltas[
            "adjustment_strength"
        ].gt(0.0)
    ].copy()

    rows: list[dict[str, Any]] = []

    for (
        team,
        stage,
        strength,
    ), group in nonzero.groupby(
        [
            "team",
            "stage",
            "adjustment_strength",
        ],
        sort=True,
    ):
        values = group[
            "probability_delta"
        ].to_numpy(float)

        signs = [
            sign_value(value)
            for value in values
        ]

        positive_seed_count = sum(
            sign > 0
            for sign in signs
        )

        negative_seed_count = sum(
            sign < 0
            for sign in signs
        )

        zero_seed_count = sum(
            sign == 0
            for sign in signs
        )

        dominant_sign_count = max(
            positive_seed_count,
            negative_seed_count,
            zero_seed_count,
        )

        rows.append(
            {
                "team": team,
                "stage": stage,
                "adjustment_strength":
                    strength,
                "seed_count": len(values),
                "mean_probability_delta":
                    float(
                        values.mean()
                    ),
                "median_probability_delta":
                    float(
                        np.median(values)
                    ),
                "standard_deviation":
                    float(
                        values.std(
                            ddof=1
                        )
                    ),
                "minimum_probability_delta":
                    float(
                        values.min()
                    ),
                "maximum_probability_delta":
                    float(
                        values.max()
                    ),
                "mean_absolute_probability_delta":
                    float(
                        np.abs(values).mean()
                    ),
                "positive_seed_count":
                    int(
                        positive_seed_count
                    ),
                "negative_seed_count":
                    int(
                        negative_seed_count
                    ),
                "zero_seed_count":
                    int(
                        zero_seed_count
                    ),
                "dominant_direction_fraction":
                    float(
                        dominant_sign_count
                        / len(values)
                    ),
                "all_nonzero_seeds_same_direction":
                    bool(
                        (
                            positive_seed_count > 0
                            and negative_seed_count == 0
                        )
                        or (
                            negative_seed_count > 0
                            and positive_seed_count == 0
                        )
                    ),
            }
        )

    return pd.DataFrame(rows)


def build_strength_stability_summary(
    seed_deltas: pd.DataFrame,
) -> pd.DataFrame:
    nonzero = seed_deltas.loc[
        seed_deltas[
            "adjustment_strength"
        ].gt(0.0)
    ]

    rows: list[dict[str, Any]] = []

    for (
        strength,
        stage,
    ), group in nonzero.groupby(
        [
            "adjustment_strength",
            "stage",
        ],
        sort=True,
    ):
        values = group[
            "probability_delta"
        ].to_numpy(float)

        seed_totals = (
            group
            .groupby(
                "seed"
            )[
                "absolute_probability_delta"
            ]
            .sum()
        )

        rows.append(
            {
                "adjustment_strength":
                    strength,
                "stage": stage,
                "observation_count":
                    int(len(group)),
                "mean_signed_probability_delta":
                    float(
                        values.mean()
                    ),
                "mean_absolute_probability_delta":
                    float(
                        np.abs(values).mean()
                    ),
                "maximum_absolute_probability_delta":
                    float(
                        np.abs(values).max()
                    ),
                "mean_seed_total_absolute_delta":
                    float(
                        seed_totals.mean()
                    ),
                "standard_deviation_seed_total_absolute_delta":
                    float(
                        seed_totals.std(
                            ddof=1
                        )
                    ),
                "minimum_seed_total_absolute_delta":
                    float(
                        seed_totals.min()
                    ),
                "maximum_seed_total_absolute_delta":
                    float(
                        seed_totals.max()
                    ),
            }
        )

    return pd.DataFrame(rows)


def build_directional_consistency_summary(
    team_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (
        strength,
        stage,
    ), group in team_summary.groupby(
        [
            "adjustment_strength",
            "stage",
        ],
        sort=True,
    ):
        rows.append(
            {
                "adjustment_strength":
                    strength,
                "stage": stage,
                "team_count":
                    int(len(group)),
                "all_nonzero_same_direction_count":
                    int(
                        group[
                            "all_nonzero_seeds_same_direction"
                        ].sum()
                    ),
                "dominant_direction_at_least_80pct_count":
                    int(
                        group[
                            "dominant_direction_fraction"
                        ].ge(0.8).sum()
                    ),
                "mean_dominant_direction_fraction":
                    float(
                        group[
                            "dominant_direction_fraction"
                        ].mean()
                    ),
                "median_dominant_direction_fraction":
                    float(
                        group[
                            "dominant_direction_fraction"
                        ].median()
                    ),
            }
        )

    return pd.DataFrame(rows)


def validate_outputs(
    *,
    seed_deltas: pd.DataFrame,
    team_summary: pd.DataFrame,
) -> None:
    expected_rows = (
        len(SEEDS)
        * len(ADJUSTMENT_STRENGTHS)
        * len(STAGE_KEYS)
        * len(WORLD_CUP_TEAMS)
    )

    if len(seed_deltas) != expected_rows:
        raise AssertionError(
            "Unexpected seed-delta row count: "
            f"{len(seed_deltas)} vs "
            f"{expected_rows}."
        )

    zero = seed_deltas.loc[
        seed_deltas[
            "adjustment_strength"
        ].eq(0.0)
    ]

    if not zero[
        "count_delta"
    ].eq(0).all():
        raise AssertionError(
            "Zero-strength seed-level counts differ."
        )

    if not np.allclose(
        zero[
            "probability_delta"
        ].to_numpy(float),
        0.0,
        atol=0.0,
        rtol=0.0,
    ):
        raise AssertionError(
            "Zero-strength seed-level probabilities differ."
        )

    numeric_columns = (
        "baseline_probability",
        "candidate_probability",
        "probability_delta",
        "absolute_probability_delta",
    )

    for column in numeric_columns:
        values = seed_deltas[
            column
        ].to_numpy(float)

        if not np.isfinite(
            values
        ).all():
            raise AssertionError(
                f"{column} contains non-finite values."
            )

    if team_summary.empty:
        raise AssertionError(
            "Team stability summary is empty."
        )

    if not (
        team_summary[
            "dominant_direction_fraction"
        ].between(
            0.0,
            1.0,
            inclusive="both",
        )
        .all()
    ):
        raise AssertionError(
            "Directional consistency lies outside [0, 1]."
        )


def build_metadata(
    *,
    seed_deltas: pd.DataFrame,
    team_summary: pd.DataFrame,
) -> dict[str, Any]:
    nonzero = seed_deltas.loc[
        seed_deltas[
            "adjustment_strength"
        ].gt(0.0)
    ]

    champion_summary = team_summary.loc[
        team_summary[
            "stage"
        ].eq(
            "champion"
        )
    ]

    return {
        "study_id": "101A",
        "study_name": (
            "Multi-Seed World Cup Tournament Stability"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "adjustment_strengths": list(
            ADJUSTMENT_STRENGTHS
        ),
        "seeds": list(SEEDS),
        "seed_count": len(SEEDS),
        "simulation_count_per_seed":
            SIMULATION_COUNT_PER_SEED,
        "simulation_count_per_strength": (
            len(SEEDS)
            * SIMULATION_COUNT_PER_SEED
        ),
        "total_tournament_simulations": (
            len(SEEDS)
            * len(ADJUSTMENT_STRENGTHS)
            * SIMULATION_COUNT_PER_SEED
        ),
        "team_count": len(
            WORLD_CUP_TEAMS
        ),
        "stage_count": len(
            STAGE_KEYS
        ),
        "maximum_absolute_probability_delta":
            float(
                nonzero[
                    "absolute_probability_delta"
                ].max()
            ),
        "mean_champion_directional_consistency":
            float(
                champion_summary[
                    "dominant_direction_fraction"
                ].mean()
            ),
        "champion_teams_consistent_in_all_nonzero_seeds":
            int(
                champion_summary[
                    "all_nonzero_seeds_same_direction"
                ].sum()
            ),
        "paired_within_seed": True,
        "zero_strength_parity": True,
        "preferred_strength_selected": False,
        "predictive_superiority_established": False,
        "production_configuration_changed": False,
        "production_repository_replaced": False,
        "interpretation_boundary": (
            "This study estimates Monte Carlo stability and "
            "directional reproducibility. It does not establish "
            "predictive validity or select a production strength."
        ),
    }


def write_report(
    *,
    metadata: dict[str, Any],
    team_summary: pd.DataFrame,
    strength_summary: pd.DataFrame,
    directional_summary: pd.DataFrame,
) -> None:
    champion_summary = (
        team_summary.loc[
            team_summary[
                "stage"
            ].eq(
                "champion"
            )
        ]
        .sort_values(
            [
                "adjustment_strength",
                "dominant_direction_fraction",
                "mean_absolute_probability_delta",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .groupby(
            "adjustment_strength",
            group_keys=False,
        )
        .head(15)
    )

    report = f"""# Study 101A — Multi-Seed World Cup Tournament Stability

## Status

**PASS**

## Purpose

Estimate whether contextual tournament effects are reproducible across
independent random seeds.

## Experimental design

- Strengths: {metadata["adjustment_strengths"]}
- Seeds: {metadata["seeds"]}
- Simulations per seed and strength:
  {metadata["simulation_count_per_seed"]}
- Simulations per strength:
  {metadata["simulation_count_per_strength"]}
- Total tournament simulations:
  {metadata["total_tournament_simulations"]}

## Validation

- Paired comparisons within each seed: PASS
- Zero-strength parity within every seed: PASS
- Complete 48-team output population: PASS
- Finite probability outputs: PASS
- Production configuration changed: NO
- Preferred strength selected: NO
- Predictive superiority established: NO

## Strength-level stability

{strength_summary.to_markdown(index=False)}

## Directional consistency

{directional_summary.to_markdown(index=False)}

## Most directionally stable champion effects

{champion_summary.to_markdown(index=False)}

## Interpretation boundary

This study measures simulation stability. It does not determine whether
the contextual adjustment is empirically correct, nor does it promote a
strength to production.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 101A — MULTI-SEED WORLD CUP "
        "TOURNAMENT STABILITY"
    )
    print("=" * 88)

    validate_inputs()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = run_all_replays()

    seed_deltas = (
        build_seed_level_deltas(
            results
        )
    )

    team_summary = (
        build_team_stability_summary(
            seed_deltas
        )
    )

    strength_summary = (
        build_strength_stability_summary(
            seed_deltas
        )
    )

    directional_summary = (
        build_directional_consistency_summary(
            team_summary
        )
    )

    validate_outputs(
        seed_deltas=seed_deltas,
        team_summary=team_summary,
    )

    seed_deltas.to_csv(
        SEED_LEVEL_DELTAS_PATH,
        index=False,
    )

    team_summary.to_csv(
        TEAM_STABILITY_SUMMARY_PATH,
        index=False,
    )

    strength_summary.to_csv(
        STRENGTH_STABILITY_SUMMARY_PATH,
        index=False,
    )

    directional_summary.to_csv(
        DIRECTIONAL_CONSISTENCY_PATH,
        index=False,
    )

    metadata = build_metadata(
        seed_deltas=seed_deltas,
        team_summary=team_summary,
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
        team_summary=team_summary,
        strength_summary=strength_summary,
        directional_summary=(
            directional_summary
        ),
    )

    print()
    print("Stability summary")
    print("-" * 88)
    print(
        f"  Seeds: "
        f"{metadata['seed_count']}"
    )
    print(
        f"  Simulations per seed: "
        f"{SIMULATION_COUNT_PER_SEED}"
    )
    print(
        f"  Simulations per strength: "
        f"{metadata['simulation_count_per_strength']}"
    )
    print(
        f"  Total simulations: "
        f"{metadata['total_tournament_simulations']}"
    )
    print(
        "  Paired comparisons within seed: PASS"
    )
    print(
        "  Zero-strength parity: PASS"
    )
    print(
        "  Complete output populations: PASS"
    )
    print(
        "  Finite probability outputs: PASS"
    )
    print(
        "  Preferred strength selected: NO"
    )
    print(
        "  Predictive superiority established: NO"
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
#run_contextual_world_cup_sensitivity

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
    / "study_100_context_realization"
    / "study_100c"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_100c_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_100C_REPORT.md"
)

PROBABILITY_DELTAS_PATH = (
    OUTPUT_DIRECTORY
    / "tournament_probability_deltas.csv"
)

TEAM_STAGE_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "team_stage_sensitivity_summary.csv"
)

STRENGTH_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "strength_tournament_summary.csv"
)

STATISTICS_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "simulation_statistics_comparison.csv"
)


ADJUSTMENT_STRENGTHS = (
    0.000,
    0.025,
    0.050,
    0.075,
)

SIMULATION_COUNT = 10000
SIMULATION_SEED = 42

STAGE_KEYS = (
    "champion",
    "runner_up",
    "semifinal",
    "quarterfinal",
    "round_of_16",
)

STAGE_ORDER = {
    "round_of_16": 1,
    "quarterfinal": 2,
    "semifinal": 3,
    "runner_up": 4,
    "champion": 5,
}

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
    strength: float,
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / (
            "replay_strength_"
            f"{strength_label(strength)}"
        )
    )


def validate_input_repositories() -> None:
    missing = [
        repository_path(strength)
        for strength in ADJUSTMENT_STRENGTHS
        if not repository_path(
            strength
        ).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Study 100B repositories are missing:\n"
            + "\n".join(
                str(path)
                for path in missing
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
                "Unexpected repository population at "
                f"strength={strength:.3f}: "
                f"{len(repository)} teams."
            )

        if repository[
            "nation"
        ].duplicated().any():
            raise AssertionError(
                "Duplicate team names at "
                f"strength={strength:.3f}."
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
            "Input repository schemas differ."
        )

    if any(
        population != populations[0]
        for population in populations[1:]
    ):
        raise AssertionError(
            "Input repository team populations differ."
        )


def run_all_replays() -> dict[
    float,
    dict[str, Any],
]:
    results: dict[
        float,
        dict[str, Any],
    ] = {}

    for strength in ADJUSTMENT_STRENGTHS:
        path = repository_path(
            strength
        )

        print()
        print(
            "Running contextual World Cup replay: "
            f"strength={strength:.3f}, "
            f"simulations={SIMULATION_COUNT}, "
            f"seed={SIMULATION_SEED}"
        )

        result = (
            run_monte_carlo_repository(
                repository_path=path,
                n=SIMULATION_COUNT,
                seed=SIMULATION_SEED,
            )
        )

        results[strength] = result

        write_outputs(
            result,
            output_dir=str(
                replay_output_directory(
                    strength
                )
            ),
        )

    return results


def probability_frame(
    result: dict[str, Any],
    stage: str,
) -> pd.DataFrame:
    """
    Return a complete 48-team stage table.

    Monte Carlo outputs may omit teams with zero occurrences,
    especially during small smoke runs. Missing stage records
    therefore represent zero counts and zero probability rather
    than a different tournament population.
    """

    frame = pd.DataFrame(
        result[stage]
    )

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
            f"{stage!r} output is missing columns: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()

    frame["team"] = (
        frame["team"]
        .astype(str)
        .map(normalize_team_name)
    )

    if frame["team"].duplicated().any():
        duplicates = (
            frame.loc[
                frame["team"].duplicated(
                    keep=False
                ),
                "team",
            ]
            .unique()
            .tolist()
        )

        raise AssertionError(
            f"{stage!r} output contains duplicate teams: "
            f"{duplicates}"
        )

    unexpected_teams = sorted(
        set(frame["team"])
        - set(WORLD_CUP_TEAMS)
    )

    if unexpected_teams:
        raise AssertionError(
            f"{stage!r} output contains unexpected teams: "
            f"{unexpected_teams}"
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
        / SIMULATION_COUNT
    )

    if not np.allclose(
        completed["probability"].to_numpy(float),
        expected_probability.to_numpy(float),
        atol=1e-12,
        rtol=0.0,   
    ):
        raise AssertionError(
            f"{stage!r} probabilities do not match "
            "count / simulation_count."
        )
    return (
        completed
        .reset_index()
        .rename(
            columns={
                "index": "team",
            }
        )
    )


def build_probability_deltas(
    results: dict[
        float,
        dict[str, Any],
    ],
) -> pd.DataFrame:
    baseline_frames = {
        stage: (
            probability_frame(
                results[0.0],
                stage,
            )
            .set_index("team")
        )
        for stage in STAGE_KEYS
    }

    rows: list[dict[str, Any]] = []

    for strength in ADJUSTMENT_STRENGTHS:
        for stage in STAGE_KEYS:
            baseline = (
                baseline_frames[
                    stage
                ]
            )

            candidate = (
                probability_frame(
                    results[strength],
                    stage,
                )
                .set_index("team")
            )

            if tuple(baseline.index) != tuple(
                candidate.index
            ):
                raise AssertionError(
                    "Completed tournament output population differs for "
                    f"stage={stage!r}, "
                    f"strength={strength:.3f}."
                )

            for team in baseline.index:
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

                rows.append(
                    {
                        "team": team,
                        "adjustment_strength":
                            strength,
                        "strength_label":
                            strength_label(
                                strength
                            ),
                        "stage": stage,
                        "stage_order":
                            STAGE_ORDER[stage],
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
                        "probability_delta": (
                            candidate_probability
                            - baseline_probability
                        ),
                        "absolute_probability_delta":
                            abs(
                                candidate_probability
                                - baseline_probability
                            ),
                    }
                )

    return pd.DataFrame(rows)


def build_team_stage_summary(
    probability_deltas: pd.DataFrame,
) -> pd.DataFrame:
    nonzero = probability_deltas.loc[
        probability_deltas[
            "adjustment_strength"
        ].gt(0.0)
    ].copy()

    rows: list[dict[str, Any]] = []

    for (
        team,
        strength,
    ), group in nonzero.groupby(
        [
            "team",
            "adjustment_strength",
        ],
        sort=True,
    ):
        by_stage = group.set_index(
            "stage"
        )

        rows.append(
            {
                "team": team,
                "adjustment_strength":
                    strength,
                "champion_probability_delta":
                    float(
                        by_stage.loc[
                            "champion",
                            "probability_delta",
                        ]
                    ),
                "runner_up_probability_delta":
                    float(
                        by_stage.loc[
                            "runner_up",
                            "probability_delta",
                        ]
                    ),
                "semifinal_probability_delta":
                    float(
                        by_stage.loc[
                            "semifinal",
                            "probability_delta",
                        ]
                    ),
                "quarterfinal_probability_delta":
                    float(
                        by_stage.loc[
                            "quarterfinal",
                            "probability_delta",
                        ]
                    ),
                "round_of_16_probability_delta":
                    float(
                        by_stage.loc[
                            "round_of_16",
                            "probability_delta",
                        ]
                    ),
                "maximum_absolute_stage_delta":
                    float(
                        group[
                            "absolute_probability_delta"
                        ].max()
                    ),
                "mean_absolute_stage_delta":
                    float(
                        group[
                            "absolute_probability_delta"
                        ].mean()
                    ),
                "net_stage_probability_delta":
                    float(
                        group[
                            "probability_delta"
                        ].sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


def build_strength_summary(
    probability_deltas: pd.DataFrame,
) -> pd.DataFrame:
    nonzero = probability_deltas.loc[
        probability_deltas[
            "adjustment_strength"
        ].gt(0.0)
    ]

    rows: list[dict[str, Any]] = []

    for strength, group in nonzero.groupby(
        "adjustment_strength",
        sort=True,
    ):
        champion = group.loc[
            group["stage"].eq(
                "champion"
            )
        ]

        rows.append(
            {
                "adjustment_strength":
                    strength,
                "team_count":
                    int(
                        group["team"].nunique()
                    ),
                "changed_team_stage_count":
                    int(
                        group[
                            "count_delta"
                        ].ne(0).sum()
                    ),
                "changed_champion_team_count":
                    int(
                        champion[
                            "count_delta"
                        ].ne(0).sum()
                    ),
                "maximum_absolute_probability_delta":
                    float(
                        group[
                            "absolute_probability_delta"
                        ].max()
                    ),
                "mean_absolute_probability_delta":
                    float(
                        group[
                            "absolute_probability_delta"
                        ].mean()
                    ),
                "maximum_absolute_champion_delta":
                    float(
                        champion[
                            "absolute_probability_delta"
                        ].max()
                    ),
                "mean_absolute_champion_delta":
                    float(
                        champion[
                            "absolute_probability_delta"
                        ].mean()
                    ),
                "total_absolute_champion_delta":
                    float(
                        champion[
                            "absolute_probability_delta"
                        ].sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


def flatten_statistics(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            flattened.update(
                flatten_statistics(
                    child,
                    prefix=child_prefix,
                )
            )

        return flattened

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        for index, child in enumerate(
            value
        ):
            child_prefix = (
                f"{prefix}[{index}]"
            )

            flattened.update(
                flatten_statistics(
                    child,
                    prefix=child_prefix,
                )
            )

        return flattened

    flattened[prefix] = value
    return flattened


def build_statistics_comparison(
    results: dict[
        float,
        dict[str, Any],
    ],
) -> pd.DataFrame:
    baseline = flatten_statistics(
        results[0.0][
            "statistics_summary"
        ]
    )

    rows: list[dict[str, Any]] = []

    for strength in ADJUSTMENT_STRENGTHS:
        candidate = flatten_statistics(
            results[strength][
                "statistics_summary"
            ]
        )

        keys = sorted(
            set(baseline)
            | set(candidate)
        )

        for key in keys:
            baseline_value = baseline.get(
                key
            )

            candidate_value = candidate.get(
                key
            )

            numeric = (
                isinstance(
                    baseline_value,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    ),
                )
                and isinstance(
                    candidate_value,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    ),
                )
            )

            if numeric:
                delta = (
                    float(candidate_value)
                    - float(baseline_value)
                )
            else:
                delta = None

            rows.append(
                {
                    "adjustment_strength":
                        strength,
                    "statistic": key,
                    "baseline_value":
                        baseline_value,
                    "candidate_value":
                        candidate_value,
                    "numeric_delta":
                        delta,
                    "matched": (
                        baseline_value
                        == candidate_value
                    ),
                }
            )

    return pd.DataFrame(rows)


def validate_probability_outputs(
    probability_deltas: pd.DataFrame,
) -> None:
    expected_rows = (
        48
        * len(STAGE_KEYS)
        * len(ADJUSTMENT_STRENGTHS)
    )

    if len(
        probability_deltas
    ) != expected_rows:
        raise AssertionError(
            "Unexpected probability-delta row count: "
            f"{len(probability_deltas)} vs "
            f"{expected_rows}."
        )

    zero = probability_deltas.loc[
        probability_deltas[
            "adjustment_strength"
        ].eq(0.0)
    ]

    if not zero[
        "count_delta"
    ].eq(0).all():
        raise AssertionError(
            "Zero-strength tournament counts differ "
            "from their own baseline."
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
            "Zero-strength tournament probabilities "
            "differ from their own baseline."
        )

    probability_columns = (
        "baseline_probability",
        "candidate_probability",
    )

    for column in probability_columns:
        values = probability_deltas[
            column
        ].to_numpy(float)

        if not np.isfinite(values).all():
            raise AssertionError(
                f"{column} contains non-finite values."
            )

        if (
            (values < 0.0)
            | (values > 1.0)
        ).any():
            raise AssertionError(
                f"{column} contains values outside [0, 1]."
            )

    for (
        strength,
        stage,
    ), group in probability_deltas.groupby(
        [
            "adjustment_strength",
            "stage",
        ],
        sort=True,
    ):
        candidate_total = float(
            group[
                "candidate_count"
            ].sum()
        )

        if stage in {
            "champion",
            "runner_up",
        }:
            expected_total = (
                SIMULATION_COUNT
            )

        elif stage == "semifinal":
            expected_total = (
                SIMULATION_COUNT
                * 4
            )

        elif stage == "quarterfinal":
            expected_total = (
                SIMULATION_COUNT
                * 8
            )

        elif stage == "round_of_16":
            expected_total = (
                SIMULATION_COUNT
                * 16
            )

        else:
            raise AssertionError(
                f"Unknown stage: {stage}"
            )

        if candidate_total != expected_total:
            raise AssertionError(
                "Tournament count conservation failed. "
                f"Strength={strength:.3f}, "
                f"stage={stage!r}, "
                f"observed={candidate_total}, "
                f"expected={expected_total}."
            )


def build_metadata(
    *,
    probability_deltas: pd.DataFrame,
    strength_summary: pd.DataFrame,
) -> dict[str, Any]:
    nonzero = probability_deltas.loc[
        probability_deltas[
            "adjustment_strength"
        ].gt(0.0)
    ]

    changed_team_stage_count = int(
        nonzero[
            "count_delta"
        ].ne(0).sum()
    )

    changed_teams = int(
        nonzero.loc[
            nonzero[
                "count_delta"
            ].ne(0),
            "team",
        ].nunique()
    )

    return {
        "study_id": "100C",
        "study_name": (
            "Context-Aware World Cup Tournament Sensitivity"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "adjustment_strengths": list(
            ADJUSTMENT_STRENGTHS
        ),
        "simulation_count_per_strength":
            SIMULATION_COUNT,
        "simulation_seed":
            SIMULATION_SEED,
        "team_count": 48,
        "stage_count": len(
            STAGE_KEYS
        ),
        "changed_team_stage_count":
            changed_team_stage_count,
        "changed_team_count":
            changed_teams,
        "maximum_absolute_probability_delta":
            float(
                nonzero[
                    "absolute_probability_delta"
                ].max()
            ),
        "maximum_absolute_champion_delta":
            float(
                strength_summary[
                    "maximum_absolute_champion_delta"
                ].max()
            ),
        "paired_seed_used": True,
        "zero_strength_parity": True,
        "tournament_count_conservation":
            True,
        "contextual_adjustment_applied":
            True,
        "match_engine_modified":
            False,
        "tournament_simulator_modified":
            False,
        "production_repository_replaced":
            False,
        "production_configuration_changed":
            False,
        "preferred_strength_selected":
            False,
        "interpretation_boundary": (
            "This study measures tournament sensitivity to "
            "predeclared context-realization strengths. It does "
            "not establish predictive superiority or select a "
            "production parameter."
        ),
    }


def write_report(
    *,
    metadata: dict[str, Any],
    probability_deltas: pd.DataFrame,
    team_summary: pd.DataFrame,
    strength_summary: pd.DataFrame,
) -> None:
    champion_deltas = (
        probability_deltas.loc[
            (
                probability_deltas[
                    "stage"
                ].eq(
                    "champion"
                )
                & probability_deltas[
                    "adjustment_strength"
                ].gt(0.0)
            )
        ]
        .sort_values(
            [
                "adjustment_strength",
                "absolute_probability_delta",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "adjustment_strength",
            group_keys=False,
        )
        .head(10)
    )

    largest_stage_effects = (
        team_summary
        .sort_values(
            [
                "adjustment_strength",
                "maximum_absolute_stage_delta",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "adjustment_strength",
            group_keys=False,
        )
        .head(10)
    )

    report = f"""# Study 100C — Context-Aware World Cup Tournament Sensitivity

## Status

**PASS**

## Purpose

Measure how predeclared context-realization strengths affect World Cup
tournament probabilities while holding the simulation architecture and
random seed fixed.

## Experimental design

- Strengths: {metadata["adjustment_strengths"]}
- Simulations per strength:
  {metadata["simulation_count_per_strength"]}
- Seed: {metadata["simulation_seed"]}
- Tournament teams: {metadata["team_count"]}
- Stages compared: {metadata["stage_count"]}

## Validation

- Paired random seed: PASS
- Zero-strength replay parity: PASS
- Tournament count conservation: PASS
- Finite probability outputs: PASS
- Match engine modified: NO
- Tournament simulator modified: NO
- Production repository replaced: NO
- Preferred strength selected: NO

## Strength-level sensitivity

{strength_summary.to_markdown(index=False)}

## Largest champion-probability changes

{champion_deltas.to_markdown(index=False)}

## Largest stage-level team effects

{largest_stage_effects.to_markdown(index=False)}

## Interpretation boundary

This study evaluates tournament sensitivity, not predictive validity.
A larger or smaller tournament effect does not imply a better football
model. No contextual strength is promoted to production by this study.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 100C — CONTEXT-AWARE WORLD CUP "
        "TOURNAMENT SENSITIVITY"
    )
    print("=" * 88)

    validate_input_repositories()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = run_all_replays()

    probability_deltas = (
        build_probability_deltas(
            results
        )
    )

    team_summary = (
        build_team_stage_summary(
            probability_deltas
        )
    )

    strength_summary = (
        build_strength_summary(
            probability_deltas
        )
    )

    statistics_comparison = (
        build_statistics_comparison(
            results
        )
    )

    validate_probability_outputs(
        probability_deltas
    )

    probability_deltas.to_csv(
        PROBABILITY_DELTAS_PATH,
        index=False,
    )

    team_summary.to_csv(
        TEAM_STAGE_SUMMARY_PATH,
        index=False,
    )

    strength_summary.to_csv(
        STRENGTH_SUMMARY_PATH,
        index=False,
    )

    statistics_comparison.to_csv(
        STATISTICS_COMPARISON_PATH,
        index=False,
    )

    metadata = build_metadata(
        probability_deltas=(
            probability_deltas
        ),
        strength_summary=(
            strength_summary
        ),
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
        probability_deltas=(
            probability_deltas
        ),
        team_summary=team_summary,
        strength_summary=(
            strength_summary
        ),
    )

    print()
    print("Tournament sensitivity summary")
    print("-" * 88)
    print(
        f"  Tournament teams: "
        f"{metadata['team_count']}"
    )
    print(
        f"  Simulations per strength: "
        f"{SIMULATION_COUNT}"
    )
    print(
        f"  Seed: "
        f"{SIMULATION_SEED}"
    )
    print(
        f"  Teams with changed outcomes: "
        f"{metadata['changed_team_count']}"
    )
    print(
        "  Paired seed used: PASS"
    )
    print(
        "  Zero-strength parity: PASS"
    )
    print(
        "  Tournament count conservation: PASS"
    )
    print(
        "  Probability outputs finite: PASS"
    )
    print(
        "  Contextual adjustment applied: YES"
    )
    print(
        "  Match engine modified: NO"
    )
    print(
        "  Tournament simulator modified: NO"
    )
    print(
        "  Production repository replaced: NO"
    )
    print(
        "  Preferred strength selected: NO"
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
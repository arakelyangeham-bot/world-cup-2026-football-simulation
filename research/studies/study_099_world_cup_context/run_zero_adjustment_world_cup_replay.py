#run_zero_adjustment_world_cup_replay

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_player_intelligence_team_repository import (
    build_repository_dataframe,
    create_default_roster_builder,
    resolve_representation_builder,
)
from scripts.monte_carlo_driver import (
    run_monte_carlo_repository,
    write_outputs,
)
from shared.national_team_priors import (
    load_fifa_points,
)

from scripts.wc2026_data import GROUPS
from shared.team_name_normalizer import (
    normalize_team_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_099_world_cup_context"
    / "study_099b"
)

LEGACY_REPOSITORY_PATH = (
    OUTPUT_DIRECTORY
    / "expected_xi_legacy_repository.csv"
)

CONTRIBUTION_REPOSITORY_PATH = (
    OUTPUT_DIRECTORY
    / "expected_xi_contribution_zero_repository.csv"
)

REPOSITORY_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "repository_comparison.csv"
)

TOURNAMENT_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "tournament_output_comparison.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_099b_metadata.json"
)

SIMULATION_COUNT = 10000
SIMULATION_SEED = 42

SIMULATION_RELEVANT_COLUMNS = (
    "nation",
    "att_composite",
    "mid_composite",
    "def_composite",
    "gk_composite",
    "poisson_attack_adj",
    "poisson_defense_adj",
    "fifa_points",
)


PROBABILITY_KEYS = (
    "champion",
    "runner_up",
    "semifinal",
    "quarterfinal",
    "round_of_16",
)

WORLD_CUP_TEAMS = {
    normalize_team_name(team)
    for teams in GROUPS.values()
    for team in teams
}

EXPECTED_GROUP_COUNT = 12
EXPECTED_TEAM_COUNT = 48


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
            "Unexpected World Cup roster count: "
            f"{raw_team_count} vs "
            f"{EXPECTED_TEAM_COUNT}."
        )

    if len(WORLD_CUP_TEAMS) != EXPECTED_TEAM_COUNT:
        raise AssertionError(
            "World Cup team names are not unique after "
            "normalization. "
            f"Unique={len(WORLD_CUP_TEAMS)}, "
            f"expected={EXPECTED_TEAM_COUNT}."
        )

def build_repositories() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    roster_builder = (
        create_default_roster_builder()
    )

    fifa_lookup = load_fifa_points()

    legacy = build_repository_dataframe(
        roster_builder=roster_builder,
        representation_builder=(
            resolve_representation_builder(
                "expected_xi_legacy"
            )
        ),
        fifa_lookup=fifa_lookup,
        included_teams=WORLD_CUP_TEAMS,
    )

    contribution = build_repository_dataframe(
        roster_builder=roster_builder,
        representation_builder=(
            resolve_representation_builder(
                "expected_xi_contribution_zero"
            )
        ),
        fifa_lookup=fifa_lookup,
        included_teams=WORLD_CUP_TEAMS,
    )

    return legacy, contribution

def compare_repositories(
    legacy: pd.DataFrame,
    contribution: pd.DataFrame,
) -> pd.DataFrame:
    legacy = (
        legacy
        .sort_values("nation")
        .reset_index(drop=True)
    )

    contribution = (
        contribution
        .sort_values("nation")
        .reset_index(drop=True)
    )

    if not legacy["nation"].equals(
        contribution["nation"]
    ):
        raise AssertionError(
            "Repository team populations differ."
        )

    rows: list[dict[str, object]] = []

    for column in SIMULATION_RELEVANT_COLUMNS:
        if column == "nation":
            continue

        a = pd.to_numeric(
            legacy[column],
            errors="coerce",
        ).to_numpy(float)

        b = pd.to_numeric(
            contribution[column],
            errors="coerce",
        ).to_numpy(float)

        difference = np.abs(
            a - b
        )

        finite_difference = difference[
            np.isfinite(difference)
        ]

        maximum_difference = (
            float(finite_difference.max())
            if finite_difference.size
            else 0.0
        )

        rows.append(
            {
                "column": column,
                "matched": bool(
                    np.allclose(
                        a,
                        b,
                        equal_nan=True,
                        atol=1e-12,
                        rtol=0.0,
                    )
                ),
                "maximum_absolute_difference":
                    maximum_difference,
            }
        )

    return pd.DataFrame(rows)

def write_repository(
    repository: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    repository.to_csv(
        path,
        index=False,
    )

def validate_results(
    *,
    legacy_repository: pd.DataFrame,
    contribution_repository: pd.DataFrame,
    repository_comparison: pd.DataFrame,
    tournament_comparison: pd.DataFrame,
) -> None:
    if legacy_repository.empty:
        raise AssertionError(
            "Legacy repository is empty."
        )

    if contribution_repository.empty:
        raise AssertionError(
            "Contribution repository is empty."
        )

    expected_teams = set(
        WORLD_CUP_TEAMS
    )

    legacy_teams = set(
        legacy_repository["nation"]
        .astype(str)
    )

    contribution_teams = set(
        contribution_repository["nation"]
        .astype(str)
    )

    if legacy_teams != expected_teams:
        missing = sorted(
            expected_teams - legacy_teams
        )

        extra = sorted(
            legacy_teams - expected_teams
        )

        raise AssertionError(
            "Legacy World Cup repository population "
            "does not match the expected tournament teams. "
            f"Missing={missing}, extra={extra}"
        )

    if contribution_teams != expected_teams:
        missing = sorted(
            expected_teams - contribution_teams
        )

        extra = sorted(
            contribution_teams - expected_teams
        )

        raise AssertionError(
            "Contribution World Cup repository population "
            "does not match the expected tournament teams. "
            f"Missing={missing}, extra={extra}"
        )

    if not repository_comparison[
        "matched"
    ].all():
        failures = repository_comparison.loc[
            ~repository_comparison["matched"]
        ]

        raise AssertionError(
            "Zero-adjustment repositories differ:\n"
            + failures.to_string(
                index=False
            )
        )

    if not tournament_comparison[
        "team_population_matched"
    ].fillna(True).all():
        raise AssertionError(
            "Paired tournament output populations differ."
        )

    if not tournament_comparison[
        "counts_matched"
    ].fillna(True).all():
        failures = tournament_comparison.loc[
            ~tournament_comparison[
                "counts_matched"
            ].fillna(True)
        ]

        raise AssertionError(
            "Paired tournament counts differ:\n"
            + failures.to_string(
                index=False
            )
        )

    if not tournament_comparison[
        "probabilities_matched"
    ].fillna(True).all():
        failures = tournament_comparison.loc[
            ~tournament_comparison[
                "probabilities_matched"
            ].fillna(True)
        ]

        raise AssertionError(
            "Paired tournament probabilities differ:\n"
            + failures.to_string(
                index=False
            )
        )

def build_metadata(
    *,
    legacy_repository: pd.DataFrame,
    repository_comparison: pd.DataFrame,
    tournament_comparison: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "study_id": "099B",
        "study_name": (
            "Zero-Adjustment World Cup Replay"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "simulation_count": SIMULATION_COUNT,
        "simulation_seed": SIMULATION_SEED,
        "team_count": len(
            legacy_repository
        ),
        "repository_columns_compared": len(
            repository_comparison
        ),
        "repository_values_identical": bool(
            repository_comparison[
                "matched"
            ].all()
        ),
        "tournament_outputs_identical": bool(
            tournament_comparison[
                "counts_matched"
            ].fillna(True).all()
            and tournament_comparison[
                "probabilities_matched"
            ].fillna(True).all()
        ),
        "contextual_adjustment_applied": False,
        "match_engine_modified": False,
        "tournament_simulator_modified": False,
        "production_repository_replaced": False,
        "production_configuration_changed": False,
    }

def probability_frame(
    rows: list[dict],
) -> pd.DataFrame:
    return (
        pd.DataFrame(rows)
        .sort_values("team")
        .reset_index(drop=True)
    )

def compare_tournament_outputs(
    legacy_results: dict,
    contribution_results: dict,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for output_name in PROBABILITY_KEYS:
        legacy = probability_frame(
            legacy_results[output_name]
        )

        contribution = probability_frame(
            contribution_results[
                output_name
            ]
        )

        same_teams = legacy[
            "team"
        ].equals(
            contribution["team"]
        )

        same_counts = (
            same_teams
            and legacy[
                "count"
            ].equals(
                contribution["count"]
            )
        )

        same_probabilities = (
            same_teams
            and np.allclose(
                legacy["probability"],
                contribution["probability"],
                atol=0.0,
                rtol=0.0,
            )
        )

        rows.append(
            {
                "output": output_name,
                "team_population_matched":
                    same_teams,
                "counts_matched":
                    same_counts,
                "probabilities_matched":
                    same_probabilities,
            }
        )

    legacy_summary = (
        legacy_results[
            "statistics_summary"
        ]
    )

    contribution_summary = (
        contribution_results[
            "statistics_summary"
        ]
    )

    rows.append(
        {
            "output": "statistics_summary",
            "team_population_matched":
                None,
            "counts_matched": bool(
                legacy_summary
                == contribution_summary
            ),
            "probabilities_matched": bool(
                legacy_summary
                == contribution_summary
            ),
        }
    )

    return pd.DataFrame(rows)

def main() -> None:
    validate_world_cup_configuration()
    print("=" * 88)
    print(
        "STUDY 099B — ZERO-ADJUSTMENT "
        "WORLD CUP REPLAY"
    )
    print("=" * 88)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Building paired repositories...")

    (
        legacy_repository,
        contribution_repository,
    ) = build_repositories()

    write_repository(
        legacy_repository,
        LEGACY_REPOSITORY_PATH,
    )

    write_repository(
        contribution_repository,
        CONTRIBUTION_REPOSITORY_PATH,
    )

    repository_comparison = (
        compare_repositories(
            legacy_repository,
            contribution_repository,
        )
    )

    repository_comparison.to_csv(
        REPOSITORY_COMPARISON_PATH,
        index=False,
    )

    if not repository_comparison[
        "matched"
    ].all():
        failures = repository_comparison.loc[
            ~repository_comparison["matched"]
        ]

        raise AssertionError(
            "Repository parity failed before simulation:\n"
            + failures.to_string(
                index=False
            )
        )

    print(
        "Repository simulation fields matched: PASS"
    )

    print()
    print(
        f"Running legacy replay "
        f"({SIMULATION_COUNT} tournaments)..."
    )

    legacy_results = (
        run_monte_carlo_repository(
            repository_path=(
                LEGACY_REPOSITORY_PATH
            ),
            n=SIMULATION_COUNT,
            seed=SIMULATION_SEED,
        )
    )

    print()
    print(
        f"Running contribution replay "
        f"({SIMULATION_COUNT} tournaments)..."
    )

    contribution_results = (
        run_monte_carlo_repository(
            repository_path=(
                CONTRIBUTION_REPOSITORY_PATH
            ),
            n=SIMULATION_COUNT,
            seed=SIMULATION_SEED,
        )
    )

    tournament_comparison = (
        compare_tournament_outputs(
            legacy_results,
            contribution_results,
        )
    )

    tournament_comparison.to_csv(
        TOURNAMENT_COMPARISON_PATH,
        index=False,
    )

    validate_results(
        legacy_repository=(
            legacy_repository
        ),
        contribution_repository=(
            contribution_repository
        ),
        repository_comparison=(
            repository_comparison
        ),
        tournament_comparison=(
            tournament_comparison
        ),
    )

    write_outputs(
        legacy_results,
        output_dir=str(
            OUTPUT_DIRECTORY
            / "legacy_replay"
        ),
    )

    write_outputs(
        contribution_results,
        output_dir=str(
            OUTPUT_DIRECTORY
            / "contribution_zero_replay"
        ),
    )

    metadata = build_metadata(
        legacy_repository=(
            legacy_repository
        ),
        repository_comparison=(
            repository_comparison
        ),
        tournament_comparison=(
            tournament_comparison
        ),
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Replay summary")
    print("-" * 88)
    print(
        f"  Teams: "
        f"{metadata['team_count']}"
    )
    print(
        f"  Simulations per path: "
        f"{SIMULATION_COUNT}"
    )
    print(
        f"  Seed: "
        f"{SIMULATION_SEED}"
    )
    print(
        "  Repository team population matched: PASS"
    )
    print(
        "  Repository simulation fields matched: PASS"
    )

    for row in tournament_comparison.itertuples(
        index=False
    ):
        label = str(
            row.output
        ).replace(
            "_",
            " ",
        ).title()

        print(
            f"  {label} matched: PASS"
        )

    print(
        "  Contextual adjustment applied: NO"
    )
    print(
        "  Match engine modified: NO"
    )
    print(
        "  Tournament simulator modified: NO"
    )
    print(
        "  Production configuration changed: NO"
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
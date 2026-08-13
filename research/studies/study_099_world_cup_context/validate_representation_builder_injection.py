#validate_representation_builder_injection

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.player_intelligence.team_representation_builder import (
    build_team_representation_from_squad,
)
from research.player_intelligence.team_repository_builder import (
    project_representation_to_repository_entry,
)
from scripts.build_player_intelligence_team_repository import (
    REPOSITORY_COLUMNS,
    build_repository_dataframe,
    create_default_roster_builder,
    resolve_representation_builder,
)
from shared.national_team_priors import load_fifa_points
from shared.team_name_normalizer import normalize_team_name


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_099_world_cup_context"
    / "study_099a"
)

CONTROL_PATH = (
    OUTPUT_DIRECTORY
    / "original_hardcoded_control_repository.csv"
)

INJECTED_PATH = (
    OUTPUT_DIRECTORY
    / "injected_default_repository.csv"
)

COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "repository_parity_comparison.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_099a_metadata.json"
)


def build_original_hardcoded_control(
    *,
    roster_builder,
    fifa_lookup: dict[str, float],
) -> pd.DataFrame:
    """
    Reproduce the pre-refactor implementation using current inputs.

    This intentionally hardcodes
    build_team_representation_from_squad().
    """

    rows: list[dict[str, object]] = []

    for team in roster_builder.list_teams():
        canonical_team = normalize_team_name(
            team
        )

        squad = roster_builder.get_squad(
            team
        )

        if not squad.players:
            continue

        representation = (
            build_team_representation_from_squad(
                squad
            )
        )

        entry = (
            project_representation_to_repository_entry(
                representation=representation,
                fifa_points=None,
            )
        )

        fifa_points = fifa_lookup.get(
            canonical_team
        )

        if fifa_points is None or pd.isna(
            fifa_points
        ):
            continue

        rows.append(
            {
                "nation": canonical_team,
                "att_composite": entry.attack,
                "mid_composite": entry.midfield,
                "def_composite": entry.defense,
                "gk_composite": entry.gk,
                "poisson_attack_adj":
                    entry.poisson_attack,
                "poisson_defense_adj":
                    entry.poisson_defense,
                "representation_type":
                    entry.representation_type,
                "aggregation_profile":
                    entry.aggregation_profile,
                "player_count":
                    entry.player_count,
                "available_player_count":
                    entry.available_player_count,
                "squad_quality":
                    representation.squad_quality,
                "evidence_score":
                    representation.evidence_score,
                "attack_depth":
                    representation.attack_depth,
                "midfield_depth":
                    representation.midfield_depth,
                "defense_depth":
                    representation.defense_depth,
                "fifa_points": float(
                    fifa_points
                ),
            }
        )

    return (
        pd.DataFrame(
            rows,
            columns=REPOSITORY_COLUMNS,
        )
        .sort_values("nation")
        .reset_index(drop=True)
    )


def build_parity_comparison(
    control: pd.DataFrame,
    injected: pd.DataFrame,
) -> pd.DataFrame:
    if list(control.columns) != list(
        injected.columns
    ):
        raise AssertionError(
            "Control and injected schemas differ."
        )

    if not control[
        "nation"
    ].equals(
        injected["nation"]
    ):
        raise AssertionError(
            "Control and injected team populations differ."
        )

    rows: list[dict[str, object]] = []

    for column in control.columns:
        if column == "nation":
            continue

        if (
            pd.api.types.is_numeric_dtype(
                control[column]
            )
            and pd.api.types.is_numeric_dtype(
                injected[column]
            )
        ):
            control_values = pd.to_numeric(
                control[column],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            injected_values = pd.to_numeric(
                injected[column],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            differences = np.abs(
                control_values
                - injected_values
            )

            finite_differences = differences[
                np.isfinite(
                    differences
                )
            ]

            maximum_difference = (
                float(
                    finite_differences.max()
                )
                if finite_differences.size
                else 0.0
            )

            matched = bool(
                np.allclose(
                    control_values,
                    injected_values,
                    equal_nan=True,
                    atol=1e-12,
                    rtol=0.0,
                )
            )

            comparison_type = "numeric"

        else:
            matched = bool(
                control[column]
                .fillna("<missing>")
                .astype(str)
                .equals(
                    injected[column]
                    .fillna("<missing>")
                    .astype(str)
                )
            )

            maximum_difference = None
            comparison_type = "text"

        rows.append(
            {
                "column": column,
                "comparison_type":
                    comparison_type,
                "matched": matched,
                "maximum_absolute_difference":
                    maximum_difference,
            }
        )

    return pd.DataFrame(rows)


def validate_parity(
    *,
    control: pd.DataFrame,
    injected: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    if control.empty or injected.empty:
        raise AssertionError(
            "Parity repositories must not be empty."
        )

    if len(control) != len(injected):
        raise AssertionError(
            "Repository row counts differ."
        )

    if not comparison[
        "matched"
    ].all():
        failures = comparison.loc[
            ~comparison["matched"]
        ]

        raise AssertionError(
            "Representation-builder injection changed "
            "repository output:\n"
            + failures.to_string(
                index=False
            )
        )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 099A — REPRESENTATION-BUILDER "
        "INJECTION PARITY"
    )
    print("=" * 88)

    roster_builder = (
        create_default_roster_builder()
    )

    fifa_lookup = load_fifa_points()

    control = (
        build_original_hardcoded_control(
            roster_builder=roster_builder,
            fifa_lookup=fifa_lookup,
        )
    )

    injected = (
        build_repository_dataframe(
            roster_builder=roster_builder,
            representation_builder=(
                resolve_representation_builder(
                    "full_squad_legacy"
                )
            ),
            fifa_lookup=fifa_lookup,
        )
    )

    comparison = build_parity_comparison(
        control=control,
        injected=injected,
    )

    validate_parity(
        control=control,
        injected=injected,
        comparison=comparison,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    control.to_csv(
        CONTROL_PATH,
        index=False,
    )

    injected.to_csv(
        INJECTED_PATH,
        index=False,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    metadata = {
        "study_id": "099A",
        "study_name": (
            "Representation-Builder Injection Parity"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "team_count": len(control),
        "column_count": len(
            control.columns
        ),
        "all_columns_matched": bool(
            comparison["matched"].all()
        ),
        "maximum_numeric_difference": float(
            comparison[
                "maximum_absolute_difference"
            ]
            .dropna()
            .max()
        ),
        "representation_policy":
            "full_squad_legacy",
        "production_repository_replaced":
            False,
        "simulation_runtime_modified":
            False,
        "production_behavior_changed":
            False,
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Parity summary")
    print("-" * 88)
    print(
        f"  Teams: {metadata['team_count']}"
    )
    print(
        f"  Columns: {metadata['column_count']}"
    )
    print(
        "  Team population matched: PASS"
    )
    print(
        "  Repository schema matched: PASS"
    )
    print(
        "  Repository values matched: PASS"
    )
    print(
        "  Representation-builder injection: PASS"
    )
    print(
        "  Production repository replaced: NO"
    )
    print(
        "  Simulation runtime modified: NO"
    )
    print(
        "  Production behavior changed: NO"
    )

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
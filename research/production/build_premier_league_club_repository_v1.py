#build_premier_league_club_repository_v1

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared.team_name_normalizer import (
    normalize_team_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_OBSERVATIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_060_clubelo_enriched_observations"
    / "full_squad_observations_with_clubelo.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_071a_premier_league_club_repository_v1"
)

REPOSITORY_PATH = (
    OUTPUT_DIRECTORY
    / "premier_league_club_repository_v1.csv"
)

REPRESENTATION_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "club_representation_selection_audit.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "study_report.md"
)


REPRESENTATION_FIELDS = (
    "attack",
    "midfield",
    "defense",
    "goalkeeper",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
    "squad_quality",
    "evidence_score",
)


HOME_COLUMN_MAP = {
    "home_team": "club",
    "home_attack": "attack",
    "home_midfield": "midfield",
    "home_defense": "defense",
    "home_goalkeeper": "goalkeeper",
    "home_attack_depth": "attack_depth",
    "home_midfield_depth": "midfield_depth",
    "home_defense_depth": "defense_depth",
    "home_squad_quality": "squad_quality",
    "home_evidence_score": "evidence_score",
    "home_representation_type": "representation_type",
    "home_representation_source": "representation_source",
    "home_representation_season_id":
        "representation_season_id",
    "home_representation_player_count":
        "representation_player_count",
    "home_representation_available_player_count":
        "available_player_count",
}


AWAY_COLUMN_MAP = {
    "away_team": "club",
    "away_attack": "attack",
    "away_midfield": "midfield",
    "away_defense": "defense",
    "away_goalkeeper": "goalkeeper",
    "away_attack_depth": "attack_depth",
    "away_midfield_depth": "midfield_depth",
    "away_defense_depth": "defense_depth",
    "away_squad_quality": "squad_quality",
    "away_evidence_score": "evidence_score",
    "away_representation_type": "representation_type",
    "away_representation_source": "representation_source",
    "away_representation_season_id":
        "representation_season_id",
    "away_representation_player_count":
        "representation_player_count",
    "away_representation_available_player_count":
        "available_player_count",
}


def load_observations() -> pd.DataFrame:
    if not SOURCE_OBSERVATIONS_PATH.exists():
        raise FileNotFoundError(
            "Study 060 observations do not exist: "
            f"{SOURCE_OBSERVATIONS_PATH}"
        )

    dataframe = pd.read_csv(
        SOURCE_OBSERVATIONS_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Study 060 observations are empty."
        )

    required_columns = {
        "event_id",
        "date",
        *HOME_COLUMN_MAP,
        *AWAY_COLUMN_MAP,
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Study 060 observations are missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    return dataframe


def build_team_appearance_rows(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    shared_columns = [
        "event_id",
        "date",
    ]

    home = observations[
        shared_columns + list(HOME_COLUMN_MAP)
    ].rename(
        columns=HOME_COLUMN_MAP
    )

    home["match_role"] = "home"

    away = observations[
        shared_columns + list(AWAY_COLUMN_MAP)
    ].rename(
        columns=AWAY_COLUMN_MAP
    )

    away["match_role"] = "away"

    appearances = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    appearances["club"] = (
        appearances["club"]
        .astype(str)
        .map(normalize_team_name)
    )

    if appearances["club"].eq("").any():
        raise ValueError(
            "One or more club names became empty after "
            "normalization."
        )

    numeric_columns = [
        *REPRESENTATION_FIELDS,
        "representation_player_count",
        "available_player_count",
    ]

    for column in numeric_columns:
        appearances[column] = pd.to_numeric(
            appearances[column],
            errors="raise",
        )

    if appearances[numeric_columns].isna().any().any():
        raise ValueError(
            "Team appearances contain missing required "
            "numeric representation values."
        )

    if not np.isfinite(
        appearances[numeric_columns].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Team appearances contain non-finite values."
        )

    return (
        appearances
        .sort_values(
            [
                "club",
                "date",
                "event_id",
                "match_role",
            ]
        )
        .reset_index(drop=True)
    )


def validate_representation_consistency(
    appearances: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for club, group in appearances.groupby(
        "club",
        sort=True,
    ):
        unique_counts = {
            field: int(group[field].nunique())
            for field in REPRESENTATION_FIELDS
        }

        maximum_unique_count = max(
            unique_counts.values()
        )

        records.append(
            {
                "club": club,
                "appearance_count": int(len(group)),
                "first_appearance_date":
                    group["date"].min().date().isoformat(),
                "last_appearance_date":
                    group["date"].max().date().isoformat(),
                "representation_season_count": int(
                    group[
                        "representation_season_id"
                    ].nunique()
                ),
                "maximum_field_unique_count":
                    maximum_unique_count,
                "representation_constant_within_source":
                    maximum_unique_count == 1,
                **{
                    f"{field}_unique_count": count
                    for field, count
                    in unique_counts.items()
                },
            }
        )

    return pd.DataFrame(records)


def select_production_representations(
    appearances: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the most recent validated representation for
    every club.

    The source observation dataset may contain repeated
    appearances of the same season-level representation.
    The final chronological appearance is selected
    deterministically.
    """

    latest_rows = (
        appearances
        .sort_values(
            [
                "club",
                "date",
                "event_id",
                "match_role",
            ]
        )
        .groupby(
            "club",
            as_index=False,
            sort=True,
        )
        .tail(1)
        .copy()
    )

    repository_columns = [
        "club",
        *REPRESENTATION_FIELDS,
        "representation_type",
        "representation_source",
        "representation_season_id",
        "representation_player_count",
        "available_player_count",
        "date",
        "event_id",
    ]

    repository = latest_rows[
        repository_columns
    ].rename(
        columns={
            "date": "representation_selected_from_date",
            "event_id":
                "representation_selected_from_event_id",
        }
    )

    repository[
        "repository_version"
    ] = "1.0"

    repository[
        "repository_scope"
    ] = "premier_league"

    repository[
        "representation_selected_from_date"
    ] = (
        repository[
            "representation_selected_from_date"
        ]
        .dt.strftime("%Y-%m-%d")
    )

    repository = (
        repository
        .sort_values("club")
        .reset_index(drop=True)
    )

    if repository["club"].duplicated().any():
        raise AssertionError(
            "Production repository contains duplicate "
            "clubs."
        )

    return repository


def validate_repository(
    repository: pd.DataFrame,
) -> None:
    required_columns = {
        "club",
        *REPRESENTATION_FIELDS,
        "repository_version",
        "repository_scope",
    }

    missing = required_columns - set(repository.columns)

    if missing:
        raise AssertionError(
            "Built repository is missing columns: "
            f"{sorted(missing)}"
        )

    if repository.empty:
        raise AssertionError(
            "Built repository is empty."
        )

    if repository["club"].duplicated().any():
        raise AssertionError(
            "Built repository contains duplicate clubs."
        )

    if repository[
        list(REPRESENTATION_FIELDS)
    ].isna().any().any():
        raise AssertionError(
            "Built repository contains missing football "
            "intelligence values."
        )

    if not np.isfinite(
        repository[
            list(REPRESENTATION_FIELDS)
        ].to_numpy(dtype=float)
    ).all():
        raise AssertionError(
            "Built repository contains non-finite football "
            "intelligence values."
        )


def build_metadata(
    observations: pd.DataFrame,
    appearances: pd.DataFrame,
    repository: pd.DataFrame,
    audit: pd.DataFrame,
) -> dict[str, object]:
    return {
        "study_id": "071A",
        "study_name": (
            "Premier League Production Club Repository v1"
        ),
        "source_observations": str(
            SOURCE_OBSERVATIONS_PATH
        ),
        "repository_path": str(
            REPOSITORY_PATH
        ),
        "source_match_count": int(
            len(observations)
        ),
        "team_appearance_count": int(
            len(appearances)
        ),
        "club_count": int(
            len(repository)
        ),
        "repository_version": "1.0",
        "repository_scope": "premier_league",
        "representation_fields": list(
            REPRESENTATION_FIELDS
        ),
        "constant_representation_club_count": int(
            audit[
                "representation_constant_within_source"
            ].sum()
        ),
        "varying_representation_club_count": int(
            (
                ~audit[
                    "representation_constant_within_source"
                ]
            ).sum()
        ),
        "unique_club_contract_pass": True,
        "finite_value_contract_pass": True,
        "latest_representation_selection_pass": True,
        "overall_result": "PASS",
    }


def write_report(
    metadata: dict[str, object],
) -> None:
    report = f"""# Study 071A — Premier League Production Club Repository v1

## Purpose

Promote the validated full-squad club representations used by
the Integrated Club Goal Model research programme into a clean
runtime repository.

## Scope

- Competition scope: Premier League
- Repository version: 1.0
- Source matches: {metadata["source_match_count"]}
- Team appearances: {metadata["team_appearance_count"]}
- Clubs: {metadata["club_count"]}

## Source

`{metadata["source_observations"]}`

## Output

`{metadata["repository_path"]}`

## Football-intelligence fields

{chr(10).join(
    f"- `{field}`"
    for field in metadata["representation_fields"]
)}

## Selection policy

For each club, the most recent validated full-squad
representation in the Study 060 observation population is
selected deterministically.

This repository does not store the production rating prior.
Prediction-date rating priors are resolved separately through
the temporally valid ClubElo repository.

## Representation consistency

- Clubs with constant representations:
  {metadata["constant_representation_club_count"]}
- Clubs with variation in at least one field:
  {metadata["varying_representation_club_count"]}

Variation is permitted because the deterministic policy selects
the most recent validated representation.

## Validation

- Study 060 source loading: PASS
- Home/away representation projection: PASS
- Club-name normalization: PASS
- Required numeric values: PASS
- Finite representation values: PASS
- Most-recent representation selection: PASS
- Unique-club contract: PASS
- Repository schema contract: PASS

## Result

**OVERALL RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    observations = load_observations()

    appearances = build_team_appearance_rows(
        observations
    )

    audit = validate_representation_consistency(
        appearances
    )

    repository = select_production_representations(
        appearances
    )

    validate_repository(repository)

    metadata = build_metadata(
        observations=observations,
        appearances=appearances,
        repository=repository,
        audit=audit,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    repository.to_csv(
        REPOSITORY_PATH,
        index=False,
    )

    audit.to_csv(
        REPRESENTATION_AUDIT_PATH,
        index=False,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(metadata)

    print(
        "Study 071A — Premier League Production "
        "Club Repository v1"
    )
    print("=" * 76)
    print()
    print(
        f"Source matches: {len(observations)}"
    )
    print(
        f"Team appearances: {len(appearances)}"
    )
    print(
        f"Clubs: {len(repository)}"
    )
    print(
        "Constant-representation clubs: "
        f"{metadata['constant_representation_club_count']}"
    )
    print(
        "Varying-representation clubs: "
        f"{metadata['varying_representation_club_count']}"
    )
    print()
    print("Source loading: PASS")
    print("Home/away projection: PASS")
    print("Club-name normalization: PASS")
    print("Finite representation values: PASS")
    print("Latest representation selection: PASS")
    print("Unique-club contract: PASS")
    print("Repository schema contract: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Repository written to: {REPOSITORY_PATH}"
    )


if __name__ == "__main__":
    main()
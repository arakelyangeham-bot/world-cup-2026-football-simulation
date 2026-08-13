#build_robust_observation_candidate

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.football_observatory.build_observation_dataset import (
    build_observation_dataset,
    load_historical_matches,
)
from research.player_intelligence.competition_player_repository import (
    CompetitionPlayerRepository,
)
from research.player_intelligence.competition_roster_builder import (
    CompetitionRosterBuilder,
)
from research.player_intelligence.competition_team_repository import (
    CompetitionTeamRepository,
)
from research.player_intelligence.player_repository import (
    PlayerRepository,
)
from research.player_intelligence.previous_season_competition_representation_provider import (
    PreviousSeasonCompetitionRepresentationProvider,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MATCH_INPUT_PATH = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "historical_matches"
    / "premier_league"
    / "premier_league_2024_completed_matches.csv"
)

ROBUST_PLAYER_RATINGS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "study_092c1"
    / "player_ratings_robust_zscore.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_096_premier_league_robust_candidate"
    / "study_096a_observations"
)

FULL_SQUAD_OBSERVATION_PATH = (
    OUTPUT_DIRECTORY
    / "full_squad_observations_robust_zscore.csv"
)

FULL_SQUAD_EXCLUSION_PATH = (
    OUTPUT_DIRECTORY
    / "full_squad_exclusions_robust_zscore.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_096a_metadata.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "STUDY_096A_RESULTS.md"
)

FROZEN_GLOBAL_OBSERVATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_048_club_observation_dataset"
    / "full_squad_observations.csv"
)

def build_representation_provider(
) -> PreviousSeasonCompetitionRepresentationProvider:
    """
    Assemble the existing competition representation pipeline
    with an explicitly injected robust player-ratings artifact.

    No production defaults are modified.
    """

    player_repository = PlayerRepository(
        player_features_path=(
            ROBUST_PLAYER_RATINGS_PATH
        )
    )

    competition_player_repository = (
        CompetitionPlayerRepository(
            player_repository=(
                player_repository
            )
        )
    )

    roster_builder = CompetitionRosterBuilder(
        repository=(
            competition_player_repository
        )
    )

    team_repository = CompetitionTeamRepository(
        roster_builder=roster_builder
    )

    return (
        PreviousSeasonCompetitionRepresentationProvider(
            team_repository=team_repository,
            representation_type="full_squad",
        )
    )

def validate_inputs() -> None:
    if not MATCH_INPUT_PATH.exists():
        raise FileNotFoundError(
            "Historical Premier League match input does "
            f"not exist: {MATCH_INPUT_PATH}"
        )

    if not ROBUST_PLAYER_RATINGS_PATH.exists():
        raise FileNotFoundError(
            "Robust player-ratings artifact does not "
            f"exist: {ROBUST_PLAYER_RATINGS_PATH}"
        )

    if MATCH_INPUT_PATH.stat().st_size <= 0:
        raise ValueError(
            "Historical match input is empty."
        )

    if ROBUST_PLAYER_RATINGS_PATH.stat().st_size <= 0:
        raise ValueError(
            "Robust player-ratings artifact is empty."
        )

def validate_observations(
    *,
    observations: pd.DataFrame,
    exclusions: pd.DataFrame,
    historical_match_count: int,
) -> None:
    if observations.empty:
        raise AssertionError(
            "Robust observation dataset is empty."
        )

    if observations[
        "event_id"
    ].duplicated().any():
        raise AssertionError(
            "Robust observations contain duplicate "
            "event IDs."
        )

    if len(observations) + len(exclusions) != (
        historical_match_count
    ):
        raise AssertionError(
            "Accepted and excluded populations do not "
            "reconcile with the historical match input."
        )

    if not observations[
        "home_temporal_validity_pass"
    ].all():
        raise AssertionError(
            "At least one home representation failed "
            "temporal validation."
        )

    if not observations[
        "away_temporal_validity_pass"
    ].all():
        raise AssertionError(
            "At least one away representation failed "
            "temporal validation."
        )

    if not (
        observations[
            "representation_season_start_year"
        ]
        == (
            observations[
                "prediction_season_start_year"
            ]
            - 1
        )
    ).all():
        raise AssertionError(
            "At least one robust observation violates "
            "the previous-season representation rule."
        )

    expected_representation_type = (
        "full_squad"
    )

    if not observations[
        "home_representation_type"
    ].eq(
        expected_representation_type
    ).all():
        raise AssertionError(
            "Unexpected home representation type."
        )

    if not observations[
        "away_representation_type"
    ].eq(
        expected_representation_type
    ).all():
        raise AssertionError(
            "Unexpected away representation type."
        )

    required_numeric_columns = [
        "home_attack",
        "home_midfield",
        "home_defense",
        "home_goalkeeper",
        "home_attack_depth",
        "home_midfield_depth",
        "home_defense_depth",
        "home_squad_quality",
        "home_evidence_score",

        "away_attack",
        "away_midfield",
        "away_defense",
        "away_goalkeeper",
        "away_attack_depth",
        "away_midfield_depth",
        "away_defense_depth",
        "away_squad_quality",
        "away_evidence_score",

        "attack_diff",
        "midfield_diff",
        "defense_diff",
        "goalkeeper_diff",
        "attack_depth_diff",
        "midfield_depth_diff",
        "defense_depth_diff",
        "squad_quality_diff",
        "evidence_score_diff",

        "home_score",
        "away_score",
    ]

    if observations[
        required_numeric_columns
    ].isna().any().any():
        missing_counts = (
            observations[
                required_numeric_columns
            ]
            .isna()
            .sum()
        )

        missing_counts = (
            missing_counts[
                missing_counts > 0
            ]
            .to_dict()
        )

        raise AssertionError(
            "Robust observations contain missing numeric "
            f"values: {missing_counts}"
        )

    if observations[
        "rating_prior_available"
    ].any():
        raise AssertionError(
            "Rating priors should not yet be available "
            "in Study 096A."
        )

    if observations[
        [
            "home_rating_prior",
            "away_rating_prior",
            "rating_prior_diff",
        ]
    ].notna().any().any():
        raise AssertionError(
            "Rating-prior values entered Study 096A "
            "before ClubElo enrichment."
        )

def build_population_comparison(
    robust_observations: pd.DataFrame,
) -> dict[str, object]:
    if not FROZEN_GLOBAL_OBSERVATION_PATH.exists():
        raise FileNotFoundError(
            "Frozen Study 048 full-squad observations "
            f"do not exist: {FROZEN_GLOBAL_OBSERVATION_PATH}"
        )

    frozen = pd.read_csv(
        FROZEN_GLOBAL_OBSERVATION_PATH,
        low_memory=False,
    )

    if frozen.empty:
        raise ValueError(
            "Frozen Study 048 observation dataset is empty."
        )

    frozen_events = set(
        frozen[
            "event_id"
        ].astype(str)
    )

    robust_events = set(
        robust_observations[
            "event_id"
        ].astype(str)
    )

    only_frozen = sorted(
        frozen_events
        - robust_events
    )

    only_robust = sorted(
        robust_events
        - frozen_events
    )

    if only_frozen or only_robust:
        raise AssertionError(
            "Robust and frozen observation event "
            "populations differ. "
            f"Only frozen={only_frozen[:20]}, "
            f"only robust={only_robust[:20]}."
        )

    return {
        "frozen_observation_rows":
            len(frozen),
        "robust_observation_rows":
            len(robust_observations),
        "event_population_equal":
            True,
        "event_count":
            len(robust_events),
    }

def write_report(
    *,
    metadata: dict[str, object],
) -> None:
    report = f"""# Study 096A — Premier League Robust Observation Candidate

## Status

**PASS**

## Purpose

Build an isolated Premier League full-squad observation
dataset using the validated `robust_zscore` player-ratings
artifact.

## Architecture

The candidate is assembled through existing dependency
injection:

`PlayerRepository`
→ `CompetitionPlayerRepository`
→ `CompetitionRosterBuilder`
→ `CompetitionTeamRepository`
→ `PreviousSeasonCompetitionRepresentationProvider`
→ `build_observation_dataset`

No production class or default path was modified.

## Inputs

- Historical matches:
  `{metadata["match_input"]}`
- Player ratings:
  `{metadata["player_ratings_path"]}`

## Population

- Historical matches:
  {metadata["historical_match_count"]}
- Accepted observations:
  {metadata["accepted_observation_count"]}
- Excluded matches:
  {metadata["excluded_match_count"]}
- Accepted event population equal to frozen Study 048:
  {metadata["event_population_equal"]}

## Representation

- Transformation provenance: `robust_zscore`
- Representation type: `full_squad`
- Representation season policy:
  previous competition season
- Rating prior available:
  no

## Validation

- Input artifacts present: PASS
- Explicit robust ratings injection: PASS
- Previous-season temporal validity: PASS
- Required representation values: PASS
- Accepted/excluded population reconciliation: PASS
- Frozen Study 048 event population match: PASS
- Production defaults modified: NO
- Canonical files overwritten: NO

## Result

**OVERALL RESULT: PASS**
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

def main() -> None:
    print("=" * 88)
    print(
        "STUDY 096A — PREMIER LEAGUE ROBUST "
        "OBSERVATION CANDIDATE"
    )
    print("=" * 88)

    validate_inputs()

    matches = load_historical_matches(
        MATCH_INPUT_PATH
    )

    provider = build_representation_provider()

    result = build_observation_dataset(
        matches=matches,
        representation_provider=provider,
    )

    validate_observations(
        observations=result.observations,
        exclusions=result.exclusions,
        historical_match_count=len(matches),
    )

    population_comparison = (
        build_population_comparison(
            result.observations
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    observation_output = (
        result.observations.copy()
    )

    if (
        "date"
        in observation_output.columns
    ):
        observation_output[
            "date"
        ] = (
            pd.to_datetime(
                observation_output[
                    "date"
                ],
                errors="raise",
                utc=True,
            )
            .dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )

    exclusion_output = (
        result.exclusions.copy()
    )

    if (
        not exclusion_output.empty
        and "date"
        in exclusion_output.columns
    ):
        exclusion_output[
            "date"
        ] = (
            pd.to_datetime(
                exclusion_output[
                    "date"
                ],
                errors="raise",
                utc=True,
            )
            .dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )

    observation_output.to_csv(
        FULL_SQUAD_OBSERVATION_PATH,
        index=False,
    )

    exclusion_output.to_csv(
        FULL_SQUAD_EXCLUSION_PATH,
        index=False,
    )

    metadata = {
        "study_id":
            "096A",
        "study_name": (
            "Premier League Robust Observation Candidate"
        ),
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "status":
            "PASS",

        "match_input":
            str(
                MATCH_INPUT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        "player_ratings_path":
            str(
                ROBUST_PLAYER_RATINGS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        "player_representation_transformation":
            "robust_zscore",
        "representation_type":
            "full_squad",

        "historical_match_count":
            len(matches),
        "accepted_observation_count":
            len(
                result.observations
            ),
        "excluded_match_count":
            len(
                result.exclusions
            ),

        **population_comparison,

        "rating_prior_available":
            False,
        "production_defaults_modified":
            False,
        "canonical_files_overwritten":
            False,
        "dependency_injection_used":
            True,

        "outputs": [
            FULL_SQUAD_OBSERVATION_PATH.name,
            FULL_SQUAD_EXCLUSION_PATH.name,
            METADATA_PATH.name,
            REPORT_PATH.name,
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        metadata=metadata
    )

    print()
    print("Candidate population")
    print("-" * 88)
    print(
        f"  Historical matches: "
        f"{len(matches)}"
    )
    print(
        f"  Accepted observations: "
        f"{len(result.observations)}"
    )
    print(
        f"  Excluded matches: "
        f"{len(result.exclusions)}"
    )
    print(
        "  Frozen Study 048 event population: "
        "MATCH"
    )

    print()
    print("Validation summary")
    print("  Input artifacts present: PASS")
    print("  Robust ratings injection: PASS")
    print("  Observation construction: PASS")
    print("  Previous-season temporal validity: PASS")
    print("  Required representation values: PASS")
    print("  Population reconciliation: PASS")
    print("  Frozen event population comparison: PASS")
    print("  Production defaults modified: NO")
    print("  Canonical files overwritten: NO")

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
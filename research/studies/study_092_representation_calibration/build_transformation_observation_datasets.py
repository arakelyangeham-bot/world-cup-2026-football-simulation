from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.data_pipeline.build_clubelo_enriched_observations import (
    build_enriched_observations,
    build_population_audit,
    normalize_dates,
    remove_rating_prior_placeholders,
    validate_population_audit,
    validate_rating_priors,
    validate_source_observations,
    validate_temporal_provenance,
)
from research.studies.study_083_bundesliga_production_replay.run_bundesliga_production_replay import (
    load_replay_population,
    validate_repository_population,
)
from simulation.live_match_observation_builder import (
    ProductionClubRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STUDY_092C1_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "study_092c1"
)

REPOSITORY_DIRECTORY = (
    STUDY_092C1_DIRECTORY
    / "club_repositories"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "study_092c2"
    / "observation_datasets"
)

RATING_PRIOR_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "study_092c2"
    / "bundesliga_match_rating_priors.csv"
)

TRANSFORMATIONS = (
    "global_zscore",
    "percentile_normal",
    "robust_zscore",
)

REPOSITORY_PATHS = {
    transformation: (
        REPOSITORY_DIRECTORY
        / (
            "bundesliga_club_repository_"
            f"{transformation}.csv"
        )
    )
    for transformation in TRANSFORMATIONS
}

RAW_OBSERVATION_PATHS = {
    transformation: (
        OUTPUT_DIRECTORY
        / (
            "bundesliga_observations_"
            f"{transformation}.csv"
        )
    )
    for transformation in TRANSFORMATIONS
}

ENRICHED_OBSERVATION_PATHS = {
    transformation: (
        OUTPUT_DIRECTORY
        / (
            "bundesliga_observations_"
            f"{transformation}_with_clubelo.csv"
        )
    )
    for transformation in TRANSFORMATIONS
}

POPULATION_AUDIT_PATHS = {
    transformation: (
        OUTPUT_DIRECTORY
        / (
            "clubelo_population_audit_"
            f"{transformation}.csv"
        )
    )
    for transformation in TRANSFORMATIONS
}

TEMPORAL_AUDIT_PATHS = {
    transformation: (
        OUTPUT_DIRECTORY
        / (
            "clubelo_temporal_audit_"
            f"{transformation}.csv"
        )
    )
    for transformation in TRANSFORMATIONS
}

BRANCH_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "transformation_observation_audit.csv"
)

DIFFERENCE_PATH = (
    OUTPUT_DIRECTORY
    / "transformation_observation_differences.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_092c2a_metadata.json"
)


REPOSITORY_VALUE_COLUMNS = (
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

REPRESENTATION_VALUE_COLUMNS = tuple(
    [
        *[
            f"home_{column}"
            for column in REPOSITORY_VALUE_COLUMNS
        ],
        *[
            f"away_{column}"
            for column in REPOSITORY_VALUE_COLUMNS
        ],
        *[
            f"{column}_diff"
            for column in REPOSITORY_VALUE_COLUMNS
        ],
    ]
)

MATCH_AND_TARGET_COLUMNS = (
    "competition_key",
    "season_start_year",
    "event_id",
    "date",
    "stage",
    "round",
    "round_number",
    "home_team",
    "home_team_id",
    "away_team",
    "away_team_id",
    "home_score",
    "away_score",
    "goal_difference",
    "total_goals",
    "outcome",
)


def load_repository_dataframe(
    path: Path,
    *,
    transformation: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Transformation repository does not exist: "
            f"{path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"{transformation}: repository is empty."
        )

    required_columns = {
        "club",
        "representation_type",
        *REPOSITORY_VALUE_COLUMNS,
    }

    missing = required_columns - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"{transformation}: repository is missing "
            f"columns: {sorted(missing)}"
        )

    if dataframe["club"].duplicated().any():
        raise ValueError(
            f"{transformation}: repository contains "
            "duplicate clubs."
        )

    numeric = dataframe[
        list(REPOSITORY_VALUE_COLUMNS)
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if numeric.isna().any().any():
        raise ValueError(
            f"{transformation}: repository contains "
            "missing representation values."
        )

    if not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{transformation}: repository contains "
            "non-finite representation values."
        )

    dataframe = dataframe.copy()

    for column in REPOSITORY_VALUE_COLUMNS:
        dataframe[column] = numeric[column]

    return dataframe


def repository_side_columns(
    repository: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    rename_mapping = {
        "club": f"{prefix}_repository_club",
        "representation_type":
            f"{prefix}_representation_type",
        **{
            column: f"{prefix}_{column}"
            for column in REPOSITORY_VALUE_COLUMNS
        },
    }

    optional_columns = [
        column
        for column in (
            "representation_season_id",
            "repository_version",
            "repository_scope",
            "aggregation_profile",
            "player_count",
            "available_player_count",
        )
        if column in repository.columns
    ]

    rename_mapping.update(
        {
            column: f"{prefix}_{column}"
            for column in optional_columns
        }
    )

    selected_columns = [
        "club",
        "representation_type",
        *REPOSITORY_VALUE_COLUMNS,
        *optional_columns,
    ]

    return repository[
        selected_columns
    ].rename(
        columns=rename_mapping
    )


def build_projected_observations(
    fixtures: pd.DataFrame,
    repository: pd.DataFrame,
    *,
    transformation: str,
) -> pd.DataFrame:
    home_repository = repository_side_columns(
        repository,
        prefix="home",
    )

    away_repository = repository_side_columns(
        repository,
        prefix="away",
    )

    observations = fixtures.merge(
        home_repository,
        left_on="home_team",
        right_on="home_repository_club",
        how="left",
        validate="many_to_one",
    )

    observations = observations.merge(
        away_repository,
        left_on="away_team",
        right_on="away_repository_club",
        how="left",
        validate="many_to_one",
    )

    if len(observations) != len(fixtures):
        raise AssertionError(
            f"{transformation}: repository projection "
            "changed fixture row count."
        )

    if observations["event_id"].duplicated().any():
        raise AssertionError(
            f"{transformation}: repository projection "
            "created duplicate event IDs."
        )

    missing_home = observations[
        "home_repository_club"
    ].isna()

    missing_away = observations[
        "away_repository_club"
    ].isna()

    if missing_home.any() or missing_away.any():
        raise ValueError(
            f"{transformation}: repository projection "
            "failed for one or more fixture clubs. "
            f"Missing home rows={int(missing_home.sum())}; "
            f"missing away rows={int(missing_away.sum())}."
        )

    observations[
        "home_representation_source"
    ] = (
        "static_bundesliga_repository_"
        f"{transformation}"
    )

    observations[
        "away_representation_source"
    ] = (
        "static_bundesliga_repository_"
        f"{transformation}"
    )

    observations[
        "home_temporal_validity_pass"
    ] = True

    observations[
        "away_temporal_validity_pass"
    ] = True

    observations[
        "rating_prior_available"
    ] = False

    observations[
        "rating_prior_source"
    ] = "unavailable"

    observations[
        "home_rating_prior"
    ] = np.nan

    observations[
        "away_rating_prior"
    ] = np.nan

    observations[
        "rating_prior_diff"
    ] = np.nan

    for column in REPOSITORY_VALUE_COLUMNS:
        observations[
            f"{column}_diff"
        ] = (
            observations[
                f"home_{column}"
            ]
            - observations[
                f"away_{column}"
            ]
        )

    observations[
        "result"
    ] = np.select(
        [
            observations[
                "home_score"
            ].gt(
                observations[
                    "away_score"
                ]
            ),
            observations[
                "away_score"
            ].gt(
                observations[
                    "home_score"
                ]
            ),
        ],
        [
            "home_win",
            "away_win",
        ],
        default="draw",
    )

    observations[
        "is_draw"
    ] = observations[
        "home_score"
    ].eq(
        observations[
            "away_score"
        ]
    ).astype(int)

    observations[
        "is_home_win"
    ] = observations[
        "home_score"
    ].gt(
        observations[
            "away_score"
        ]
    ).astype(int)

    observations[
        "is_away_win"
    ] = observations[
        "away_score"
    ].gt(
        observations[
            "home_score"
        ]
    ).astype(int)

    observations[
        "both_teams_scored"
    ] = (
        observations[
            "home_score"
        ].gt(0)
        & observations[
            "away_score"
        ].gt(0)
    ).astype(int)

    observations[
        "is_clean_sheet"
    ] = (
        observations[
            "home_score"
        ].eq(0)
        | observations[
            "away_score"
        ].eq(0)
    ).astype(int)

    observations[
        "is_high_scoring"
    ] = observations[
        "total_goals"
    ].ge(5).astype(int)

    observations[
        "is_blowout"
    ] = observations[
        "goal_difference"
    ].abs().ge(3).astype(int)

    observations[
        "representation_provider"
    ] = (
        "retrospective_static_repository_"
        f"{transformation}"
    )

    observations[
        "prediction_season_start_year"
    ] = observations[
        "season_start_year"
    ]

    observations[
        "representation_season_start_year"
    ] = observations[
        "season_start_year"
    ]

    numeric_values = observations[
        list(
            REPRESENTATION_VALUE_COLUMNS
        )
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise AssertionError(
            f"{transformation}: projected observations "
            "contain non-finite representation values."
        )

    return (
        observations
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )


def load_rating_priors(
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    if not RATING_PRIOR_PATH.exists():
        raise FileNotFoundError(
            "Rating-prior file does not exist: "
            f"{RATING_PRIOR_PATH}"
        )

    priors = pd.read_csv(
        RATING_PRIOR_PATH,
        low_memory=False,
    )

    if priors.empty:
        raise ValueError(
            "Rating-prior dataset is empty."
        )

    required_columns = {
        "event_id",
        "date",
        "home_team",
        "away_team",
        "home_rating_prior",
        "away_rating_prior",
        "rating_prior_diff",
        "rating_prior_source",
        "rating_prior_available",
        "home_rating_effective_from",
        "home_rating_effective_to",
        "away_rating_effective_from",
        "away_rating_effective_to",
    }

    missing = required_columns - set(
        priors.columns
    )

    if missing:
        raise ValueError(
            "Rating-prior dataset is missing columns: "
            f"{sorted(missing)}"
        )

    priors = priors.copy()

    priors["event_id"] = pd.to_numeric(
        priors["event_id"],
        errors="raise",
    )

    fixture_event_ids = set(
        pd.to_numeric(
            fixtures["event_id"],
            errors="raise",
        ).tolist()
    )

    matching_priors = priors.loc[
        priors["event_id"].isin(
            fixture_event_ids
        )
    ].copy()

    if matching_priors[
        "event_id"
    ].duplicated().any():
        duplicates = (
            matching_priors.loc[
                matching_priors[
                    "event_id"
                ].duplicated(
                    keep=False
                ),
                [
                    "event_id",
                    "home_team",
                    "away_team",
                ],
            ]
            .sort_values("event_id")
        )

        raise ValueError(
            "Filtered Bundesliga rating priors contain "
            "duplicate event IDs:\n"
            f"{duplicates.to_string(index=False)}"
        )

    matched_event_ids = set(
        matching_priors[
            "event_id"
        ].tolist()
    )

    missing_event_ids = sorted(
        fixture_event_ids
        - matched_event_ids
    )

    unexpected_event_ids = sorted(
        matched_event_ids
        - fixture_event_ids
    )

    if missing_event_ids:
        raise ValueError(
            "Rating-prior repository does not cover the "
            "complete Bundesliga fixture population. "
            f"Missing count: {len(missing_event_ids)}; "
            f"examples: {missing_event_ids[:20]}"
        )

    if unexpected_event_ids:
        raise AssertionError(
            "Filtered rating priors unexpectedly contain "
            "events outside the Bundesliga population."
        )

    if len(matching_priors) != len(
        fixtures
    ):
        raise AssertionError(
            "Filtered rating-prior row count does not "
            "equal the Bundesliga fixture count. "
            f"Priors={len(matching_priors)}; "
            f"fixtures={len(fixtures)}."
        )

    return (
        matching_priors
        .sort_values("event_id")
        .reset_index(drop=True)
    )


def enrich_with_clubelo(
    observations: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    transformation: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    observations, normalized_priors = (
        normalize_dates(
            observations=observations,
            priors=priors,
        )
    )

    validate_source_observations(
        observations
    )

    observations_without_placeholders = (
        remove_rating_prior_placeholders(
            observations
        )
    )

    validate_rating_priors(
        normalized_priors
    )

    population_audit = (
        build_population_audit(
            observations=(
                observations_without_placeholders
            ),
            priors=normalized_priors,
        )
    )

    validate_population_audit(
        audit=population_audit,
        expected_rows=len(
            observations_without_placeholders
        ),
    )

    enriched = build_enriched_observations(
        observations=(
            observations_without_placeholders
        ),
        priors=normalized_priors,
    )

    temporal_audit = (
        validate_temporal_provenance(
            enriched
        )
    )

    if len(enriched) != len(observations):
        raise AssertionError(
            f"{transformation}: ClubElo enrichment "
            "changed row count."
        )

    return (
        enriched,
        population_audit,
        temporal_audit,
    )


def assert_same_schema(
    frames: dict[str, pd.DataFrame],
    *,
    label: str,
) -> None:
    baseline_columns = list(
        frames[
            "global_zscore"
        ].columns
    )

    for transformation, dataframe in (
        frames.items()
    ):
        if list(
            dataframe.columns
        ) != baseline_columns:
            raise AssertionError(
                f"{label} schema differs for "
                f"{transformation}."
            )


def assert_same_population_and_targets(
    frames: dict[str, pd.DataFrame],
) -> None:
    baseline = (
        frames[
            "global_zscore"
        ]
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )

    for transformation, dataframe in (
        frames.items()
    ):
        candidate = (
            dataframe
            .sort_values(
                [
                    "date",
                    "event_id",
                ]
            )
            .reset_index(drop=True)
        )

        if len(candidate) != len(
            baseline
        ):
            raise AssertionError(
                "Observation row counts differ for "
                f"{transformation}."
            )

        for column in (
            MATCH_AND_TARGET_COLUMNS
        ):
            if column not in baseline.columns:
                continue

            left = baseline[column]
            right = candidate[column]

            if pd.api.types.is_numeric_dtype(
                left
            ):
                equal = np.allclose(
                    pd.to_numeric(
                        left,
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    pd.to_numeric(
                        right,
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    equal_nan=True,
                    atol=0.0,
                    rtol=0.0,
                )
            else:
                equal = (
                    left.fillna("<missing>")
                    .astype(str)
                    .equals(
                        right.fillna("<missing>")
                        .astype(str)
                    )
                )

            if not equal:
                raise AssertionError(
                    "Fixture or target column differs "
                    "across representations. "
                    f"Transformation={transformation}; "
                    f"column={column}."
                )


def build_branch_audit(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    baseline_events = set(
        frames[
            "global_zscore"
        ]["event_id"]
    )

    records: list[
        dict[str, object]
    ] = []

    for transformation, dataframe in (
        frames.items()
    ):
        representation_values = dataframe[
            list(
                REPRESENTATION_VALUE_COLUMNS
            )
        ].to_numpy(
            dtype=float
        )

        records.append(
            {
                "transformation":
                    transformation,
                "row_count":
                    len(dataframe),
                "unique_event_count":
                    dataframe[
                        "event_id"
                    ].nunique(),
                "event_population_match":
                    set(
                        dataframe[
                            "event_id"
                        ]
                    )
                    == baseline_events,
                "unique_home_club_count":
                    dataframe[
                        "home_team"
                    ].nunique(),
                "unique_away_club_count":
                    dataframe[
                        "away_team"
                    ].nunique(),
                "duplicate_event_count":
                    int(
                        dataframe[
                            "event_id"
                        ].duplicated().sum()
                    ),
                "missing_representation_count":
                    int(
                        dataframe[
                            list(
                                REPRESENTATION_VALUE_COLUMNS
                            )
                        ]
                        .isna()
                        .sum()
                        .sum()
                    ),
                "non_finite_representation_count":
                    int(
                        (
                            ~np.isfinite(
                                representation_values
                            )
                        ).sum()
                    ),
                "rating_prior_available_rate":
                    float(
                        dataframe[
                            "rating_prior_available"
                        ].astype(bool).mean()
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


def build_difference_summary(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    baseline = (
        frames[
            "global_zscore"
        ]
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )

    records: list[
        dict[str, object]
    ] = []

    for transformation in (
        "percentile_normal",
        "robust_zscore",
    ):
        candidate = (
            frames[
                transformation
            ]
            .sort_values(
                [
                    "date",
                    "event_id",
                ]
            )
            .reset_index(drop=True)
        )

        for column in (
            REPRESENTATION_VALUE_COLUMNS
        ):
            differences = (
                candidate[
                    column
                ].to_numpy(dtype=float)
                - baseline[
                    column
                ].to_numpy(dtype=float)
            )

            records.append(
                {
                    "candidate_transformation":
                        transformation,
                    "representation_field":
                        column,
                    "match_count":
                        len(differences),
                    "mean_difference":
                        float(
                            differences.mean()
                        ),
                    "mean_absolute_difference":
                        float(
                            np.abs(
                                differences
                            ).mean()
                        ),
                    "maximum_absolute_difference":
                        float(
                            np.abs(
                                differences
                            ).max()
                        ),
                    "changed_match_count":
                        int(
                            (
                                np.abs(
                                    differences
                                )
                                > 1e-12
                            ).sum()
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


def format_dates_for_output(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    if pd.api.types.is_datetime64_any_dtype(
        output["date"]
    ):
        output["date"] = (
            output["date"]
            .dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )

    return output


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 092C2A — REPOSITORY "
        "OBSERVATION PROJECTION"
    )
    print("=" * 88)

    fixtures = load_replay_population()

    priors = load_rating_priors(
        fixtures
    )

    print()
    print(
        "ClubElo prior population selected: "
        f"{len(priors)} events"
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_observations: dict[
        str,
        pd.DataFrame,
    ] = {}

    enriched_observations: dict[
        str,
        pd.DataFrame,
    ] = {}

    for transformation in (
        TRANSFORMATIONS
    ):
        print()
        print(
            f"Projecting {transformation} repository..."
        )

        repository_path = (
            REPOSITORY_PATHS[
                transformation
            ]
        )

        repository_frame = (
            load_repository_dataframe(
                repository_path,
                transformation=transformation,
            )
        )

        runtime_repository = (
            ProductionClubRepository(
                repository_path=(
                    repository_path
                )
            )
        )

        validate_repository_population(
            fixtures,
            runtime_repository.list_clubs(),
        )

        raw = build_projected_observations(
            fixtures,
            repository_frame,
            transformation=transformation,
        )

        (
            enriched,
            population_audit,
            temporal_audit,
        ) = enrich_with_clubelo(
            raw,
            priors,
            transformation=transformation,
        )

        raw_observations[
            transformation
        ] = raw

        enriched_observations[
            transformation
        ] = enriched

        format_dates_for_output(
            raw
        ).to_csv(
            RAW_OBSERVATION_PATHS[
                transformation
            ],
            index=False,
        )

        format_dates_for_output(
            enriched
        ).to_csv(
            ENRICHED_OBSERVATION_PATHS[
                transformation
            ],
            index=False,
        )

        population_audit.to_csv(
            POPULATION_AUDIT_PATHS[
                transformation
            ],
            index=False,
        )

        temporal_audit.to_csv(
            TEMPORAL_AUDIT_PATHS[
                transformation
            ],
            index=False,
        )

        print(
            f"  Fixtures projected: {len(raw)}"
        )

        print(
            "  ClubElo-enriched rows: "
            f"{len(enriched)}"
        )

    assert_same_schema(
        enriched_observations,
        label="Enriched observation datasets",
    )

    assert_same_population_and_targets(
        enriched_observations
    )

    branch_audit = build_branch_audit(
        enriched_observations
    )

    differences = (
        build_difference_summary(
            enriched_observations
        )
    )

    if not branch_audit[
        "event_population_match"
    ].all():
        raise AssertionError(
            "Observation event populations differ."
        )

    if branch_audit[
        "missing_representation_count"
    ].sum() != 0:
        raise AssertionError(
            "Projected observations contain missing "
            "representation values."
        )

    if branch_audit[
        "non_finite_representation_count"
    ].sum() != 0:
        raise AssertionError(
            "Projected observations contain non-finite "
            "representation values."
        )

    if not np.allclose(
        branch_audit[
            "rating_prior_available_rate"
        ].to_numpy(dtype=float),
        np.ones(
            len(branch_audit)
        ),
    ):
        raise AssertionError(
            "ClubElo coverage is incomplete."
        )

    if differences[
        "changed_match_count"
    ].sum() == 0:
        raise AssertionError(
            "Alternative representations do not differ "
            "from the global-z-score control."
        )

    branch_audit.to_csv(
        BRANCH_AUDIT_PATH,
        index=False,
    )

    differences.to_csv(
        DIFFERENCE_PATH,
        index=False,
    )

    metadata = {
        "study_id": "092C2A",
        "study_name": (
            "Repository Observation Projection"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "study_type": (
            "controlled_retrospective_"
            "representation_comparison"
        ),
        "transformations": list(
            TRANSFORMATIONS
        ),
        "fixture_count": len(
            fixtures
        ),
        "club_count": 18,
        "event_population_match": True,
        "fixture_and_target_columns_match": True,
        "observation_schema_match": True,
        "clubelo_enrichment_applied": True,
        "clubelo_temporal_validity_pass": True,
        "alternative_representation_values_detected": True,
        "goal_models_fitted": False,
        "canonical_study_048_outputs_changed": False,
        "canonical_study_060_outputs_changed": False,
        "canonical_study_083_outputs_changed": False,
        "methodological_boundary": (
            "Static season-level Bundesliga repositories "
            "are projected retrospectively onto the 2024-25 "
            "fixture population. Player evidence is not "
            "prediction-date frozen. ClubElo priors remain "
            "fixture-date valid."
        ),
        "outputs": [
            *[
                path.name
                for path in (
                    RAW_OBSERVATION_PATHS.values()
                )
            ],
            *[
                path.name
                for path in (
                    ENRICHED_OBSERVATION_PATHS.values()
                )
            ],
            *[
                path.name
                for path in (
                    POPULATION_AUDIT_PATHS.values()
                )
            ],
            *[
                path.name
                for path in (
                    TEMPORAL_AUDIT_PATHS.values()
                )
            ],
            BRANCH_AUDIT_PATH.name,
            DIFFERENCE_PATH.name,
            METADATA_PATH.name,
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Observation branch audit")
    print("-" * 88)
    print(
        branch_audit.to_string(
            index=False
        )
    )

    print()
    print("Validation summary")
    print("  Bundesliga fixture population: PASS")
    print("  Repository club alignment: PASS")
    print("  Repository projection: PASS")
    print("  Observation schemas matched: PASS")
    print("  Fixture and target columns matched: PASS")
    print("  ClubElo population match: PASS")
    print("  ClubElo temporal validity: PASS")
    print("  Finite representation values: PASS")
    print("  Alternative representation values: PASS")
    print("  Goal models fitted: NO")
    print("  Canonical study outputs changed: NO")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
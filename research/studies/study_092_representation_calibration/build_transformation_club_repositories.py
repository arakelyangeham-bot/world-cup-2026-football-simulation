#build_transformation_club_repositories

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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
from research.production.production_club_repository_builder import (
    ProductionClubRepositoryBuilder,
)
from research.production.production_repository_config import (
    ProductionRepositoryConfig,
)
from simulation.live_match_observation_builder import (
    ProductionClubRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_092_representation_calibration"
    / "study_092c1"
)

OUTPUT_DIRECTORY = (
    INPUT_DIRECTORY
    / "club_repositories"
)

TRANSFORMATIONS = (
    "global_zscore",
    "percentile_normal",
    "robust_zscore",
)

RATINGS_PATHS = {
    transformation: (
        INPUT_DIRECTORY
        / f"player_ratings_{transformation}.csv"
    )
    for transformation in TRANSFORMATIONS
}

OUTPUT_PATHS = {
    transformation: (
        OUTPUT_DIRECTORY
        / f"bundesliga_club_repository_{transformation}.csv"
    )
    for transformation in TRANSFORMATIONS
}

AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "transformation_club_repository_audit.csv"
)

DIFFERENCE_PATH = (
    OUTPUT_DIRECTORY
    / "transformation_club_repository_differences.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_092c1b_metadata.json"
)


TARGET_COMPETITION = "Bundesliga"
TARGET_SEASON_YEAR = "24/25"

REPRESENTATION_TYPE = "full_squad"
EXPECTED_CLUB_COUNT = 18


NUMERIC_REPRESENTATION_COLUMNS = (
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


def normalize_competition_name(
    value: object,
) -> str:
    return (
        str(value)
        .strip()
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
    )


def normalize_season_year(
    value: object,
) -> str:
    text = str(value).strip()

    if "-" in text:
        start_text, end_text = text.split(
            "-",
            maxsplit=1,
        )

        return (
            f"{int(start_text)}/"
            f"{str(int(end_text))[-2:]}"
        )

    if "/" not in text:
        return text

    start_text, end_text = text.split(
        "/",
        maxsplit=1,
    )

    start_year = (
        2000 + int(start_text)
        if len(start_text) == 2
        else int(start_text)
    )

    if len(end_text) == 4:
        end_text = end_text[-2:]

    return f"{start_year}/{end_text}"


def build_team_repository(
    ratings_path: Path,
) -> CompetitionTeamRepository:
    if not ratings_path.exists():
        raise FileNotFoundError(
            "Transformation-specific ratings file "
            f"does not exist: {ratings_path}"
        )

    player_repository = PlayerRepository(
        player_features_path=ratings_path
    )

    competition_repository = (
        CompetitionPlayerRepository(
            player_repository=player_repository
        )
    )

    roster_builder = CompetitionRosterBuilder(
        repository=competition_repository
    )

    return CompetitionTeamRepository(
        roster_builder=roster_builder
    )


def resolve_competition_season(
    team_repository: CompetitionTeamRepository,
) -> tuple[int, int, str]:
    seasons = (
        team_repository
        .roster_builder
        .list_competition_seasons()
        .copy()
    )

    required_columns = {
        "competition",
        "competition_id",
        "season_id",
        "season_year",
    }

    missing = required_columns - set(
        seasons.columns
    )

    if missing:
        raise ValueError(
            "Competition-season table is missing "
            f"required columns: {sorted(missing)}"
        )

    selected = seasons.loc[
        seasons["competition"]
        .map(normalize_competition_name)
        .eq(
            normalize_competition_name(
                TARGET_COMPETITION
            )
        )
        & seasons["season_year"]
        .map(normalize_season_year)
        .eq(
            normalize_season_year(
                TARGET_SEASON_YEAR
            )
        )
    ][
        [
            "competition_id",
            "season_id",
            "season_year",
        ]
    ].drop_duplicates()

    if len(selected) != 1:
        raise ValueError(
            "Could not uniquely resolve the configured "
            "Bundesliga competition-season."
        )

    row = selected.iloc[0]

    return (
        int(row["competition_id"]),
        int(row["season_id"]),
        str(row["season_year"]),
    )


def load_teams(
    team_repository: CompetitionTeamRepository,
    *,
    competition_id: int,
    season_id: int,
) -> pd.DataFrame:
    teams = (
        team_repository
        .roster_builder
        .list_teams(
            competition_id=competition_id,
            season_id=season_id,
        )
        .copy()
    )

    required_columns = {
        "team_id",
        "team",
    }

    missing = required_columns - set(
        teams.columns
    )

    if missing:
        raise ValueError(
            "Competition team table is missing "
            f"required columns: {sorted(missing)}"
        )

    teams = (
        teams[
            [
                "team_id",
                "team",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            by="team",
            key=lambda values: (
                values.astype(str).str.casefold()
            ),
        )
        .reset_index(drop=True)
    )

    if len(teams) != EXPECTED_CLUB_COUNT:
        raise ValueError(
            "Unexpected Bundesliga club count. "
            f"Expected {EXPECTED_CLUB_COUNT}, "
            f"received {len(teams)}."
        )

    return teams


def validate_runtime_reload(
    *,
    output_path: Path,
    expected_clubs: tuple[str, ...],
) -> None:
    runtime_repository = (
        ProductionClubRepository(
            repository_path=output_path
        )
    )

    loaded_clubs = set(
        runtime_repository.list_clubs()
    )

    if loaded_clubs != set(
        expected_clubs
    ):
        raise AssertionError(
            "Runtime reload changed the club population."
        )

    for club in expected_clubs:
        runtime_repository.resolve_club(
            club
        )


def build_repository(
    *,
    transformation: str,
    ratings_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    team_repository = build_team_repository(
        ratings_path
    )

    (
        competition_id,
        season_id,
        season_year,
    ) = resolve_competition_season(
        team_repository
    )

    teams = load_teams(
        team_repository,
        competition_id=competition_id,
        season_id=season_id,
    )

    name_to_team_id = {
        str(row.team):
            int(row.team_id)
        for row in teams.itertuples(
            index=False
        )
    }

    config = ProductionRepositoryConfig(
        competition_id=competition_id,
        competition_name=TARGET_COMPETITION,
        season_id=str(season_id),
        repository_version=(
            f"study_092c1_{transformation}"
        ),
        repository_scope=(
            "bundesliga_2024_25_"
            f"{transformation}"
        ),
        representation_type=(
            REPRESENTATION_TYPE
        ),
        expected_club_count=(
            EXPECTED_CLUB_COUNT
        ),
        output_path=output_path,
    )

    builder = ProductionClubRepositoryBuilder(
        config
    )

    def representation_provider(
        club: str,
    ):
        team_id = name_to_team_id[
            club
        ]

        return (
            team_repository
            .get_team_representation(
                competition_id=competition_id,
                season_id=season_id,
                team_id=team_id,
                representation_type=(
                    REPRESENTATION_TYPE
                ),
            )
        )

    dataframe = builder.build(
        clubs=tuple(
            teams["team"].tolist()
        ),
        representation_provider=(
            representation_provider
        ),
    )

    validate_runtime_reload(
        output_path=output_path,
        expected_clubs=tuple(
            teams["team"].tolist()
        ),
    )

    dataframe.insert(
        0,
        "transformation",
        transformation,
    )

    return dataframe


def build_repository_audit(
    repositories: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    baseline_clubs = set(
        repositories[
            "global_zscore"
        ]["club"]
    )

    for transformation, dataframe in (
        repositories.items()
    ):
        numeric = dataframe[
            list(
                NUMERIC_REPRESENTATION_COLUMNS
            )
        ].to_numpy(
            dtype=float
        )

        records.append(
            {
                "transformation":
                    transformation,
                "club_count":
                    len(dataframe),
                "unique_club_count":
                    dataframe[
                        "club"
                    ].nunique(),
                "club_population_match":
                    set(
                        dataframe[
                            "club"
                        ]
                    )
                    == baseline_clubs,
                "missing_value_count":
                    int(
                        dataframe.isna().sum().sum()
                    ),
                "non_finite_numeric_count":
                    int(
                        (
                            ~np.isfinite(
                                numeric
                            )
                        ).sum()
                    ),
                "representation_type_count":
                    dataframe[
                        "representation_type"
                    ].nunique(),
                "aggregation_profile_count":
                    dataframe[
                        "aggregation_profile"
                    ].nunique(),
            }
        )

    return pd.DataFrame(records)


def build_difference_summary(
    repositories: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    baseline = (
        repositories[
            "global_zscore"
        ]
        .sort_values("club")
        .reset_index(drop=True)
    )

    records: list[dict[str, object]] = []

    for transformation in (
        "percentile_normal",
        "robust_zscore",
    ):
        candidate = (
            repositories[
                transformation
            ]
            .sort_values("club")
            .reset_index(drop=True)
        )

        if not baseline[
            "club"
        ].equals(
            candidate[
                "club"
            ]
        ):
            raise AssertionError(
                "Club order or population differs across "
                "representation branches."
            )

        for column in (
            NUMERIC_REPRESENTATION_COLUMNS
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
                    "club_count":
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
                    "changed_club_count":
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

    return pd.DataFrame(records)


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 092C1B — TRANSFORMATION-SPECIFIC "
        "BUNDESLIGA REPOSITORIES"
    )
    print("=" * 88)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    repositories: dict[
        str,
        pd.DataFrame,
    ] = {}

    for transformation in TRANSFORMATIONS:
        print()
        print(
            f"Building {transformation} repository..."
        )

        repositories[
            transformation
        ] = build_repository(
            transformation=transformation,
            ratings_path=(
                RATINGS_PATHS[
                    transformation
                ]
            ),
            output_path=(
                OUTPUT_PATHS[
                    transformation
                ]
            ),
        )

    audit = build_repository_audit(
        repositories
    )

    differences = build_difference_summary(
        repositories
    )

    if not audit[
        "club_population_match"
    ].all():
        raise AssertionError(
            "Transformation repositories do not contain "
            "the same clubs."
        )

    if audit[
        "missing_value_count"
    ].sum() != 0:
        raise AssertionError(
            "Transformation repositories contain "
            "missing values."
        )

    if audit[
        "non_finite_numeric_count"
    ].sum() != 0:
        raise AssertionError(
            "Transformation repositories contain "
            "non-finite values."
        )

    if differences[
        "changed_club_count"
    ].sum() == 0:
        raise AssertionError(
            "Alternative repositories do not differ "
            "from the global-z-score control."
        )

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    differences.to_csv(
        DIFFERENCE_PATH,
        index=False,
    )

    metadata = {
        "study_id": "092C1B",
        "study_name": (
            "Transformation-Specific Bundesliga "
            "Repository Generation"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "transformations": list(
            TRANSFORMATIONS
        ),
        "club_count": EXPECTED_CLUB_COUNT,
        "club_population_match": True,
        "runtime_reload_pass": True,
        "alternative_values_detected": True,
        "canonical_study_078_artifact_changed": False,
        "observations_generated": False,
        "goal_models_fitted": False,
        "outputs": [
            path.name
            for path in OUTPUT_PATHS.values()
        ] + [
            AUDIT_PATH.name,
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
    print("Repository audit")
    print("-" * 88)
    print(
        audit.to_string(
            index=False
        )
    )

    print()
    print("Validation summary")
    print("  Ratings-path injection: PASS")
    print("  Bundesliga club population: PASS")
    print("  Repository schemas: PASS")
    print("  Finite representation values: PASS")
    print("  Runtime reload: PASS")
    print("  Alternative values detected: PASS")
    print("  Canonical Study 078 artifact changed: NO")
    print("  Goal models fitted: NO")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
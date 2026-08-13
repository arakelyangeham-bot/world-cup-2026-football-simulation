#build_bundesliga_production_repository

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.player_intelligence.competition_team_repository import (
    CompetitionTeamRepository,
)
from research.production.production_club_repository_builder import (
    ProductionClubRepositoryBuilder,
)
from research.production.production_repository_config import (
    ProductionRepositoryConfig,
)

# This runtime loader already exists in the project.
from simulation.live_match_observation_builder import (
    ProductionClubRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_078_bundesliga_production_repository"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "bundesliga_club_repository_v1.csv"
)


TARGET_COMPETITION = "Bundesliga"
TARGET_SEASON_YEAR = "24/25"

REPRESENTATION_TYPE = "full_squad"
REPOSITORY_VERSION = "v1"
REPOSITORY_SCOPE = "bundesliga_2024_25"

EXPECTED_CLUB_COUNT = 18


def normalize_competition_name(
    value: object,
) -> str:
    """
    Normalize a competition label for safe comparison.

    This normalization is deliberately local to study-level
    configuration resolution. It does not alter persisted club or
    competition names.
    """

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
    """
    Normalize common domestic season labels.

    Examples
    --------
    24/25       -> 2024/25
    2024/25     -> 2024/25
    2024-2025   -> 2024/25
    """

    text = str(value).strip()

    if "-" in text:
        parts = text.split("-", maxsplit=1)

        if len(parts) == 2:
            start_year = int(parts[0])
            end_year = int(parts[1])

            return (
                f"{start_year}/"
                f"{str(end_year)[-2:]}"
            )

    if "/" not in text:
        return text

    start_text, end_text = text.split(
        "/",
        maxsplit=1,
    )

    if len(start_text) == 2:
        start_year = 2000 + int(start_text)
    else:
        start_year = int(start_text)

    if len(end_text) == 4:
        end_text = end_text[-2:]

    return f"{start_year}/{end_text}"


def resolve_competition_season(
    team_repository: CompetitionTeamRepository,
) -> tuple[int, int, str]:
    """
    Resolve the configured Bundesliga season from the canonical
    competition-season table.

    Returns
    -------
    tuple
        competition_id, season_id, persisted season label
    """

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
            "Competition-season table is missing required "
            f"columns: {sorted(missing)}"
        )

    target_competition = (
        normalize_competition_name(
            TARGET_COMPETITION
        )
    )

    target_season = normalize_season_year(
        TARGET_SEASON_YEAR
    )

    competition_matches = (
        seasons["competition"]
        .map(normalize_competition_name)
        .eq(target_competition)
    )

    season_matches = (
        seasons["season_year"]
        .map(normalize_season_year)
        .eq(target_season)
    )

    selected = seasons[
        competition_matches
        & season_matches
    ].copy()

    if selected.empty:
        available = (
            seasons[
                [
                    "competition",
                    "season_year",
                    "competition_id",
                    "season_id",
                ]
            ]
            .sort_values(
                [
                    "competition",
                    "season_year",
                ]
            )
            .to_string(index=False)
        )

        raise KeyError(
            "Could not resolve the requested competition-season.\n"
            f"Competition: {TARGET_COMPETITION!r}\n"
            f"Season: {TARGET_SEASON_YEAR!r}\n\n"
            "Available competition-seasons:\n"
            f"{available}"
        )

    identity_columns = [
        "competition_id",
        "season_id",
        "season_year",
    ]

    selected = selected[
        identity_columns
    ].drop_duplicates()

    if len(selected) != 1:
        raise ValueError(
            "Competition-season resolution is ambiguous. "
            f"Matching rows:\n"
            f"{selected.to_string(index=False)}"
        )

    row = selected.iloc[0]

    return (
        int(row["competition_id"]),
        int(row["season_id"]),
        str(row["season_year"]),
    )


def load_competition_teams(
    team_repository: CompetitionTeamRepository,
    *,
    competition_id: int,
    season_id: int,
) -> pd.DataFrame:
    """
    Load and validate the canonical team table for one
    competition-season.
    """

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
            "Competition team table is missing required "
            f"columns: {sorted(missing)}"
        )

    teams = teams[
        [
            "team_id",
            "team",
        ]
    ].drop_duplicates()

    if teams.empty:
        raise ValueError(
            "No teams were found for the resolved "
            "competition-season."
        )

    teams["team_id"] = pd.to_numeric(
        teams["team_id"],
        errors="raise",
    ).astype(int)

    teams["team"] = (
        teams["team"]
        .astype(str)
        .str.strip()
    )

    if teams["team"].eq("").any():
        raise ValueError(
            "Competition team table contains an empty "
            "team name."
        )

    if teams["team_id"].duplicated().any():
        duplicates = (
            teams.loc[
                teams["team_id"].duplicated(
                    keep=False
                ),
                [
                    "team_id",
                    "team",
                ],
            ]
            .sort_values("team_id")
        )

        raise ValueError(
            "Competition team table contains duplicate "
            f"team IDs:\n{duplicates.to_string(index=False)}"
        )

    if (
        teams["team"]
        .str.casefold()
        .duplicated()
        .any()
    ):
        duplicates = teams.loc[
            teams[
                "team"
            ].str.casefold().duplicated(
                keep=False
            ),
            [
                "team_id",
                "team",
            ],
        ]

        raise ValueError(
            "Competition team table contains duplicate "
            f"team names:\n{duplicates.to_string(index=False)}"
        )

    if len(teams) != EXPECTED_CLUB_COUNT:
        raise ValueError(
            "Unexpected Bundesliga club count. "
            f"Expected {EXPECTED_CLUB_COUNT}, "
            f"received {len(teams)}."
        )

    return (
        teams
        .sort_values(
            by="team",
            key=lambda series: (
                series.str.casefold()
            ),
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_name_to_team_id(
    teams: pd.DataFrame,
) -> dict[str, int]:
    """
    Create the study-level bridge between production club names
    and Sofascore team identifiers.
    """

    return {
        str(row.team): int(row.team_id)
        for row in teams.itertuples(
            index=False
        )
    }


def validate_runtime_reload(
    output_path: Path,
    expected_clubs: tuple[str, ...],
) -> ProductionClubRepository:
    """
    Verify that the existing runtime repository accepts the newly
    generated artifact without modification.
    """

    runtime_repository = (
        ProductionClubRepository(
            repository_path=output_path
        )
    )

    loaded_clubs = (
        runtime_repository.list_clubs()
    )

    expected_normalized = tuple(
        sorted(
            expected_clubs,
            key=str.casefold,
        )
    )

    if len(loaded_clubs) != len(
        expected_normalized
    ):
        raise AssertionError(
            "Runtime reload changed the repository club "
            "count. "
            f"Expected {len(expected_normalized)}, "
            f"loaded {len(loaded_clubs)}."
        )

    for club in expected_clubs:
        representation = (
            runtime_repository.resolve_club(
                club
            )
        )

        if representation.club not in loaded_clubs:
            raise AssertionError(
                "Runtime repository resolved a club that "
                "is absent from its own club index. "
                f"Requested club: {club!r}. "
                f"Resolved club: "
                f"{representation.club!r}."
            )

    return runtime_repository


def main() -> None:
    print("=" * 72)
    print(
        "STUDY 078 — BUNDESLIGA PRODUCTION REPOSITORY"
    )
    print("=" * 72)

    team_repository = (
        CompetitionTeamRepository()
    )

    (
        competition_id,
        season_id,
        season_year,
    ) = resolve_competition_season(
        team_repository
    )

    print()
    print("Resolved competition-season")
    print(
        f"  Competition: {TARGET_COMPETITION}"
    )
    print(
        f"  Competition ID: {competition_id}"
    )
    print(
        f"  Season: {season_year}"
    )
    print(
        f"  Season ID: {season_id}"
    )

    teams = load_competition_teams(
        team_repository,
        competition_id=competition_id,
        season_id=season_id,
    )

    print()
    print(
        f"Resolved {len(teams)} clubs:"
    )

    for row in teams.itertuples(
        index=False
    ):
        print(
            f"  {int(row.team_id):>8}  "
            f"{row.team}"
        )

    name_to_team_id = (
        build_name_to_team_id(
            teams
        )
    )

    config = ProductionRepositoryConfig(
        competition_id=competition_id,
        competition_name=TARGET_COMPETITION,
        season_id=str(season_id),
        repository_version=(
            REPOSITORY_VERSION
        ),
        repository_scope=(
            REPOSITORY_SCOPE
        ),
        representation_type=(
            REPRESENTATION_TYPE
        ),
        expected_club_count=(
            EXPECTED_CLUB_COUNT
        ),
        output_path=OUTPUT_PATH,
    )

    builder = (
        ProductionClubRepositoryBuilder(
            config
        )
    )

    def representation_provider(
        club: str,
    ):
        try:
            team_id = name_to_team_id[
                club
            ]
        except KeyError as error:
            raise KeyError(
                "Representation provider received an "
                f"unknown club: {club!r}."
            ) from error

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

    print()
    print(
        "Building full-squad representations..."
    )

    dataframe = builder.build(
        clubs=tuple(
            teams["team"].tolist()
        ),
        representation_provider=(
            representation_provider
        ),
    )

    print()
    print(
        "Reloading artifact through "
        "ProductionClubRepository..."
    )

    runtime_repository = (
        validate_runtime_reload(
            output_path=OUTPUT_PATH,
            expected_clubs=tuple(
                teams["team"].tolist()
            ),
        )
    )

    print()
    print("Validation summary")
    print(
        f"  Repository rows: {len(dataframe)}"
    )
    print(
        "  Unique clubs: "
        f"{dataframe['club'].nunique()}"
    )
    print(
        "  Runtime clubs: "
        f"{len(runtime_repository.list_clubs())}"
    )
    print(
        "  Missing required values: "
        f"{int(dataframe.isna().sum().sum())}"
    )
    print(
        "  Representation type: "
        f"{REPRESENTATION_TYPE}"
    )
    print(
        f"  Output: {OUTPUT_PATH}"
    )

    print()
    print(dataframe.to_string(index=False))

    print()
    print("=" * 72)
    print("OVERALL RESULT: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
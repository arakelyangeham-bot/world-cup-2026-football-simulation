#sofascore_season_loader

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "sofascore_league_seasons.csv"
)

REQUIRED_COLUMNS = {
    "competition_key",
    "unique_tournament_id",
    "season_start_year",
    "season_id",
}


@dataclass(frozen=True)
class SofascoreSeason:
    competition_key: str
    year: int
    unique_tournament_id: int
    season_id: int

    @property
    def dataset_id(self) -> str:
        return f"{self.competition_key}_{self.year}"

    @property
    def output_filename(self) -> str:
        return f"{self.dataset_id}_match_results.csv"


def load_sofascore_seasons(
    registry_path: Path = REGISTRY_PATH,
) -> list[SofascoreSeason]:
    """
    Load and validate Sofascore competition-season identifiers
    from the canonical CSV registry.
    """

    if not registry_path.exists():
        raise FileNotFoundError(
            "Sofascore season registry was not found:\n"
            f"{registry_path}"
        )

    seasons: list[SofascoreSeason] = []

    with registry_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Sofascore season registry has no header row."
            )

        available_columns = {
            column.strip()
            for column in reader.fieldnames
            if column
        }

        missing_columns = (
            REQUIRED_COLUMNS - available_columns
        )

        if missing_columns:
            raise ValueError(
                "Sofascore season registry is missing "
                f"required columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            competition_key = str(
                row.get("competition_key", "")
            ).strip()

            if not competition_key:
                raise ValueError(
                    "Empty competition_key in Sofascore "
                    f"season registry row {row_number}."
                )

            try:
                year = int(
                    str(
                        row.get(
                            "season_start_year",
                            "",
                        )
                    ).strip()
                )

                unique_tournament_id = int(
                    str(
                        row.get(
                            "unique_tournament_id",
                            "",
                        )
                    ).strip()
                )

                season_id = int(
                    str(
                        row.get(
                            "season_id",
                            "",
                        )
                    ).strip()
                )

            except ValueError as exc:
                raise ValueError(
                    "Invalid numeric value in Sofascore "
                    f"season registry row {row_number}: "
                    f"{row}"
                ) from exc

            seasons.append(
                SofascoreSeason(
                    competition_key=competition_key,
                    year=year,
                    unique_tournament_id=(
                        unique_tournament_id
                    ),
                    season_id=season_id,
                )
            )

    if not seasons:
        raise ValueError(
            "Sofascore season registry contains no data rows."
        )

    validate_sofascore_seasons(seasons)

    return sorted(
        seasons,
        key=lambda season: (
            season.competition_key,
            season.year,
        ),
    )


def validate_sofascore_seasons(
    seasons: list[SofascoreSeason],
) -> None:
    """
    Validate uniqueness and consistency of loaded registry rows.
    """

    seen_dataset_keys: set[
        tuple[str, int]
    ] = set()

    seen_source_keys: set[
        tuple[int, int]
    ] = set()

    competition_tournament_ids: dict[
        str,
        int,
    ] = {}

    for season in seasons:
        dataset_key = (
            season.competition_key,
            season.year,
        )

        if dataset_key in seen_dataset_keys:
            raise ValueError(
                "Duplicate competition-season registry "
                f"entry: {dataset_key}"
            )

        seen_dataset_keys.add(dataset_key)

        source_key = (
            season.unique_tournament_id,
            season.season_id,
        )

        if source_key in seen_source_keys:
            raise ValueError(
                "Duplicate Sofascore tournament-season "
                f"identifier pair: {source_key}"
            )

        seen_source_keys.add(source_key)

        existing_tournament_id = (
            competition_tournament_ids.get(
                season.competition_key
            )
        )

        if existing_tournament_id is None:
            competition_tournament_ids[
                season.competition_key
            ] = season.unique_tournament_id

        elif (
            existing_tournament_id
            != season.unique_tournament_id
        ):
            raise ValueError(
                "Competition uses inconsistent unique "
                "tournament IDs: "
                f"{season.competition_key!r} has both "
                f"{existing_tournament_id} and "
                f"{season.unique_tournament_id}."
            )


def build_season_lookup(
    registry_path: Path = REGISTRY_PATH,
) -> dict[tuple[str, int], SofascoreSeason]:
    """
    Build a lookup indexed by:
        (competition_key, season_start_year)
    """

    seasons = load_sofascore_seasons(
        registry_path=registry_path
    )

    return {
        (
            season.competition_key,
            season.year,
        ): season
        for season in seasons
    }


def get_sofascore_season(
    competition_key: str,
    year: int,
    registry_path: Path = REGISTRY_PATH,
) -> SofascoreSeason:
    """
    Return one registered Sofascore season.
    """

    normalized_competition_key = (
        competition_key.strip().lower()
    )

    lookup = build_season_lookup(
        registry_path=registry_path
    )

    key = (
        normalized_competition_key,
        year,
    )

    try:
        return lookup[key]

    except KeyError as exc:
        available = sorted(
            lookup,
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

        raise KeyError(
            "No registered Sofascore season exists for "
            f"competition={normalized_competition_key!r}, "
            f"year={year}. "
            f"Available entries: {available}"
        ) from exc
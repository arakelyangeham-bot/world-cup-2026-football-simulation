#discover_sofascore_competition_seasons

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from scripts.sofascore_utils import BASE_URL, get_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "sofascore_league_seasons.csv"
)

LEAGUES = {
    "premier_league": {
        "display_name": "Premier League",
        "unique_tournament_id": 17,
    },
    "championship": {
        "display_name": "Championship",
        "unique_tournament_id": 18,
    },
    "la_liga": {
        "display_name": "La Liga",
        "unique_tournament_id": 8,
    },
    "serie_a": {
        "display_name": "Serie A",
        "unique_tournament_id": 23,
    },
    "bundesliga": {
        "display_name": "Bundesliga",
        "unique_tournament_id": 35,
    },
    "2_bundesliga": {
        "display_name": "2. Bundesliga",
        "unique_tournament_id": 44,
    },
    "ligue_1": {
        "display_name": "Ligue 1",
        "unique_tournament_id": 34,
    },
    "liga_portugal": {
        "display_name": "Liga Portugal",
        "unique_tournament_id": 238,
    },
    "eredivisie": {
        "display_name": "Eredivisie",
        "unique_tournament_id": 37,
    },
}

DEFAULT_COMPETITIONS = [
    "premier_league",
    "la_liga",
    "serie_a",
    "bundesliga",
    "ligue_1",
]

TARGET_START_YEARS = {
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
}

OUTPUT_COLUMNS = [
    "competition_key",
    "competition_name",
    "unique_tournament_id",
    "season_name",
    "season_year_label",
    "season_start_year",
    "season_id",
]


def fetch_seasons(
    unique_tournament_id: int,
) -> list[dict[str, Any]]:
    url = (
        f"{BASE_URL}/unique-tournament/"
        f"{unique_tournament_id}/seasons"
    )

    payload = get_json(url)
    seasons = payload.get("seasons", [])

    if not isinstance(seasons, list):
        raise TypeError(
            "Sofascore seasons payload is not a list for "
            f"unique tournament ID {unique_tournament_id}."
        )

    if not seasons:
        raise RuntimeError(
            "No seasons were returned for unique tournament "
            f"ID {unique_tournament_id}."
        )

    return seasons


def parse_season_start_year(
    year_label: str,
) -> int | None:
    """
    Convert Sofascore labels such as '23/24' into 2023.

    Four-digit labels are also supported.
    """

    cleaned = str(year_label).strip()

    if not cleaned:
        return None

    first_part = cleaned.split("/")[0]

    try:
        value = int(first_part)
    except ValueError:
        return None

    if value >= 1000:
        return value

    if 0 <= value <= 99:
        return 2000 + value

    return None


def build_registry_rows(
    competition_keys: list[str],
    target_start_years: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for competition_key in competition_keys:
        if competition_key not in LEAGUES:
            known = ", ".join(sorted(LEAGUES))

            raise KeyError(
                f"Unknown competition key: {competition_key!r}. "
                f"Known keys: {known}"
            )

        league = LEAGUES[competition_key]
        display_name = str(league["display_name"])
        unique_tournament_id = int(
            league["unique_tournament_id"]
        )

        print()
        print(
            f"Fetching {display_name} "
            f"(unique tournament ID "
            f"{unique_tournament_id})..."
        )

        seasons = fetch_seasons(
            unique_tournament_id
        )

        matched_count = 0

        for season in seasons:
            year_label = str(
                season.get("year", "")
            ).strip()

            season_start_year = (
                parse_season_start_year(
                    year_label
                )
            )

            if (
                season_start_year
                not in target_start_years
            ):
                continue

            season_id = season.get("id")

            if season_id is None:
                raise ValueError(
                    f"{display_name} season "
                    f"{season.get('name')!r} "
                    "has no season ID."
                )

            rows.append(
                {
                    "competition_key": (
                        competition_key
                    ),
                    "competition_name": (
                        display_name
                    ),
                    "unique_tournament_id": (
                        unique_tournament_id
                    ),
                    "season_name": str(
                        season.get("name", "")
                    ).strip(),
                    "season_year_label": (
                        year_label
                    ),
                    "season_start_year": (
                        season_start_year
                    ),
                    "season_id": int(
                        season_id
                    ),
                }
            )

            matched_count += 1

        print(
            "  Target seasons found: "
            f"{matched_count}"
        )

    return sorted(
        rows,
        key=lambda row: (
            row["competition_key"],
            row["season_start_year"],
        ),
    )


def validate_registry_rows(
    rows: list[dict[str, Any]],
    competition_keys: list[str],
    target_start_years: set[int],
) -> None:
    expected_count = (
        len(competition_keys)
        * len(target_start_years)
    )

    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} registry rows, "
            f"but found {len(rows)}."
        )

    seen_dataset_keys: set[
        tuple[str, int]
    ] = set()

    seen_source_keys: set[
        tuple[int, int]
    ] = set()

    for row in rows:
        dataset_key = (
            str(row["competition_key"]),
            int(row["season_start_year"]),
        )

        if dataset_key in seen_dataset_keys:
            raise ValueError(
                "Duplicate competition-season row: "
                f"{dataset_key}"
            )

        seen_dataset_keys.add(
            dataset_key
        )

        source_key = (
            int(row["unique_tournament_id"]),
            int(row["season_id"]),
        )

        if source_key in seen_source_keys:
            raise ValueError(
                "Duplicate Sofascore tournament-season "
                f"identifier pair: {source_key}"
            )

        seen_source_keys.add(
            source_key
        )

    expected_dataset_keys = {
        (competition_key, year)
        for competition_key
        in competition_keys
        for year in target_start_years
    }

    missing_keys = sorted(
        expected_dataset_keys
        - seen_dataset_keys
    )

    if missing_keys:
        raise ValueError(
            "Missing competition-season rows: "
            f"{missing_keys}"
        )



def read_registry_csv(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        missing_columns = (
            set(OUTPUT_COLUMNS)
            - set(reader.fieldnames or [])
        )

        if missing_columns:
            raise ValueError(
                "Existing registry is missing required "
                "columns: "
                f"{sorted(missing_columns)}"
            )

        return [
            dict(row)
            for row in reader
        ]


def merge_registry_rows(
    *,
    existing_rows: list[dict[str, Any]],
    refreshed_rows: list[dict[str, Any]],
    refreshed_competition_keys: list[str],
) -> list[dict[str, Any]]:
    refreshed_keys = set(
        refreshed_competition_keys
    )

    preserved_rows = [
        row
        for row in existing_rows
        if row["competition_key"]
        not in refreshed_keys
    ]

    merged_rows = (
        preserved_rows
        + refreshed_rows
    )

    return sorted(
        merged_rows,
        key=lambda row: (
            row["competition_key"],
            int(row["season_start_year"]),
        ),
    )

def write_registry_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_competition_list(
    value: str,
) -> list[str]:
    competition_keys = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    if not competition_keys:
        raise argparse.ArgumentTypeError(
            "At least one competition key "
            "must be supplied."
        )

    unknown = sorted(
        set(competition_keys)
        - set(LEAGUES)
    )

    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown competition keys: {unknown}. "
            f"Known: {sorted(LEAGUES)}"
        )

    return competition_keys


def parse_year_list(
    value: str,
) -> set[int]:
    try:
        years = {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Years must be comma-separated integers."
        ) from exc

    if not years:
        raise argparse.ArgumentTypeError(
            "At least one year is required."
        )

    return years


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover Sofascore season IDs and build "
            "the canonical domestic-league season registry CSV."
        )
    )

    parser.add_argument(
        "--competitions",
        type=parse_competition_list,
        default=DEFAULT_COMPETITIONS,
        help=(
            "Comma-separated competition keys. "
            "Defaults to all registered domestic leagues."
        ),
    )

    parser.add_argument(
        "--years",
        type=parse_year_list,
        default=TARGET_START_YEARS,
        help=(
            "Comma-separated season start years. "
            "Defaults to 2023,2024,2025."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the canonical registry CSV.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    competition_keys = list(
        arguments.competitions
    )

    target_start_years = set(
        arguments.years
    )

    print(
        "Sofascore Domestic League "
        "Season Registry Builder"
    )
    print(
        "=========================================="
    )
    print(
        "Competitions: "
        f"{competition_keys}"
    )
    print(
        "Season start years: "
        f"{sorted(target_start_years)}"
    )

    rows = build_registry_rows(
        competition_keys=competition_keys,
        target_start_years=(
            target_start_years
        ),
    )

    validate_registry_rows(
        rows=rows,
        competition_keys=competition_keys,
        target_start_years=(
            target_start_years
        ),
    )

    existing_rows = read_registry_csv(
        arguments.output
    )

    rows = merge_registry_rows(
        existing_rows=existing_rows,
        refreshed_rows=rows,
        refreshed_competition_keys=competition_keys,
    )

    write_registry_csv(
        rows=rows,
        output_path=arguments.output,
    )

    print()
    print("Registry Summary")
    print("----------------")
    print(f"Rows written: {len(rows)}")
    print(
        "Competitions represented: "
        f"{len(competition_keys)}"
    )
    print(
        "Seasons per competition: "
        f"{len(target_start_years)}"
    )
    print(f"Output: {arguments.output}")
    print()
    print("Validation Result")
    print("-----------------")
    print("PASSED")
    print(
        "Canonical Sofascore domestic-league "
        "season registry written successfully."
    )


if __name__ == "__main__":
    main()
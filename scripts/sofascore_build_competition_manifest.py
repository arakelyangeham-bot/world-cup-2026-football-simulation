import argparse
from pathlib import Path

import pandas as pd
from scripts.sofascore_utils import OUT_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLUB_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "sofascore_league_seasons.csv"
)
INTL_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_international_seasons.csv"
OUT_FILE = OUT_DIR / "raw" / "sofascore" / "competition_manifest.csv"

PRODUCTION_DOMESTIC_COMPETITIONS = {
    "Premier League",
    "Bundesliga",
    "La Liga",
    "Serie A",
    "Ligue 1",
}

PRODUCTION_DOMESTIC_SEASONS_PER_COMPETITION = 5

# Recency weights are part of the player-evidence methodology.
# Expanding competition coverage must preserve these established
# weights unless a separate research study justifies changing them.

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a validated Sofascore competition manifest "
            "from domestic and international competition registries."
        )
    )

    parser.add_argument(
        "--club-file",
        type=Path,
        default=CLUB_FILE,
        help=(
            "Domestic competition registry. Defaults to the "
            "canonical sofascore_league_seasons.csv."
        ),
    )

    parser.add_argument(
        "--international-file",
        type=Path,
        default=INTL_FILE,
        help=(
            "International competition registry. Defaults to "
            "the canonical Sofascore international-season file."
        ),
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUT_FILE,
        help=(
            "Destination competition manifest. Defaults to the "
            "canonical competition_manifest.csv."
        ),
    )

    parser.add_argument(
        "--derive-domestic-contract",
        action="store_true",
        help=(
            "Derive the expected domestic competition population "
            "and season counts from the supplied club registry. "
            "Intended for isolated research candidates only. "
            "Without this flag, the canonical Big Five production "
            "contract is enforced."
        ),
    )

    return parser.parse_args()

def get_recency_weight(season_year):
    weights = {
        # Domestic league seasons
        "21/22": 0.50,
        "22/23": 0.50,
        "23/24": 0.50,
        "24/25": 0.75,
        "25/26": 1.00,

        # Calendar-year competitions
        "2023": 0.65,
        "2024": 0.75,
        "2025": 0.90,
        "2026": 1.00,
    }

    return weights.get(
        str(season_year),
        0.50,
    )


def get_competition_importance(competition_type):
    weights = {
        "world_cup": 1.50,
        "continental_international": 1.30,
        "international": 1.15,
        "international_qualifier": 1.10,
        "continental_club": 1.15,
        "club_league": 1.00,
        "domestic_cup": 0.75,
        "friendly": 0.40,
    }
    return weights.get(competition_type, 1.00)


def classify_international_type(row):
    key = str(row.get("competition_key", "")).lower()
    name = str(row.get("competition", "")).lower()

    # Qualifiers FIRST
    if "qual" in key or "qual" in name:
        return "international_qualifier"

    # Then the actual World Cup
    if key == "world_cup" or name == "fifa world cup":
        return "world_cup"

    continental_keys = [
        "euro",
        "copa",
        "afcon",
        "asian",
        "gold",
        "concacaf",
    ]

    if any(k in key or k in name for k in continental_keys):
        return "continental_international"

    return "international"

def validate_manifest(
    manifest,
    expected_domestic_counts,
    source_international_row_count,
):
    required_columns = {
        "competition",
        "competition_type",
        "competition_id",
        "season_id",
        "season_year",
        "priority",
        "importance",
        "recency_weight",
        "competition_importance",
        "enabled",
        "scrape_teams",
        "scrape_players",
        "scrape_stats",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Competition manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    identifier_columns = [
        "competition",
        "competition_id",
        "season_id",
        "season_year",
    ]

    null_identifier_rows = manifest[
        identifier_columns
    ].isna().any(axis=1)

    if null_identifier_rows.any():
        invalid_rows = manifest.loc[
            null_identifier_rows,
            identifier_columns,
        ]

        raise ValueError(
            "Competition manifest contains rows with missing "
            "competition or season identifiers:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    duplicate_source_keys = manifest.duplicated(
        subset=[
            "competition_id",
            "season_id",
        ],
        keep=False,
    )

    if duplicate_source_keys.any():
        duplicates = manifest.loc[
            duplicate_source_keys,
            [
                "competition",
                "competition_id",
                "season_id",
                "season_year",
            ],
        ].sort_values(
            [
                "competition_id",
                "season_id",
            ]
        )

        raise ValueError(
            "Duplicate Sofascore competition-season identifiers "
            "were found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    domestic = manifest[
        manifest["competition_type"]
        == "club_league"
    ].copy()

    domestic_counts = (
        domestic
        .groupby("competition")
        .size()
        .to_dict()
    )

    expected_domestic_competitions = set(
        expected_domestic_counts
    )

    missing_domestic_competitions = sorted(
        expected_domestic_competitions
        - set(domestic_counts)
    )

    if missing_domestic_competitions:
        raise ValueError(
            "Missing expected domestic competitions: "
            f"{missing_domestic_competitions}"
        )

    unexpected_domestic_competitions = sorted(
        set(domestic_counts)
        - expected_domestic_competitions
    )

    if unexpected_domestic_competitions:
        raise ValueError(
            "Unexpected domestic competitions were found: "
            f"{unexpected_domestic_competitions}"
        )

    invalid_domestic_counts = {
        competition: {
            "expected": expected_domestic_counts[
                competition
            ],
            "observed": count,
        }
        for competition, count
        in domestic_counts.items()
        if (
            competition
            in expected_domestic_counts
            and count
            != expected_domestic_counts[
                competition
            ]
        )
    }

    if invalid_domestic_counts:
        raise ValueError(
            "Domestic competitions do not contain the expected "
            "number of seasons: "
            f"{invalid_domestic_counts}"
        )

    expected_domestic_row_count = sum(
        expected_domestic_counts.values()
    )

    if len(domestic) != expected_domestic_row_count:
        raise ValueError(
            "Expected "
            f"{expected_domestic_row_count} domestic rows, "
            f"but found {len(domestic)}."
        )

    international = manifest[
        manifest["competition_type"]
        != "club_league"
    ]

    if len(international) != source_international_row_count:
        raise ValueError(
            "International row count changed during manifest "
            "construction. "
            f"Source rows: {source_international_row_count}; "
            f"manifest rows: {len(international)}."
        )

args = parse_args()

club_file = args.club_file
international_file = (
    args.international_file
)
output_file = args.output_file

output_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)

club = pd.read_csv(
    club_file
)

club_manifest = club.rename(
    columns={
        "competition_name": "competition",
        "unique_tournament_id": "competition_id",
        "season_year_label": "season_year",
    }
).copy()

required_club_columns = {
    "competition",
    "competition_id",
    "season_id",
    "season_year",
}

missing_club_columns = (
    required_club_columns
    - set(club_manifest.columns)
)

if missing_club_columns:
    raise ValueError(
        "Canonical domestic competition registry is missing "
        f"required columns: {sorted(missing_club_columns)}"
    )

club_manifest["competition_type"] = "club_league"

club_manifest = club_manifest[
    [
        "competition",
        "competition_type",
        "competition_id",
        "season_id",
        "season_year",
    ]
]

intl = pd.read_csv(
    international_file
)

source_international_row_count = len(intl)

intl_manifest = intl.copy()

intl_manifest["competition_type"] = intl_manifest.apply(
    classify_international_type,
    axis=1,
)

intl_manifest = intl_manifest[
    [
        "competition",
        "competition_type",
        "competition_id",
        "season_id",
        "season_year",
    ]
]

manifest = pd.concat(
    [club_manifest, intl_manifest],
    ignore_index=True,
)

manifest["priority"] = 100
manifest.loc[
    manifest["competition_type"] == "world_cup",
    "priority"
] = 200

manifest["importance"] = 1.0
manifest["recency_weight"] = manifest["season_year"].apply(get_recency_weight)
manifest["competition_importance"] = manifest["competition_type"].apply(
    get_competition_importance
)

manifest["enabled"] = True
manifest["scrape_teams"] = True
manifest["scrape_players"] = True
manifest["scrape_stats"] = True

manifest = manifest[
    [
        "competition",
        "competition_type",
        "competition_id",
        "season_id",
        "season_year",
        "priority",
        "importance",
        "recency_weight",
        "competition_importance",
        "enabled",
        "scrape_teams",
        "scrape_players",
        "scrape_stats",
    ]
]

if args.derive_domestic_contract:
    expected_domestic_counts = (
        club_manifest
        .groupby("competition")
        .size()
        .astype(int)
        .to_dict()
    )

    contract_mode = (
        "derived_from_supplied_registry"
    )

else:
    expected_domestic_counts = {
        competition:
            PRODUCTION_DOMESTIC_SEASONS_PER_COMPETITION
        for competition
        in PRODUCTION_DOMESTIC_COMPETITIONS
    }

    contract_mode = (
        "production_big_five"
    )

manifest = manifest.sort_values(
    ["priority", "competition", "season_year"],
    ascending=[False, True, True],
)

validate_manifest(
    manifest=manifest,
    expected_domestic_counts=(
        expected_domestic_counts
    ),
    source_international_row_count=(
        source_international_row_count
    ),
)

domestic_summary = (
    manifest[
        manifest["competition_type"]
        == "club_league"
    ]
    .groupby("competition")
    .agg(
        seasons=("season_id", "count"),
        first_season=("season_year", "min"),
        last_season=("season_year", "max"),
    )
    .sort_index()
)

international_summary = (
    manifest[
        manifest["competition_type"]
        != "club_league"
    ]
    ["competition_type"]
    .value_counts()
    .sort_index()
)

print()
print("Domestic Competition Summary")
print("----------------------------")
print(domestic_summary.to_string())

print()
print("International Competition Summary")
print("---------------------------------")
print(international_summary.to_string())

print()
print("Manifest Validation")
print("-------------------")
print("PASSED")
print(
    f"Domestic contract mode: "
    f"{contract_mode}"
)
print(f"Domestic rows: {len(manifest[manifest['competition_type'] == 'club_league'])}")
print(f"International rows: {len(manifest[manifest['competition_type'] != 'club_league'])}")
print(f"Total rows: {len(manifest)}")

manifest.to_csv(
    output_file,
    index=False,
)

print(f"Saved {len(manifest)} rows to {OUT_FILE}")
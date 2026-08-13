#build_player_registry.py

# ARCHITECTURE NOTE:
#
# This script builds a source-oriented player registry and is retained for
# historical/general-purpose workflows.
#
# The modern Player Intelligence production identity branch uses:
#
#     data/raw/sofascore/sofascore_player_profiles.csv
#         -> scripts.build_canonical_player_registry
#         -> data/processed/canonical_player_registry.csv
#
# This script is not a required stage of
# scripts.run_player_intelligence_pipeline.

from pathlib import Path
import pandas as pd

import argparse
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_player_dataset.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "player_registry.csv"

ALIAS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_id_aliases.csv"
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a player registry from an aggregated "
            "player dataset."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=IN_FILE,
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUT_FILE,
    )

    return parser.parse_args()

arguments = parse_arguments()

input_file = arguments.input_file
output_file = arguments.output_file

df = pd.read_csv(
    input_file,
    dtype={"season_year": str},
)

registry = (
    df[
        [
            "player_id",
            "player",
            "player_slug",
            "country",
            "country_alpha2",
            "country_alpha3",
            "positions_detailed",
            "eligible_roles",
            "position",
            "current_team",
        ]
    ]
    .drop_duplicates(subset=["player_id"])
    .copy()
)

registry["canonical_player_id"] = registry["player_id"]

if ALIAS_FILE.exists() and ALIAS_FILE.stat().st_size > 0:
    aliases = pd.read_csv(ALIAS_FILE)

    required_alias_columns = {
        "source_player_id",
        "canonical_player_id",
        "review_status",
    }

    missing_alias_columns = (
        required_alias_columns
        - set(aliases.columns)
    )

    if missing_alias_columns:
        raise ValueError(
            "Player ID alias file is missing required columns: "
            f"{sorted(missing_alias_columns)}"
        )

    reviewed_aliases = aliases[
        aliases["review_status"] == "reviewed"
    ].copy()

    if reviewed_aliases["source_player_id"].duplicated().any():
        duplicate_ids = (
            reviewed_aliases.loc[
                reviewed_aliases[
                    "source_player_id"
                ].duplicated(keep=False),
                "source_player_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate reviewed source player IDs in alias file: "
            f"{duplicate_ids}"
        )

    alias_map = dict(
        zip(
            reviewed_aliases["source_player_id"].astype(int),
            reviewed_aliases["canonical_player_id"].astype(int),
        )
    )

    registry["canonical_player_id"] = (
        registry["player_id"]
        .astype(int)
        .map(alias_map)
        .fillna(registry["player_id"])
        .astype(int)
    )

registry = registry[
    [
        "canonical_player_id",
        "player_id",
        "player",
        "player_slug",
        "country",
        "country_alpha2",
        "country_alpha3",
        "positions_detailed",
        "eligible_roles",
        "position",
        "current_team",
    ]
]

output_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)

registry.to_csv(
    output_file,
    index=False,
)

print(registry.head())
print(registry.shape)
print(f"Wrote: {output_file}")
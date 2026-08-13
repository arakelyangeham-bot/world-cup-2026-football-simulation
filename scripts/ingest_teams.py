import argparse
from time import sleep
from pathlib import Path
import pandas as pd

from scripts.sofascore_utils import BASE_URL, OUT_DIR, get_json

REQUEST_DELAY = 3

MANIFEST_FILE = OUT_DIR / "raw" / "sofascore" / "competition_manifest.csv"
OUT_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_teams.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest Sofascore teams for selected competition-seasons."
    )

    parser.add_argument(
        "--competition",
        type=str,
        help=(
            "Optional competition name from the manifest, "
            "for example 'Bundesliga'."
        ),
    )

    parser.add_argument(
        "--season-year",
        type=str,
        help=(
            "Optional season label from the manifest, "
            "for example '24/25'."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the selected competition-seasons without making "
            "requests or writing output."
        ),
    )

    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=MANIFEST_FILE,
        help=(
            "Competition-manifest input CSV. Defaults to the "
            "canonical competition_manifest.csv."
        ),
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUT_FILE,
        help=(
            "Team-season output CSV. Defaults to the "
            "canonical sofascore_teams.csv."
        ),
    )

    return parser.parse_args()


def get_teams_for_competition(row):
    url = (
        f"{BASE_URL}/unique-tournament/{row['competition_id']}/"
        f"season/{row['season_id']}/standings/total"
    )

    data = get_json(url)

    teams = []

    for standing in data.get("standings", []):
        for standing_row in standing.get("rows", []):
            team = standing_row["team"]

            teams.append({
                "competition": row["competition"],
                "competition_type": row["competition_type"],
                "competition_id": row["competition_id"],
                "season_id": row["season_id"],
                "season_year": row["season_year"],
                "team_id": team.get("id"),
                "team": team.get("name"),
                "team_slug": team.get("slug"),
            })

    return teams


if __name__ == "__main__":
    args = parse_args()

    manifest_file = args.manifest_file
    output_file = args.output_file

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        manifest_file
    )

    manifest = manifest[
        (manifest["enabled"] == True) &
        (manifest["scrape_teams"] == True)
    ].sort_values(
        ["priority", "competition", "season_year"],
        ascending=[False, True, True],
    )

    if args.competition:
        manifest = manifest[
            manifest["competition"].str.casefold()
            == args.competition.casefold()
        ]

    if args.season_year:
        manifest = manifest[
            manifest["season_year"].astype(str)
            == args.season_year
        ]

    if manifest.empty:
        raise ValueError(
            "No enabled competition-seasons matched the supplied filters. "
            f"competition={args.competition!r}, "
            f"season_year={args.season_year!r}"
        )

    if args.dry_run:
        columns = [
            "competition",
            "competition_id",
            "season_id",
            "season_year",
        ]

        print("Selected competition-seasons:")
        print(
            manifest[columns].to_string(index=False)
        )

        raise SystemExit(0)

    all_teams = []
    successful_scope = set()
    for idx, row in manifest.iterrows():
        print(
            f"[{idx+1}/{len(manifest)}] "
            f"Ingesting teams for {row['competition']} {row['season_year']}"
        )

        try:
            teams = get_teams_for_competition(row)
            all_teams.extend(teams)

            if not teams:
                raise RuntimeError(
                    "No teams were returned for "
                    f"{row['competition']} "
                    f"{row['season_year']}."
                )

            successful_scope.add(
                (
                    int(row["competition_id"]),
                    int(row["season_id"]),
                )
            )

        except Exception as e:
            print(
                f"FAILED teams for {row['competition']} "
                f"{row['season_year']} -> {e}"
            )
            

        sleep(REQUEST_DELAY)

    if not all_teams:
        raise RuntimeError(
            "Team ingestion produced no rows for the selected "
            "competition-seasons."
        )
    
    new_df = pd.DataFrame(
        all_teams
    ).drop_duplicates(
        subset=[
            "competition_id",
            "season_id",
            "team_id",
        ]
    )

    if output_file.exists() and output_file.stat().st_size > 0:
        existing_df = pd.read_csv(
            output_file
        )

        required_columns = {
            "competition_id",
            "season_id",
            "team_id",
        }

        missing_columns = (
            required_columns
            - set(existing_df.columns)
        )

        if missing_columns:
            raise ValueError(
                "Existing team dataset is missing required "
                "columns: "
                f"{sorted(missing_columns)}"
            )

        existing_scope_keys = list(
            zip(
                existing_df[
                    "competition_id"
                ].astype(int),
                existing_df[
                    "season_id"
                ].astype(int),
            )
        )

        keep_existing_rows = [
            scope_key not in successful_scope
            for scope_key in existing_scope_keys
        ]

        preserved_df = existing_df.loc[
            keep_existing_rows
        ].copy()

    else:
        preserved_df = pd.DataFrame(
            columns=new_df.columns
        )

    combined_df = pd.concat(
        [
            preserved_df,
            new_df,
        ],
        ignore_index=True,
    )

    combined_df = (
        combined_df
        .drop_duplicates(
            subset=[
                "competition_id",
                "season_id",
                "team_id",
            ],
            keep="last",
        )
        .sort_values(
            [
                "competition",
                "season_year",
                "team",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    print()
    print("Team Ingestion Summary")
    print("----------------------")
    print(
        f"Newly scraped rows: "
        f"{len(new_df)}"
    )
    print(
        f"Preserved existing rows: "
        f"{len(preserved_df)}"
    )
    print(
        f"Final combined rows: "
        f"{len(combined_df)}"
    )

    combined_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved {len(combined_df)} "
        f"team-season rows to {output_file}"
    )
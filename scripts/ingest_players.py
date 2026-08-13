import argparse
from time import sleep
from pathlib import Path

import pandas as pd

from scripts.sofascore_utils import BASE_URL, OUT_DIR, get_json

REQUEST_DELAY = 3

MANIFEST_FILE = OUT_DIR / "raw" / "sofascore" / "competition_manifest.csv"
OUT_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_players.csv"
FAILED_FILE = (
    OUT_DIR
    / "raw"
    / "sofascore"
    / "sofascore_players_failed.csv"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ingest Sofascore player-team memberships for "
            "selected competition-seasons."
        )
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
            "Player-team-season output CSV. Defaults to the "
            "canonical sofascore_players.csv."
        ),
    )

    parser.add_argument(
        "--failed-file",
        type=Path,
        default=FAILED_FILE,
        help=(
            "Player-ingestion failure CSV. Defaults to the "
            "canonical failure artifact."
        ),
    )

    return parser.parse_args()

def get_players_for_competition(row):
    first_url = (
        f"{BASE_URL}/unique-tournament/{row['competition_id']}/"
        f"season/{row['season_id']}/statistics"
    )

    first_data = get_json(first_url)
    pages = first_data.get("pages", 0)

    players = []

    for offset in range(0, pages * 10, 10):
        url = (
            f"{BASE_URL}/unique-tournament/{row['competition_id']}/"
            f"season/{row['season_id']}/statistics?offset={offset}"
        )

        data = get_json(url)
        results = data.get("results", [])

        if not results:
            break

        for item in results:
            player = item["player"]
            team = item["team"]

            players.append({
                "competition": row["competition"],
                "competition_type": row["competition_type"],
                "competition_id": row["competition_id"],
                "season_id": row["season_id"],
                "season_year": row["season_year"],
                "player_id": player.get("id"),
                "player": player.get("name"),
                "player_slug": player.get("slug"),
                "team_id": team.get("id"),
                "team": team.get("name"),
                "team_slug": team.get("slug"),
            })

    if not players:
        raise RuntimeError(
            "No player-team membership rows were returned for "
            f"{row['competition']} {row['season_year']} "
            f"(competition_id={row['competition_id']}, "
            f"season_id={row['season_id']})."
        )

    return players


if __name__ == "__main__":
    args = parse_args()

    manifest_file = args.manifest_file
    output_file = args.output_file
    failed_file = args.failed_file

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    failed_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(manifest_file)

    manifest = manifest[
        (manifest["enabled"] == True) &
        (manifest["scrape_players"] == True)
    ].sort_values(
        ["priority", "competition", "season_year"],
        ascending=[False, True, True]
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

    all_players = []
    failed_rows = []
    successful_scope = set()

    for idx, row in manifest.iterrows():
        print(
            f"[{idx+1}/{len(manifest)}] "
            f"Ingesting players for {row['competition']} {row['season_year']}"
        )

        try:
            players = get_players_for_competition(row)
            all_players.extend(players)

            successful_scope.add(
                (
                    int(row["competition_id"]),
                    int(row["season_id"]),
                )
            )

        except Exception as exc:
            print(
                f"FAILED players for {row['competition']} "
                f"{row['season_year']} -> {exc}"
            )

            failed_rows.append(
                {
                    "competition": row["competition"],
                    "competition_type": row["competition_type"],
                    "competition_id": row["competition_id"],
                    "season_id": row["season_id"],
                    "season_year": row["season_year"],
                    "error": str(exc),
                }
            )

        sleep(REQUEST_DELAY)

    failed_df = pd.DataFrame(failed_rows)

    if failed_df.empty:
        if failed_file.exists():
            failed_file.unlink()

        print("No player-ingestion failures recorded.")

    else:
        failed_df.to_csv(
            failed_file,
            index=False,
        )

        print(
            f"Saved {len(failed_df)} player-ingestion failures "
            f"to {failed_file}"
        )


    if not all_players:
        raise RuntimeError(
            "Player ingestion produced no rows for the selected "
            "competition-seasons. Review the terminal output and "
            f"{failed_file} for failure details."
        )
    
    new_df = pd.DataFrame(all_players).drop_duplicates(
        subset=[
            "competition_id",
            "season_id",
            "player_id",
            "team_id",
        ]
    )

    if output_file.exists():
        existing_df = pd.read_csv(
            output_file
        )

        required_existing_columns = {
            "competition_id",
            "season_id",
            "player_id",
            "team_id",
        }

        missing_existing_columns = (
            required_existing_columns
            - set(existing_df.columns)
        )

        if missing_existing_columns:
            raise ValueError(
                "Existing player dataset is missing required "
                "columns: "
                f"{sorted(missing_existing_columns)}"
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
            scope_key
            not in successful_scope
            for scope_key
            in existing_scope_keys
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

    combined_df = combined_df.drop_duplicates(
        subset=[
            "competition_id",
            "season_id",
            "player_id",
            "team_id",
        ],
        keep="last",
    )

    combined_df = combined_df.sort_values(
        [
            "competition",
            "season_year",
            "team",
            "player",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    print()
    print("Player Ingestion Summary")
    print("------------------------")
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
        f"Saved {len(combined_df)} player-team-season "
        f"rows to {output_file}"
    )
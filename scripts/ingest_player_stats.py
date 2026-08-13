#ingest_player_stats.py

import argparse
from time import sleep
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

from scripts.sofascore_utils import BASE_URL, OUT_DIR, get_json

REQUEST_DELAY = 0.75
CHECKPOINT_EVERY = 100

IN_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_players.csv"
OUT_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_player_stats.csv"
FAILED_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_player_stats_failed.csv"

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ingest Sofascore player statistics for selected "
            "competition-seasons."
        )
    )

    parser.add_argument(
        "--competition",
        type=str,
        help=(
            "Optional competition name from sofascore_players.csv, "
            "for example 'Bundesliga'."
        ),
    )

    parser.add_argument(
        "--season-year",
        type=str,
        help=(
            "Optional season label, for example '24/25'."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the selected player-stat tasks without making "
            "requests or writing output."
        ),
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=IN_FILE,
        help=(
            "Player-team-season input CSV. Defaults to the "
            "canonical sofascore_players.csv artifact."
        ),
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUT_FILE,
        help=(
            "Player-stat output CSV. Defaults to the canonical "
            "sofascore_player_stats.csv artifact."
        ),
    )

    parser.add_argument(
        "--failed-file",
        type=Path,
        default=FAILED_FILE,
        help=(
            "Failed-task CSV. Defaults to the canonical "
            "player-stat failure artifact."
        ),
    )

    return parser.parse_args()

def get_player_stats(row):
    url = (
        f"{BASE_URL}/player/{row['player_id']}/unique-tournament/"
        f"{row['competition_id']}/season/{row['season_id']}/statistics/overall"
    )
    return get_json(url)


def flatten_player_stats(player_row, data):
    stats = data.get("statistics") or {}

    # Sofascore returns one aggregate record per player and
    # competition-season. If a player represented multiple clubs
    # during the season, these team fields provide membership context
    # and do not imply that the statistics are split by club.

    row = {
        "competition": player_row["competition"],
        "competition_type": player_row["competition_type"],
        "competition_id": player_row["competition_id"],
        "season_id": player_row["season_id"],
        "season_year": player_row["season_year"],
        "player_id": player_row["player_id"],
        "player": player_row["player"],
        "player_slug": player_row["player_slug"],
        "team_id": player_row["team_id"],
        "team": player_row["team"],
        "team_slug": player_row["team_slug"],
    }

    for key, value in stats.items():
        if not isinstance(value, (dict, list)):
            row[key] = value

    return row


def make_task_key(row):
    return (
        int(row["competition_id"]),
        int(row["season_id"]),
        int(row["player_id"]),
    )

def normalize_task_dataframe(
    records,
    *,
    keep_last=True,
):
    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        return dataframe

    task_columns = [
        "competition_id",
        "season_id",
        "player_id",
    ]

    missing_columns = (
        set(task_columns)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Task dataframe is missing required key columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        dataframe
        .drop_duplicates(
            subset=task_columns,
            keep="last" if keep_last else "first",
        )
        .reset_index(drop=True)
    )

if __name__ == "__main__":
    args = parse_args()

    input_file = args.input_file
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

    players_df = pd.read_csv(input_file)

    if args.competition:
        players_df = players_df[
            players_df["competition"].str.casefold()
            == args.competition.casefold()
        ]

    if args.season_year:
        players_df = players_df[
            players_df["season_year"].astype(str)
            == args.season_year
        ]

    players_df = (
        players_df
        .drop_duplicates(
            subset=[
                "competition_id",
                "season_id",
                "player_id",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if players_df.empty:
        raise ValueError(
            "No player-stat tasks matched the supplied filters. "
            f"competition={args.competition!r}, "
            f"season_year={args.season_year!r}"
        )

    if args.dry_run:
        task_columns = [
            "competition",
            "competition_id",
            "season_id",
            "season_year",
            "player_id",
            "player",
            "team_id",
            "team",
        ]

        print("Selected player-stat tasks:")
        print(
            players_df[
                task_columns
            ].head(25).to_string(
                index=False
            )
        )

        print()
        print(
            f"Total selected tasks: "
            f"{len(players_df)}"
        )

        summary = (
            players_df
            .groupby(
                [
                    "competition",
                    "season_year",
                ]
            )
            .size()
            .rename("tasks")
            .reset_index()
        )

        print()
        print("Task Summary")
        print("------------")
        print(
            summary.to_string(
                index=False
            )
        )

        raise SystemExit(0)

    # Player statistics are treated as a resumable task cache.
    # Existing successful player-season tasks are preserved and skipped.
    # Failed or previously unseen tasks remain eligible for ingestion.

    all_stats = []
    failed_tasks = []
    completed_keys = set()

    if output_file.exists() and output_file.stat().st_size > 0:
        existing = pd.read_csv(output_file)
        all_stats = existing.to_dict("records")
        completed_keys.update(
            zip(
                existing["competition_id"].astype(int),
                existing["season_id"].astype(int),
                existing["player_id"].astype(int),
            )
        )
        print(f"Resuming from {len(completed_keys)} completed stat rows")

    if failed_file.exists() and failed_file.stat().st_size > 0:
        try:
            failed = pd.read_csv(
                failed_file
            )

        except EmptyDataError:
            print(
                "Existing player-stat failure file is empty; "
                "removing stale file."
            )

            failed_file.unlink()
            failed = pd.DataFrame()

        if not failed.empty:
            failed_tasks = failed.to_dict(
                "records"
            )

            print(
                f"Loaded {len(failed_tasks)} "
                "failed stat tasks for retry"
            )

    else:
        failed_tasks = []

    for idx, player in players_df.iterrows():
        task_key = make_task_key(player)

        if task_key in completed_keys:
            print(f"[{idx+1}/{len(players_df)}] Skipping completed: {player['player']}")
            continue

        failed_tasks = [
            failure
            for failure in failed_tasks
            if (
                int(failure["competition_id"]),
                int(failure["season_id"]),
                int(failure["player_id"]),
            )
            != task_key
        ]

        print(
            f"[{idx+1}/{len(players_df)}] "
            f"Scraping {player['player']} — {player['competition']} {player['season_year']}"
        )

        try:
            data = get_player_stats(player)
            flat = flatten_player_stats(player, data)

            all_stats.append(flat)
            completed_keys.add(task_key)

        except Exception as e:
            print(f"FAILED: {player['player']} -> {e}")

            failed_tasks.append({
                "competition": player["competition"],
                "competition_type": player["competition_type"],
                "competition_id": player["competition_id"],
                "season_id": player["season_id"],
                "season_year": player["season_year"],
                "player_id": player["player_id"],
                "player": player["player"],
                "team_id": player["team_id"],
                "team": player["team"],
                "error": str(e),
            })

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            stats_df = normalize_task_dataframe(
                all_stats
            )

            failed_df = normalize_task_dataframe(
                failed_tasks
            )

            stats_df.to_csv(
                output_file,
                index=False,
            )

            if failed_df.empty:
                if failed_file.exists():
                    failed_file.unlink()
            else:
                failed_df.to_csv(
                    failed_file,
                    index=False,
                )

            print(
                f"Checkpoint saved at row {idx + 1}"
            )
        sleep(REQUEST_DELAY)

    stats_df = normalize_task_dataframe(
        all_stats
    )

    failed_df = normalize_task_dataframe(
        failed_tasks
    )

    stats_df.to_csv(
        output_file,
        index=False,
    )

    if failed_df.empty:
        if failed_file.exists():
            failed_file.unlink()
    else:
        failed_df.to_csv(
            failed_file,
            index=False,
        )

    print("Done.")
    print(
        f"Successful stat rows: "
        f"{len(stats_df)}"
    )
    print(
        f"Failed tasks: "
        f"{len(failed_df)}"
    )
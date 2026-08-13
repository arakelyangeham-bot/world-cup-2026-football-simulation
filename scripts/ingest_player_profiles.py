#ingest_player_profiles.py

import argparse
from time import sleep
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

from scripts.sofascore_utils import BASE_URL, OUT_DIR, get_json


REQUEST_DELAY = 0.75
CHECKPOINT_EVERY = 100

IN_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_players.csv"
OUT_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_player_profiles.csv"
FAILED_FILE = OUT_DIR / "raw" / "sofascore" / "sofascore_player_profiles_failed.csv"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest Sofascore player profiles from "
            "a configurable player population."
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

    parser.add_argument(
        "--failed-file",
        type=Path,
        default=FAILED_FILE,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Display the selected player population "
            "without making requests."
        ),
    )

    return parser.parse_args()

def get_player_profile(player_id):
    url = f"{BASE_URL}/player/{player_id}"
    return get_json(url)


# This is a current-profile snapshot only.
# Historical club membership must come from
# sofascore_players.csv competition-season rows.

def flatten_player_profile(data):
    player = data.get("player", {})

    team = player.get("team") or {}
    country = player.get("country") or {}

    return {
        "player_id": player.get("id"),
        "player": player.get("name"),
        "player_slug": player.get("slug"),
        "position": player.get("position"),
        "positions_detailed": player.get("positionsDetailed"),
        "jersey_number": player.get("jerseyNumber"),
        "height": player.get("height"),
        "preferred_foot": player.get("preferredFoot"),
        "date_of_birth": player.get("dateOfBirth"),
        "date_of_birth_timestamp": player.get("dateOfBirthTimestamp"),
        "country": country.get("name"),
        "country_alpha2": country.get("alpha2"),
        "country_alpha3": country.get("alpha3"),
        "current_team_id": team.get("id"),
        "current_team": team.get("name"),
        "current_team_slug": team.get("slug"),
    }


if __name__ == "__main__":
    arguments = parse_arguments()

    input_file = arguments.input_file
    output_file = arguments.output_file
    failed_file = arguments.failed_file

    players_df = pd.read_csv(
        input_file
    )

    unique_players = (
        players_df[["player_id", "player"]]
        .drop_duplicates(subset=["player_id"])
        .sort_values("player")
    )

    print(
        f"Selected unique players: "
        f"{len(unique_players)}"
    )

    if arguments.dry_run:
        print()
        print(
            unique_players.to_string(
                index=False
            )
        )
        raise SystemExit(0)

    profiles = []
    failed = []
    completed_ids = set()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    failed_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_file.exists() and output_file.stat().st_size > 0:
        existing = pd.read_csv(output_file)
        profiles = existing.to_dict("records")
        completed_ids.update(existing["player_id"].dropna().astype(int).tolist())
        print(f"Resuming from {len(completed_ids)} completed profiles")

    if failed_file.exists() and failed_file.stat().st_size > 0:
        try:
            existing_failed = pd.read_csv(
                failed_file
            )

        except EmptyDataError:
            print(
                "Existing profile failure file is empty; "
                "removing stale file."
            )

            failed_file.unlink()
            existing_failed = pd.DataFrame()

        if not existing_failed.empty:
            failed = existing_failed.to_dict(
                "records"
            )

            print(
                f"Loaded {len(existing_failed)} "
                "failed profiles for retry"
            )

    for idx, row in unique_players.iterrows():
        player_id = int(row["player_id"])
        player_name = row["player"]

        if player_id in completed_ids:
            print(f"Skipping completed: {player_name}")
            continue

        failed = [
            failure
            for failure in failed
            if int(failure["player_id"]) != player_id
        ]

        print(f"[{len(profiles) + len(failed) + 1}/{len(unique_players)}] Scraping profile: {player_name}")

        try:
            data = get_player_profile(player_id)
            profile = flatten_player_profile(data)
            profiles.append(profile)
            completed_ids.add(player_id)

        except Exception as e:
            print(f"FAILED: {player_name} -> {e}")

            failed.append({
                "player_id": player_id,
                "player": player_name,
                "error": str(e),
            })

        if (len(profiles) + len(failed)) % CHECKPOINT_EVERY == 0:

            profiles_df = (
                pd.DataFrame(profiles)
                .drop_duplicates(
                    subset=["player_id"],
                    keep="last",
                )
            )

            failed_df = (
                pd.DataFrame(failed)
                .drop_duplicates(
                    subset=["player_id"],
                    keep="last",
                )
            )

            profiles_df.to_csv(
                OUT_FILE,
                index=False,
            )

            if failed_df.empty:
                if FAILED_FILE.exists():
                    FAILED_FILE.unlink()
            else:
                failed_df.to_csv(
                    FAILED_FILE,
                    index=False,
                )

            print("Checkpoint saved")

        sleep(REQUEST_DELAY)


    profiles_df = (
        pd.DataFrame(profiles)
        .drop_duplicates(
            subset=["player_id"],
            keep="last",
        )
    )

    failed_df = (
        pd.DataFrame(failed)
        .drop_duplicates(
            subset=["player_id"],
            keep="last",
        )
    )

    profiles_df.to_csv(
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
    print(f"Successful profiles: {len(profiles_df)}")
    print(f"Failed profiles: {len(failed_df)}")
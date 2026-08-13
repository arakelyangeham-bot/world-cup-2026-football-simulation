from time import sleep
import pandas as pd
from sofascore_utils import BASE_URL, OUT_DIR, get_json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROSTER_FILE = PROJECT_ROOT / "data" / "roster" / "world_cup_2026_roster_with_sofascore_ids.csv"

REQUEST_DELAY = 5
CHECKPOINT_EVERY = 100

STATS_FILE = OUT_DIR / "sofascore_wc_player_stats.csv"
FAILED_FILE = OUT_DIR / "sofascore_wc_player_stats_failed.csv"

WORLD_CUP_UNIQUE_TOURNAMENT_ID = 16
WORLD_CUP_2026_ID = 58210

def get_player_stats(player_id):
    url = (
        f"{BASE_URL}/player/{player_id}/unique-tournament/"
        f"{WORLD_CUP_UNIQUE_TOURNAMENT_ID}/season/{WORLD_CUP_2026_ID}/statistics/overall"
    )

    data = get_json(url)
    return data

def flatten_player_stats(player_row, data):
    stats = data.get("statistics") or {}

    row = {
        "player_name": player_row["player_name"],
        "player_id": player_row["sofascore_player_id"],
        "current_team": player_row["current_team"],
        "current_team_id": player_row["current_team_id"],
        "nation": player_row["nation"],
        "group": player_row["group"]
    }

    for key, value in stats.items():
        if not isinstance(value, (dict, list)):
            row[key] = value

    return row

if __name__ == "__main__":
    players_df = pd.read_csv(ROSTER_FILE)

    all_stats = []
    failed_players = []

    completed_player_ids = set()

    if STATS_FILE.exists() and STATS_FILE.stat().st_size > 0:
        existing_stats = pd.read_csv(STATS_FILE)
        all_stats = existing_stats.to_dict("records")
        completed_player_ids.update(existing_stats["player_id"].astype(int))
        print(f"Resuming from existing stats file: {len(completed_player_ids)} completed players")

    if FAILED_FILE.exists() and FAILED_FILE.stat().st_size > 0:
        existing_failed = pd.read_csv(FAILED_FILE)
        failed_players = existing_failed.to_dict("records")
        print(f"Loaded failed-player log: {len(existing_failed)} failed players")

    for idx, player in players_df.iterrows():
        player_id = int(player["sofascore_player_id"])
        player_name = player["player_name"]

        if player_id in completed_player_ids:
            print(f"[{idx+1}/{len(players_df)}] Skipping completed: {player_name}")
            continue

        print(f"[{idx+1}/{len(players_df)}] Scraping: {player_name}")

        try:
            data = get_player_stats(player_id)

            flat = flatten_player_stats(player, data)
            all_stats.append(flat)
            completed_player_ids.add(player_id)

        except Exception as e:
            print(f"FAILED: {player_name} -> {e}")

            failed_players.append({
                "player_name": player["player_name"],
                "player_id": player["sofascore_player_id"],
                "current_team": player["current_team"],
                "current_team_id": player["current_team_id"],
                "nation": player["nation"],
                "group": player["group"],
                "error": str(e),
            })
        
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(all_stats).to_csv(STATS_FILE, index=False)
            pd.DataFrame(failed_players).to_csv(FAILED_FILE, index=False)
            print(f"Checkpoint saved at player {idx+1}")
        
        sleep(REQUEST_DELAY)

    pd.DataFrame(all_stats).to_csv(STATS_FILE, index=False)
    pd.DataFrame(failed_players).to_csv(FAILED_FILE, index=False)

    print("Done.")
    print(f"Successful stat rows: {len(all_stats)}")
    print(f"Failed players: {len(failed_players)}")

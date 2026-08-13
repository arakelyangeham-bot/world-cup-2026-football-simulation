
import pandas as pd
from sofascore_utils import BASE_URL, OUT_DIR, get_json

PLAYER_ID = 750
UNIQUE_TOURNAMENT_ID = 955
SEASON_ID = 80443
OUT_FILE = OUT_DIR / "single_player_competition_stats.csv"


def get_player_competition_stats(player_id, unique_tournament_id, season_id):
    url = (
        f"{BASE_URL}/player/{player_id}/unique-tournament/"
        f"{unique_tournament_id}/season/{season_id}/statistics/overall"
    )
    return get_json(url), url


def flatten_stats(data, player_id, unique_tournament_id, season_id, url):
    stats = data.get("statistics") or {}

    row = {
        "player_id": player_id,
        "unique_tournament_id": unique_tournament_id,
        "season_id": season_id,
        "stats_url": url,
    }

    for key, value in stats.items():
        if not isinstance(value, (dict, list)):
            row[key] = value

    return row


if __name__ == "__main__":
    data, url = get_player_competition_stats(
        PLAYER_ID,
        UNIQUE_TOURNAMENT_ID,
        SEASON_ID,
    )

    flat = flatten_stats(
        data,
        PLAYER_ID,
        UNIQUE_TOURNAMENT_ID,
        SEASON_ID,
        url,
    )

    df = pd.DataFrame([flat])
    df.to_csv(OUT_FILE, index=False)

    print("Done.")
    print(f"Wrote: {OUT_FILE}")
    print(f"URL: {url}")



'''
from sofascore_utils import BASE_URL, get_json
import json


PLAYER_ID = 134029

url = f"{BASE_URL}/player/{PLAYER_ID}"
data = get_json(url)

with open("player_profile.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
'''
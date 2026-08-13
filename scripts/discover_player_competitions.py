from __future__ import annotations

"""
Discover valid Sofascore player competition rows.

Purpose
-------
This script creates the bridge table needed before scraping player stats:

    player_id + league_id + season_id

It reads the reviewed roster with Sofascore player IDs, looks at recent Sofascore
player events to discover which unique tournaments/seasons the player actually
appeared in, validates those combinations against the statistics/overall endpoint,
and saves the valid rows.

Expected project layout
-----------------------
World Cup 2026 Data Science Project/
    scripts/
        discover_player_competitions.py
        sofascore_utils.py
    data/
        roster/
            world_cup_2026_roster_with_sofascore_ids.csv
            league_and_season_ids_filled.csv
        raw/
            sofascore/
                sofascore_player_competitions.csv
                sofascore_player_competitions_candidates.csv
                sofascore_player_competitions_failed.csv

Run from project root:
    python scripts/discover_player_competitions.py
"""

from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd

from sofascore_utils import BASE_URL, OUT_DIR, get_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROSTER_FILE = PROJECT_ROOT / "data" / "roster" / "world_cup_2026_roster_with_sofascore_ids.csv"
LEAGUE_SEASON_FILE = PROJECT_ROOT / "data" / "roster" / "league_and_season_ids_filled.csv"

OUT_FILE = OUT_DIR / "sofascore_player_competitions.csv"
CANDIDATES_FILE = OUT_DIR / "sofascore_player_competitions_candidates.csv"
FAILED_FILE = OUT_DIR / "sofascore_player_competitions_failed.csv"

# Sofascore's player event endpoint is paginated. More pages = better transfer/history coverage.
# Start with 6. If too many older 2023 rows are missing, raise to 8 or 10.
MAX_EVENT_PAGES = 12
REQUEST_DELAY = 2
CHECKPOINT_EVERY = 10

# Resume behavior.
# If True, players already present in FAILED_FILE are skipped on reruns.
# Set to False when you want to retry temporary failures after a full run.
SKIP_FAILED_PLAYERS = False

# These are your model target seasons, not necessarily literal calendar-year seasons.
TARGET_SEASON_COLUMNS = ["season_id_2023", "season_id_2024", "season_id_2025"]

# Final validation endpoint. If this works, the row is safe for the stats scraper.
STATS_PATH = (
    "{base}/player/{player_id}/unique-tournament/"
    "{league_id}/season/{season_id}/statistics/overall"
)


def first_existing_column(df: pd.DataFrame, options: list[str]) -> str:
    for col in options:
        if col in df.columns:
            return col
    raise ValueError(f"None of these columns exist: {options}")


def safe_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None

def read_csv_if_valid(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def nested_get(obj: dict[str, Any], path: list[str], default: Any = "") -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def load_allowed_league_seasons() -> tuple[set[tuple[int, int]], dict[tuple[int, int], dict[str, Any]]]:
    """Return allowed (league_id, season_id) pairs from league_and_season_ids_filled.csv."""
    df = pd.read_csv(LEAGUE_SEASON_FILE)
    required = {"league_name", "league_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"League-season file is missing columns: {sorted(missing)}")

    allowed: set[tuple[int, int]] = set()
    metadata: dict[tuple[int, int], dict[str, Any]] = {}

    for _, row in df.iterrows():
        league_id = safe_int(row.get("league_id"))
        if league_id is None:
            continue

        for col in TARGET_SEASON_COLUMNS:
            season_id = safe_int(row.get(col))
            if season_id is None:
                continue

            target_year = col.replace("season_id_", "")
            key = (league_id, season_id)
            allowed.add(key)
            metadata[key] = {
                "league_name_lookup": row.get("league_name", ""),
                "target_year": target_year,
            }

    if not allowed:
        raise ValueError(
            f"No league-season pairs found in {LEAGUE_SEASON_FILE}. "
            "Run fetch_sofascore_league_season_ids.py first."
        )

    return allowed, metadata


def extract_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    events = data.get("events")
    if isinstance(events, list):
        return [e for e in events if isinstance(e, dict)]
    return []


def discover_candidates_from_events(
        player_id: int,
        base_player: dict[str, Any],
        allowed_pairs: set[tuple[int, int]],
        pair_metadata: dict[tuple[int, int], dict[str, Any]],
    ) -> list[dict[str, Any]]:
    """Use /player/{id}/events/last/{page} to discover possible league-season rows."""
    candidates_by_key: dict[tuple[int, int], dict[str, Any]] = {}

    for page in range(MAX_EVENT_PAGES):
        url = f"{BASE_URL}/player/{player_id}/events/last/{page}"
        data = get_json(url)
        events = extract_events(data)
        if not events:
            break

        for event in events:
            unique_tournament = nested_get(event, ["tournament", "uniqueTournament"], {})
            if not isinstance(unique_tournament, dict):
                continue

            league_id = safe_int(unique_tournament.get("id"))
            season_id = safe_int(nested_get(event, ["season", "id"], None))
            if league_id is None or season_id is None:
                continue

            pair_key = (league_id, season_id)
            if pair_key not in allowed_pairs:
                continue

            home_team = nested_get(event, ["homeTeam"], {})
            away_team = nested_get(event, ["awayTeam"], {})
            if not isinstance(home_team, dict):
                home_team = {}
            if not isinstance(away_team, dict):
                away_team = {}

            candidate_key = (player_id, league_id, season_id)
            if candidate_key not in candidates_by_key:
                meta = pair_metadata.get(pair_key, {})
                candidates_by_key[candidate_key] = {
                    **base_player,
                    "league_id": league_id,
                    "league_name": unique_tournament.get("name", ""),
                    "league_name_lookup": meta.get("league_name_lookup", ""),
                    "season_id": season_id,
                    "season_name": nested_get(event, ["season", "name"], ""),
                    "season_year": nested_get(event, ["season", "year"], ""),
                    "target_year": meta.get("target_year", ""),
                    "sample_event_id": event.get("id", ""),
                    "sample_event_date_ts": event.get("startTimestamp", ""),
                    "sample_home_team_id": home_team.get("id", ""),
                    "sample_home_team_name": home_team.get("name", ""),
                    "sample_away_team_id": away_team.get("id", ""),
                    "sample_away_team_name": away_team.get("name", ""),
                    "event_count_seen": 0,
                    "discovery_source": url,
                }

            candidates_by_key[candidate_key]["event_count_seen"] += 1

        sleep(REQUEST_DELAY)

    return list(candidates_by_key.values())


def validate_stats_endpoint(row: dict[str, Any]) -> tuple[bool, str, str]:
    """Confirm the stats endpoint accepts this player/league/season row."""
    player_id = safe_int(row.get("sofascore_player_id"))
    league_id = safe_int(row.get("league_id"))
    season_id = safe_int(row.get("season_id"))
    if player_id is None or league_id is None or season_id is None:
        return False, "missing_ids", ""

    url = STATS_PATH.format(
        base=BASE_URL,
        player_id=player_id,
        league_id=league_id,
        season_id=season_id,
    )

    try:
        data = get_json(url)
    except Exception as exc:
        return False, str(exc), url

    # The shape can vary. If we got a 200 and a dict back, keep the row.
    # Later stats scraper can normalize the actual stat fields.
    if isinstance(data, dict):
        return True, "validated_stats_endpoint", url
    return False, "unexpected_response_shape", url


def save_checkpoint(valid_rows, candidate_rows, failed_rows) -> None:
    """Write all progress files. Existing rows are preserved through in-memory loads on resume."""
    pd.DataFrame(valid_rows).drop_duplicates(
        subset=["sofascore_player_id", "league_id", "season_id"], keep="last"
    ).to_csv(OUT_FILE, index=False)

    pd.DataFrame(candidate_rows).drop_duplicates(
        subset=["sofascore_player_id", "league_id", "season_id"], keep="last"
    ).to_csv(CANDIDATES_FILE, index=False)

    pd.DataFrame(failed_rows).drop_duplicates(
        subset=["sofascore_player_id"], keep="last"
    ).to_csv(FAILED_FILE, index=False)


def player_ids_from_rows(rows: list[dict[str, Any]]) -> set[int]:
    ids: set[int] = set()
    for row in rows:
        player_id = safe_int(row.get("sofascore_player_id"))
        if player_id is not None:
            ids.add(player_id)
    return ids


def main() -> None:
    roster = pd.read_csv(ROSTER_FILE)

    if "roster_row_id" not in roster.columns:
        roster = roster.reset_index(names="roster_row_id")
    
    required_cols = {
        "player_name",
        "sofascore_player_id",
        "current_team",
        "current_team_id",
        "nation",
    }

    missing = required_cols - set(roster.columns)

    if missing:
        raise ValueError(
            f"Roster file is missing columns: {sorted(missing)}"
        )
    
    player_id_col = first_existing_column(
        roster,
        ["sofascore_player_id", "player_id", "sofascore_id"],
    )

    allowed_pairs, pair_metadata = load_allowed_league_seasons()
    print(f"Loaded {len(allowed_pairs)} allowed league-season pairs")

    valid_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    completed_player_ids: set[int] = set()

    existing = read_csv_if_valid(OUT_FILE)

    if not existing.empty:
        valid_rows = existing.to_dict("records")
        completed_player_ids.update(player_ids_from_rows(valid_rows))
        print(f"Loaded existing valid rows: {len(valid_rows)}")

    existing_candidates = read_csv_if_valid(CANDIDATES_FILE)

    if not existing_candidates.empty:
        candidate_rows = existing_candidates.to_dict("records")
        print(f"Loaded existing candidate rows: {len(candidate_rows)}")

    existing_failed = read_csv_if_valid(FAILED_FILE)

    if not existing_failed.empty:
        failed_rows = existing_failed.to_dict("records")

    if SKIP_FAILED_PLAYERS:
        completed_player_ids.update(player_ids_from_rows(failed_rows))

    print(f"Loaded existing failed rows: {len(failed_rows)}")


    for idx, row in roster.iterrows():
        player_id = safe_int(row.get(player_id_col))
        player_name = row.get("player_name", row.get("name", ""))
        nation = row.get("nation", "")

        if player_id is None:
            failed_rows.append({**row.to_dict(), "error": "missing_sofascore_player_id"})
            save_checkpoint(valid_rows, candidate_rows, failed_rows)
            continue

        if player_id in completed_player_ids:
            print(f"[{idx + 1}/{len(roster)}] Skipping completed: {player_name} ({player_id})")
            continue

        print(f"[{idx + 1}/{len(roster)}] Discovering competitions: {player_name} ({nation}) [{player_id}]")

        base_player = row.to_dict()
        base_player["sofascore_player_id"] = player_id

        try:
            candidates = discover_candidates_from_events(
                player_id=player_id,
                base_player=base_player,
                allowed_pairs=allowed_pairs,
                pair_metadata=pair_metadata,
            )

            if not candidates:
                failed_rows.append({**base_player, "error": "no_allowed_competitions_discovered"})
                completed_player_ids.add(player_id)
                save_checkpoint(valid_rows, candidate_rows, failed_rows)
                continue

            for cand in candidates:
                ok, status, stats_url = validate_stats_endpoint(cand)
                cand["validation_status"] = status
                cand["stats_url"] = stats_url
                candidate_rows.append(cand)

                if ok:
                    valid_rows.append(cand)

                sleep(REQUEST_DELAY)

            completed_player_ids.add(player_id)

        except Exception as exc:
            print(f"FAILED: {player_name} ({player_id}) -> {exc}")
            failed_rows.append({**base_player, "error": str(exc)})
            completed_player_ids.add(player_id)
            save_checkpoint(valid_rows, candidate_rows, failed_rows)

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(valid_rows, candidate_rows, failed_rows)
            print(f"Checkpoint saved at roster row {idx + 1}")

        sleep(REQUEST_DELAY)

    save_checkpoint(valid_rows, candidate_rows, failed_rows)

    print("Done.")
    print(f"Valid competition rows: {len(valid_rows)}")
    print(f"Candidate rows checked: {len(candidate_rows)}")
    print(f"Failed/player review rows: {len(failed_rows)}")
    print(f"Output: {OUT_FILE}")
    print(f"Candidates: {CANDIDATES_FILE}")
    print(f"Failed: {FAILED_FILE}")


if __name__ == "__main__":
    main()

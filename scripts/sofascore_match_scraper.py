# sofascore_match_scraper.py  —  Stage 2 of the prediction pipeline
#
# Scrapes all WC 2026 fixtures and per-match team statistics from Sofascore.
#
# Outputs (both resumable):
#   data/raw/sofascore/wc_2026_match_results.csv
#       One row per match. Columns: event_id, date, round, stage,
#       home_team, home_team_id, away_team, away_team_id,
#       home_score, away_score, status, winner
#
#   data/raw/sofascore/wc_2026_match_team_stats.csv
#       One row per team per match (2 rows per game).
#       All statistical items Sofascore returns for the full match period.
#
# The 2026 WC has 104 matches total. Sofascore paginates events at 20
# per page, so we walk pages until we've seen all fixtures.
#
# Run this after each matchday to keep results current. The scraper is
# fully resumable — completed event IDs are skipped on re-run.

from time import sleep
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from sofascore_utils import BASE_URL, get_json
from pandas.errors import EmptyDataError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sofascore"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE   = RAW_DIR / "wc_2026_match_results.csv"
TEAM_STATS_FILE= RAW_DIR / "wc_2026_match_team_stats.csv"
FAILED_FILE    = RAW_DIR / "wc_2026_match_stats_failed.csv"

WORLD_CUP_UNIQUE_TOURNAMENT_ID = 16
WORLD_CUP_2026_ID              = 58210

# Seconds between requests — be conservative; match stats are a secondary
# endpoint and less likely to be cached by Sofascore's CDN.
REQUEST_DELAY = 3

# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------

def fetch_events_page(page: int, direction: str) -> dict:
    """
    direction: "last" (completed) or "next" (upcoming)
    Returns raw JSON from Sofascore.
    """
    url = (
        f"{BASE_URL}/unique-tournament/{WORLD_CUP_UNIQUE_TOURNAMENT_ID}"
        f"/season/{WORLD_CUP_2026_ID}/events/{direction}/{page}"
    )
    return get_json(url)


def fetch_all_events() -> list[dict]:
    """
    Pages through both past and upcoming events until Sofascore returns
    empty pages, then deduplicates by event ID.
    """
    all_events = {}

    for direction in ("last", "next"):
        page = 0
        while True:
            try:
                data = fetch_events_page(page, direction)
            except Exception as e:
                print(f"  [{direction} p{page}] fetch failed: {e}")
                break

            events = data.get("events", [])
            if not events:
                break

            for ev in events:
                all_events[ev["id"]] = ev

            print(f"  [{direction} p{page}] {len(events)} events (total unique so far: {len(all_events)})")

            # Sofascore returns fewer than 20 on the final page
            if len(events) < 20:
                break

            page += 1
            sleep(REQUEST_DELAY)

    return list(all_events.values())


def flatten_event(ev: dict) -> dict:
    """Extract match metadata from a raw Sofascore event object."""
    home = ev.get("homeTeam", {})
    away = ev.get("awayTeam", {})
    home_score = ev.get("homeScore", {})
    away_score = ev.get("awayScore", {})
    tournament = ev.get("tournament", {})
    round_info = ev.get("roundInfo", {})
    status     = ev.get("status", {})

    # Convert Unix timestamp to UTC datetime string
    ts = ev.get("startTimestamp")
    date_str = (
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        if ts else None
    )

    # Determine winner
    winner_code = ev.get("winnerCode")  # 1=home, 2=away, 3=draw
    winner_map  = {1: home.get("name"), 2: away.get("name"), 3: "Draw"}
    winner      = winner_map.get(winner_code)

    return {
        "event_id":     ev["id"],
        "date":         date_str,
        "stage":        tournament.get("name"),
        "round":        round_info.get("name") or round_info.get("round"),
        "round_number": round_info.get("round"),
        "home_team":    home.get("name"),
        "home_team_id": home.get("id"),
        "away_team":    away.get("name"),
        "away_team_id": away.get("id"),
        "home_score":   home_score.get("current"),
        "away_score":   away_score.get("current"),
        "status_code":  status.get("code"),
        "status_desc":  status.get("description"),
        "winner":       winner,
    }


def fetch_match_team_stats(event_id: int) -> list[dict]:
    """
    Fetches per-team statistics for a finished match.

    Sofascore returns statistics grouped by period (ALL, 1ST, 2ND) and
    category (Match overview, Shots, Duels, etc.). We extract the ALL
    period only and flatten into one row per team.
    """
    url = f"{BASE_URL}/event/{event_id}/statistics"
    data = get_json(url)

    stats_periods = data.get("statistics", [])

    # Find the full-match period
    full_match = next(
        (p for p in stats_periods if p.get("period", "").upper() == "ALL"),
        None,
    )
    if not full_match:
        return []

    home_row = {"event_id": event_id, "side": "home"}
    away_row = {"event_id": event_id, "side": "away"}

    for group in full_match.get("groups", []):
        for item in group.get("statisticsItems", []):
            key = (
                item.get("key")
                or item.get("name", "unknown").lower().replace(" ", "_")
            )
            home_row[key] = item.get("homeValue")
            away_row[key] = item.get("awayValue")

    return [home_row, away_row]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load any existing progress ---
    completed_stats_ids: set[int] = set()
    all_team_stats: list[dict]    = []
    failed: list[dict]            = []

    if TEAM_STATS_FILE.exists() and TEAM_STATS_FILE.stat().st_size > 0:
        try:
            existing = pd.read_csv(TEAM_STATS_FILE)
            if "event_id" in existing.columns:
                all_team_stats = existing.to_dict("records")
                completed_stats_ids = set(existing["event_id"].astype(int).unique())
                print(f"Resuming: {len(completed_stats_ids)} matches already have team stats")
            else:
                print(f"Team stats file missing event_id; starting stats resume fresh.")
        except EmptyDataError:
            print(f"Team stats file exists but is empty/corrupt; starting stats resume fresh.")

    if FAILED_FILE.exists() and FAILED_FILE.stat().st_size > 0:
        try:
            failed = pd.read_csv(FAILED_FILE).to_dict("records")
            print(f"Loaded failed log: {len(failed)} entries")
        except EmptyDataError:
            print(f"Failed log exists but is empty/corrupt; starting fresh: {FAILED_FILE}")
            failed = []

    # --- Step 1: fetch all fixture metadata ---
    print("\nFetching all WC 2026 fixtures...")
    events = fetch_all_events()
    print(f"Total fixtures found: {len(events)}")

    results = [flatten_event(ev) for ev in events]
    results_df = pd.DataFrame(results).sort_values("date")
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Match results saved → {RESULTS_FILE}")

    # --- Step 2: fetch team stats for completed matches ---
    # status_code 100 = finished in Sofascore
    finished = results_df[results_df["status_code"] == 100].copy()
    pending  = finished[~finished["event_id"].isin(completed_stats_ids)]

    print(f"\nFinished matches : {len(finished)}")
    print(f"Stats already fetched : {len(completed_stats_ids)}")
    print(f"Remaining to fetch    : {len(pending)}\n")

    for i, (_, match) in enumerate(pending.iterrows(), 1):
        event_id  = int(match["event_id"])
        matchup   = f"{match['home_team']} vs {match['away_team']}"

        print(f"[{i}/{len(pending)}] {match['date']}  {matchup}")

        try:
            rows = fetch_match_team_stats(event_id)

            if not rows:
                raise ValueError("No statistics returned (match may not have stats yet)")

            # Attach match context to each team row
            for row in rows:
                side = row["side"]
                row["home_team"]    = match["home_team"]
                row["away_team"]    = match["away_team"]
                row["home_team_id"] = match["home_team_id"]
                row["away_team_id"] = match["away_team_id"]
                row["team"]         = match["home_team"] if side == "home" else match["away_team"]
                row["team_id"]      = match["home_team_id"] if side == "home" else match["away_team_id"]
                row["opponent"]     = match["away_team"] if side == "home" else match["home_team"]
                row["date"]         = match["date"]
                row["round"]        = match["round"]
                row["stage"]        = match["stage"]
                row["goals_scored"] = match["home_score"] if side == "home" else match["away_score"]
                row["goals_conceded"]= match["away_score"] if side == "home" else match["home_score"]
                row["result"]       = (
                    "win"  if match["winner"] == row["team"] else
                    "draw" if match["winner"] == "Draw"       else
                    "loss"
                )

            all_team_stats.extend(rows)
            completed_stats_ids.add(event_id)

        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append({
                "event_id": event_id,
                "matchup":  matchup,
                "date":     match["date"],
                "error":    str(e),
            })

        sleep(REQUEST_DELAY)

    # --- Save ---
    pd.DataFrame(all_team_stats).to_csv(TEAM_STATS_FILE, index=False)
    pd.DataFrame(failed).to_csv(FAILED_FILE, index=False)

    print(f"\nDone.")
    print(f"  Match results   → {RESULTS_FILE}  ({len(results_df)} rows)")
    print(f"  Team stats rows → {TEAM_STATS_FILE}  ({len(all_team_stats)} rows)")
    print(f"  Failed          → {FAILED_FILE}  ({len(failed)} entries)")

    # --- Quick summary ---
    if all_team_stats:
        stats_df = pd.DataFrame(all_team_stats)
        print(f"\nColumns in team stats: {len(stats_df.columns)}")

        # Show which xG-related columns came back
        xg_cols = [c for c in stats_df.columns if "xg" in c.lower() or "expected" in c.lower()]
        print(f"xG-related columns: {xg_cols if xg_cols else 'none found — check column names'}")
        print(f"\nSample columns: {list(stats_df.columns[:20])}")


if __name__ == "__main__":
    main()
# sofascore_historical_scraper.py

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
from scripts.sofascore_utils import BASE_URL, get_json
from pandas.errors import EmptyDataError
from shared.sofascore_season_loader import (
    build_season_lookup,
    load_sofascore_seasons,
)
from shared.competition_registry import get_competition
import argparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sofascore"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE   = RAW_DIR / "wc_2026_match_results.csv"
TEAM_STATS_FILE= RAW_DIR / "wc_2026_match_team_stats.csv"
FAILED_FILE    = RAW_DIR / "wc_2026_match_stats_failed.csv"

# Seconds between requests — be conservative; match stats are a secondary
# endpoint and less likely to be cached by Sofascore's CDN.
REQUEST_DELAY = 3
SEASON_LOOKUP = build_season_lookup()
# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------

def fetch_events_page(
    unique_tournament_id: int,
    season_id: int,
    page: int,
    direction: str,
) -> dict:
    url = (
        f"{BASE_URL}/unique-tournament/{unique_tournament_id}"
        f"/season/{season_id}/events/{direction}/{page}"
    )
    return get_json(url)


def fetch_all_events(
    unique_tournament_id: int,
    season_id: int,
) -> list[dict]:
    all_events = {}

    for direction in ("last", "next"):
        page = 0

        while True:
            try:
                data = fetch_events_page(
                    unique_tournament_id,
                    season_id,
                    page,
                    direction,
                )
            except Exception as e:
                print(f"  [{direction} p{page}] fetch failed: {e}")
                break

            events = data.get("events", [])
            if not events:
                break

            for ev in events:
                all_events[ev["id"]] = ev

            print(
                f"  [{direction} p{page}] {len(events)} events "
                f"(total unique so far: {len(all_events)})"
            )

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

def select_seasons(
    competition_key: str,
    year: int | None,
) -> list:
    """
    Select registered Sofascore seasons for one competition.

    When year is omitted, all registered seasons for the selected
    competition are returned.

    When year is supplied, exactly one matching season must exist.
    """

    competition_key = (
        competition_key
        .strip()
        .lower()
    )

    seasons = load_sofascore_seasons()

    competition_seasons = [
        season
        for season in seasons
        if season.competition_key == competition_key
    ]

    if not competition_seasons:
        known_competitions = sorted(
            {
                season.competition_key
                for season in seasons
            }
        )

        raise ValueError(
            f"No Sofascore seasons are registered for "
            f"competition {competition_key!r}. "
            f"Registered competition keys: "
            f"{known_competitions}"
        )

    if year is not None:
        key = (
            competition_key,
            year,
        )

        try:
            return [
                SEASON_LOOKUP[key]
            ]

        except KeyError as exc:
            available_years = sorted(
                season.year
                for season in competition_seasons
            )

            raise ValueError(
                f"No registered season found for "
                f"{competition_key!r} and year {year}. "
                f"Available years: {available_years}"
            ) from exc

    return sorted(
        competition_seasons,
        key=lambda season: season.year,
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build historical Sofascore match-result datasets "
            "for registered competitions and seasons."
        )
    )

    parser.add_argument(
        "--competition",
        required=True,
        help=(
            "Registered competition key, such as "
            "'premier_league', 'world_cup', or 'euro'."
        ),
    )

    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            "Optional season start year. "
            "For domestic leagues, 2024 means 2024–25. "
            "When omitted, all registered seasons for the "
            "selected competition are processed."
        ),
    )

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    arguments = parse_arguments()

    selected_seasons = select_seasons(
        competition_key=arguments.competition,
        year=arguments.year,
    )

    competition = get_competition(
        arguments.competition
    )

    print("Sofascore Competition Dataset Builder")
    print("=====================================")
    print(
        f"Competition: "
        f"{competition.display_name} "
        f"({competition.key})"
    )
    print(
        "Selected seasons: "
        f"{[season.year for season in selected_seasons]}"
    )
    print()

    for season in selected_seasons:
        year = season.year
        season_id = season.season_id
        unique_tournament_id = (
            season.unique_tournament_id
        )

        print()
        print("=" * 60)
        print(
            f"Fetching {season.dataset_id} "
            f"({competition.display_name})"
        )
        print(
            "Unique tournament ID: "
            f"{unique_tournament_id}"
        )
        print(f"Season ID: {season_id}")
        print("=" * 60)

        events = fetch_all_events(
            unique_tournament_id=(
                unique_tournament_id
            ),
            season_id=season_id,
        )

        print(
            f"Total unique fixtures found: "
            f"{len(events)}"
        )

        if not events:
            raise RuntimeError(
                f"No events were returned for "
                f"{season.dataset_id}."
            )

        results = [
            flatten_event(event)
            for event in events
        ]

        results_df = pd.DataFrame(results)

        if results_df.empty:
            raise RuntimeError(
                f"Flattening produced an empty dataset "
                f"for {season.dataset_id}."
            )

        results_df = (
            results_df
            .sort_values(
                ["date", "event_id"],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        if results_df["event_id"].duplicated().any():
            duplicate_ids = (
                results_df.loc[
                    results_df[
                        "event_id"
                    ].duplicated(
                        keep=False
                    ),
                    "event_id",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Duplicate event IDs remain after collection: "
                f"{duplicate_ids[:20]}"
            )

        filename = (
            competition
            .filename_pattern
            .format(year=year)
        )

        output_path = RAW_DIR / filename

        results_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )

        completed_count = int(
            results_df["home_score"]
            .notna()
            .mul(
                results_df["away_score"].notna()
            )
            .sum()
        )

        unfinished_count = (
            len(results_df)
            - completed_count
        )

        print()
        print("Dataset Summary")
        print("---------------")
        print(
            f"Dataset ID: "
            f"{season.dataset_id}"
        )
        print(
            f"Rows written: "
            f"{len(results_df)}"
        )
        print(
            f"Completed-score rows: "
            f"{completed_count}"
        )
        print(
            f"Unfinished-score rows: "
            f"{unfinished_count}"
        )
        print(f"Output: {output_path}")
        print()

    print(
        "All selected competition datasets "
        "were written successfully."
    )

if __name__ == "__main__":
    main()
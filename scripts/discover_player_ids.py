from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import quote_plus

import pandas as pd

from sofascore_utils import BASE_URL, OUT_DIR, get_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROSTER_FILE = PROJECT_ROOT / "data" / "roster" / "world_cup_2026_roster.csv"
OVERRIDES_FILE = PROJECT_ROOT / "data" / "roster" / "sofascore_player_id_overrides.csv"

ACCEPTED_FILE = PROJECT_ROOT / "data" / "roster" / "world_cup_2026_roster_with_sofascore_ids.csv"
REVIEW_FILE = OUT_DIR / "sofascore_player_id_review.csv"
CANDIDATES_FILE = OUT_DIR / "sofascore_player_id_candidates.csv"
FAILED_FILE = OUT_DIR / "sofascore_player_id_failed.csv"

REQUEST_DELAY = 1.5
CHECKPOINT_EVERY = 25
MAX_CANDIDATES_PER_PLAYER = 8

# Tune these if the review file is too large or too aggressive.
AUTO_ACCEPT_SCORE = 0.92
AUTO_ACCEPT_GAP = 0.08
EXACT_NAME_SCORE = 0.985

TEAM_COLS = ["team_2025", "team_2024", "team_2023"]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = (
        text.replace("ø", "o")
        .replace("ö", "o")
        .replace("ó", "o")
        .replace("ò", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ä", "a")
        .replace("á", "a")
        .replace("à", "a")
        .replace("â", "a")
        .replace("ã", "a")
        .replace("å", "a")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("ë", "e")
        .replace("í", "i")
        .replace("ì", "i")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ú", "u")
        .replace("ù", "u")
        .replace("û", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
        .replace("ç", "c")
        .replace("ğ", "g")
        .replace("ş", "s")
        .replace("ı", "i")
        .replace("ł", "l")
        .replace("đ", "d")
        .replace("ð", "d")
        .replace("þ", "th")
        .replace("æ", "ae")
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_similarity(a: Any, b: Any) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def token_overlap(a: Any, b: Any) -> float:
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def walk_json(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_json(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_json(value)


def extract_player_entity(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Return a Sofascore player-like entity from changing search response shapes."""
    entity = obj.get("entity", obj)

    if isinstance(entity, dict) and isinstance(entity.get("player"), dict):
        entity = entity["player"]

    if not isinstance(entity, dict):
        return None

    entity_type = str(entity.get("type", obj.get("type", ""))).lower()
    has_player_fields = entity.get("id") is not None and entity.get("name") is not None

    # Sofascore search/all commonly includes entity.type == "player"; some shapes omit type.
    if entity_type and entity_type != "player":
        return None
    if not has_player_fields:
        return None

    # Avoid tournament/team rows that only look player-like because they have id/name.
    sport = entity.get("sport")
    if isinstance(sport, dict):
        sport_name = normalize_text(sport.get("name", ""))
        if sport_name and sport_name not in {"football", "soccer"}:
            return None

    return entity


def player_candidates_from_search(data: dict[str, Any], roster_row: pd.Series) -> list[dict[str, Any]]:
    seen_ids: set[int] = set()
    candidates: list[dict[str, Any]] = []

    target_name = roster_row.get("player_name", "")
    target_nation = roster_row.get("nation", "")
    target_teams = [roster_row.get(col, "") for col in TEAM_COLS if col in roster_row.index]

    for obj in walk_json(data):
        player = extract_player_entity(obj)
        if not player:
            continue

        player_id = player.get("id")
        try:
            player_id = int(player_id)
        except (TypeError, ValueError):
            continue
        if player_id in seen_ids:
            continue
        seen_ids.add(player_id)

        team = player.get("team") or obj.get("team") or {}
        country = player.get("country") or obj.get("country") or {}
        if not isinstance(team, dict):
            team = {}
        if not isinstance(country, dict):
            country = {}

        player_name = player.get("name", "")
        candidate_team = team.get("name", "")
        candidate_country = country.get("name", "")

        n_score = name_similarity(target_name, player_name)
        club_score = max([token_overlap(t, candidate_team) for t in target_teams] or [0.0])
        nation_score = token_overlap(target_nation, candidate_country)

        # Name is by far the most important field. Club/country are only tie-breakers
        # because Sofascore search sometimes returns stale/missing team metadata.
        score = n_score + (0.05 * club_score) + (0.03 * nation_score)

        candidates.append(
            {
                "candidate_rank": None,
                "candidate_score": round(score, 4),
                "name_score": round(n_score, 4),
                "club_score": round(club_score, 4),
                "nation_score": round(nation_score, 4),
                "sofascore_player_id": player_id,
                "sofascore_player_name": player_name,
                "sofascore_slug": player.get("slug", ""),
                "sofascore_team_id": team.get("id", ""),
                "sofascore_team_name": candidate_team,
                "sofascore_country": candidate_country,
            }
        )

    candidates.sort(key=lambda x: x["candidate_score"], reverse=True)
    for rank, cand in enumerate(candidates, start=1):
        cand["candidate_rank"] = rank
    return candidates[:MAX_CANDIDATES_PER_PLAYER]


def search_player(player_name: str) -> tuple[list[dict[str, Any]], str]:
    url = f"{BASE_URL}/search/all?q={quote_plus(player_name)}"
    data = get_json(url)
    return data, url


def auto_accept(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    if not candidates:
        return None, "no_candidates"

    best = candidates[0]
    second_score = candidates[1]["candidate_score"] if len(candidates) > 1 else 0.0
    gap = best["candidate_score"] - second_score

    if best["name_score"] >= EXACT_NAME_SCORE:
        return best, "auto_exact_name"
    if best["candidate_score"] >= AUTO_ACCEPT_SCORE and gap >= AUTO_ACCEPT_GAP:
        return best, "auto_high_confidence"
    return None, "needs_review"


def load_overrides() -> dict[tuple[str, str], dict[str, Any]]:
    if not OVERRIDES_FILE.exists() or OVERRIDES_FILE.stat().st_size == 0:
        return {}

    df = pd.read_csv(OVERRIDES_FILE)
    required = {"player_name", "nation", "sofascore_player_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Overrides file is missing columns: {sorted(missing)}")

    overrides = {}
    for _, row in df.iterrows():
        key = (normalize_text(row["player_name"]), normalize_text(row["nation"]))
        overrides[key] = row.to_dict()
    return overrides


def add_match_columns(base: dict[str, Any], candidate: dict[str, Any] | None, status: str, source_url: str = "") -> dict[str, Any]:
    out = dict(base)
    out["lookup_status"] = status
    out["search_url"] = source_url

    fields = [
        "candidate_score",
        "name_score",
        "club_score",
        "nation_score",
        "sofascore_player_id",
        "sofascore_player_name",
        "sofascore_slug",
        "sofascore_team_id",
        "sofascore_team_name",
        "sofascore_country",
    ]
    for field in fields:
        out[field] = candidate.get(field, "") if candidate else ""
    return out


def save_checkpoint(accepted_rows, review_rows, candidate_rows, failed_rows) -> None:
    pd.DataFrame(accepted_rows).to_csv(ACCEPTED_FILE, index=False)
    pd.DataFrame(review_rows).to_csv(REVIEW_FILE, index=False)
    pd.DataFrame(candidate_rows).to_csv(CANDIDATES_FILE, index=False)
    pd.DataFrame(failed_rows).to_csv(FAILED_FILE, index=False)


def main() -> None:
    roster = pd.read_csv(ROSTER_FILE)
    roster = roster.reset_index(names="roster_row_id")

    overrides = load_overrides()

    accepted_rows = []
    review_rows = []
    candidate_rows = []
    failed_rows = []
    completed_ids: set[int] = set()

    if ACCEPTED_FILE.exists() and ACCEPTED_FILE.stat().st_size > 0:
        existing = pd.read_csv(ACCEPTED_FILE)
        accepted_rows = existing.to_dict("records")
        if "roster_row_id" in existing.columns:
            completed_ids.update(existing["roster_row_id"].dropna().astype(int).tolist())
        print(f"Loaded existing accepted file: {len(accepted_rows)} rows")

    if REVIEW_FILE.exists() and REVIEW_FILE.stat().st_size > 0:
        existing = pd.read_csv(REVIEW_FILE)
        review_rows = existing.to_dict("records")
        if "roster_row_id" in existing.columns:
            completed_ids.update(existing["roster_row_id"].dropna().astype(int).tolist())
        print(f"Loaded existing review file: {len(review_rows)} rows")

    if CANDIDATES_FILE.exists() and CANDIDATES_FILE.stat().st_size > 0:
        existing = pd.read_csv(CANDIDATES_FILE)
        candidate_rows = existing.to_dict("records")
        print(f"Loaded existing candidate file: {len(candidate_rows)} rows")

    if FAILED_FILE.exists() and FAILED_FILE.stat().st_size > 0:
        existing = pd.read_csv(FAILED_FILE)
        failed_rows = existing.to_dict("records")
        if "roster_row_id" in existing.columns:
            completed_ids.update(existing["roster_row_id"].dropna().astype(int).tolist())
        print(f"Loaded existing failed file: {len(failed_rows)} rows")

    for idx, row in roster.iterrows():
        roster_row_id = int(row["roster_row_id"])
        if roster_row_id in completed_ids:
            print(f"[{idx + 1}/{len(roster)}] Skipping completed: {row['player_name']}")
            continue

        base = row.to_dict()
        player_name = str(row["player_name"])
        nation = str(row.get("nation", ""))
        override_key = (normalize_text(player_name), normalize_text(nation))

        print(f"[{idx + 1}/{len(roster)}] Searching Sofascore player ID: {player_name} ({nation})")

        if override_key in overrides:
            override = overrides[override_key]
            candidate = {
                "candidate_score": 1.0,
                "name_score": 1.0,
                "club_score": "",
                "nation_score": "",
                "sofascore_player_id": int(override["sofascore_player_id"]),
                "sofascore_player_name": override.get("sofascore_player_name", player_name),
                "sofascore_slug": override.get("sofascore_slug", ""),
                "sofascore_team_id": override.get("sofascore_team_id", ""),
                "sofascore_team_name": override.get("sofascore_team_name", ""),
                "sofascore_country": override.get("sofascore_country", ""),
            }
            accepted_rows.append(add_match_columns(base, candidate, "manual_override"))
            completed_ids.add(roster_row_id)
            continue

        try:
            data, search_url = search_player(player_name)
            candidates = player_candidates_from_search(data, row)

            for cand in candidates:
                candidate_rows.append({**base, **cand, "search_url": search_url})

            accepted, status = auto_accept(candidates)
            if accepted:
                accepted_rows.append(add_match_columns(base, accepted, status, search_url))
            else:
                # Put the best candidate on the roster-level review row for easier filtering.
                review_rows.append(add_match_columns(base, candidates[0] if candidates else None, status, search_url))

            completed_ids.add(roster_row_id)

        except Exception as exc:
            print(f"FAILED: {player_name} -> {exc}")
            failed_rows.append({**base, "error": str(exc)})
            if "Permanent HTTP" in str(exc):
                completed_ids.add(roster_row_id)

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(accepted_rows, review_rows, candidate_rows, failed_rows)
            print(f"Checkpoint saved at roster row {idx + 1}")

        sleep(REQUEST_DELAY)

    save_checkpoint(accepted_rows, review_rows, candidate_rows, failed_rows)

    print("Done.")
    print(f"Accepted rows: {len(accepted_rows)}")
    print(f"Review rows: {len(review_rows)}")
    print(f"Candidate rows: {len(candidate_rows)}")
    print(f"Failed rows: {len(failed_rows)}")
    print(f"Accepted output: {ACCEPTED_FILE}")
    print(f"Review output: {REVIEW_FILE}")


if __name__ == "__main__":
    main()

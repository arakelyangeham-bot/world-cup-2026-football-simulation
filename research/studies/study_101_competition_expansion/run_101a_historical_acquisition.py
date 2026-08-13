#run_101a_historical_acquisition

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STUDY_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_101a_competition_expansion"
)

PLAYER_INPUT_PATH = (
    STUDY_DIRECTORY
    / "candidate_players.csv"
)

PILOT_STATS_PATH = (
    STUDY_DIRECTORY
    / "pilot_24_25_player_stats.csv"
)

CANDIDATE_STATS_PATH = (
    STUDY_DIRECTORY
    / "candidate_player_stats.csv"
)

FAILED_PATH = (
    STUDY_DIRECTORY
    / "candidate_player_stats_failed.csv"
)

COMPETITIONS = (
    "Eredivisie",
    "Liga Portugal",
)

SEASONS = (
    "21/22",
    "22/23",
    "23/24",
    "24/25",
    "25/26",
)


def validate_inputs() -> None:
    required = (
        PLAYER_INPUT_PATH,
        PILOT_STATS_PATH,
    )

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Study 101A inputs:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


def seed_candidate_cache() -> None:
    if CANDIDATE_STATS_PATH.exists():
        print(
            "Candidate stat cache already exists; "
            "preserving it for resumability."
        )
        return

    shutil.copy2(
        PILOT_STATS_PATH,
        CANDIDATE_STATS_PATH,
    )

    print(
        "Seeded candidate stat cache from "
        "the validated 24/25 pilot."
    )


def run_scope(
    *,
    competition: str,
    season: str,
) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.ingest_player_stats",
        "--input-file",
        str(PLAYER_INPUT_PATH),
        "--output-file",
        str(CANDIDATE_STATS_PATH),
        "--failed-file",
        str(FAILED_PATH),
        "--competition",
        competition,
        "--season-year",
        season,
    ]

    print()
    print("=" * 88)
    print(
        f"STUDY 101A ACQUISITION — "
        f"{competition} {season}"
    )
    print("=" * 88)

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Player-stat ingestion failed for "
            f"{competition} {season} "
            f"with exit code "
            f"{completed.returncode}."
        )


def validate_final_cache() -> None:
    stats = pd.read_csv(
        CANDIDATE_STATS_PATH,
        dtype={
            "season_year": str,
        },
        low_memory=False,
    )

    expected_scopes = {
        (competition, season)
        for competition in COMPETITIONS
        for season in SEASONS
    }

    observed_scopes = set(
        zip(
            stats["competition"].astype(str),
            stats["season_year"].astype(str),
        )
    )

    missing_scopes = (
        expected_scopes
        - observed_scopes
    )

    unexpected_scopes = (
        observed_scopes
        - expected_scopes
    )

    if missing_scopes:
        raise AssertionError(
            "Candidate stat cache is missing scopes: "
            f"{sorted(missing_scopes)}"
        )

    if unexpected_scopes:
        raise AssertionError(
            "Candidate stat cache contains unexpected scopes: "
            f"{sorted(unexpected_scopes)}"
        )

    duplicate_count = int(
        stats.duplicated(
            [
                "competition_id",
                "season_id",
                "player_id",
            ]
        ).sum()
    )

    if duplicate_count:
        raise AssertionError(
            "Candidate stat cache contains "
            f"{duplicate_count} duplicate task keys."
        )

    scope_summary = (
        stats.groupby(
            [
                "competition",
                "season_year",
            ]
        )
        .agg(
            rows=("player_id", "size"),
            players=(
                "player_id",
                "nunique",
            ),
        )
        .sort_index()
    )

    print()
    print("Final Study 101A acquisition")
    print("-" * 88)
    print(
        scope_summary.to_string()
    )

    print()
    print(
        f"Successful rows: {len(stats)}"
    )
    print(
        f"Unique players: "
        f"{stats['player_id'].nunique()}"
    )
    print(
        "Duplicate task keys: 0"
    )

    if FAILED_PATH.exists():
        try:
            failed = pd.read_csv(
                FAILED_PATH
            )
        except pd.errors.EmptyDataError:
            failed = pd.DataFrame()

        print(
            f"Outstanding failed tasks: "
            f"{len(failed)}"
        )

    else:
        print(
            "Outstanding failed tasks: 0"
        )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 101A — HISTORICAL "
        "COMPETITION EXPANSION ACQUISITION"
    )
    print("=" * 88)

    validate_inputs()
    seed_candidate_cache()

    for competition in COMPETITIONS:
        for season in SEASONS:
            run_scope(
                competition=competition,
                season=season,
            )

    validate_final_cache()

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
#run_101b_big_five_backfill

from __future__ import annotations

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

MANIFEST_PATH = (
    STUDY_DIRECTORY
    / "candidate_competition_manifest.csv"
)

CANONICAL_STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_stats.csv"
)

BACKFILL_PLAYERS_PATH = (
    STUDY_DIRECTORY
    / "big_five_backfill_players.csv"
)

BACKFILL_PLAYER_FAILURES_PATH = (
    STUDY_DIRECTORY
    / "big_five_backfill_player_failures.csv"
)

BACKFILL_STATS_PATH = (
    STUDY_DIRECTORY
    / "big_five_backfill_player_stats.csv"
)

BACKFILL_STATS_FAILURES_PATH = (
    STUDY_DIRECTORY
    / "big_five_backfill_player_stats_failed.csv"
)

BIG_FIVE = (
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
)

TARGET_SEASONS = (
    "21/22",
    "22/23",
    "23/24",
    "24/25",
    "25/26",
)


def run_command(
    command: list[str],
    *,
    label: str,
) -> None:
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code "
            f"{completed.returncode}."
        )


def discover_missing_scopes() -> pd.DataFrame:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype={"season_year": str},
    )

    canonical = pd.read_csv(
        CANONICAL_STATS_PATH,
        dtype={"season_year": str},
        low_memory=False,
    )

    expected = manifest.loc[
        manifest["competition"].isin(
            BIG_FIVE
        )
        & manifest["season_year"].isin(
            TARGET_SEASONS
        ),
        [
            "competition",
            "competition_id",
            "season_id",
            "season_year",
        ],
    ].drop_duplicates()

    if len(expected) != 25:
        raise AssertionError(
            "Expected 25 Big Five competition-season "
            f"scopes, found {len(expected)}."
        )

    observed = canonical[
        [
            "competition",
            "competition_id",
            "season_id",
            "season_year",
        ]
    ].drop_duplicates()

    audit = expected.merge(
        observed,
        on=[
            "competition",
            "competition_id",
            "season_id",
            "season_year",
        ],
        how="left",
        indicator=True,
    )

    missing = audit.loc[
        audit["_merge"].eq(
            "left_only"
        ),
        [
            "competition",
            "competition_id",
            "season_id",
            "season_year",
        ],
    ].copy()

    return missing.sort_values(
        [
            "competition",
            "season_year",
        ]
    ).reset_index(drop=True)


def acquire_player_memberships(
    missing_scopes: pd.DataFrame,
) -> None:
    for row in missing_scopes.itertuples(
        index=False
    ):
        command = [
            sys.executable,
            "-m",
            "scripts.ingest_players",
            "--manifest-file",
            str(MANIFEST_PATH),
            "--output-file",
            str(BACKFILL_PLAYERS_PATH),
            "--failed-file",
            str(
                BACKFILL_PLAYER_FAILURES_PATH
            ),
            "--competition",
            str(row.competition),
            "--season-year",
            str(row.season_year),
        ]

        run_command(
            command,
            label=(
                "BIG FIVE PLAYER MEMBERSHIP — "
                f"{row.competition} "
                f"{row.season_year}"
            ),
        )


def acquire_player_stats(
    missing_scopes: pd.DataFrame,
) -> None:
    for row in missing_scopes.itertuples(
        index=False
    ):
        command = [
            sys.executable,
            "-m",
            "scripts.ingest_player_stats",
            "--input-file",
            str(BACKFILL_PLAYERS_PATH),
            "--output-file",
            str(BACKFILL_STATS_PATH),
            "--failed-file",
            str(
                BACKFILL_STATS_FAILURES_PATH
            ),
            "--competition",
            str(row.competition),
            "--season-year",
            str(row.season_year),
        ]

        run_command(
            command,
            label=(
                "BIG FIVE PLAYER STATS — "
                f"{row.competition} "
                f"{row.season_year}"
            ),
        )


def validate_backfill(
    missing_scopes: pd.DataFrame,
) -> None:
    stats = pd.read_csv(
        BACKFILL_STATS_PATH,
        dtype={"season_year": str},
        low_memory=False,
    )

    observed_scopes = set(
        zip(
            stats["competition"].astype(str),
            stats["season_year"].astype(str),
        )
    )

    expected_scopes = set(
        zip(
            missing_scopes[
                "competition"
            ].astype(str),
            missing_scopes[
                "season_year"
            ].astype(str),
        )
    )

    missing_after_backfill = (
        expected_scopes
        - observed_scopes
    )

    unexpected_scopes = (
        observed_scopes
        - expected_scopes
    )

    if missing_after_backfill:
        raise AssertionError(
            "Backfill is missing expected scopes: "
            f"{sorted(missing_after_backfill)}"
        )

    if unexpected_scopes:
        raise AssertionError(
            "Backfill contains unexpected scopes: "
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
            "Backfill contains duplicate player-stat "
            f"task keys: {duplicate_count}."
        )

    summary = (
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
    print("Big Five backfill summary")
    print("-" * 88)
    print(
        summary.to_string()
    )

    print()
    print(
        f"Backfilled scopes: "
        f"{len(observed_scopes)}"
    )
    print(
        f"Successful stat rows: "
        f"{len(stats)}"
    )
    print(
        "Duplicate stat-task keys: 0"
    )

    if (
        BACKFILL_STATS_FAILURES_PATH.exists()
    ):
        try:
            failures = pd.read_csv(
                BACKFILL_STATS_FAILURES_PATH
            )
        except pd.errors.EmptyDataError:
            failures = pd.DataFrame()

        print(
            "Outstanding stat failures: "
            f"{len(failures)}"
        )

    else:
        print(
            "Outstanding stat failures: 0"
        )


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 101B — BIG FIVE PLAYER-STAT "
        "BACKFILL"
    )
    print("=" * 88)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing candidate manifest: "
            f"{MANIFEST_PATH}"
        )

    if not CANONICAL_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Missing canonical stats: "
            f"{CANONICAL_STATS_PATH}"
        )

    missing_scopes = (
        discover_missing_scopes()
    )

    print()
    print("Missing Big Five scopes")
    print("-" * 88)
    print(
        missing_scopes.to_string(
            index=False
        )
    )

    print()
    print(
        f"Scopes requiring backfill: "
        f"{len(missing_scopes)}"
    )

    if missing_scopes.empty:
        print(
            "Nothing to backfill."
        )
        return

    acquire_player_memberships(
        missing_scopes
    )

    acquire_player_stats(
        missing_scopes
    )

    validate_backfill(
        missing_scopes
    )

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
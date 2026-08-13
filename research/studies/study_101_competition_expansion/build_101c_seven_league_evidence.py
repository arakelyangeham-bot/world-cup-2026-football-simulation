#build_101c_seven_league_evidence

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STUDY_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_101a_competition_expansion"
)

CANONICAL_STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_stats.csv"
)

BIG_FIVE_BACKFILL_PATH = (
    STUDY_DIRECTORY
    / "big_five_backfill_player_stats.csv"
)

EXPANSION_STATS_PATH = (
    STUDY_DIRECTORY
    / "candidate_player_stats.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_101c_seven_league_evidence"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "seven_league_player_stats.csv"
)

SCOPE_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "seven_league_scope_audit.csv"
)

LEAGUES = (
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
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

TASK_KEY = [
    "competition_id",
    "season_id",
    "player_id",
]


def load_stats(
    path: Path,
    source_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {source_name}: {path}"
        )

    dataframe = pd.read_csv(
        path,
        dtype={"season_year": str},
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"{source_name} is empty."
        )

    dataframe["_evidence_source"] = (
        source_name
    )

    return dataframe


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 101C — SEVEN-LEAGUE "
        "PLAYER-EVIDENCE ASSEMBLY"
    )
    print("=" * 88)

    canonical = load_stats(
        CANONICAL_STATS_PATH,
        "canonical",
    )

    backfill = load_stats(
        BIG_FIVE_BACKFILL_PATH,
        "big_five_backfill",
    )

    expansion = load_stats(
        EXPANSION_STATS_PATH,
        "competition_expansion",
    )

    canonical = canonical.loc[
        canonical["competition"].isin(
            LEAGUES
        )
        & canonical["season_year"].isin(
            SEASONS
        )
    ].copy()

    backfill = backfill.loc[
        backfill["competition"].isin(
            LEAGUES
        )
        & backfill["season_year"].isin(
            SEASONS
        )
    ].copy()

    expansion = expansion.loc[
        expansion["competition"].isin(
            LEAGUES
        )
        & expansion["season_year"].isin(
            SEASONS
        )
    ].copy()

    combined = pd.concat(
        [
            canonical,
            backfill,
            expansion,
        ],
        ignore_index=True,
        sort=False,
    )

    #
    # Existing canonical evidence has priority,
    # followed by backfill and expansion evidence.
    #
    source_priority = {
        "competition_expansion": 1,
        "big_five_backfill": 2,
        "canonical": 3,
    }

    combined["_source_priority"] = (
        combined["_evidence_source"]
        .map(source_priority)
    )

    combined = (
        combined
        .sort_values(
            "_source_priority"
        )
        .drop_duplicates(
            subset=TASK_KEY,
            keep="last",
        )
        .reset_index(drop=True)
    )

    expected_scopes = {
        (league, season)
        for league in LEAGUES
        for season in SEASONS
    }

    observed_scopes = set(
        zip(
            combined["competition"].astype(str),
            combined["season_year"].astype(str),
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
            "Seven-league evidence is missing scopes: "
            f"{sorted(missing_scopes)}"
        )

    if unexpected_scopes:
        raise AssertionError(
            "Seven-league evidence contains unexpected "
            f"scopes: {sorted(unexpected_scopes)}"
        )

    if len(observed_scopes) != 35:
        raise AssertionError(
            "Expected 35 league-season scopes, "
            f"found {len(observed_scopes)}."
        )

    duplicate_count = int(
        combined.duplicated(
            TASK_KEY
        ).sum()
    )

    if duplicate_count:
        raise AssertionError(
            "Seven-league evidence contains "
            f"{duplicate_count} duplicate task keys."
        )

    scope_audit = (
        combined.groupby(
            [
                "competition",
                "season_year",
            ],
            as_index=False,
        )
        .agg(
            rows=("player_id", "size"),
            players=(
                "player_id",
                "nunique",
            ),
        )
        .sort_values(
            [
                "competition",
                "season_year",
            ]
        )
    )

    if len(scope_audit) != 35:
        raise AssertionError(
            "Scope audit must contain exactly "
            f"35 rows; found {len(scope_audit)}."
        )

    if (
        scope_audit["rows"]
        != scope_audit["players"]
    ).any():
        raise AssertionError(
            "One or more scopes contain multiple "
            "rows for the same player."
        )

    source_summary = (
        combined["_evidence_source"]
        .value_counts()
        .sort_index()
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = combined.drop(
        columns=[
            "_source_priority",
        ]
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    scope_audit.to_csv(
        SCOPE_AUDIT_PATH,
        index=False,
    )

    print()
    print("Evidence source summary")
    print("-" * 88)
    print(
        source_summary.to_string()
    )

    print()
    print("League-season scope audit")
    print("-" * 88)
    print(
        scope_audit.to_string(
            index=False
        )
    )

    print()
    print("Validation summary")
    print("-" * 88)
    print(
        "Expected league-season scopes: 35"
    )
    print(
        f"Observed league-season scopes: "
        f"{len(observed_scopes)}"
    )
    print(
        f"Player-stat rows: {len(output)}"
    )
    print(
        f"Unique players: "
        f"{output['player_id'].nunique()}"
    )
    print(
        "Duplicate task keys: 0"
    )

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)

    print()
    print(
        f"Evidence artifact: {OUTPUT_PATH}"
    )
    print(
        f"Scope audit: {SCOPE_AUDIT_PATH}"
    )


if __name__ == "__main__":
    main()
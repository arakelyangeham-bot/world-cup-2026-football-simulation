#audit_competition_player_membership

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_players.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_046_club_membership"
)

SMALL_SQUAD_THRESHOLD = 15
LARGE_SQUAD_THRESHOLD = 60

REQUIRED_COLUMNS = {
    "competition",
    "competition_type",
    "competition_id",
    "season_id",
    "season_year",
    "team_id",
    "team",
    "player_id",
    "player",
}

MEMBERSHIP_KEY = [
    "competition_id",
    "season_id",
    "team_id",
    "player_id",
]

COMPETITION_SEASON_KEY = [
    "competition",
    "competition_type",
    "competition_id",
    "season_id",
    "season_year",
]

CLUB_SEASON_KEY = [
    "competition",
    "competition_type",
    "competition_id",
    "season_id",
    "season_year",
    "team_id",
    "team",
]


def load_membership_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Membership input file does not exist: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "season_year": str,
        },
        low_memory=False,
    )

    if df.empty:
        raise ValueError(
            f"Membership input file is empty: {path}"
        )

    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Membership dataset is missing required columns: "
            f"{sorted(missing)}"
        )


def missing_count(
    group: pd.DataFrame,
    column: str,
) -> int:
    return int(group[column].isna().sum())


def build_duplicate_memberships(
    df: pd.DataFrame,
) -> pd.DataFrame:
    duplicates = (
        df.groupby(
            MEMBERSHIP_KEY,
            dropna=False,
        )
        .size()
        .reset_index(name="duplicate_count")
    )

    duplicates = duplicates[
        duplicates["duplicate_count"] > 1
    ].copy()

    return duplicates.sort_values(
        MEMBERSHIP_KEY,
        na_position="last",
    ).reset_index(drop=True)


def build_multi_club_memberships(
    df: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "competition",
        "competition_type",
        "competition_id",
        "season_id",
        "season_year",
        "player_id",
    ]

    rows: list[dict] = []

    for keys, group in df.groupby(
        group_columns,
        dropna=False,
    ):
        unique_team_ids = sorted(
            {
                str(value)
                for value in group["team_id"].dropna().unique()
            }
        )

        unique_team_names = sorted(
            {
                str(value)
                for value in group["team"].dropna().unique()
            }
        )

        if len(unique_team_ids) <= 1:
            continue

        (
            competition,
            competition_type,
            competition_id,
            season_id,
            season_year,
            player_id,
        ) = keys

        player_names = sorted(
            {
                str(value)
                for value in group["player"].dropna().unique()
            }
        )

        rows.append(
            {
                "competition": competition,
                "competition_type": competition_type,
                "competition_id": competition_id,
                "season_id": season_id,
                "season_year": season_year,
                "player_id": player_id,
                "player": " | ".join(player_names),
                "team_count": len(unique_team_ids),
                "team_ids": " | ".join(unique_team_ids),
                "teams": " | ".join(unique_team_names),
                "membership_rows": len(group),
            }
        )

    columns = [
        "competition",
        "competition_type",
        "competition_id",
        "season_id",
        "season_year",
        "player_id",
        "player",
        "team_count",
        "team_ids",
        "teams",
        "membership_rows",
    ]

    result = pd.DataFrame(rows, columns=columns)

    if result.empty:
        return result

    return result.sort_values(
        [
            "competition",
            "season_year",
            "player",
        ],
        na_position="last",
    ).reset_index(drop=True)


def build_identifier_consistency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    context_columns = [
        "competition_id",
        "season_id",
    ]

    for keys, group in df.groupby(
        context_columns + ["team_id"],
        dropna=False,
    ):
        competition_id, season_id, team_id = keys

        names = sorted(
            {
                str(value)
                for value in group["team"].dropna().unique()
            }
        )

        if len(names) > 1:
            rows.append(
                {
                    "entity_type": "team",
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "entity_id": team_id,
                    "name_count": len(names),
                    "names": " | ".join(names),
                }
            )

    for keys, group in df.groupby(
        context_columns + ["player_id"],
        dropna=False,
    ):
        competition_id, season_id, player_id = keys

        names = sorted(
            {
                str(value)
                for value in group["player"].dropna().unique()
            }
        )

        if len(names) > 1:
            rows.append(
                {
                    "entity_type": "player",
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "entity_id": player_id,
                    "name_count": len(names),
                    "names": " | ".join(names),
                }
            )

    columns = [
        "entity_type",
        "competition_id",
        "season_id",
        "entity_id",
        "name_count",
        "names",
    ]

    result = pd.DataFrame(rows, columns=columns)

    if result.empty:
        return result

    return result.sort_values(
        [
            "entity_type",
            "competition_id",
            "season_id",
            "entity_id",
        ],
        na_position="last",
    ).reset_index(drop=True)


def build_competition_summary(
    df: pd.DataFrame,
    duplicate_memberships: pd.DataFrame,
    multi_club_memberships: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for keys, group in df.groupby(
        COMPETITION_SEASON_KEY,
        dropna=False,
    ):
        (
            competition,
            competition_type,
            competition_id,
            season_id,
            season_year,
        ) = keys

        duplicate_count = 0

        if not duplicate_memberships.empty:
            duplicate_count = int(
                duplicate_memberships[
                    (
                        duplicate_memberships["competition_id"]
                        == competition_id
                    )
                    & (
                        duplicate_memberships["season_id"]
                        == season_id
                    )
                ]["duplicate_count"]
                .sub(1)
                .sum()
            )

        multi_club_count = 0

        if not multi_club_memberships.empty:
            multi_club_count = int(
                len(
                    multi_club_memberships[
                        (
                            multi_club_memberships["competition_id"]
                            == competition_id
                        )
                        & (
                            multi_club_memberships["season_id"]
                            == season_id
                        )
                    ]
                )
            )

        unique_memberships = (
            group.drop_duplicates(
                subset=MEMBERSHIP_KEY,
            )
            .shape[0]
        )

        rows.append(
            {
                "competition": competition,
                "competition_type": competition_type,
                "competition_id": competition_id,
                "season_id": season_id,
                "season_year": season_year,
                "row_count": len(group),
                "team_count": group["team_id"].nunique(
                    dropna=True
                ),
                "player_count": group["player_id"].nunique(
                    dropna=True
                ),
                "unique_membership_count": unique_memberships,
                "missing_team_id": missing_count(
                    group,
                    "team_id",
                ),
                "missing_team_name": missing_count(
                    group,
                    "team",
                ),
                "missing_player_id": missing_count(
                    group,
                    "player_id",
                ),
                "missing_player_name": missing_count(
                    group,
                    "player",
                ),
                "duplicate_membership_rows": duplicate_count,
                "multi_club_player_count": multi_club_count,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "competition",
                "season_year",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def classify_squad_size(
    unique_players: int,
) -> str:
    if unique_players < SMALL_SQUAD_THRESHOLD:
        return "small"

    if unique_players > LARGE_SQUAD_THRESHOLD:
        return "large"

    return "plausible"


def build_club_summary(
    df: pd.DataFrame,
    multi_club_memberships: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    multi_club_keys: set[
        tuple[object, object, object]
    ] = set()

    if not multi_club_memberships.empty:
        multi_club_keys = {
            (
                row.competition_id,
                row.season_id,
                row.player_id,
            )
            for row in multi_club_memberships.itertuples(
                index=False
            )
        }

    for keys, group in df.groupby(
        CLUB_SEASON_KEY,
        dropna=False,
    ):
        (
            competition,
            competition_type,
            competition_id,
            season_id,
            season_year,
            team_id,
            team,
        ) = keys

        unique_players = int(
            group["player_id"].nunique(
                dropna=True
            )
        )

        duplicate_player_rows = int(
            len(group)
            - len(
                group.drop_duplicates(
                    subset=["player_id"]
                )
            )
        )

        multi_club_player_count = sum(
            1
            for player_id
            in group["player_id"].dropna().unique()
            if (
                competition_id,
                season_id,
                player_id,
            )
            in multi_club_keys
        )

        rows.append(
            {
                "competition": competition,
                "competition_type": competition_type,
                "competition_id": competition_id,
                "season_id": season_id,
                "season_year": season_year,
                "team_id": team_id,
                "team": team,
                "membership_rows": len(group),
                "unique_players": unique_players,
                "duplicate_player_rows": duplicate_player_rows,
                "multi_club_players": multi_club_player_count,
                "squad_size_flag": classify_squad_size(
                    unique_players
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "competition",
                "season_year",
                "team",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def calculate_status(
    competition_summary: pd.DataFrame,
    club_summary: pd.DataFrame,
    duplicate_memberships: pd.DataFrame,
    identifier_consistency: pd.DataFrame,
    multi_club_memberships: pd.DataFrame,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    missing_identifier_total = int(
        competition_summary[
            [
                "missing_team_id",
                "missing_player_id",
            ]
        ].sum().sum()
    )

    if missing_identifier_total > 0:
        failures.append(
            "Missing team or player identifiers were detected."
        )

    if not identifier_consistency.empty:
        failures.append(
            "Identifier-to-name inconsistencies were detected."
        )

    if not duplicate_memberships.empty:
        warnings.append(
            "Exact duplicate membership keys were detected."
        )

    suspicious_squads = club_summary[
        club_summary["squad_size_flag"] != "plausible"
    ]

    if not suspicious_squads.empty:
        warnings.append(
            "One or more club-season squad sizes were flagged."
        )

    if not multi_club_memberships.empty:
        warnings.append(
            "Players with multiple clubs in the same "
            "competition-season were detected."
        )

    if failures:
        return "FAIL", failures + warnings

    if warnings:
        return "PASS_WITH_WARNINGS", warnings

    return "PASS", []


def build_metadata(
    df: pd.DataFrame,
    competition_summary: pd.DataFrame,
    club_summary: pd.DataFrame,
    output_files: list[str],
    status: str,
) -> dict:
    return {
        "study_id": "046",
        "study_name": (
            "Competition-Season Club Membership Audit"
        ),
        "input_path": str(
            INPUT_FILE.relative_to(PROJECT_ROOT)
        ),
        "input_row_count": int(len(df)),
        "competition_season_count": int(
            len(competition_summary)
        ),
        "club_season_count": int(
            len(club_summary)
        ),
        "unique_player_count": int(
            df["player_id"].nunique(
                dropna=True
            )
        ),
        "small_squad_threshold": (
            SMALL_SQUAD_THRESHOLD
        ),
        "large_squad_threshold": (
            LARGE_SQUAD_THRESHOLD
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": status,
        "output_files": output_files,
    }


def write_results_markdown(
    path: Path,
    df: pd.DataFrame,
    competition_summary: pd.DataFrame,
    club_summary: pd.DataFrame,
    duplicate_memberships: pd.DataFrame,
    multi_club_memberships: pd.DataFrame,
    identifier_consistency: pd.DataFrame,
    status: str,
    findings: list[str],
) -> None:
    squad_sizes = club_summary["unique_players"]

    flagged_squads = club_summary[
        club_summary["squad_size_flag"]
        != "plausible"
    ]

    lines = [
        "# Study 046 Results",
        "",
        "## Competition-Season Club Membership Audit",
        "",
        f"**Status:** `{status}`",
        "",
        "## Dataset summary",
        "",
        f"- Input rows: {len(df)}",
        (
            "- Competition-seasons: "
            f"{len(competition_summary)}"
        ),
        f"- Club-season records: {len(club_summary)}",
        (
            "- Unique clubs: "
            f"{df['team_id'].nunique(dropna=True)}"
        ),
        (
            "- Unique players: "
            f"{df['player_id'].nunique(dropna=True)}"
        ),
        "",
        "## Squad-size distribution",
        "",
        f"- Minimum: {squad_sizes.min():.0f}",
        f"- First quartile: {squad_sizes.quantile(0.25):.2f}",
        f"- Median: {squad_sizes.median():.2f}",
        f"- Mean: {squad_sizes.mean():.2f}",
        f"- Third quartile: {squad_sizes.quantile(0.75):.2f}",
        f"- Maximum: {squad_sizes.max():.0f}",
        f"- Standard deviation: {squad_sizes.std():.2f}",
        (
            "- Flagged club-seasons: "
            f"{len(flagged_squads)}"
        ),
        "",
        "## Data-quality findings",
        "",
        (
            "- Duplicate membership keys: "
            f"{len(duplicate_memberships)}"
        ),
        (
            "- Multi-club player records: "
            f"{len(multi_club_memberships)}"
        ),
        (
            "- Identifier inconsistencies: "
            f"{len(identifier_consistency)}"
        ),
        "",
    ]

    if findings:
        lines.extend(
            [
                "## Warnings or failures",
                "",
            ]
        )

        lines.extend(
            f"- {finding}"
            for finding in findings
        )

        lines.append("")

    lines.extend(
        [
            "## Interpretation gate",
            "",
            (
                "The Competition Player Repository and "
                "Club Roster Builder should not be "
                "implemented until these outputs have "
                "been reviewed."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_outputs(
    df: pd.DataFrame,
    competition_summary: pd.DataFrame,
    club_summary: pd.DataFrame,
    duplicate_memberships: pd.DataFrame,
    multi_club_memberships: pd.DataFrame,
    identifier_consistency: pd.DataFrame,
    status: str,
    findings: list[str],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "competition_membership_summary.csv":
            competition_summary,
        "club_membership_summary.csv":
            club_summary,
        "duplicate_memberships.csv":
            duplicate_memberships,
        "multi_club_memberships.csv":
            multi_club_memberships,
        "identifier_consistency.csv":
            identifier_consistency,
    }

    for filename, dataframe in outputs.items():
        dataframe.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    metadata = build_metadata(
        df=df,
        competition_summary=competition_summary,
        club_summary=club_summary,
        output_files=[
            *outputs.keys(),
            "study_metadata.json",
            "STUDY_046_RESULTS.md",
        ],
        status=status,
    )

    with (
        OUTPUT_DIR
        / "study_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    write_results_markdown(
        path=(
            OUTPUT_DIR
            / "STUDY_046_RESULTS.md"
        ),
        df=df,
        competition_summary=competition_summary,
        club_summary=club_summary,
        duplicate_memberships=duplicate_memberships,
        multi_club_memberships=multi_club_memberships,
        identifier_consistency=identifier_consistency,
        status=status,
        findings=findings,
    )


def print_summary(
    df: pd.DataFrame,
    competition_summary: pd.DataFrame,
    club_summary: pd.DataFrame,
    duplicate_memberships: pd.DataFrame,
    multi_club_memberships: pd.DataFrame,
    identifier_consistency: pd.DataFrame,
    status: str,
    findings: list[str],
) -> None:
    print("Study 046")
    print("=" * 46)
    print()
    print(
        f"Competition-seasons .......... "
        f"{len(competition_summary)}"
    )
    print(
        f"Club-season records ......... "
        f"{len(club_summary)}"
    )
    print(
        f"Unique clubs ................ "
        f"{df['team_id'].nunique(dropna=True)}"
    )
    print(
        f"Unique players .............. "
        f"{df['player_id'].nunique(dropna=True)}"
    )
    print(
        f"Average squad size .......... "
        f"{club_summary['unique_players'].mean():.2f}"
    )
    print(
        f"Duplicate membership keys ... "
        f"{len(duplicate_memberships)}"
    )
    print(
        f"Multi-club players .......... "
        f"{len(multi_club_memberships)}"
    )
    print(
        f"Identifier inconsistencies .. "
        f"{len(identifier_consistency)}"
    )
    print()
    print("Status")
    print("-" * 46)
    print(status)

    if findings:
        print()
        print("Findings")
        print("-" * 46)

        for finding in findings:
            print(f"- {finding}")

    print()
    print(f"Outputs written to: {OUTPUT_DIR}")


def main() -> None:
    df = load_membership_data(
        INPUT_FILE
    )

    validate_schema(df)

    duplicate_memberships = (
        build_duplicate_memberships(df)
    )

    multi_club_memberships = (
        build_multi_club_memberships(df)
    )

    identifier_consistency = (
        build_identifier_consistency(df)
    )

    competition_summary = (
        build_competition_summary(
            df=df,
            duplicate_memberships=(
                duplicate_memberships
            ),
            multi_club_memberships=(
                multi_club_memberships
            ),
        )
    )

    club_summary = build_club_summary(
        df=df,
        multi_club_memberships=(
            multi_club_memberships
        ),
    )

    status, findings = calculate_status(
        competition_summary=(
            competition_summary
        ),
        club_summary=club_summary,
        duplicate_memberships=(
            duplicate_memberships
        ),
        identifier_consistency=(
            identifier_consistency
        ),
        multi_club_memberships=(
            multi_club_memberships
        ),
    )

    write_outputs(
        df=df,
        competition_summary=(
            competition_summary
        ),
        club_summary=club_summary,
        duplicate_memberships=(
            duplicate_memberships
        ),
        multi_club_memberships=(
            multi_club_memberships
        ),
        identifier_consistency=(
            identifier_consistency
        ),
        status=status,
        findings=findings,
    )

    print_summary(
        df=df,
        competition_summary=(
            competition_summary
        ),
        club_summary=club_summary,
        duplicate_memberships=(
            duplicate_memberships
        ),
        multi_club_memberships=(
            multi_club_memberships
        ),
        identifier_consistency=(
            identifier_consistency
        ),
        status=status,
        findings=findings,
    )


if __name__ == "__main__":
    main()
#validate_global_club_prior_dataset

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "global_club_prior_dataset.csv"
)

REQUIRED_COLUMNS = {
    "club_id",
    "opta_id",
    "club",
    "club_short",
    "club_full",
    "club_code",
    "opta_rating",
    "global_rank",
    "competition_id",
    "rating_prior",
    "rating_prior_method",
    "snapshot_date",
    "rating_source",
    "source_url",
}

PREMIER_LEAGUE_TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Global Club Prior Dataset not found: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            "Global Club Prior Dataset is empty."
        )

    return dataframe


def validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def validate_required_values(
    dataframe: pd.DataFrame,
) -> None:
    required_non_null_columns = [
        "club_id",
        "club",
        "opta_rating",
        "global_rank",
        "snapshot_date",
        "rating_source",
        "source_url",
    ]

    for column in required_non_null_columns:
        missing_count = int(dataframe[column].isna().sum())

        if missing_count:
            raise ValueError(
                f"Column {column!r} contains "
                f"{missing_count} missing values."
            )

    empty_club_ids = (
        dataframe["club_id"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_club_ids:
        raise ValueError(
            f"Dataset contains {empty_club_ids} empty club IDs."
        )

    empty_club_names = (
        dataframe["club"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_club_names:
        raise ValueError(
            f"Dataset contains {empty_club_names} empty club names."
        )


def validate_unique_identifiers(
    dataframe: pd.DataFrame,
) -> None:
    duplicate_club_ids = dataframe[
        dataframe["club_id"].duplicated(keep=False)
    ]

    if not duplicate_club_ids.empty:
        duplicate_values = sorted(
            duplicate_club_ids["club_id"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate stable club IDs found: "
            f"{duplicate_values[:20]}"
        )

    duplicate_ranks = dataframe[
        dataframe["global_rank"].duplicated(keep=False)
    ]

    if not duplicate_ranks.empty:
        duplicate_values = sorted(
            duplicate_ranks["global_rank"]
            .astype(int)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate global ranks found: "
            f"{duplicate_values[:20]}"
        )


def validate_numeric_ranges(
    dataframe: pd.DataFrame,
) -> None:
    ratings = pd.to_numeric(
        dataframe["opta_rating"],
        errors="coerce",
    )

    ranks = pd.to_numeric(
        dataframe["global_rank"],
        errors="coerce",
    )

    if ratings.isna().any():
        raise ValueError(
            "One or more Opta ratings are not numeric."
        )

    if ranks.isna().any():
        raise ValueError(
            "One or more global ranks are not numeric."
        )

    invalid_ratings = dataframe[
        ~ratings.between(0.0, 100.0)
    ]

    if not invalid_ratings.empty:
        raise ValueError(
            "One or more Opta ratings fall outside 0–100."
        )

    invalid_ranks = dataframe[ranks < 1]

    if not invalid_ranks.empty:
        raise ValueError(
            "One or more global ranks are non-positive."
        )


def validate_rank_continuity(
    dataframe: pd.DataFrame,
) -> list[int]:
    actual_ranks = set(
        dataframe["global_rank"]
        .astype(int)
        .tolist()
    )

    maximum_rank = max(actual_ranks)

    expected_ranks = set(
        range(1, maximum_rank + 1)
    )

    return sorted(expected_ranks - actual_ranks)


def find_duplicate_values(
    dataframe: pd.DataFrame,
    column: str,
) -> list[tuple[str, int]]:
    values = (
        dataframe[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    counts = Counter(values)

    return sorted(
        [
            (value, count)
            for value, count in counts.items()
            if value and count > 1
        ],
        key=lambda item: (
            -item[1],
            item[0].casefold(),
        ),
    )


def validate_rating_prior_policy(
    dataframe: pd.DataFrame,
) -> None:
    assigned_count = int(
        dataframe["rating_prior"]
        .notna()
        .sum()
    )

    methods = set(
        dataframe["rating_prior_method"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    if assigned_count != 0:
        raise ValueError(
            "Study 041 should not contain calibrated rating priors yet. "
            f"Found {assigned_count} assigned values."
        )

    if methods != {"unassigned"}:
        raise ValueError(
            "Expected every rating_prior_method to equal "
            f"'unassigned', but found: {sorted(methods)}"
        )


def find_premier_league_coverage(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Match each expected Premier League club to one canonical Opta row.

    When several clubs share the same canonical name, the highest-ranked
    record is selected. All additional matches are returned separately as
    name collisions for inspection.
    """

    selected_rows: list[pd.Series] = []
    collision_rows: list[pd.DataFrame] = []
    missing_teams: list[str] = []

    for team in PREMIER_LEAGUE_TEAMS:
        matches = dataframe[
            dataframe["club"].eq(team)
        ].copy()

        if matches.empty:
            missing_teams.append(team)
            continue

        matches = matches.sort_values(
            ["global_rank", "opta_rating"],
            ascending=[True, False],
        )

        selected_rows.append(matches.iloc[0])

        if len(matches) > 1:
            additional_matches = matches.iloc[1:].copy()
            additional_matches.insert(
                0,
                "matched_premier_league_name",
                team,
            )
            collision_rows.append(additional_matches)

    if selected_rows:
        premier_league_rows = pd.DataFrame(
            selected_rows
        ).sort_values("global_rank")
    else:
        premier_league_rows = pd.DataFrame(
            columns=dataframe.columns
        )

    if collision_rows:
        name_collisions = pd.concat(
            collision_rows,
            ignore_index=True,
        ).sort_values(
            [
                "matched_premier_league_name",
                "global_rank",
            ]
        )
    else:
        name_collisions = pd.DataFrame(
            columns=[
                "matched_premier_league_name",
                *dataframe.columns,
            ]
        )

    return (
        premier_league_rows,
        missing_teams,
        name_collisions,
    )

def print_duplicate_summary(
    title: str,
    duplicates: list[tuple[str, int]],
    limit: int = 15,
) -> None:
    print(title)
    print("-" * len(title))

    if not duplicates:
        print("None")
        print()
        return

    print(f"Duplicated values: {len(duplicates)}")

    for value, count in duplicates[:limit]:
        print(f"{value}: {count}")

    if len(duplicates) > limit:
        print(
            f"... and {len(duplicates) - limit} more"
        )

    print()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the canonical Global Club Prior Dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to global_club_prior_dataset.csv.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    dataframe = load_dataset(arguments.input)

    validate_required_columns(dataframe)
    validate_required_values(dataframe)
    validate_unique_identifiers(dataframe)
    validate_numeric_ranges(dataframe)
    validate_rating_prior_policy(dataframe)

    missing_ranks = validate_rank_continuity(
        dataframe
    )

    duplicate_names = find_duplicate_values(
        dataframe,
        "club",
    )

    duplicate_full_names = find_duplicate_values(
        dataframe,
        "club_full",
    )

    duplicate_codes = find_duplicate_values(
        dataframe,
        "club_code",
    )

    (
        premier_league_rows,
        missing_premier_league,
        premier_league_name_collisions,
    ) = find_premier_league_coverage(dataframe)

    print("Global Club Prior Dataset Validation")
    print("====================================")
    print(f"Input: {arguments.input}")
    print(f"Records: {len(dataframe)}")
    print(
        "Opta rating range: "
        f"{dataframe['opta_rating'].min():.4f}–"
        f"{dataframe['opta_rating'].max():.4f}"
    )
    print(
        "Global rank range: "
        f"{int(dataframe['global_rank'].min())}–"
        f"{int(dataframe['global_rank'].max())}"
    )
    print(
        "Snapshot dates: "
        f"{dataframe['snapshot_date'].nunique()}"
    )
    print()

    print("Rank Continuity")
    print("---------------")

    if missing_ranks:
        print("Status: WARNING")
        print(f"Missing rank count: {len(missing_ranks)}")
        print(
            "Missing ranks: "
            f"{missing_ranks[:20]}"
        )
        print(
            "Rank gaps are preserved as source characteristics "
            "and are not treated as structural failures."
        )
    else:
        print("Status: PASS")
        print("Ranks are continuous.")

    print()

    print_duplicate_summary(
        "Duplicate Canonical Club Names",
        duplicate_names,
    )

    print_duplicate_summary(
        "Duplicate Full Club Names",
        duplicate_full_names,
    )

    print_duplicate_summary(
        "Duplicate Club Codes",
        duplicate_codes,
    )

    print("Premier League Coverage")
    print("-----------------------")
    print(
        "Unique expected clubs matched: "
        f"{len(premier_league_rows)}/"
        f"{len(PREMIER_LEAGUE_TEAMS)}"
    )

    if missing_premier_league:
        print(
            "Status: FAIL"
        )
        print(
            "Missing clubs: "
            f"{missing_premier_league}"
        )
    else:
        print("Status: PASS")
        print("All Premier League clubs matched.")

    print(
        "Additional records sharing Premier League names: "
        f"{len(premier_league_name_collisions)}"
    )
    print()

    print(
        f"{'Rank':>5}  "
        f"{'Club':<28} "
        f"{'Rating':>10}  "
        f"{'Club ID':<25}"
    )

    for _, row in premier_league_rows.iterrows():
        print(
            f"{int(row['global_rank']):>5}  "
            f"{str(row['club']):<28} "
            f"{float(row['opta_rating']):>10.4f}  "
            f"{str(row['club_id']):<25}"
        )
    
    if not premier_league_name_collisions.empty:
        print()
        print("Premier League Name Collisions")
        print("------------------------------")
        print(
            "These additional records share names with expected "
            "Premier League clubs but were not selected."
        )
        print()

        print(
            f"{'Expected Name':<28} "
            f"{'Rank':>6}  "
            f"{'Club Full Name':<40} "
            f"{'Rating':>10}"
        )

        for _, row in premier_league_name_collisions.iterrows():
            print(
                f"{str(row['matched_premier_league_name']):<28} "
                f"{int(row['global_rank']):>6}  "
                f"{str(row['club_full']):<40} "
                f"{float(row['opta_rating']):>10.4f}"
            )

    if missing_premier_league:
        raise ValueError(
            "Premier League coverage validation failed."
        )

    print()
    print("Validation Summary")
    print("------------------")
    print("PASS: Required schema and values")
    print("PASS: Stable club ID uniqueness")
    print("PASS: Global rank uniqueness")
    print("PASS: Numeric rating and rank ranges")
    print("PASS: Study 041 rating-prior policy")
    print("PASS: Premier League canonical coverage")

    if missing_ranks:
        print(
            f"WARNING: Source ranking contains "
            f"{len(missing_ranks)} gap(s)"
        )

    print(
        f"WARNING: {len(duplicate_names)} duplicated "
        "canonical club-name values"
    )
    print(
        f"WARNING: {len(duplicate_full_names)} duplicated "
        "full club-name values"
    )
    print(
        f"WARNING: {len(duplicate_codes)} duplicated "
        "club-code values"
    )
    print(
        "INFO: Duplicate human-readable names and codes are "
        "expected in a global dataset."
    )
    print()
    print("All fatal structural checks passed.")


if __name__ == "__main__":
    main()
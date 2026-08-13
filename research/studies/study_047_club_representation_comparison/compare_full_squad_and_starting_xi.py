#compare_full_squad_and_starting_xi

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.player_intelligence.competition_team_repository import (
    CompetitionTeamRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_047_club_representation_comparison"
)

FORMATION = "4-3-3"

COMPARISON_FIELDS = (
    "attack",
    "midfield",
    "defense",
    "goalkeeper",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
)

UNAVAILABLE_FIELDS = (
    "squad_quality",
    "evidence_score",
)


def build_representation_comparison(
    repository: CompetitionTeamRepository,
) -> pd.DataFrame:
    seasons = (
        repository
        .roster_builder
        .list_competition_seasons()
    )

    club_seasons = seasons[
        seasons["competition_type"]
        .astype(str)
        .str.contains(
            "club",
            case=False,
            na=False,
        )
    ].copy()

    if club_seasons.empty:
        raise RuntimeError(
            "No club competition-seasons were found."
        )

    rows: list[dict] = []

    for season in club_seasons.itertuples(index=False):
        competition_id = int(
            season.competition_id
        )
        season_id = int(
            season.season_id
        )

        teams = (
            repository
            .roster_builder
            .list_teams(
                competition_id=competition_id,
                season_id=season_id,
            )
        )

        for team in teams.itertuples(index=False):
            team_id = int(team.team_id)

            context = repository.get_context(
                competition_id=competition_id,
                season_id=season_id,
                team_id=team_id,
            )

            full_squad = (
                repository
                .get_full_squad_representation(
                    competition_id=competition_id,
                    season_id=season_id,
                    team_id=team_id,
                )
            )

            starting_xi = (
                repository
                .get_starting_xi_representation(
                    competition_id=competition_id,
                    season_id=season_id,
                    team_id=team_id,
                    formation=FORMATION,
                )
            )

            row = {
                "competition": context.competition,
                "competition_type": (
                    context.competition_type
                ),
                "competition_id": competition_id,
                "season_id": season_id,
                "season_year": context.season_year,
                "team_id": team_id,
                "team": context.team,
                "formation": FORMATION,
                "full_squad_player_count": (
                    full_squad.player_count
                ),
                "starting_xi_player_count": (
                    starting_xi.player_count
                ),
            }

            for field in COMPARISON_FIELDS:
                full_value = float(
                    getattr(full_squad, field)
                )
                xi_value = float(
                    getattr(starting_xi, field)
                )

                row[f"full_squad_{field}"] = (
                    full_value
                )
                row[f"starting_xi_{field}"] = (
                    xi_value
                )
                row[f"{field}_difference"] = (
                    xi_value - full_value
                )
                row[f"{field}_absolute_difference"] = (
                    abs(xi_value - full_value)
                )

            rows.append(row)

    result = pd.DataFrame(rows)

    return (
        result
        .sort_values(
            [
                "competition",
                "season_year",
                "team",
            ]
        )
        .reset_index(drop=True)
    )


def build_difference_summary(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for field in COMPARISON_FIELDS:
        difference_column = (
            f"{field}_difference"
        )
        absolute_column = (
            f"{field}_absolute_difference"
        )

        differences = comparison[
            difference_column
        ]
        absolute_differences = comparison[
            absolute_column
        ]

        rows.append(
            {
                "dimension": field,
                "comparison_count": int(
                    differences.count()
                ),
                "mean_difference": float(
                    differences.mean()
                ),
                "median_difference": float(
                    differences.median()
                ),
                "standard_deviation": float(
                    differences.std()
                ),
                "minimum_difference": float(
                    differences.min()
                ),
                "maximum_difference": float(
                    differences.max()
                ),
                "mean_absolute_difference": float(
                    absolute_differences.mean()
                ),
                "median_absolute_difference": float(
                    absolute_differences.median()
                ),
                "starting_xi_higher_count": int(
                    (differences > 0).sum()
                ),
                "full_squad_higher_count": int(
                    (differences < 0).sum()
                ),
                "equal_count": int(
                    (differences == 0).sum()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "mean_absolute_difference",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_largest_shifts(
    comparison: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    rows: list[dict] = []

    for field in COMPARISON_FIELDS:
        ordered = comparison.sort_values(
            f"{field}_absolute_difference",
            ascending=False,
        ).head(top_n)

        for rank, row in enumerate(
            ordered.itertuples(index=False),
            start=1,
        ):
            rows.append(
                {
                    "dimension": field,
                    "rank": rank,
                    "competition": row.competition,
                    "season_year": row.season_year,
                    "team": row.team,
                    "full_squad_value": getattr(
                        row,
                        f"full_squad_{field}",
                    ),
                    "starting_xi_value": getattr(
                        row,
                        f"starting_xi_{field}",
                    ),
                    "difference": getattr(
                        row,
                        f"{field}_difference",
                    ),
                    "absolute_difference": getattr(
                        row,
                        f"{field}_absolute_difference",
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_team_shift_summary(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    result = comparison[
        [
            "competition",
            "season_year",
            "team_id",
            "team",
        ]
    ].copy()

    absolute_columns = []

    for field in COMPARISON_FIELDS:
        column = (
            f"{field}_absolute_difference"
        )
        absolute_columns.append(column)

    result["mean_absolute_shift"] = (
        comparison[absolute_columns]
        .mean(axis=1)
    )

    result["maximum_absolute_shift"] = (
        comparison[absolute_columns]
        .max(axis=1)
    )

    result["most_changed_dimension"] = (
        comparison[absolute_columns]
        .idxmax(axis=1)
        .str.replace(
            "_absolute_difference",
            "",
            regex=False,
        )
    )

    return (
        result
        .sort_values(
            "mean_absolute_shift",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def determine_status(
    comparison: pd.DataFrame,
) -> str:
    if comparison.empty:
        return "FAIL"

    if (
        comparison["starting_xi_player_count"]
        != 11
    ).any():
        return "FAIL"

    required_columns = [
        f"{field}_difference"
        for field in COMPARISON_FIELDS
    ]

    if comparison[required_columns].isna().any().any():
        return "FAIL"

    return "PASS"


def write_results_markdown(
    path: Path,
    comparison: pd.DataFrame,
    difference_summary: pd.DataFrame,
    team_shift_summary: pd.DataFrame,
    status: str,
) -> None:
    most_sensitive = (
        difference_summary.iloc[0]
    )

    largest_team_shift = (
        team_shift_summary.iloc[0]
    )

    lines = [
        "# Study 047 Results",
        "",
        "## Full Squad vs Expected Starting XI",
        "",
        f"**Status:** `{status}`",
        "",
        "## Dataset",
        "",
        (
            f"- Club-season comparisons: "
            f"{len(comparison)}"
        ),
        (
            f"- Competitions: "
            f"{comparison['competition'].nunique()}"
        ),
        (
            f"- Seasons: "
            f"{comparison['season_id'].nunique()}"
        ),
        (
            f"- Teams: "
            f"{comparison['team_id'].nunique()}"
        ),
        f"- Formation: `{FORMATION}`",
        "",
        "## Most sensitive representation dimension",
        "",
        (
            f"- Dimension: "
            f"`{most_sensitive['dimension']}`"
        ),
        (
            "- Mean absolute difference: "
            f"{most_sensitive['mean_absolute_difference']:.6f}"
        ),
        (
            "- Mean signed difference: "
            f"{most_sensitive['mean_difference']:.6f}"
        ),
        "",
        "## Largest overall club-season shift",
        "",
        (
            f"- Team: "
            f"{largest_team_shift['team']}"
        ),
        (
            f"- Season: "
            f"{largest_team_shift['season_year']}"
        ),
        (
            "- Mean absolute shift: "
            f"{largest_team_shift['mean_absolute_shift']:.6f}"
        ),
        (
            "- Most changed dimension: "
            f"{largest_team_shift['most_changed_dimension']}"
        ),
        "",
        "## Interpretation warning",
        "",
        (
            "`squad_quality` and `evidence_score` were "
            "excluded because the upstream overall-rating "
            "path currently produces zero values."
        ),
        "",
        "## Decision gate",
        "",
        (
            "These results should be interpreted before "
            "selecting the default representation type for "
            "the first club prediction model."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_outputs(
    comparison: pd.DataFrame,
    difference_summary: pd.DataFrame,
    largest_shifts: pd.DataFrame,
    team_shift_summary: pd.DataFrame,
    status: str,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "representation_comparison.csv",
        index=False,
    )

    difference_summary.to_csv(
        OUTPUT_DIR
        / "representation_difference_summary.csv",
        index=False,
    )

    largest_shifts.to_csv(
        OUTPUT_DIR
        / "largest_representation_shifts.csv",
        index=False,
    )

    team_shift_summary.to_csv(
        OUTPUT_DIR
        / "team_representation_shift_summary.csv",
        index=False,
    )

    metadata = {
        "study_id": "047",
        "study_name": (
            "Full Squad vs Expected Starting XI"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": status,
        "formation": FORMATION,
        "comparison_count": int(
            len(comparison)
        ),
        "competition_count": int(
            comparison["competition"].nunique()
        ),
        "season_count": int(
            comparison["season_id"].nunique()
        ),
        "team_count": int(
            comparison["team_id"].nunique()
        ),
        "comparison_fields": list(
            COMPARISON_FIELDS
        ),
        "excluded_fields": list(
            UNAVAILABLE_FIELDS
        ),
        "output_files": [
            "representation_comparison.csv",
            "representation_difference_summary.csv",
            "largest_representation_shifts.csv",
            "team_representation_shift_summary.csv",
            "study_metadata.json",
            "STUDY_047_RESULTS.md",
        ],
    }

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
            / "STUDY_047_RESULTS.md"
        ),
        comparison=comparison,
        difference_summary=(
            difference_summary
        ),
        team_shift_summary=(
            team_shift_summary
        ),
        status=status,
    )


def print_summary(
    comparison: pd.DataFrame,
    difference_summary: pd.DataFrame,
    team_shift_summary: pd.DataFrame,
    status: str,
) -> None:
    print("Study 047")
    print("=" * 52)
    print()
    print(
        "Club-season comparisons: "
        f"{len(comparison)}"
    )
    print(
        "Competitions: "
        f"{comparison['competition'].nunique()}"
    )
    print(
        "Seasons: "
        f"{comparison['season_id'].nunique()}"
    )
    print(
        "Unique clubs: "
        f"{comparison['team_id'].nunique()}"
    )
    print()

    print("Representation Difference Summary")
    print("-" * 52)
    print(
        difference_summary[
            [
                "dimension",
                "mean_difference",
                "mean_absolute_difference",
                "starting_xi_higher_count",
                "full_squad_higher_count",
                "equal_count",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Largest Overall Team Shifts")
    print("-" * 52)
    print(
        team_shift_summary[
            [
                "competition",
                "season_year",
                "team",
                "mean_absolute_shift",
                "most_changed_dimension",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Status")
    print("-" * 52)
    print(status)
    print()
    print(
        f"Outputs written to: {OUTPUT_DIR}"
    )


def main() -> None:
    repository = CompetitionTeamRepository()

    comparison = (
        build_representation_comparison(
            repository
        )
    )

    difference_summary = (
        build_difference_summary(
            comparison
        )
    )

    largest_shifts = build_largest_shifts(
        comparison,
        top_n=10,
    )

    team_shift_summary = (
        build_team_shift_summary(
            comparison
        )
    )

    status = determine_status(
        comparison
    )

    write_outputs(
        comparison=comparison,
        difference_summary=(
            difference_summary
        ),
        largest_shifts=largest_shifts,
        team_shift_summary=(
            team_shift_summary
        ),
        status=status,
    )

    print_summary(
        comparison=comparison,
        difference_summary=(
            difference_summary
        ),
        team_shift_summary=(
            team_shift_summary
        ),
        status=status,
    )

    if status != "PASS":
        raise AssertionError(
            "Study 047 did not pass."
        )


if __name__ == "__main__":
    main()
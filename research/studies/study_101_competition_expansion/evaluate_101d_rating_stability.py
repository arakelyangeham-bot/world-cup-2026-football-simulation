#evaluate_101d_rating_stability

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STUDY_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "study_101d_expanded_player_intelligence"
)

CANONICAL_RATINGS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_ratings.csv"
)

CANDIDATE_RATINGS = (
    PROJECT_ROOT
    / "outputs"
    / "study_101e_registry_provenance_repair"
    / "candidate_player_ratings.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_101e_registry_provenance_repair"
    / "stability_evaluation"
)

LARGEST_MOVERS_PATH = (
    OUTPUT_DIRECTORY
    / "largest_existing_player_movers.csv"
)

STABILITY_BY_EVIDENCE_PATH = (
    OUTPUT_DIRECTORY
    / "stability_by_evidence.csv"
)

NEW_PLAYER_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "new_player_coverage_summary.csv"
)

DECISION_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "study_101d_decision_summary.csv"
)


def evidence_band(
    value: object,
) -> str:
    if pd.isna(value):
        return "missing"

    value = float(value)

    if value >= 1.0:
        return "full"

    if value >= 0.70:
        return "high"

    if value >= 0.30:
        return "medium"

    return "low"


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    canonical = pd.read_csv(
        CANONICAL_RATINGS,
        low_memory=False,
    )

    candidate = pd.read_csv(
        CANDIDATE_RATINGS,
        low_memory=False,
    )

    if canonical.empty:
        raise ValueError(
            "Canonical player ratings are empty."
        )

    if candidate.empty:
        raise ValueError(
            "Candidate player ratings are empty."
        )

    key = "canonical_player_id"

    if key not in canonical.columns:
        raise ValueError(
            "Canonical ratings are missing "
            "'canonical_player_id'."
        )

    if key not in candidate.columns:
        raise ValueError(
            "Candidate ratings are missing "
            "'canonical_player_id'."
        )

    if canonical[key].duplicated().any():
        raise AssertionError(
            "Canonical ratings contain duplicate "
            "canonical player IDs."
        )

    if candidate[key].duplicated().any():
        raise AssertionError(
            "Candidate ratings contain duplicate "
            "canonical player IDs."
        )

    return canonical, candidate


def build_existing_player_comparison(
    canonical: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    key = "canonical_player_id"

    canonical_rating_columns = {
        column
        for column in canonical.columns
        if (
            column.startswith("rating_")
            and not column.startswith(
                "raw_rating_"
            )
        )
    }

    candidate_rating_columns = {
        column
        for column in candidate.columns
        if (
            column.startswith("rating_")
            and not column.startswith(
                "raw_rating_"
            )
        )
    }

    rating_columns = sorted(
        canonical_rating_columns
        & candidate_rating_columns
    )

    if not rating_columns:
        raise AssertionError(
            "No common role-rating columns were found."
        )

    metadata_columns = [
        "player_id",
        "player",
        "country",
        "current_team",
        "position",
        "eligible_roles",
        "minutesPlayed",
        "total_weighted_evidence",
        "evidence_confidence",
        "source_competitions",
        "competition_count",
        "season_count",
        "best_role",
        "best_rating",
    ]

    canonical_columns = [
        key,
        *[
            column
            for column in metadata_columns
            if column in canonical.columns
        ],
        *rating_columns,
    ]

    candidate_columns = [
        key,
        *[
            column
            for column in metadata_columns
            if column in candidate.columns
        ],
        *rating_columns,
    ]

    comparison = candidate[
        candidate_columns
    ].merge(
        canonical[
            canonical_columns
        ],
        on=key,
        how="inner",
        suffixes=(
            "_candidate",
            "_canonical",
        ),
        validate="one_to_one",
    )

    delta_columns: list[str] = []

    for rating_column in rating_columns:
        candidate_column = (
            f"{rating_column}_candidate"
        )

        canonical_column = (
            f"{rating_column}_canonical"
        )

        delta_column = (
            f"delta_{rating_column}"
        )

        candidate_values = pd.to_numeric(
            comparison[candidate_column],
            errors="coerce",
        )

        canonical_values = pd.to_numeric(
            comparison[canonical_column],
            errors="coerce",
        )

        comparable = (
            candidate_values.notna()
            & canonical_values.notna()
        )

        comparison[delta_column] = np.where(
            comparable,
            candidate_values
            - canonical_values,
            np.nan,
        )

        delta_columns.append(
            delta_column
        )

    comparison[
        "comparable_role_count"
    ] = comparison[
        delta_columns
    ].notna().sum(axis=1)

    comparison[
        "mean_absolute_role_rating_change"
    ] = (
        comparison[
            delta_columns
        ]
        .abs()
        .mean(axis=1)
    )

    comparison[
        "maximum_absolute_role_rating_change"
    ] = (
        comparison[
            delta_columns
        ]
        .abs()
        .max(axis=1)
    )

    comparison[
        "canonical_evidence_band"
    ] = comparison[
        "evidence_confidence_canonical"
    ].map(
        evidence_band
    )

    comparison[
        "candidate_evidence_band"
    ] = comparison[
        "evidence_confidence_candidate"
    ].map(
        evidence_band
    )

    for column in (
        "minutesPlayed",
        "total_weighted_evidence",
        "evidence_confidence",
        "competition_count",
        "season_count",
    ):
        canonical_column = (
            f"{column}_canonical"
        )

        candidate_column = (
            f"{column}_candidate"
        )

        if (
            canonical_column
            in comparison.columns
            and candidate_column
            in comparison.columns
        ):
            comparison[
                f"delta_{column}"
            ] = (
                pd.to_numeric(
                    comparison[
                        candidate_column
                    ],
                    errors="coerce",
                )
                - pd.to_numeric(
                    comparison[
                        canonical_column
                    ],
                    errors="coerce",
                )
            )

    return (
        comparison,
        delta_columns,
    )


def write_largest_movers(
    comparison: pd.DataFrame,
) -> None:
    movers = (
        comparison.loc[
            comparison[
                "comparable_role_count"
            ].gt(0)
        ]
        .sort_values(
            "maximum_absolute_role_rating_change",
            ascending=False,
        )
        .head(50)
        .copy()
    )

    priority_columns = [
        "canonical_player_id",
        "player_candidate",
        "current_team_candidate",
        "country_candidate",
        "position_candidate",
        "canonical_evidence_band",
        "candidate_evidence_band",
        "evidence_confidence_canonical",
        "evidence_confidence_candidate",
        "delta_evidence_confidence",
        "total_weighted_evidence_canonical",
        "total_weighted_evidence_candidate",
        "delta_total_weighted_evidence",
        "competition_count_canonical",
        "competition_count_candidate",
        "delta_competition_count",
        "season_count_canonical",
        "season_count_candidate",
        "delta_season_count",
        "source_competitions_canonical",
        "source_competitions_candidate",
        "comparable_role_count",
        "mean_absolute_role_rating_change",
        "maximum_absolute_role_rating_change",
    ]

    remaining_columns = [
        column
        for column in movers.columns
        if (
            column.startswith("delta_rating_")
            and column
            not in priority_columns
        )
    ]

    selected_columns = [
        column
        for column in (
            priority_columns
            + remaining_columns
        )
        if column in movers.columns
    ]

    movers[
        selected_columns
    ].to_csv(
        LARGEST_MOVERS_PATH,
        index=False,
    )


def build_stability_by_evidence(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    usable = comparison.loc[
        comparison[
            "comparable_role_count"
        ].gt(0)
    ].copy()

    rows: list[
        dict[str, object]
    ] = []

    band_order = [
        "low",
        "medium",
        "high",
        "full",
        "missing",
    ]

    for band in band_order:
        group = usable.loc[
            usable[
                "canonical_evidence_band"
            ].eq(band)
        ]

        if group.empty:
            continue

        change = group[
            "mean_absolute_role_rating_change"
        ]

        rows.append(
            {
                "canonical_evidence_band":
                    band,
                "players":
                    len(group),
                "mean_absolute_role_rating_change":
                    float(
                        change.mean()
                    ),
                "median_absolute_role_rating_change":
                    float(
                        change.median()
                    ),
                "p75_absolute_role_rating_change":
                    float(
                        change.quantile(
                            0.75
                        )
                    ),
                "p90_absolute_role_rating_change":
                    float(
                        change.quantile(
                            0.90
                        )
                    ),
                "p95_absolute_role_rating_change":
                    float(
                        change.quantile(
                            0.95
                        )
                    ),
                "mean_evidence_confidence_change":
                    float(
                        group[
                            "delta_evidence_confidence"
                        ].mean()
                    )
                    if (
                        "delta_evidence_confidence"
                        in group.columns
                    )
                    else np.nan,
                "mean_weighted_evidence_change":
                    float(
                        group[
                            "delta_total_weighted_evidence"
                        ].mean()
                    )
                    if (
                        "delta_total_weighted_evidence"
                        in group.columns
                    )
                    else np.nan,
                "mean_competition_count_change":
                    float(
                        group[
                            "delta_competition_count"
                        ].mean()
                    )
                    if (
                        "delta_competition_count"
                        in group.columns
                    )
                    else np.nan,
                "mean_season_count_change":
                    float(
                        group[
                            "delta_season_count"
                        ].mean()
                    )
                    if (
                        "delta_season_count"
                        in group.columns
                    )
                    else np.nan,
            }
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        STABILITY_BY_EVIDENCE_PATH,
        index=False,
    )

    return output


def build_new_player_summary(
    canonical: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    key = "canonical_player_id"

    canonical_ids = set(
        canonical[key]
    )

    new_players = candidate.loc[
        ~candidate[key].isin(
            canonical_ids
        )
    ].copy()

    rating_columns = [
        column
        for column in candidate.columns
        if (
            column.startswith(
                "rating_"
            )
            and not column.startswith(
                "raw_rating_"
            )
        )
    ]

    new_players[
        "usable_role_rating_count"
    ] = new_players[
        rating_columns
    ].notna().sum(axis=1)

    new_players[
        "has_usable_role_rating"
    ] = new_players[
        "usable_role_rating_count"
    ].gt(0)

    new_players[
        "evidence_band"
    ] = new_players[
        "evidence_confidence"
    ].map(
        evidence_band
    )

    rows: list[
        dict[str, object]
    ] = []

    #
    # Overall summary.
    #
    rows.extend(
        [
            {
                "dimension": "overall",
                "group": "all_new_players",
                "players": len(
                    new_players
                ),
                "mean_evidence_confidence":
                    float(
                        new_players[
                            "evidence_confidence"
                        ].mean()
                    ),
                "full_confidence_rate":
                    float(
                        new_players[
                            "evidence_confidence"
                        ].ge(1.0).mean()
                    ),
                "usable_role_rating_rate":
                    float(
                        new_players[
                            "has_usable_role_rating"
                        ].mean()
                    ),
                "mean_weighted_evidence":
                    float(
                        new_players[
                            "total_weighted_evidence"
                        ].mean()
                    ),
                "mean_season_count":
                    float(
                        new_players[
                            "season_count"
                        ].mean()
                    ),
            }
        ]
    )

    #
    # Evidence-band summary.
    #
    for (
        band,
        group,
    ) in new_players.groupby(
        "evidence_band",
        sort=True,
    ):
        rows.append(
            {
                "dimension":
                    "evidence_band",
                "group":
                    band,
                "players":
                    len(group),
                "mean_evidence_confidence":
                    float(
                        group[
                            "evidence_confidence"
                        ].mean()
                    ),
                "full_confidence_rate":
                    float(
                        group[
                            "evidence_confidence"
                        ].ge(1.0).mean()
                    ),
                "usable_role_rating_rate":
                    float(
                        group[
                            "has_usable_role_rating"
                        ].mean()
                    ),
                "mean_weighted_evidence":
                    float(
                        group[
                            "total_weighted_evidence"
                        ].mean()
                    ),
                "mean_season_count":
                    float(
                        group[
                            "season_count"
                        ].mean()
                    ),
            }
        )

    #
    # Source-competition contribution.
    #
    competition_counts: dict[
        str,
        int,
    ] = {}

    for value in new_players[
        "source_competitions"
    ].dropna():
        competitions = {
            item.strip()
            for item in str(
                value
            ).split(";")
            if item.strip()
        }

        for competition in competitions:
            competition_counts[
                competition
            ] = (
                competition_counts.get(
                    competition,
                    0,
                )
                + 1
            )

    for (
        competition,
        count,
    ) in sorted(
        competition_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):
        rows.append(
            {
                "dimension":
                    "source_competition",
                "group":
                    competition,
                "players":
                    count,
                "mean_evidence_confidence":
                    np.nan,
                "full_confidence_rate":
                    np.nan,
                "usable_role_rating_rate":
                    np.nan,
                "mean_weighted_evidence":
                    np.nan,
                "mean_season_count":
                    np.nan,
            }
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        NEW_PLAYER_SUMMARY_PATH,
        index=False,
    )

    return (
        output,
        new_players,
    )


def build_decision_summary(
    canonical: pd.DataFrame,
    candidate: pd.DataFrame,
    comparison: pd.DataFrame,
    new_players: pd.DataFrame,
) -> pd.DataFrame:
    usable_existing = (
        comparison.loc[
            comparison[
                "comparable_role_count"
            ].gt(0)
        ]
    )

    high_confidence_existing = (
        usable_existing.loc[
            usable_existing[
                "evidence_confidence_canonical"
            ].ge(0.70)
        ]
    )

    full_confidence_existing = (
        usable_existing.loc[
            usable_existing[
                "evidence_confidence_canonical"
            ].ge(1.0)
        ]
    )

    metrics = [
        {
            "metric":
                "canonical_players",
            "value":
                len(canonical),
        },
        {
            "metric":
                "candidate_players",
            "value":
                len(candidate),
        },
        {
            "metric":
                "population_growth",
            "value":
                len(candidate)
                - len(canonical),
        },
        {
            "metric":
                "population_growth_percent",
            "value":
                (
                    (
                        len(candidate)
                        - len(canonical)
                    )
                    / len(canonical)
                    * 100.0
                ),
        },
        {
            "metric":
                "canonical_player_retention_rate",
            "value":
                (
                    len(comparison)
                    / len(canonical)
                ),
        },
        {
            "metric":
                "existing_players_with_comparable_ratings",
            "value":
                len(
                    usable_existing
                ),
        },
        {
            "metric":
                "existing_median_absolute_role_change",
            "value":
                float(
                    usable_existing[
                        "mean_absolute_role_rating_change"
                    ].median()
                ),
        },
        {
            "metric":
                "existing_p90_absolute_role_change",
            "value":
                float(
                    usable_existing[
                        "mean_absolute_role_rating_change"
                    ].quantile(
                        0.90
                    )
                ),
        },
        {
            "metric":
                "high_confidence_existing_players",
            "value":
                len(
                    high_confidence_existing
                ),
        },
        {
            "metric":
                "high_confidence_median_absolute_role_change",
            "value":
                float(
                    high_confidence_existing[
                        "mean_absolute_role_rating_change"
                    ].median()
                )
                if not high_confidence_existing.empty
                else np.nan,
        },
        {
            "metric":
                "high_confidence_p90_absolute_role_change",
            "value":
                float(
                    high_confidence_existing[
                        "mean_absolute_role_rating_change"
                    ].quantile(
                        0.90
                    )
                )
                if not high_confidence_existing.empty
                else np.nan,
        },
        {
            "metric":
                "full_confidence_existing_players",
            "value":
                len(
                    full_confidence_existing
                ),
        },
        {
            "metric":
                "full_confidence_median_absolute_role_change",
            "value":
                float(
                    full_confidence_existing[
                        "mean_absolute_role_rating_change"
                    ].median()
                )
                if not full_confidence_existing.empty
                else np.nan,
        },
        {
            "metric":
                "new_candidate_players",
            "value":
                len(
                    new_players
                ),
        },
        {
            "metric":
                "new_player_full_confidence_rate",
            "value":
                float(
                    new_players[
                        "evidence_confidence"
                    ].ge(1.0).mean()
                ),
        },
        {
            "metric":
                "new_player_usable_role_rating_rate",
            "value":
                float(
                    new_players[
                        "has_usable_role_rating"
                    ].mean()
                ),
        },
    ]

    output = pd.DataFrame(
        metrics
    )

    output.to_csv(
        DECISION_SUMMARY_PATH,
        index=False,
    )

    return output


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 101D — EXPANDED PLAYER "
        "INTELLIGENCE STABILITY EVALUATION"
    )
    print("=" * 88)

    canonical, candidate = (
        load_inputs()
    )

    comparison, _ = (
        build_existing_player_comparison(
            canonical,
            candidate,
        )
    )

    if len(comparison) != len(
        canonical
    ):
        raise AssertionError(
            "Not every canonical player was retained "
            "in the candidate population."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_largest_movers(
        comparison
    )

    stability = (
        build_stability_by_evidence(
            comparison
        )
    )

    (
        new_summary,
        new_players,
    ) = build_new_player_summary(
        canonical,
        candidate,
    )

    decision = (
        build_decision_summary(
            canonical,
            candidate,
            comparison,
            new_players,
        )
    )

    print()
    print("Population")
    print("-" * 88)
    print(
        f"Canonical players: {len(canonical)}"
    )
    print(
        f"Candidate players: {len(candidate)}"
    )
    print(
        f"Retained canonical players: "
        f"{len(comparison)}"
    )
    print(
        f"New candidate players: "
        f"{len(new_players)}"
    )

    print()
    print("Stability by canonical evidence")
    print("-" * 88)
    print(
        stability.to_string(
            index=False
        )
    )

    print()
    print("New-player coverage")
    print("-" * 88)

    overall_new = (
        new_summary.loc[
            new_summary[
                "dimension"
            ].eq("overall")
        ]
    )

    print(
        overall_new.to_string(
            index=False
        )
    )

    print()
    print("Decision summary")
    print("-" * 88)
    print(
        decision.to_string(
            index=False
        )
    )

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)

    print()
    print(
        f"Largest movers: "
        f"{LARGEST_MOVERS_PATH}"
    )
    print(
        f"Stability by evidence: "
        f"{STABILITY_BY_EVIDENCE_PATH}"
    )
    print(
        f"New-player coverage: "
        f"{NEW_PLAYER_SUMMARY_PATH}"
    )
    print(
        f"Decision summary: "
        f"{DECISION_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
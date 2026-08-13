#validate_101f_rating_impact

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STUDY_101E_RATINGS = (
    PROJECT_ROOT
    / "outputs"
    / "study_101e_registry_provenance_repair"
    / "candidate_player_ratings.csv"
)

STUDY_101F_RATINGS = (
    PROJECT_ROOT
    / "outputs"
    / "study_101f_weighted_player_ratings.csv"
)

STUDY_101F_FEATURES = (
    PROJECT_ROOT
    / "outputs"
    / "study_101f_canonical_competition_season_features.csv"
)


def main() -> None:
    old = pd.read_csv(
        STUDY_101E_RATINGS,
        low_memory=False,
    )

    new = pd.read_csv(
        STUDY_101F_RATINGS,
        low_memory=False,
    )

    features = pd.read_csv(
        STUDY_101F_FEATURES,
        low_memory=False,
    )

    old_rating_columns = {
        column
        for column in old.columns
        if (
            column.startswith("rating_")
            and not column.startswith("raw_rating_")
        )
    }

    new_rating_columns = {
        column
        for column in new.columns
        if (
            column.startswith("rating_")
            and not column.startswith("raw_rating_")
        )
    }

    common_rating_columns = sorted(
        old_rating_columns
        & new_rating_columns
    )

    old_ids = set(
        old["canonical_player_id"]
    )

    new_ids = set(
        new["canonical_player_id"]
    )

    print(
        f"101E rating rows: {len(old):,}"
    )

    print(
        "101E unique canonical IDs: "
        f"{old['canonical_player_id'].nunique():,}"
    )

    print(
        f"101F rating rows: {len(new):,}"
    )

    print(
        "101F unique canonical IDs: "
        f"{new['canonical_player_id'].nunique():,}"
    )

    print(
        f"Matched canonical IDs: "
        f"{len(old_ids & new_ids):,}"
    )

    print(
        f"101E-only canonical IDs: "
        f"{len(old_ids - new_ids):,}"
    )

    print(
        f"101F-only canonical IDs: "
        f"{len(new_ids - old_ids):,}"
    )

    print(
        "Common role-rating columns: "
        f"{len(common_rating_columns)}"
    )

    print(
        "101F feature rows: "
        f"{len(features):,}"
    )

    print(
        "101F feature canonical players: "
        f"{features['canonical_player_id'].nunique():,}"
    )

    scope_counts = (
        features.groupby(
            "canonical_player_id"
        )
        .size()
        .rename("evidence_scope_count")
        .reset_index()
    )

    scope_counts["scope_group"] = (
        scope_counts[
            "evidence_scope_count"
        ]
        .gt(1)
        .map(
            {
                False: "single_scope",
                True: "multi_scope",
            }
        )
    )

    old_subset = old[
        [
            "canonical_player_id",
        ]
        + common_rating_columns
    ].copy()

    new_subset = new[
        [
            "canonical_player_id",
        ]
        + common_rating_columns
    ].copy()

    comparison = new_subset.merge(
        old_subset,
        on="canonical_player_id",
        how="inner",
        suffixes=(
            "_101f",
            "_101e",
        ),
        validate="one_to_one",
    )

    comparison = comparison.merge(
        scope_counts,
        on="canonical_player_id",
        how="left",
        validate="one_to_one",
    )

    delta_columns = []

    for rating_column in common_rating_columns:
        new_column = (
            f"{rating_column}_101f"
        )

        old_column = (
            f"{rating_column}_101e"
        )

        delta_column = (
            f"delta_{rating_column}"
        )

        comparison[delta_column] = (
            pd.to_numeric(
                comparison[new_column],
                errors="coerce",
            )
            - pd.to_numeric(
                comparison[old_column],
                errors="coerce",
            )
        )

        delta_columns.append(
            delta_column
        )

    comparison[
        "mean_absolute_role_rating_change"
    ] = (
        comparison[
            delta_columns
        ]
        .abs()
        .mean(
            axis=1,
            skipna=True,
        )
    )

    comparison[
        "maximum_absolute_role_rating_change"
    ] = (
        comparison[
            delta_columns
        ]
        .abs()
        .max(
            axis=1,
            skipna=True,
        )
    )

    print()
    print("101E -> 101F matched comparison")
    print("-" * 88)
    print(
        f"Matched players: "
        f"{len(comparison):,}"
    )

    print()
    print("Evidence-scope population")
    print("-" * 88)
    print(
        comparison[
            "scope_group"
        ]
        .value_counts()
        .to_string()
    )

    def summarize_rating_change(
        frame: pd.DataFrame,
        label: str,
    ) -> dict[str, object]:
        change = pd.to_numeric(
            frame[
                "mean_absolute_role_rating_change"
            ],
            errors="coerce",
        ).dropna()

        maximum_change = pd.to_numeric(
            frame[
                "maximum_absolute_role_rating_change"
            ],
            errors="coerce",
        ).dropna()

        return {
            "population": label,
            "players": len(frame),
            "players_with_comparable_ratings": len(change),
            "mean_absolute_role_rating_change": (
                float(change.mean())
                if not change.empty
                else float("nan")
            ),
            "median_absolute_role_rating_change": (
                float(change.median())
                if not change.empty
                else float("nan")
            ),
            "p90_absolute_role_rating_change": (
                float(change.quantile(0.90))
                if not change.empty
                else float("nan")
            ),
            "maximum_observed_role_rating_change": (
                float(maximum_change.max())
                if not maximum_change.empty
                else float("nan")
            ),
        }

    summary_rows = [
        summarize_rating_change(
            comparison,
            "all_matched",
        ),
        summarize_rating_change(
            comparison.loc[
                comparison[
                    "scope_group"
                ].eq("single_scope")
            ],
            "single_scope",
        ),
        summarize_rating_change(
            comparison.loc[
                comparison[
                    "scope_group"
                ].eq("multi_scope")
            ],
            "multi_scope",
        ),
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("Rating-impact summary")
    print("-" * 88)
    print(
        summary.to_string(
            index=False
        )
    )

    player_metadata = new[
        [
            column
            for column in [
                "canonical_player_id",
                "player",
                "country",
                "current_team",
                "best_role",
                "best_rating",
            ]
            if column in new.columns
        ]
    ].copy()

    comparison = comparison.merge(
        player_metadata,
        on="canonical_player_id",
        how="left",
        validate="one_to_one",
    )

    top_movers = (
        comparison.loc[
            comparison[
                "mean_absolute_role_rating_change"
            ].notna()
        ]
        .sort_values(
            "mean_absolute_role_rating_change",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    display_columns = [
        column
        for column in [
            "canonical_player_id",
            "player",
            "country",
            "current_team",
            "scope_group",
            "evidence_scope_count",
            "best_role",
            "best_rating",
            "mean_absolute_role_rating_change",
            "maximum_absolute_role_rating_change",
        ]
        if column in top_movers.columns
    ]

    print()
    print("Top 20 rating movers")
    print("-" * 88)
    print(
        top_movers[
            display_columns
        ].to_string(
            index=False
        )
    )

    top_single_scope = (
        comparison.loc[
            comparison[
                "scope_group"
            ].eq("single_scope")
            & comparison[
                "mean_absolute_role_rating_change"
            ].notna()
        ]
        .sort_values(
            "mean_absolute_role_rating_change",
            ascending=False,
        )
        .head(10)
    )

    print()
    print("Top 10 single-scope movers")
    print("-" * 88)
    print(
        top_single_scope[
            display_columns
        ].to_string(
            index=False
        )
    )

if __name__ == "__main__":
    main()
#audit_club_observation_datasets

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_048_club_observation_dataset"
)

FULL_SQUAD_PATH = (
    INPUT_DIR
    / "full_squad_observations.csv"
)

EXPECTED_XI_PATH = (
    INPUT_DIR
    / "expected_starting_xi_observations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_049_club_observation_audit"
)


BASE_DIMENSIONS = (
    "attack",
    "midfield",
    "defense",
    "goalkeeper",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
)

DIFFERENCE_FEATURES = tuple(
    f"{dimension}_diff"
    for dimension in BASE_DIMENSIONS
)

SIDE_FEATURES = tuple(
    f"{side}_{dimension}"
    for side in ("home", "away")
    for dimension in BASE_DIMENSIONS
)

TARGET_COLUMNS = (
    "home_score",
    "away_score",
    "total_goals",
    "goal_difference",
    "is_draw",
    "is_home_win",
    "is_away_win",
    "both_teams_scored",
    "is_clean_sheet",
    "is_high_scoring",
    "is_blowout",
)

IDENTITY_COLUMNS = (
    "competition_key",
    "prediction_season_start_year",
    "representation_season_start_year",
    "event_id",
    "date",
    "home_team",
    "home_team_id",
    "away_team",
    "away_team_id",
)


def load_dataset(
    path: Path,
    expected_representation_type: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Observation dataset does not exist: {path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Observation dataset is empty: {path}"
        )

    required_columns = {
        *IDENTITY_COLUMNS,
        *SIDE_FEATURES,
        *DIFFERENCE_FEATURES,
        *TARGET_COLUMNS,
        "home_representation_type",
        "away_representation_type",
    }

    missing = required_columns - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    if dataframe["event_id"].duplicated().any():
        duplicates = (
            dataframe.loc[
                dataframe["event_id"].duplicated(
                    keep=False
                ),
                "event_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{path.name} contains duplicate event IDs: "
            f"{duplicates[:20]}"
        )

    invalid_home_types = dataframe[
        "home_representation_type"
    ].ne(expected_representation_type)

    invalid_away_types = dataframe[
        "away_representation_type"
    ].ne(expected_representation_type)

    if (
        invalid_home_types.any()
        or invalid_away_types.any()
    ):
        raise ValueError(
            f"{path.name} contains an unexpected "
            "representation type."
        )

    numeric_columns = [
        *SIDE_FEATURES,
        *DIFFERENCE_FEATURES,
        *TARGET_COLUMNS,
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    return (
        dataframe
        .sort_values("event_id")
        .reset_index(drop=True)
    )


def validate_matched_populations(
    full_squad: pd.DataFrame,
    expected_xi: pd.DataFrame,
) -> None:
    full_events = set(
        full_squad["event_id"]
    )

    xi_events = set(
        expected_xi["event_id"]
    )

    if full_events != xi_events:
        only_full = sorted(
            full_events - xi_events
        )

        only_xi = sorted(
            xi_events - full_events
        )

        raise AssertionError(
            "Observation populations do not match. "
            f"Only full squad: {only_full[:20]}; "
            f"only expected XI: {only_xi[:20]}."
        )

    full_indexed = (
        full_squad
        .set_index("event_id")
        .sort_index()
    )

    xi_indexed = (
        expected_xi
        .set_index("event_id")
        .sort_index()
    )

    comparison_columns = [
        *[
            column
            for column in IDENTITY_COLUMNS
            if column != "event_id"
        ],
        *TARGET_COLUMNS,
    ]

    for column in comparison_columns:
        left = full_indexed[column]
        right = xi_indexed[column]

        if pd.api.types.is_numeric_dtype(left):
            equal = np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
            ).all()
        else:
            equal = (
                left.fillna("<missing>")
                .astype(str)
                .eq(
                    right.fillna("<missing>")
                    .astype(str)
                )
                .all()
            )

        if not equal:
            raise AssertionError(
                "Matched observation datasets disagree "
                f"on {column!r}."
            )


def build_distribution_summary(
    full_squad: pd.DataFrame,
    expected_xi: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    feature_columns = [
        *SIDE_FEATURES,
        *DIFFERENCE_FEATURES,
    ]

    datasets = {
        "full_squad": full_squad,
        "expected_starting_xi": expected_xi,
    }

    for representation_type, dataframe in (
        datasets.items()
    ):
        for feature in feature_columns:
            values = dataframe[feature]

            rows.append(
                {
                    "representation_type":
                        representation_type,
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "standard_deviation": float(
                        values.std()
                    ),
                    "minimum": float(values.min()),
                    "first_quartile": float(
                        values.quantile(0.25)
                    ),
                    "median": float(
                        values.median()
                    ),
                    "third_quartile": float(
                        values.quantile(0.75)
                    ),
                    "maximum": float(values.max()),
                    "zero_count": int(
                        values.eq(0).sum()
                    ),
                    "zero_rate": float(
                        values.eq(0).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_paired_feature_comparison(
    full_squad: pd.DataFrame,
    expected_xi: pd.DataFrame,
) -> pd.DataFrame:
    full_indexed = (
        full_squad
        .set_index("event_id")
        .sort_index()
    )

    xi_indexed = (
        expected_xi
        .set_index("event_id")
        .sort_index()
    )

    rows: list[dict[str, object]] = []

    feature_columns = [
        *SIDE_FEATURES,
        *DIFFERENCE_FEATURES,
    ]

    for feature in feature_columns:
        full_values = full_indexed[feature]
        xi_values = xi_indexed[feature]

        paired_difference = (
            xi_values - full_values
        )

        pearson_correlation = (
            full_values.corr(
                xi_values,
                method="pearson",
            )
        )

        spearman_correlation = (
            full_values.corr(
                xi_values,
                method="spearman",
            )
        )

        rows.append(
            {
                "feature": feature,
                "comparison_count": int(
                    paired_difference.count()
                ),
                "full_squad_mean": float(
                    full_values.mean()
                ),
                "expected_xi_mean": float(
                    xi_values.mean()
                ),
                "mean_difference": float(
                    paired_difference.mean()
                ),
                "median_difference": float(
                    paired_difference.median()
                ),
                "mean_absolute_difference": float(
                    paired_difference.abs().mean()
                ),
                "maximum_absolute_difference": float(
                    paired_difference.abs().max()
                ),
                "expected_xi_higher_count": int(
                    paired_difference.gt(0).sum()
                ),
                "full_squad_higher_count": int(
                    paired_difference.lt(0).sum()
                ),
                "equal_count": int(
                    paired_difference.eq(0).sum()
                ),
                "pearson_correlation": float(
                    pearson_correlation
                ),
                "spearman_correlation": float(
                    spearman_correlation
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


def build_feature_target_relationships(
    full_squad: pd.DataFrame,
    expected_xi: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    relationship_targets = (
        "home_score",
        "away_score",
        "total_goals",
        "goal_difference",
        "is_draw",
        "is_home_win",
        "both_teams_scored",
    )

    datasets = {
        "full_squad": full_squad,
        "expected_starting_xi": expected_xi,
    }

    for representation_type, dataframe in (
        datasets.items()
    ):
        for feature in DIFFERENCE_FEATURES:
            for target in relationship_targets:
                pearson = dataframe[
                    feature
                ].corr(
                    dataframe[target],
                    method="pearson",
                )

                spearman = dataframe[
                    feature
                ].corr(
                    dataframe[target],
                    method="spearman",
                )

                rows.append(
                    {
                        "representation_type":
                            representation_type,
                        "feature": feature,
                        "target": target,
                        "pearson_correlation": float(
                            pearson
                        ),
                        "spearman_correlation": float(
                            spearman
                        ),
                        "absolute_pearson_correlation":
                            float(abs(pearson)),
                        "absolute_spearman_correlation":
                            float(abs(spearman)),
                    }
                )

    return pd.DataFrame(rows)


def build_largest_match_shifts(
    full_squad: pd.DataFrame,
    expected_xi: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    identity = full_squad[
        [
            "event_id",
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    full_indexed = (
        full_squad
        .set_index("event_id")
        .sort_index()
    )

    xi_indexed = (
        expected_xi
        .set_index("event_id")
        .sort_index()
    )

    shift_columns: list[str] = []

    for feature in DIFFERENCE_FEATURES:
        shift_column = (
            f"{feature}_representation_shift"
        )

        identity[shift_column] = (
            identity["event_id"]
            .map(
                xi_indexed[feature]
                - full_indexed[feature]
            )
        )

        identity[
            f"{feature}_absolute_representation_shift"
        ] = identity[
            shift_column
        ].abs()

        shift_columns.append(
            f"{feature}_absolute_representation_shift"
        )

    identity["mean_absolute_representation_shift"] = (
        identity[shift_columns]
        .mean(axis=1)
    )

    identity["maximum_absolute_representation_shift"] = (
        identity[shift_columns]
        .max(axis=1)
    )

    identity["most_changed_feature"] = (
        identity[shift_columns]
        .idxmax(axis=1)
        .str.replace(
            "_absolute_representation_shift",
            "",
            regex=False,
        )
    )

    return (
        identity
        .sort_values(
            "mean_absolute_representation_shift",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def build_target_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "metric": "matches",
            "value": len(dataframe),
        },
        {
            "metric": "mean_home_goals",
            "value": dataframe[
                "home_score"
            ].mean(),
        },
        {
            "metric": "mean_away_goals",
            "value": dataframe[
                "away_score"
            ].mean(),
        },
        {
            "metric": "mean_total_goals",
            "value": dataframe[
                "total_goals"
            ].mean(),
        },
        {
            "metric": "home_win_rate",
            "value": dataframe[
                "is_home_win"
            ].mean(),
        },
        {
            "metric": "draw_rate",
            "value": dataframe[
                "is_draw"
            ].mean(),
        },
        {
            "metric": "away_win_rate",
            "value": dataframe[
                "is_away_win"
            ].mean(),
        },
        {
            "metric": "both_teams_scored_rate",
            "value": dataframe[
                "both_teams_scored"
            ].mean(),
        },
        {
            "metric": "clean_sheet_rate",
            "value": dataframe[
                "is_clean_sheet"
            ].mean(),
        },
        {
            "metric": "high_scoring_rate",
            "value": dataframe[
                "is_high_scoring"
            ].mean(),
        },
        {
            "metric": "blowout_rate",
            "value": dataframe[
                "is_blowout"
            ].mean(),
        },
    ]

    return pd.DataFrame(rows)


def determine_status(
    full_squad: pd.DataFrame,
    expected_xi: pd.DataFrame,
    paired_comparison: pd.DataFrame,
) -> str:
    if (
        full_squad.empty
        or expected_xi.empty
    ):
        return "FAIL"

    if len(full_squad) != len(expected_xi):
        return "FAIL"

    if paired_comparison.empty:
        return "FAIL"

    if paired_comparison[
        "mean_absolute_difference"
    ].isna().any():
        return "FAIL"

    return "PASS"


def write_results_markdown(
    path: Path,
    full_squad: pd.DataFrame,
    paired_comparison: pd.DataFrame,
    largest_shifts: pd.DataFrame,
    status: str,
) -> None:
    most_changed = paired_comparison.iloc[0]

    least_changed = paired_comparison.iloc[-1]

    largest_match = largest_shifts.iloc[0]

    lines = [
        "# Study 049 Results",
        "",
        "## Club Observation Dataset Audit",
        "",
        f"**Status:** `{status}`",
        "",
        "## Population integrity",
        "",
        (
            f"- Matched observations: "
            f"{len(full_squad)}"
        ),
        (
            "- Full-squad and expected-XI event "
            "populations are identical."
        ),
        (
            "- Match identities and observed outcomes "
            "are identical across datasets."
        ),
        "",
        "## Most changed feature",
        "",
        (
            f"- Feature: "
            f"`{most_changed['feature']}`"
        ),
        (
            "- Mean absolute difference: "
            f"{most_changed['mean_absolute_difference']:.6f}"
        ),
        (
            "- Pearson agreement: "
            f"{most_changed['pearson_correlation']:.6f}"
        ),
        "",
        "## Least changed feature",
        "",
        (
            f"- Feature: "
            f"`{least_changed['feature']}`"
        ),
        (
            "- Mean absolute difference: "
            f"{least_changed['mean_absolute_difference']:.6f}"
        ),
        "",
        "## Largest match-level representation shift",
        "",
        (
            f"- Match: {largest_match['home_team']} "
            f"vs {largest_match['away_team']}"
        ),
        (
            f"- Score: {largest_match['home_score']}-"
            f"{largest_match['away_score']}"
        ),
        (
            "- Mean absolute shift: "
            f"{largest_match['mean_absolute_representation_shift']:.6f}"
        ),
        (
            "- Most changed feature: "
            f"{largest_match['most_changed_feature']}"
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "This study measures representation differences "
            "and univariate relationships only. It does not "
            "establish predictive superiority."
        ),
        "",
        "## Decision gate",
        "",
        (
            "If the two feature systems show non-trivial "
            "differences while retaining the same match "
            "population, proceed to a controlled predictive "
            "benchmark using identical train/test procedures."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    full_squad = load_dataset(
        FULL_SQUAD_PATH,
        expected_representation_type=(
            "full_squad"
        ),
    )

    expected_xi = load_dataset(
        EXPECTED_XI_PATH,
        expected_representation_type=(
            "expected_starting_xi"
        ),
    )

    validate_matched_populations(
        full_squad=full_squad,
        expected_xi=expected_xi,
    )

    distribution_summary = (
        build_distribution_summary(
            full_squad=full_squad,
            expected_xi=expected_xi,
        )
    )

    paired_comparison = (
        build_paired_feature_comparison(
            full_squad=full_squad,
            expected_xi=expected_xi,
        )
    )

    feature_target_relationships = (
        build_feature_target_relationships(
            full_squad=full_squad,
            expected_xi=expected_xi,
        )
    )

    largest_shifts = (
        build_largest_match_shifts(
            full_squad=full_squad,
            expected_xi=expected_xi,
            top_n=25,
        )
    )

    target_summary = build_target_summary(
        full_squad
    )

    status = determine_status(
        full_squad=full_squad,
        expected_xi=expected_xi,
        paired_comparison=paired_comparison,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution_summary.to_csv(
        OUTPUT_DIR
        / "feature_distribution_summary.csv",
        index=False,
    )

    paired_comparison.to_csv(
        OUTPUT_DIR
        / "paired_feature_comparison.csv",
        index=False,
    )

    feature_target_relationships.to_csv(
        OUTPUT_DIR
        / "feature_target_relationships.csv",
        index=False,
    )

    largest_shifts.to_csv(
        OUTPUT_DIR
        / "largest_match_representation_shifts.csv",
        index=False,
    )

    target_summary.to_csv(
        OUTPUT_DIR
        / "target_population_summary.csv",
        index=False,
    )

    metadata = {
        "study_id": "049",
        "study_name": (
            "Club Observation Dataset Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": status,
        "observation_count": int(
            len(full_squad)
        ),
        "event_populations_equal": True,
        "identity_columns_equal": True,
        "target_columns_equal": True,
        "base_dimensions": list(
            BASE_DIMENSIONS
        ),
        "difference_features": list(
            DIFFERENCE_FEATURES
        ),
        "rating_prior_included": False,
        "output_files": [
            "feature_distribution_summary.csv",
            "paired_feature_comparison.csv",
            "feature_target_relationships.csv",
            (
                "largest_match_"
                "representation_shifts.csv"
            ),
            "target_population_summary.csv",
            "study_metadata.json",
            "STUDY_049_RESULTS.md",
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
            / "STUDY_049_RESULTS.md"
        ),
        full_squad=full_squad,
        paired_comparison=(
            paired_comparison
        ),
        largest_shifts=largest_shifts,
        status=status,
    )

    print("Study 049")
    print("=" * 64)
    print()
    print(
        f"Matched observations: "
        f"{len(full_squad)}"
    )
    print(
        "Event population equality: PASS"
    )
    print(
        "Match identity equality: PASS"
    )
    print(
        "Observed-target equality: PASS"
    )
    print()

    print("Paired Feature Comparison")
    print("-" * 64)
    print(
        paired_comparison[
            [
                "feature",
                "full_squad_mean",
                "expected_xi_mean",
                "mean_difference",
                "mean_absolute_difference",
                "pearson_correlation",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Largest Match-Level Shifts")
    print("-" * 64)
    print(
        largest_shifts[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                (
                    "mean_absolute_"
                    "representation_shift"
                ),
                "most_changed_feature",
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
    print("-" * 64)
    print(status)
    print()
    print(
        f"Outputs written to: {OUTPUT_DIR}"
    )

    if status != "PASS":
        raise AssertionError(
            "Study 049 did not pass."
        )


if __name__ == "__main__":
    main()
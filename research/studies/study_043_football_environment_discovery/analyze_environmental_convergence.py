#analyze_environmental_convergence

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from research.studies.study_043_football_environment_discovery.ml.pca_geometry import (
    euclidean_distance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_043_football_environment_discovery"
    / "outputs"
)

COORDINATE_PATH = (
    OUTPUT_DIRECTORY
    / "pca_coordinates_reduced.csv"
)

COMPONENT_COLUMNS = [
    "PC1",
    "PC2",
]

REQUIRED_COLUMNS = {
    "observation_id",
    "competition_key",
    "competition_name",
    "season_start_year",
    "PC1",
    "PC2",
}


def load_coordinates(
    coordinate_path: Path,
) -> pd.DataFrame:
    if not coordinate_path.exists():
        raise FileNotFoundError(
            "PCA coordinate dataset was not found:\n"
            f"{coordinate_path}"
        )

    dataframe = pd.read_csv(
        coordinate_path
    )

    if dataframe.empty:
        raise ValueError(
            "PCA coordinate dataset is empty."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "PCA coordinate dataset is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["season_start_year"] = (
        pd.to_numeric(
            dataframe["season_start_year"],
            errors="raise",
        )
        .astype(int)
    )

    for component in COMPONENT_COLUMNS:
        dataframe[component] = pd.to_numeric(
            dataframe[component],
            errors="raise",
        )

    if dataframe["observation_id"].duplicated().any():
        duplicate_ids = sorted(
            dataframe.loc[
                dataframe[
                    "observation_id"
                ].duplicated(
                    keep=False
                ),
                "observation_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate PCA observation IDs found: "
            f"{duplicate_ids}"
        )

    duplicate_league_seasons = dataframe[
        dataframe.duplicated(
            subset=[
                "competition_key",
                "season_start_year",
            ],
            keep=False,
        )
    ]

    if not duplicate_league_seasons.empty:
        preview = (
            duplicate_league_seasons[
                [
                    "competition_key",
                    "season_start_year",
                ]
            ]
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate league-season coordinates found: "
            f"{preview}"
        )

    return dataframe.sort_values(
        [
            "season_start_year",
            "competition_key",
        ]
    ).reset_index(drop=True)


def validate_balanced_panel(
    dataframe: pd.DataFrame,
) -> None:
    seasons = sorted(
        dataframe[
            "season_start_year"
        ].unique()
    )

    competitions = sorted(
        dataframe[
            "competition_key"
        ].unique()
    )

    expected_observations = (
        len(seasons)
        * len(competitions)
    )

    if len(dataframe) != expected_observations:
        raise ValueError(
            "PCA coordinate dataset is not a balanced "
            "competition-season panel. Expected "
            f"{expected_observations} rows from "
            f"{len(competitions)} competitions and "
            f"{len(seasons)} seasons, but found "
            f"{len(dataframe)}."
        )

    counts = (
        dataframe.groupby(
            "competition_key"
        )[
            "season_start_year"
        ]
        .nunique()
    )

    invalid = counts[
        counts != len(seasons)
    ]

    if not invalid.empty:
        raise ValueError(
            "One or more competitions do not contain every "
            f"season: {invalid.to_dict()}"
        )


def build_season_pair_distances(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for season_start_year, season_data in (
        dataframe.groupby(
            "season_start_year",
            sort=True,
        )
    ):
        season_data = (
            season_data.sort_values(
                "competition_key"
            )
            .reset_index(drop=True)
        )

        for first_index, second_index in combinations(
            range(len(season_data)),
            2,
        ):
            first = season_data.iloc[
                first_index
            ]

            second = season_data.iloc[
                second_index
            ]

            distance = euclidean_distance(
                first[COMPONENT_COLUMNS]
                .to_numpy(dtype=float),
                second[COMPONENT_COLUMNS]
                .to_numpy(dtype=float),
            )

            rows.append(
                {
                    "season_start_year": (
                        int(season_start_year)
                    ),
                    "competition_key_a": (
                        first["competition_key"]
                    ),
                    "competition_name_a": (
                        first["competition_name"]
                    ),
                    "competition_key_b": (
                        second["competition_key"]
                    ),
                    "competition_name_b": (
                        second["competition_name"]
                    ),
                    "league_pair": (
                        " | ".join(
                            sorted(
                                [
                                    str(
                                        first[
                                            "competition_key"
                                        ]
                                    ),
                                    str(
                                        second[
                                            "competition_key"
                                        ]
                                    ),
                                ]
                            )
                        )
                    ),
                    "distance": distance,
                }
            )

    return pd.DataFrame(rows)


def summarize_pair_trends(
    pair_distances: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for league_pair, group in (
        pair_distances.groupby(
            "league_pair",
            sort=True,
        )
    ):
        ordered = (
            group.sort_values(
                "season_start_year"
            )
            .reset_index(drop=True)
        )

        first_distance = float(
            ordered.iloc[0][
                "distance"
            ]
        )

        final_distance = float(
            ordered.iloc[-1][
                "distance"
            ]
        )

        net_change = (
            final_distance
            - first_distance
        )

        if net_change < 0:
            direction = "converging"
        elif net_change > 0:
            direction = "diverging"
        else:
            direction = "unchanged"

        rows.append(
            {
                "league_pair": league_pair,
                "competition_key_a": (
                    ordered.iloc[0][
                        "competition_key_a"
                    ]
                ),
                "competition_key_b": (
                    ordered.iloc[0][
                        "competition_key_b"
                    ]
                ),
                "season_count": len(
                    ordered
                ),
                "initial_distance": (
                    first_distance
                ),
                "final_distance": (
                    final_distance
                ),
                "net_distance_change": (
                    net_change
                ),
                "absolute_net_change": abs(
                    net_change
                ),
                "mean_distance": float(
                    ordered["distance"].mean()
                ),
                "minimum_distance": float(
                    ordered["distance"].min()
                ),
                "maximum_distance": float(
                    ordered["distance"].max()
                ),
                "trend_direction": direction,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "net_distance_change"
        )
        .reset_index(drop=True)
    )


def summarize_seasons(
    pair_distances: pd.DataFrame,
) -> pd.DataFrame:
    return (
        pair_distances.groupby(
            "season_start_year",
            as_index=False,
            sort=True,
        )
        .agg(
            pair_count=(
                "distance",
                "size",
            ),
            mean_between_league_distance=(
                "distance",
                "mean",
            ),
            median_between_league_distance=(
                "distance",
                "median",
            ),
            minimum_between_league_distance=(
                "distance",
                "min",
            ),
            maximum_between_league_distance=(
                "distance",
                "max",
            ),
            standard_deviation=(
                "distance",
                "std",
            ),
        )
    )


def write_outputs(
    pair_distances: pd.DataFrame,
    pair_trends: pd.DataFrame,
    season_summary: pd.DataFrame,
    source_path: Path,
) -> dict[str, Path]:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "pair_distances": (
            OUTPUT_DIRECTORY
            / "environmental_pair_distances_by_season.csv"
        ),
        "pair_trends": (
            OUTPUT_DIRECTORY
            / "environmental_pair_convergence_summary.csv"
        ),
        "season_summary": (
            OUTPUT_DIRECTORY
            / "environmental_separation_by_season.csv"
        ),
        "metadata": (
            OUTPUT_DIRECTORY
            / "environmental_convergence_metadata.json"
        ),
    }

    pair_distances.to_csv(
        paths["pair_distances"],
        index=False,
        encoding="utf-8",
    )

    pair_trends.to_csv(
        paths["pair_trends"],
        index=False,
        encoding="utf-8",
    )

    season_summary.to_csv(
        paths["season_summary"],
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "study": (
            "study_043_football_environment_discovery"
        ),
        "experiment": (
            "environmental_convergence"
        ),
        "source_path": str(
            source_path
        ),
        "components_used": (
            COMPONENT_COLUMNS
        ),
        "distance_metric": (
            "Euclidean distance"
        ),
        "pair_distance_rows": len(
            pair_distances
        ),
        "league_pair_count": int(
            pair_distances[
                "league_pair"
            ].nunique()
        ),
        "season_count": int(
            pair_distances[
                "season_start_year"
            ].nunique()
        ),
        "outputs": {
            label: str(path)
            for label, path in paths.items()
            if label != "metadata"
        },
    }

    paths["metadata"].write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return paths


def print_summary(
    pair_trends: pd.DataFrame,
    season_summary: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    print("Study 043 — Environmental Convergence")
    print("=====================================")
    print(
        f"League pairs: "
        f"{len(pair_trends)}"
    )
    print(
        f"Seasons: "
        f"{len(season_summary)}"
    )
    print()

    print("Pairwise Convergence Summary")
    print("----------------------------")
    print(
        pair_trends[
            [
                "league_pair",
                "initial_distance",
                "final_distance",
                "net_distance_change",
                "trend_direction",
            ]
        ].to_string(
            index=False
        )
    )
    print()

    print("Overall Separation by Season")
    print("----------------------------")
    print(
        season_summary.to_string(
            index=False
        )
    )
    print()

    print("Most Convergent Pair")
    print("--------------------")

    most_convergent = (
        pair_trends.sort_values(
            "net_distance_change"
        )
        .iloc[0]
    )

    print(
        f"{most_convergent['league_pair']}: "
        f"{most_convergent['net_distance_change']:.6f}"
    )
    print()

    print("Most Divergent Pair")
    print("-------------------")

    most_divergent = (
        pair_trends.sort_values(
            "net_distance_change",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        f"{most_divergent['league_pair']}: "
        f"{most_divergent['net_distance_change']:.6f}"
    )
    print()

    print("Outputs")
    print("-------")

    for label, path in output_paths.items():
        print(f"{label}: {path}")

    print()
    print("Analysis Result")
    print("---------------")
    print("PASSED")


def main() -> None:
    coordinates = load_coordinates(
        COORDINATE_PATH
    )

    validate_balanced_panel(
        coordinates
    )

    pair_distances = (
        build_season_pair_distances(
            coordinates
        )
    )

    pair_trends = summarize_pair_trends(
        pair_distances
    )

    season_summary = summarize_seasons(
        pair_distances
    )

    output_paths = write_outputs(
        pair_distances=pair_distances,
        pair_trends=pair_trends,
        season_summary=season_summary,
        source_path=COORDINATE_PATH,
    )

    print_summary(
        pair_trends=pair_trends,
        season_summary=season_summary,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()
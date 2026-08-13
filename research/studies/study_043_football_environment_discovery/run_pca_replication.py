#run_pca_replication

from __future__ import annotations

import json
from pathlib import Path

from research.studies.study_043_football_environment_discovery.ml.pca_loader import (
    PCAFeatureMatrix,
    load_pca_feature_matrix,
)
from research.studies.study_043_football_environment_discovery.ml.pca_model import (
    PCAResult,
    fit_pca,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_043_football_environment_discovery"
    / "outputs"
)

FEATURE_SET = "reduced"

EXPECTED_OBSERVATION_COUNT = 25
EXPECTED_FEATURE_COUNT = 5


def validate_replication_input(
    feature_matrix: PCAFeatureMatrix,
) -> None:
    """
    Confirm that the replication is using the intended expanded
    Study 043 input population and unchanged reduced feature set.
    """

    if (
        feature_matrix.feature_set
        != FEATURE_SET
    ):
        raise ValueError(
            "Unexpected PCA feature set. "
            f"Expected {FEATURE_SET!r}, but received "
            f"{feature_matrix.feature_set!r}."
        )

    if (
        feature_matrix.observation_count
        != EXPECTED_OBSERVATION_COUNT
    ):
        raise ValueError(
            "Unexpected replication observation count. "
            f"Expected {EXPECTED_OBSERVATION_COUNT}, but found "
            f"{feature_matrix.observation_count}. "
            "Confirm that the League-Season Repository and "
            "standardized feature matrices were regenerated."
        )

    if (
        feature_matrix.feature_count
        != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Unexpected reduced feature count. "
            f"Expected {EXPECTED_FEATURE_COUNT}, but found "
            f"{feature_matrix.feature_count}."
        )

    if (
        feature_matrix.to_numpy().shape
        != (
            EXPECTED_OBSERVATION_COUNT,
            EXPECTED_FEATURE_COUNT,
        )
    ):
        raise ValueError(
            "Unexpected PCA matrix shape. "
            f"Observed: {feature_matrix.to_numpy().shape}. "
            "Expected: "
            f"({EXPECTED_OBSERVATION_COUNT}, "
            f"{EXPECTED_FEATURE_COUNT})."
        )


def build_output_paths(
    output_directory: Path,
    feature_set: str,
) -> dict[str, Path]:
    """
    Resolve all canonical output paths for one PCA experiment.
    """

    return {
        "coordinates": (
            output_directory
            / f"pca_coordinates_{feature_set}.csv"
        ),
        "loadings": (
            output_directory
            / f"pca_component_loadings_{feature_set}.csv"
        ),
        "explained_variance": (
            output_directory
            / f"pca_explained_variance_{feature_set}.csv"
        ),
        "component_correlations": (
            output_directory
            / f"pca_feature_correlations_{feature_set}.csv"
        ),
        "reconstruction_errors": (
            output_directory
            / f"pca_reconstruction_errors_{feature_set}.csv"
        ),
        "metadata": (
            output_directory
            / f"pca_replication_metadata_{feature_set}.json"
        ),
    }


def write_pca_outputs(
    results: PCAResult,
    feature_matrix: PCAFeatureMatrix,
    output_directory: Path,
) -> dict[str, Path]:
    """
    Write the existing PCA result tables without altering their
    contents or recomputing any mathematical quantities.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = build_output_paths(
        output_directory=output_directory,
        feature_set=results.feature_set,
    )

    results.coordinates.to_csv(
        output_paths["coordinates"],
        index=False,
        encoding="utf-8",
    )

    results.loadings.to_csv(
        output_paths["loadings"],
        index=False,
        encoding="utf-8",
    )

    results.explained_variance.to_csv(
        output_paths["explained_variance"],
        index=False,
        encoding="utf-8",
    )

    results.component_correlations.to_csv(
        output_paths["component_correlations"],
        index=False,
        encoding="utf-8",
    )

    results.reconstruction_errors.to_csv(
        output_paths["reconstruction_errors"],
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "study": (
            "study_043_football_environment_discovery"
        ),
        "experiment": "pca_replication",
        "feature_set": results.feature_set,
        "source_path": str(
            feature_matrix.source_path
        ),
        "observation_count": (
            results.observation_count
        ),
        "feature_count": (
            results.feature_count
        ),
        "component_count": (
            results.component_count
        ),
        "feature_names": list(
            results.feature_names
        ),
        "component_names": list(
            results.component_names
        ),
        "expected_original_observation_count": 10,
        "replication_observation_count": (
            results.observation_count
        ),
        "pca_implementation": (
            "Existing Study 043 fit_pca implementation; "
            "unchanged for replication."
        ),
        "preprocessing": (
            "Existing Study 043 standardized reduced feature "
            "matrix; no preprocessing applied by this script."
        ),
        "outputs": {
            label: str(path)
            for label, path in output_paths.items()
            if label != "metadata"
        },
    }

    output_paths["metadata"].write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_paths


def print_experiment_summary(
    feature_matrix: PCAFeatureMatrix,
    results: PCAResult,
    output_paths: dict[str, Path],
) -> None:
    """
    Print the numerical replication results and output locations.
    """

    explained_variance = (
        results.explained_variance
    )

    pc1_ratio = float(
        explained_variance.loc[
            explained_variance[
                "component"
            ].eq("PC1"),
            "explained_variance_ratio",
        ].iloc[0]
    )

    pc2_ratio = float(
        explained_variance.loc[
            explained_variance[
                "component"
            ].eq("PC2"),
            "explained_variance_ratio",
        ].iloc[0]
    )

    cumulative_two_components = float(
        explained_variance.loc[
            explained_variance[
                "component"
            ].eq("PC2"),
            "cumulative_explained_variance_ratio",
        ].iloc[0]
    )

    print("Study 043 — PCA Replication")
    print("===========================")
    print(f"Feature set: {results.feature_set}")
    print(
        f"Source: {feature_matrix.source_path}"
    )
    print(
        f"Observations: "
        f"{results.observation_count}"
    )
    print(
        f"Features: {results.feature_count}"
    )
    print(
        f"Principal components: "
        f"{results.component_count}"
    )
    print(
        "Feature names: "
        f"{list(results.feature_names)}"
    )
    print()

    print("Primary Replication Metrics")
    print("---------------------------")
    print(
        f"PC1 explained variance ratio: "
        f"{pc1_ratio:.6f}"
    )
    print(
        f"PC2 explained variance ratio: "
        f"{pc2_ratio:.6f}"
    )
    print(
        "Cumulative explained variance "
        f"through PC2: "
        f"{cumulative_two_components:.6f}"
    )
    print()

    print("Explained Variance")
    print("------------------")
    print(
        results.explained_variance.to_string(
            index=False
        )
    )
    print()

    print("Reconstruction Error")
    print("--------------------")
    print(
        results.reconstruction_errors.to_string(
            index=False
        )
    )
    print()

    print("Outputs")
    print("-------")

    for label, path in output_paths.items():
        print(f"{label}: {path}")

    print()
    print("Replication Result")
    print("------------------")
    print("PASSED")
    print(
        "Study 043 PCA replication completed using "
        "the expanded 25-observation research population."
    )


def main() -> None:
    feature_matrix = load_pca_feature_matrix(
        feature_set=FEATURE_SET,
    )

    validate_replication_input(
        feature_matrix=feature_matrix,
    )

    results = fit_pca(
        feature_matrix=feature_matrix,
    )

    output_paths = write_pca_outputs(
        results=results,
        feature_matrix=feature_matrix,
        output_directory=OUTPUT_DIRECTORY,
    )

    print_experiment_summary(
        feature_matrix=feature_matrix,
        results=results,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()
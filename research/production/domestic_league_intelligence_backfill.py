#domestic_league_intelligence_backfill

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import subprocess
import sys
import pandas as pd

from scripts.build_canonical_player_registry import (
    infer_eligible_roles,
    infer_primary_role,
)


@dataclass(frozen=True)
class DomesticLeagueIntelligenceBackfillConfig:
    membership_path: Path
    baseline_ratings_path: Path
    baseline_registry_path: Path
    output_directory: Path


@dataclass(frozen=True)
class IntelligenceCoverageResult:
    membership_players: int
    rated_players: int
    missing_players: int
    coverage: float

    missing_player_ids: tuple[str, ...]

@dataclass(frozen=True)
class RegistryExtensionResult:
    baseline_players: int
    requested_new_players: int
    added_players: int
    expanded_players: int

    canonicalized_new_registry: pd.DataFrame
    expanded_registry: pd.DataFrame

@dataclass(frozen=True)
class RatingsExtensionResult:
    baseline_players: int
    requested_missing_players: int
    supplementary_players: int
    added_players: int
    expanded_players: int

    added_player_ids: tuple[str, ...]
    expanded_ratings: pd.DataFrame

@dataclass(frozen=True)
class IntelligenceBackfillPipelineConfig:
    competition: str
    source_season_year: str

    membership_path: Path
    feeder_membership_path: Path

    baseline_registry_path: Path
    baseline_ratings_path: Path

    competition_manifest_path: Path
    competition_feature_manifest_path: Path
    feature_attribute_manifest_path: Path
    role_attribute_manifest_path: Path

    output_directory: Path

@dataclass(frozen=True)
class IntelligenceBackfillArtifacts:
    missing_population: Path
    missing_profiles: Path
    missing_profile_failures: Path

    supplementary_stats: Path
    supplementary_stats_failures: Path

    expanded_registry: Path

    engineered_features: Path
    canonical_features: Path
    weighted_features: Path
    attribute_scores: Path
    supplementary_ratings: Path
    expanded_ratings: Path

@dataclass(frozen=True)
class BackfillPreparationResult:
    coverage: IntelligenceCoverageResult
    artifacts: IntelligenceBackfillArtifacts

    feeder_players: int
    missing_players_found_in_feeder: int

    missing_population: pd.DataFrame
    feeder_population: pd.DataFrame

@dataclass(frozen=True)
class RegistryBackfillStageResult:
    new_players: int
    expanded_players: int
    expanded_registry: pd.DataFrame

@dataclass(frozen=True)
class FeatureBackfillStageResult:
    engineered_players: int
    canonical_players: int

    engineered_features: pd.DataFrame
    canonical_features: pd.DataFrame

@dataclass(frozen=True)
class WeightedFeatureBackfillStageResult:
    weighted_players: int
    weighted_features: pd.DataFrame

@dataclass(frozen=True)
class RatingBackfillStageResult:
    attribute_players: int
    rated_players: int

    attribute_scores: pd.DataFrame
    supplementary_ratings: pd.DataFrame

@dataclass(frozen=True)
class FinalBackfillResult:
    baseline_players: int
    added_players: int
    expanded_players: int

    membership_players: int
    rated_membership_players: int
    missing_membership_players: int
    coverage: float

    expanded_ratings: pd.DataFrame

@dataclass(frozen=True)
class IntelligenceBackfillPipelineResult:
    preparation: BackfillPreparationResult
    registry: RegistryBackfillStageResult
    features: FeatureBackfillStageResult
    weighted_features: WeightedFeatureBackfillStageResult
    ratings: RatingBackfillStageResult
    final: FinalBackfillResult

def extend_player_ratings(
    *,
    membership: pd.DataFrame,
    baseline_ratings: pd.DataFrame,
    supplementary_ratings: pd.DataFrame,
) -> RatingsExtensionResult:
    for label, frame in (
        ("membership", membership),
        ("baseline_ratings", baseline_ratings),
        ("supplementary_ratings", supplementary_ratings),
    ):
        if "player_id" not in frame.columns:
            raise ValueError(
                f"{label} does not contain player_id."
            )

    membership_ids = set(
        membership["player_id"]
        .dropna()
        .astype(str)
    )

    baseline = baseline_ratings.copy()
    supplement = supplementary_ratings.copy()

    baseline["player_id"] = (
        baseline["player_id"].astype(str)
    )

    supplement["player_id"] = (
        supplement["player_id"].astype(str)
    )

    if baseline["player_id"].duplicated().any():
        raise ValueError(
            "Baseline ratings contain duplicate player IDs."
        )

    if supplement["player_id"].duplicated().any():
        raise ValueError(
            "Supplementary ratings contain duplicate player IDs."
        )

    baseline_ids = set(
        baseline["player_id"]
    )

    supplementary_ids = set(
        supplement["player_id"]
    )

    requested_missing_ids = (
        membership_ids
        - baseline_ids
    )

    unresolved = (
        requested_missing_ids
        - supplementary_ids
    )

    if unresolved:
        raise ValueError(
            "Supplementary ratings do not cover all "
            "membership players missing from the baseline: "
            f"{sorted(unresolved)}"
        )

    #
    # Provenance rule:
    # supplement only genuinely absent baseline identities.
    #
    additions = supplement.loc[
        supplement["player_id"].isin(
            requested_missing_ids
        )
    ].copy()

    added_ids = set(
        additions["player_id"]
    )

    if added_ids != requested_missing_ids:
        raise AssertionError(
            "Selected supplementary additions do not exactly "
            "match the missing baseline population."
        )

    missing_baseline_columns = (
        set(baseline.columns)
        - set(additions.columns)
    )

    if missing_baseline_columns:
        raise ValueError(
            "Supplementary ratings cannot satisfy the baseline "
            "rating schema. Missing columns: "
            f"{sorted(missing_baseline_columns)}"
        )

    additions = additions[
        list(baseline.columns)
    ].copy()

    expanded = pd.concat(
        [
            baseline,
            additions,
        ],
        ignore_index=True,
    )

    if expanded["player_id"].duplicated().any():
        raise AssertionError(
            "Expanded ratings contain duplicate player IDs."
        )

    expanded_ids = set(
        expanded["player_id"]
    )

    if not membership_ids <= expanded_ids:
        raise AssertionError(
            "Expanded ratings still do not cover the complete "
            "target membership population."
        )

    return RatingsExtensionResult(
        baseline_players=len(
            baseline_ids
        ),
        requested_missing_players=len(
            requested_missing_ids
        ),
        supplementary_players=len(
            supplementary_ids
        ),
        added_players=len(
            added_ids
        ),
        expanded_players=len(
            expanded_ids
        ),
        added_player_ids=tuple(
            sorted(added_ids)
        ),
        expanded_ratings=expanded,
    )

def _read_player_artifact(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Player artifact does not exist: {path}"
        )

    frame = pd.read_csv(
        path,
        dtype={
            "player_id": str,
            "canonical_player_id": str,
        },
        low_memory=False,
    )

    if "player_id" not in frame.columns:
        raise ValueError(
            f"{path} does not contain player_id."
        )

    return frame


def audit_intelligence_coverage(
    config: DomesticLeagueIntelligenceBackfillConfig,
) -> IntelligenceCoverageResult:
    memberships = _read_player_artifact(
        config.membership_path
    )

    ratings = _read_player_artifact(
        config.baseline_ratings_path
    )

    membership_ids = set(
        memberships["player_id"]
        .dropna()
        .astype(str)
    )

    rating_ids = set(
        ratings["player_id"]
        .dropna()
        .astype(str)
    )

    if not membership_ids:
        raise ValueError(
            "Membership artifact contains no players."
        )

    rated_ids = (
        membership_ids
        & rating_ids
    )

    missing_ids = (
        membership_ids
        - rating_ids
    )

    coverage = (
        len(rated_ids)
        / len(membership_ids)
    )

    return IntelligenceCoverageResult(
        membership_players=len(
            membership_ids
        ),
        rated_players=len(
            rated_ids
        ),
        missing_players=len(
            missing_ids
        ),
        coverage=coverage,
        missing_player_ids=tuple(
            sorted(missing_ids)
        ),
    )


def build_missing_player_population(
    config: DomesticLeagueIntelligenceBackfillConfig,
) -> pd.DataFrame:
    memberships = _read_player_artifact(
        config.membership_path
    )

    coverage = audit_intelligence_coverage(
        config
    )

    missing_ids = set(
        coverage.missing_player_ids
    )

    missing = memberships.loc[
        memberships["player_id"].isin(
            missing_ids
        )
    ].copy()

    missing = missing.drop_duplicates(
        subset=["player_id"]
    )

    if (
        missing["player_id"].nunique()
        != coverage.missing_players
    ):
        raise AssertionError(
            "Missing-player population does not "
            "match the coverage audit."
        )

    return missing

def canonicalize_new_player_profiles(
    *,
    missing_profiles: pd.DataFrame,
    baseline_registry: pd.DataFrame,
) -> pd.DataFrame:
    required_profile_columns = {
        "player_id",
        "player",
        "player_slug",
        "position",
        "positions_detailed",
        "country",
        "country_alpha2",
        "country_alpha3",
        "current_team",
    }

    missing_columns = (
        required_profile_columns
        - set(missing_profiles.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing-player profiles are missing required "
            f"columns: {sorted(missing_columns)}"
        )

    if "player_id" not in baseline_registry.columns:
        raise ValueError(
            "Baseline registry does not contain player_id."
        )

    profiles = missing_profiles.copy()

    profiles["player_id"] = (
        profiles["player_id"]
        .astype(str)
    )

    if profiles["player_id"].duplicated().any():
        raise ValueError(
            "Missing-player profiles contain duplicate "
            "player IDs."
        )

    baseline_ids = set(
        baseline_registry["player_id"]
        .dropna()
        .astype(str)
    )

    profile_ids = set(
        profiles["player_id"]
    )

    overlap = (
        baseline_ids
        & profile_ids
    )

    if overlap:
        raise ValueError(
            "New-player profile population overlaps the "
            f"baseline registry: {len(overlap)} players."
        )

    #
    # We have already established that these identities
    # are absent from the frozen baseline. New source IDs
    # therefore begin as their own canonical identities.
    #
    profiles["canonical_player_id"] = (
        profiles["player_id"]
    )

    profiles["eligible_roles"] = (
        profiles["positions_detailed"]
        .apply(infer_eligible_roles)
    )

    profiles["position"] = (
        profiles["eligible_roles"]
        .apply(infer_primary_role)
    )

    required_registry_columns = list(
        baseline_registry.columns
    )

    missing_registry_columns = (
        set(required_registry_columns)
        - set(profiles.columns)
    )

    if missing_registry_columns:
        raise ValueError(
            "Canonicalized profiles cannot satisfy the "
            "baseline registry contract. Missing columns: "
            f"{sorted(missing_registry_columns)}"
        )

    canonicalized = profiles[
        required_registry_columns
    ].copy()

    if canonicalized["canonical_player_id"].isna().any():
        raise AssertionError(
            "Canonicalized new-player registry contains "
            "missing canonical IDs."
        )

    if canonicalized["eligible_roles"].isna().any():
        raise AssertionError(
            "Canonicalized new-player registry contains "
            "missing eligible roles."
        )

    return canonicalized

def extend_player_registry(
    *,
    baseline_registry: pd.DataFrame,
    new_canonical_registry: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baseline_registry.copy()
    new = new_canonical_registry.copy()

    baseline["player_id"] = (
        baseline["player_id"].astype(str)
    )

    new["player_id"] = (
        new["player_id"].astype(str)
    )

    baseline_ids = set(
        baseline["player_id"]
    )

    new_ids = set(
        new["player_id"]
    )

    overlap = (
        baseline_ids
        & new_ids
    )

    if overlap:
        raise ValueError(
            "Registry extension contains baseline-player "
            f"overlap: {len(overlap)} players."
        )

    expanded = pd.concat(
        [
            baseline,
            new,
        ],
        ignore_index=True,
    )

    if expanded["player_id"].duplicated().any():
        raise AssertionError(
            "Expanded registry contains duplicate player IDs."
        )

    if expanded["canonical_player_id"].isna().any():
        raise AssertionError(
            "Expanded registry contains missing canonical IDs."
        )

    if expanded["eligible_roles"].isna().any():
        raise AssertionError(
            "Expanded registry contains missing eligible roles."
        )

    return expanded

def attach_canonical_player_ids(
    *,
    features: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    if "player_id" not in features.columns:
        raise ValueError(
            "Feature artifact does not contain player_id."
        )

    required_registry_columns = {
        "player_id",
        "canonical_player_id",
    }

    missing_registry_columns = (
        required_registry_columns
        - set(registry.columns)
    )

    if missing_registry_columns:
        raise ValueError(
            "Registry is missing identity columns: "
            f"{sorted(missing_registry_columns)}"
        )

    if "canonical_player_id" in features.columns:
        raise ValueError(
            "Feature artifact already contains "
            "canonical_player_id."
        )

    identity = (
        registry[
            [
                "player_id",
                "canonical_player_id",
            ]
        ]
        .copy()
    )

    identity["player_id"] = (
        identity["player_id"]
        .astype(str)
    )

    identity["canonical_player_id"] = (
        identity["canonical_player_id"]
        .astype(str)
    )

    if identity["player_id"].duplicated().any():
        raise ValueError(
            "Registry contains ambiguous player_id "
            "mappings."
        )

    candidate = features.copy()

    candidate["player_id"] = (
        candidate["player_id"]
        .astype(str)
    )

    candidate = candidate.merge(
        identity,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    missing_identity = candidate.loc[
        candidate["canonical_player_id"].isna(),
        "player_id",
    ].drop_duplicates()

    if not missing_identity.empty:
        raise ValueError(
            "Feature players are missing canonical "
            "registry identities: "
            f"{sorted(missing_identity.tolist())}"
        )

    columns = list(candidate.columns)

    columns.remove(
        "canonical_player_id"
    )

    player_id_index = columns.index(
        "player_id"
    )

    columns.insert(
        player_id_index,
        "canonical_player_id",
    )

    return candidate[columns]


def build_backfill_artifact_layout(
    config: IntelligenceBackfillPipelineConfig,
) -> IntelligenceBackfillArtifacts:
    root = config.output_directory

    return IntelligenceBackfillArtifacts(
        missing_population=(
            root / "missing_player_population.csv"
        ),
        missing_profiles=(
            root / "missing_player_profiles.csv"
        ),
        missing_profile_failures=(
            root / "missing_player_profile_failures.csv"
        ),
        supplementary_stats=(
            root / "supplementary_player_stats.csv"
        ),
        supplementary_stats_failures=(
            root / "supplementary_player_stats_failures.csv"
        ),
        expanded_registry=(
            root / "expanded_player_registry.csv"
        ),
        engineered_features=(
            root / "supplementary_player_features.csv"
        ),
        canonical_features=(
            root / "supplementary_canonical_features.csv"
        ),
        weighted_features=(
            root / "supplementary_weighted_features.csv"
        ),
        attribute_scores=(
            root / "supplementary_attribute_scores.csv"
        ),
        supplementary_ratings=(
            root / "supplementary_player_ratings.csv"
        ),
        expanded_ratings=(
            root / "expanded_player_ratings.csv"
        ),
    )

def prepare_intelligence_backfill(
    config: IntelligenceBackfillPipelineConfig,
) -> BackfillPreparationResult:
    artifacts = build_backfill_artifact_layout(
        config
    )

    config.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage_config = (
        DomesticLeagueIntelligenceBackfillConfig(
            membership_path=config.membership_path,
            baseline_ratings_path=(
                config.baseline_ratings_path
            ),
            baseline_registry_path=(
                config.baseline_registry_path
            ),
            output_directory=(
                config.output_directory
            ),
        )
    )

    coverage = audit_intelligence_coverage(
        coverage_config
    )

    missing_population = (
        build_missing_player_population(
            coverage_config
        )
    )

    feeder = _read_player_artifact(
        config.feeder_membership_path
    )

    feeder_ids = set(
        feeder["player_id"]
        .dropna()
        .astype(str)
    )

    missing_ids = set(
        coverage.missing_player_ids
    )

    unresolved = (
        missing_ids
        - feeder_ids
    )

    if unresolved:
        raise ValueError(
            "Feeder membership does not contain all "
            "players requiring intelligence backfill: "
            f"{sorted(unresolved)}"
        )

    #
    # Preserve the full feeder evidence population for
    # the target missing identities only.
    #
    feeder_population = feeder.loc[
        feeder["player_id"]
        .astype(str)
        .isin(missing_ids)
    ].copy()

    missing_population.to_csv(
        artifacts.missing_population,
        index=False,
    )

    return BackfillPreparationResult(
        coverage=coverage,
        artifacts=artifacts,
        feeder_players=len(feeder_ids),
        missing_players_found_in_feeder=len(
            missing_ids & feeder_ids
        ),
        missing_population=missing_population,
        feeder_population=feeder_population,
    )

def _run_module(
    module: str,
    arguments: list[str],
) -> None:
    command = [
        sys.executable,
        "-m",
        module,
        *arguments,
    ]

    completed = subprocess.run(
        command,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Backfill pipeline stage failed: "
            f"{module} "
            f"(exit code {completed.returncode})"
        )

def acquire_missing_player_evidence(
    *,
    config: IntelligenceBackfillPipelineConfig,
    preparation: BackfillPreparationResult,
    dry_run: bool = False,
) -> None:
    artifacts = preparation.artifacts

    #
    # The preparation stage already proved that these
    # identities exist in the feeder membership universe.
    # Persist exactly that population as the acquisition
    # input for both profiles and historical statistics.
    #
    preparation.feeder_population.to_csv(
        artifacts.missing_population,
        index=False,
    )

    profile_arguments = [
        "--input-file",
        str(artifacts.missing_population),
        "--output-file",
        str(artifacts.missing_profiles),
        "--failed-file",
        str(
            artifacts.missing_profile_failures
        ),
    ]

    if dry_run:
        profile_arguments.append(
            "--dry-run"
        )

    _run_module(
        "scripts.ingest_player_profiles",
        profile_arguments,
    )

    stats_arguments = [
        "--competition",
        config.competition,
        "--season-year",
        config.source_season_year,
        "--input-file",
        str(artifacts.missing_population),
        "--output-file",
        str(artifacts.supplementary_stats),
        "--failed-file",
        str(
            artifacts.supplementary_stats_failures
        ),
    ]

    if dry_run:
        stats_arguments.append(
            "--dry-run"
        )

    _run_module(
        "scripts.ingest_player_stats",
        stats_arguments,
    )

    if dry_run:
        return

    if not artifacts.missing_profiles.exists():
        raise AssertionError(
            "Profile acquisition completed without "
            "producing its expected artifact."
        )

    if not artifacts.supplementary_stats.exists():
        raise AssertionError(
            "Statistics acquisition completed without "
            "producing its expected artifact."
        )

def build_expanded_registry(
    *,
    config: IntelligenceBackfillPipelineConfig,
    artifacts: IntelligenceBackfillArtifacts,
) -> RegistryBackfillStageResult:
    baseline_registry = _read_player_artifact(
        config.baseline_registry_path
    )

    missing_profiles = _read_player_artifact(
        artifacts.missing_profiles
    )

    canonicalized = canonicalize_new_player_profiles(
        missing_profiles=missing_profiles,
        baseline_registry=baseline_registry,
    )

    expanded = extend_player_registry(
        baseline_registry=baseline_registry,
        new_canonical_registry=canonicalized,
    )

    expanded.to_csv(
        artifacts.expanded_registry,
        index=False,
    )

    return RegistryBackfillStageResult(
        new_players=canonicalized[
            "player_id"
        ].nunique(),
        expanded_players=expanded[
            "player_id"
        ].nunique(),
        expanded_registry=expanded,
    )

def build_canonical_backfill_features(
    *,
    config: IntelligenceBackfillPipelineConfig,
    artifacts: IntelligenceBackfillArtifacts,
    expanded_registry: pd.DataFrame,
) -> FeatureBackfillStageResult:
    if not artifacts.supplementary_stats.exists():
        raise FileNotFoundError(
            "Supplementary statistics artifact does not exist: "
            f"{artifacts.supplementary_stats}"
        )

    _run_module(
        "scripts.sofascore_feature_engineering",
        [
            "--input-file",
            str(artifacts.supplementary_stats),
            "--output-file",
            str(artifacts.engineered_features),
        ],
    )

    engineered = _read_player_artifact(
        artifacts.engineered_features
    )

    canonical = attach_canonical_player_ids(
        features=engineered,
        registry=expanded_registry,
    )

    canonical.to_csv(
        artifacts.canonical_features,
        index=False,
    )

    return FeatureBackfillStageResult(
        engineered_players=engineered[
            "player_id"
        ].nunique(),
        canonical_players=canonical[
            "canonical_player_id"
        ].nunique(),
        engineered_features=engineered,
        canonical_features=canonical,
    )

def build_weighted_backfill_features(
    *,
    config: IntelligenceBackfillPipelineConfig,
    artifacts: IntelligenceBackfillArtifacts,
) -> WeightedFeatureBackfillStageResult:
    required_paths = (
        artifacts.canonical_features,
        artifacts.expanded_registry,
        config.competition_manifest_path,
        config.competition_feature_manifest_path,
        config.feature_attribute_manifest_path,
    )

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Weighted-feature stage is missing required "
            "artifacts: "
            f"{[str(path) for path in missing_paths]}"
        )

    _run_module(
        "scripts.build_weighted_player_features",
        [
            "--features-file",
            str(artifacts.canonical_features),

            "--competition-file",
            str(config.competition_manifest_path),

            "--competition-feature-file",
            str(
                config.competition_feature_manifest_path
            ),

            "--feature-attribute-file",
            str(
                config.feature_attribute_manifest_path
            ),

            "--registry-file",
            str(artifacts.expanded_registry),

            "--output-file",
            str(artifacts.weighted_features),
        ],
    )

    weighted = _read_player_artifact(
        artifacts.weighted_features
    )

    if "canonical_player_id" not in weighted.columns:
        raise AssertionError(
            "Weighted feature artifact does not contain "
            "canonical_player_id."
        )

    if weighted["canonical_player_id"].duplicated().any():
        raise AssertionError(
            "Weighted feature artifact contains duplicate "
            "canonical player IDs."
        )

    return WeightedFeatureBackfillStageResult(
        weighted_players=weighted[
            "canonical_player_id"
        ].nunique(),
        weighted_features=weighted,
    )

def build_supplementary_player_ratings(
    *,
    config: IntelligenceBackfillPipelineConfig,
    artifacts: IntelligenceBackfillArtifacts,
) -> RatingBackfillStageResult:
    required_paths = (
        artifacts.weighted_features,
        artifacts.expanded_registry,
        config.feature_attribute_manifest_path,
        config.role_attribute_manifest_path,
    )

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Rating stage is missing required artifacts: "
            f"{[str(path) for path in missing_paths]}"
        )

    #
    # Preserve the production Player Intelligence
    # transformation selected by Study 101D/101E.
    #
    _run_module(
        "scripts.score_player_attributes",
        [
            "--transformation-id",
            "robust_zscore",

            "--features-file",
            str(artifacts.weighted_features),

            "--feature-attribute-file",
            str(
                config.feature_attribute_manifest_path
            ),

            "--output-path",
            str(artifacts.attribute_scores),
        ],
    )

    attributes = _read_player_artifact(
        artifacts.attribute_scores
    )

    _run_module(
        "scripts.build_player_ratings_v4",
        [
            "--attribute-path",
            str(artifacts.attribute_scores),

            "--registry-path",
            str(artifacts.expanded_registry),

            "--role-attribute-path",
            str(
                config.role_attribute_manifest_path
            ),

            "--output-path",
            str(artifacts.supplementary_ratings),
        ],
    )

    ratings = _read_player_artifact(
        artifacts.supplementary_ratings
    )

    attribute_players = attributes[
        "player_id"
    ].nunique()

    rated_players = ratings[
        "player_id"
    ].nunique()

    if rated_players != attribute_players:
        raise AssertionError(
            "Rating stage changed the player population: "
            f"attributes={attribute_players}, "
            f"ratings={rated_players}."
        )

    return RatingBackfillStageResult(
        attribute_players=attribute_players,
        rated_players=rated_players,
        attribute_scores=attributes,
        supplementary_ratings=ratings,
    )

def finalize_intelligence_backfill(
    *,
    config: IntelligenceBackfillPipelineConfig,
    artifacts: IntelligenceBackfillArtifacts,
) -> FinalBackfillResult:
    membership = _read_player_artifact(
        config.membership_path
    )

    baseline_ratings = _read_player_artifact(
        config.baseline_ratings_path
    )

    supplementary_ratings = _read_player_artifact(
        artifacts.supplementary_ratings
    )

    extension = extend_player_ratings(
        membership=membership,
        baseline_ratings=baseline_ratings,
        supplementary_ratings=supplementary_ratings,
    )

    expanded = extension.expanded_ratings

    expanded.to_csv(
        artifacts.expanded_ratings,
        index=False,
    )

    membership_ids = set(
        membership["player_id"]
        .dropna()
        .astype(str)
    )

    expanded_ids = set(
        expanded["player_id"]
        .dropna()
        .astype(str)
    )

    rated_membership_ids = (
        membership_ids
        & expanded_ids
    )

    missing_membership_ids = (
        membership_ids
        - expanded_ids
    )

    if not membership_ids:
        raise ValueError(
            "Target membership population is empty."
        )

    coverage = (
        len(rated_membership_ids)
        / len(membership_ids)
    )

    if missing_membership_ids:
        raise AssertionError(
            "Intelligence backfill completed with unresolved "
            "target-membership players: "
            f"{sorted(missing_membership_ids)}"
        )

    return FinalBackfillResult(
        baseline_players=extension.baseline_players,
        added_players=extension.added_players,
        expanded_players=extension.expanded_players,
        membership_players=len(membership_ids),
        rated_membership_players=len(
            rated_membership_ids
        ),
        missing_membership_players=len(
            missing_membership_ids
        ),
        coverage=coverage,
        expanded_ratings=expanded,
    )

def run_intelligence_backfill(
    *,
    config: IntelligenceBackfillPipelineConfig,
    acquire_evidence: bool = True,
    acquisition_dry_run: bool = False,
) -> IntelligenceBackfillPipelineResult:
    preparation = prepare_intelligence_backfill(
        config
    )

    if acquire_evidence:
        acquire_missing_player_evidence(
            config=config,
            preparation=preparation,
            dry_run=acquisition_dry_run,
        )

        if acquisition_dry_run:
            raise RuntimeError(
                "A dry-run acquisition does not produce "
                "the evidence artifacts required by the "
                "downstream backfill stages."
            )

    registry = build_expanded_registry(
        config=config,
        artifacts=preparation.artifacts,
    )

    features = build_canonical_backfill_features(
        config=config,
        artifacts=preparation.artifacts,
        expanded_registry=registry.expanded_registry,
    )

    weighted_features = (
        build_weighted_backfill_features(
            config=config,
            artifacts=preparation.artifacts,
        )
    )

    ratings = build_supplementary_player_ratings(
        config=config,
        artifacts=preparation.artifacts,
    )

    final = finalize_intelligence_backfill(
        config=config,
        artifacts=preparation.artifacts,
    )

    return IntelligenceBackfillPipelineResult(
        preparation=preparation,
        registry=registry,
        features=features,
        weighted_features=weighted_features,
        ratings=ratings,
        final=final,
    )
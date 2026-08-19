#domestic_league_onboarding

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import argparse
import json

from research.production.domestic_league_intelligence_backfill import (
    DomesticLeagueIntelligenceBackfillConfig,
    audit_intelligence_coverage,
)

from research.adapters.football_model_adapter import (
    CLUBELO_CACHE_DIRECTORY,
    DOMESTIC_PRODUCTION_REPOSITORY_SOURCES,
)

from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)
from simulation.domestic_league_configs import (
    DOMESTIC_LEAGUE_CONFIGS,
)

from research.production.domestic_clubelo_identity import (
    build_clubelo_lookup_candidates,
    get_clubelo_name_override,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OnboardingStage:
    number: int
    key: str
    name: str
    program: str
    description: str


@dataclass(frozen=True)
class DomesticLeagueOnboardingSpec:
    key: str
    competition_name: str
    competition_id: int

    target_season: str
    target_season_start_year: int
    target_season_id: int

    feeder_key: str
    feeder_competition_name: str
    feeder_competition_id: int
    feeder_season: str
    feeder_season_start_year: int
    feeder_season_id: int

    participant_count: int
    matchday_count: int
    fixture_count: int
    timezone_name: str

    structural_validation_path: Path
    production_validation_path: Path
    monte_carlo_validation_path: Path
    regression_validation_path: Path

    bootstrap_directory: Path
    target_season_registry_path: Path
    feeder_season_registry_path: Path
    target_competition_manifest_path: Path
    feeder_competition_manifest_path: Path
    target_participants_path: Path
    feeder_teams_path: Path
    feeder_players_path: Path
    membership_candidate_path: Path
    membership_resolved_path: Path
    baseline_registry_path: Path
    baseline_ratings_path: Path
    expanded_registry_path: Path
    expanded_ratings_path: Path
    club_repository_path: Path
    fixture_path: Path

    repository_source: str

@dataclass(frozen=True)
class OnboardingStageStatus:
    stage_key: str
    status: str
    summary: str
    program: str

SERIE_A_2026_27_BOOTSTRAP_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "serie_a_2026_27_bootstrap"
)

SERIE_A_2026_27_ONBOARDING = DomesticLeagueOnboardingSpec(
    key="serie_a",
    competition_name="Serie A",
    competition_id=23,

    target_season="2026-27",
    target_season_start_year=2026,
    target_season_id=95836,

    feeder_key="serie_b",
    feeder_competition_name="Serie B",
    feeder_competition_id=53,
    feeder_season="25/26",
    feeder_season_start_year=2025,
    feeder_season_id=79502,

    participant_count=20,
    matchday_count=38,
    fixture_count=380,
    timezone_name="Europe/Rome",

    structural_validation_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_a_2026_27_simulation"
        / "structural_validation.json"
    ),

    production_validation_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_a_2026_27_simulation"
        / "production_validation.json"
    ),

    monte_carlo_validation_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_a_2026_27_simulation"
        / "monte_carlo_validation.json"
    ),

    regression_validation_path=(
        PROJECT_ROOT
        / "outputs"
        / "onboarding_validation"
        / "regression_suite.json"
    ),

    bootstrap_directory=(
        SERIE_A_2026_27_BOOTSTRAP_DIRECTORY
    ),

    target_season_registry_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_a_2026_27_season_registry.csv"
    ),

    feeder_season_registry_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_b_2025_26_season_registry.csv"
    ),

    target_competition_manifest_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_a_2026_27_competition_manifest.csv"
    ),

    feeder_competition_manifest_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_b_2025_26_competition_manifest.csv"
    ),

    target_participants_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_a_2026_27_target_participants.csv"
    ),

    feeder_teams_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_b_2025_26_teams.csv"
    ),

    feeder_players_path=(
        PROJECT_ROOT
        / "outputs"
        / "serie_b_2025_26_players.csv"
    ),

    membership_candidate_path=(
        SERIE_A_2026_27_BOOTSTRAP_DIRECTORY
        / "serie_a_2026_27_membership_candidate.csv"
    ),

    membership_resolved_path=(
        SERIE_A_2026_27_BOOTSTRAP_DIRECTORY
        / "serie_a_2026_27_membership_resolved.csv"
    ),

    expanded_registry_path=(
        SERIE_A_2026_27_BOOTSTRAP_DIRECTORY
        / "intelligence_backfill"
        / "expanded_player_registry.csv"
    ),

    baseline_registry_path=(
        PROJECT_ROOT
        / "outputs"
        / "study_101f_canonical_player_registry.csv"
    ),

    baseline_ratings_path=(
        PROJECT_ROOT
        / "outputs"
        / "study_101f_player_ratings.csv"
    ),

    expanded_ratings_path=(
        SERIE_A_2026_27_BOOTSTRAP_DIRECTORY
        / "intelligence_backfill"
        / "expanded_player_ratings.csv"
    ),

    club_repository_path=(
        SERIE_A_2026_27_BOOTSTRAP_DIRECTORY
        / "serie_a_2026_27_club_repository.csv"
    ),

    fixture_path=(
        SERIE_A_2026_27_BOOTSTRAP_DIRECTORY
        / "serie_a_2026_27_fixtures.csv"
    ),

    repository_source="serie_a_production_v1",
)

DOMESTIC_LEAGUE_ONBOARDING_SPECS = {
    SERIE_A_2026_27_ONBOARDING.key:
        SERIE_A_2026_27_ONBOARDING,
}

ONBOARDING_STAGES = (
    OnboardingStage(
        number=1,
        key="register_competitions",
        name="Register competitions",
        program="scripts.discover_sofascore_competition_seasons",
        description=(
            "Register the target top flight and feeder league "
            "in the Sofascore competition-season discovery layer."
        ),
    ),
    OnboardingStage(
        number=2,
        key="discover_target_season",
        name="Discover target season",
        program="scripts.discover_sofascore_competition_seasons",
        description=(
            "Resolve the authoritative Sofascore target-season "
            "tournament and season identity."
        ),
    ),
    OnboardingStage(
        number=3,
        key="discover_feeder_season",
        name="Discover feeder season",
        program="scripts.discover_sofascore_competition_seasons",
        description=(
            "Resolve the previous-season feeder competition "
            "identity used for promoted-club evidence."
        ),
    ),
    OnboardingStage(
        number=4,
        key="build_manifests",
        name="Build competition manifests",
        program="scripts.sofascore_build_competition_manifest",
        description=(
            "Build isolated target and feeder competition "
            "manifests for downstream ingestion."
        ),
    ),
    OnboardingStage(
        number=5,
        key="target_participants",
        name="Acquire target participants",
        program="scripts.ingest_teams",
        description=(
            "Build the authoritative target-season club artifact "
            "while preserving team_id, team, and team_slug."
        ),
    ),
    OnboardingStage(
        number=6,
        key="feeder_population",
        name="Acquire feeder population",
        program=(
            "scripts.ingest_teams + scripts.ingest_players"
        ),
        description=(
            "Acquire feeder-club identities and previous-season "
            "player memberships."
        ),
    ),
    OnboardingStage(
        number=7,
        key="membership_bootstrap",
        name="Build target membership",
        program=(
            "research.production."
            "domestic_league_membership_bootstrap"
        ),
        description=(
            "Combine returning top-flight memberships with "
            "promoted-club feeder memberships."
        ),
    ),
    OnboardingStage(
        number=8,
        key="membership_resolution",
        name="Resolve membership ambiguity",
        program="scripts.ingest_player_profiles",
        description=(
            "Resolve duplicate or stale memberships using "
            "current-player identity evidence."
        ),
    ),
    OnboardingStage(
        number=9,
        key="intelligence_coverage",
        name="Audit Player Intelligence coverage",
        program=(
            "research.production."
            "domestic_league_intelligence_backfill"
        ),
        description=(
            "Audit the resolved membership against the frozen "
            "Player Ratings baseline."
        ),
    ),
    OnboardingStage(
        number=10,
        key="intelligence_backfill",
        name="Backfill missing Player Intelligence",
        program=(
            "research.production."
            "domestic_league_intelligence_backfill"
        ),
        description=(
            "Conditionally backfill only genuinely missing "
            "target players. Canonical sub-pipeline: "
            "ingest_player_profiles -> ingest_player_stats -> "
            "canonicalize/extend registry -> "
            "sofascore_feature_engineering -> "
            "audit_competition_stat_coverage -> "
            "build_competition_feature_manifest -> "
            "build_weighted_player_features -> "
            "score_player_attributes -> "
            "build_player_ratings_v4 -> extend ratings."
        ),
    ),
    OnboardingStage(
        number=11,
        key="production_repository",
        name="Build production club repository",
        program=(
            "research.production."
            "production_club_repository_builder"
        ),
        description=(
            "Build and validate the simulation-ready "
            "20-row/17-column-style club repository."
        ),
    ),
    OnboardingStage(
        number=12,
        key="fixtures",
        name="Build fixture snapshot",
        program="scripts.build_sofascore_league_fixture_snapshot",
        description=(
            "Build and validate the authoritative target-season "
            "fixture schedule."
        ),
    ),
    OnboardingStage(
        number=13,
        key="identity_alignment",
        name="Audit cross-artifact club identity",
        program=(
            "research.production.domestic_league_onboarding"
        ),
        description=(
            "Require target participants, repository clubs, "
            "and fixture clubs to match exactly."
        ),
    ),
    OnboardingStage(
        number=14,
        key="simulation_config",
        name="Register simulation config",
        program="simulation.domestic_league_configs",
        description=(
            "Register the league in the shared domestic "
            "simulation configuration."
        ),
    ),
    OnboardingStage(
        number=15,
        key="structural_simulation",
        name="Run structural simulation",
        program="scripts.run_domestic_league_simulation",
        description=(
            "Validate the league framework in structural mode "
            "before production-model execution."
        ),
    ),
    OnboardingStage(
        number=16,
        key="production_routing",
        name="Register production routing",
        program="research.adapters.football_model_adapter",
        description=(
            "Add the league production repository source to the "
            "shared domestic production-model route."
        ),
    ),
    OnboardingStage(
        number=17,
        key="clubelo_preload",
        name="Preload ClubElo histories",
        program=(
            "research.production."
            "preload_domestic_league_clubelo"
        ),
        description=(
            "Resolve external ClubElo identities and materialize "
            "validated histories under canonical production names."
        ),
    ),
    OnboardingStage(
        number=18,
        key="production_simulation",
        name="Run production simulation",
        program="scripts.run_domestic_league_simulation",
        description=(
            "Run the first production season only after ClubElo "
            "coverage and upstream contracts are complete."
        ),
    ),
    OnboardingStage(
        number=19,
        key="monte_carlo",
        name="Run Monte Carlo validation",
        program="scripts.run_domestic_league_simulation",
        description=(
            "Run a larger production Monte Carlo smoke/sanity "
            "validation."
        ),
    ),
    OnboardingStage(
        number=20,
        key="regression_suite",
        name="Run regression suite",
        program="pytest",
        description=(
            "Require the full project test suite to remain green."
        ),
    ),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the canonical domestic-league "
            "onboarding contract and current status."
        )
    )

    parser.add_argument(
        "--competition",
        choices=sorted(
            DOMESTIC_LEAGUE_ONBOARDING_SPECS
        ),
        help=(
            "Domestic league onboarding specification "
            "to inspect."
        ),
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help=(
            "Evaluate the current onboarding status "
            "for the selected competition."
        ),
    )

    parser.add_argument(
        "--contract",
        action="store_true",
        help=(
            "Print the canonical onboarding-stage contract."
        ),
    )

    parser.add_argument(
        "--spec",
        action="store_true",
        help=(
            "Print the selected league onboarding "
            "specification."
        ),
    )

    return parser.parse_args()

def evaluate_target_participants(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    path = spec.target_participants_path

    if not path.exists():
        return OnboardingStageStatus(
            stage_key="target_participants",
            status="MISSING",
            summary=(
                f"Expected target-participant artifact is missing: "
                f"{path}"
            ),
            program="scripts.ingest_teams",
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        "team_id",
        "team",
        "team_slug",
    }

    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        return OnboardingStageStatus(
            stage_key="target_participants",
            status="FAIL",
            summary=(
                "Target-participant artifact is missing required "
                f"identity columns: {missing_columns}"
            ),
            program="scripts.ingest_teams",
        )

    participants = (
        frame[
            [
                "team_id",
                "team",
                "team_slug",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    invalid_identity_rows = participants.loc[
        participants[
            [
                "team_id",
                "team",
                "team_slug",
            ]
        ]
        .isna()
        .any(axis=1)
    ]

    duplicate_team_ids = (
        participants["team_id"]
        .astype(str)
        .duplicated()
        .sum()
    )

    duplicate_team_names = (
        participants["team"]
        .astype(str)
        .duplicated()
        .sum()
    )

    participant_count = len(participants)

    if participant_count != spec.participant_count:
        return OnboardingStageStatus(
            stage_key="target_participants",
            status="FAIL",
            summary=(
                f"Expected {spec.participant_count} target clubs, "
                f"found {participant_count}."
            ),
            program="scripts.ingest_teams",
        )

    if not invalid_identity_rows.empty:
        return OnboardingStageStatus(
            stage_key="target_participants",
            status="FAIL",
            summary=(
                "Target participants contain missing team_id, "
                "team, or team_slug values."
            ),
            program="scripts.ingest_teams",
        )

    if duplicate_team_ids:
        return OnboardingStageStatus(
            stage_key="target_participants",
            status="FAIL",
            summary=(
                f"Target participants contain "
                f"{duplicate_team_ids} duplicate team_id values."
            ),
            program="scripts.ingest_teams",
        )

    if duplicate_team_names:
        return OnboardingStageStatus(
            stage_key="target_participants",
            status="FAIL",
            summary=(
                f"Target participants contain "
                f"{duplicate_team_names} duplicate team names."
            ),
            program="scripts.ingest_teams",
        )

    return OnboardingStageStatus(
        stage_key="target_participants",
        status="PASS",
        summary=(
            f"{participant_count}/{spec.participant_count} "
            "target clubs with complete "
            "team_id + team + team_slug identity."
        ),
        program="scripts.ingest_teams",
    )

def evaluate_feeder_teams(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    path = spec.feeder_teams_path

    if not path.exists():
        return OnboardingStageStatus(
            stage_key="feeder_teams",
            status="MISSING",
            summary=(
                f"Expected feeder-team artifact is missing: "
                f"{path}"
            ),
            program="scripts.ingest_teams",
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        "team_id",
        "team",
        "team_slug",
    }

    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        return OnboardingStageStatus(
            stage_key="feeder_teams",
            status="FAIL",
            summary=(
                "Feeder-team artifact is missing required "
                f"identity columns: {missing_columns}"
            ),
            program="scripts.ingest_teams",
        )

    teams = (
        frame[
            [
                "team_id",
                "team",
                "team_slug",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    invalid_identity_rows = teams.loc[
        teams[
            [
                "team_id",
                "team",
                "team_slug",
            ]
        ]
        .isna()
        .any(axis=1)
    ]

    duplicate_team_ids = (
        teams["team_id"]
        .astype(str)
        .duplicated()
        .sum()
    )

    duplicate_team_names = (
        teams["team"]
        .astype(str)
        .duplicated()
        .sum()
    )

    if not invalid_identity_rows.empty:
        return OnboardingStageStatus(
            stage_key="feeder_teams",
            status="FAIL",
            summary=(
                "Feeder teams contain missing team_id, "
                "team, or team_slug values."
            ),
            program="scripts.ingest_teams",
        )

    if duplicate_team_ids or duplicate_team_names:
        return OnboardingStageStatus(
            stage_key="feeder_teams",
            status="FAIL",
            summary=(
                "Feeder teams contain duplicate team identities: "
                f"team_id duplicates={duplicate_team_ids}, "
                f"team-name duplicates={duplicate_team_names}."
            ),
            program="scripts.ingest_teams",
        )

    return OnboardingStageStatus(
        stage_key="feeder_teams",
        status="PASS",
        summary=(
            f"{len(teams)} feeder clubs with complete "
            "team_id + team + team_slug identity."
        ),
        program="scripts.ingest_teams",
    )

def evaluate_feeder_players(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    path = spec.feeder_players_path

    if not path.exists():
        return OnboardingStageStatus(
            stage_key="feeder_players",
            status="MISSING",
            summary=(
                f"Expected feeder-player artifact is missing: "
                f"{path}"
            ),
            program="scripts.ingest_players",
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        "player_id",
        "player",
        "team_id",
        "team",
        "team_slug",
    }

    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        return OnboardingStageStatus(
            stage_key="feeder_players",
            status="FAIL",
            summary=(
                "Feeder-player artifact is missing required "
                f"columns: {missing_columns}"
            ),
            program="scripts.ingest_players",
        )

    unique_players = (
        frame["player_id"]
        .astype(str)
        .nunique()
    )

    unique_teams = (
        frame["team_id"]
        .astype(str)
        .nunique()
    )

    invalid_identity_rows = frame.loc[
        frame[
            [
                "player_id",
                "player",
                "team_id",
                "team",
            ]
        ]
        .isna()
        .any(axis=1)
    ]

    if not invalid_identity_rows.empty:
        return OnboardingStageStatus(
            stage_key="feeder_players",
            status="FAIL",
            summary=(
                "Feeder-player rows contain missing player "
                "or team identity values."
            ),
            program="scripts.ingest_players",
        )

    return OnboardingStageStatus(
        stage_key="feeder_players",
        status="PASS",
        summary=(
            f"{len(frame)} membership rows, "
            f"{unique_players} unique players, "
            f"{unique_teams} feeder clubs."
        ),
        program="scripts.ingest_players",
    )

def evaluate_membership_resolved(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    path = spec.membership_resolved_path

    if not path.exists():
        return OnboardingStageStatus(
            stage_key="membership_resolution",
            status="MISSING",
            summary=(
                f"Resolved membership artifact is missing: "
                f"{path}"
            ),
            program=(
                "research.production."
                "domestic_league_membership_bootstrap"
            ),
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        "player_id",
        "player",
        "team_id",
        "team",
        "team_slug",
    }

    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        return OnboardingStageStatus(
            stage_key="membership_resolution",
            status="FAIL",
            summary=(
                "Resolved membership is missing required "
                f"columns: {missing_columns}"
            ),
            program=(
                "research.production."
                "domestic_league_membership_bootstrap"
            ),
        )

    duplicate_players = (
        frame["player_id"]
        .astype(str)
        .duplicated()
        .sum()
    )

    club_count = (
        frame["team"]
        .astype(str)
        .nunique()
    )

    if duplicate_players:
        return OnboardingStageStatus(
            stage_key="membership_resolution",
            status="FAIL",
            summary=(
                f"Resolved membership contains "
                f"{duplicate_players} duplicate player IDs."
            ),
            program=(
                "research.production."
                "domestic_league_membership_bootstrap"
            ),
        )

    if club_count != spec.participant_count:
        return OnboardingStageStatus(
            stage_key="membership_resolution",
            status="FAIL",
            summary=(
                f"Expected membership across "
                f"{spec.participant_count} target clubs; "
                f"found {club_count}."
            ),
            program=(
                "research.production."
                "domestic_league_membership_bootstrap"
            ),
        )

    return OnboardingStageStatus(
        stage_key="membership_resolution",
        status="PASS",
        summary=(
            f"{len(frame)} unique target players across "
            f"{club_count}/{spec.participant_count} clubs."
        ),
        program=(
            "research.production."
            "domestic_league_membership_bootstrap"
        ),
    )

def evaluate_population_artifacts(
    spec: DomesticLeagueOnboardingSpec,
) -> tuple[OnboardingStageStatus, ...]:
    return (
        evaluate_target_participants(spec),
        evaluate_feeder_teams(spec),
        evaluate_feeder_players(spec),
        evaluate_membership_resolved(spec),
    )

def print_onboarding_contract() -> None:
    print("Domestic League Onboarding Contract")
    print("=" * 72)

    for stage in ONBOARDING_STAGES:
        print()
        print(
            f"{stage.number:02d}. {stage.name}"
        )
        print(
            f"    Program: {stage.program}"
        )
        print(
            f"    {stage.description}"
        )

def evaluate_club_repository(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    path = spec.club_repository_path
    program = (
        "research.production."
        "production_club_repository_builder"
    )

    if not path.exists():
        return OnboardingStageStatus(
            stage_key="production_repository",
            status="MISSING",
            summary=(
                f"Production club repository is missing: {path}"
            ),
            program=program,
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        "club",
        "attack",
        "midfield",
        "defense",
        "goalkeeper",
        "attack_depth",
        "midfield_depth",
        "defense_depth",
        "squad_quality",
        "evidence_score",
        "representation_type",
        "aggregation_profile",
        "player_count",
        "available_player_count",
        "repository_version",
        "repository_scope",
        "representation_season_id",
    }

    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        return OnboardingStageStatus(
            stage_key="production_repository",
            status="FAIL",
            summary=(
                "Production repository is missing required "
                f"columns: {missing_columns}"
            ),
            program=program,
        )

    if len(frame.columns) != 17:
        return OnboardingStageStatus(
            stage_key="production_repository",
            status="FAIL",
            summary=(
                "Production repository must use the "
                f"17-column contract; found {len(frame.columns)}."
            ),
            program=program,
        )

    unique_clubs = (
        frame["club"]
        .dropna()
        .astype(str)
        .nunique()
    )

    if (
        len(frame) != spec.participant_count
        or unique_clubs != spec.participant_count
    ):
        return OnboardingStageStatus(
            stage_key="production_repository",
            status="FAIL",
            summary=(
                f"Expected {spec.participant_count} unique clubs; "
                f"found {len(frame)} rows and "
                f"{unique_clubs} unique clubs."
            ),
            program=program,
        )

    unavailable_rows = frame.loc[
        pd.to_numeric(
            frame["player_count"],
            errors="coerce",
        )
        != pd.to_numeric(
            frame["available_player_count"],
            errors="coerce",
        )
    ]

    if not unavailable_rows.empty:
        return OnboardingStageStatus(
            stage_key="production_repository",
            status="FAIL",
            summary=(
                f"{len(unavailable_rows)} clubs have "
                "player_count != available_player_count."
            ),
            program=program,
        )

    return OnboardingStageStatus(
        stage_key="production_repository",
        status="PASS",
        summary=(
            f"{unique_clubs}/{spec.participant_count} clubs, "
            "17-column production contract, and complete "
            "rated-player availability."
        ),
        program=program,
    )

def evaluate_fixture_snapshot(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    path = spec.fixture_path
    program = (
        "scripts.build_sofascore_league_fixture_snapshot"
    )

    if not path.exists():
        return OnboardingStageStatus(
            stage_key="fixtures",
            status="MISSING",
            summary=(
                f"Fixture snapshot is missing: {path}"
            ),
            program=program,
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    required_columns = {
        "fixture_id",
        "competition",
        "season",
        "matchday",
        "match_date",
        "kickoff_time",
        "home_team",
        "away_team",
        "source_url",
    }

    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        return OnboardingStageStatus(
            stage_key="fixtures",
            status="FAIL",
            summary=(
                "Fixture snapshot is missing required "
                f"columns: {missing_columns}"
            ),
            program=program,
        )

    fixture_count = len(frame)

    fixture_ids = (
        frame["fixture_id"]
        .dropna()
        .astype(str)
    )

    duplicate_fixture_ids = (
        fixture_ids.duplicated().sum()
    )

    fixture_teams = set(
        frame["home_team"]
        .dropna()
        .astype(str)
    ) | set(
        frame["away_team"]
        .dropna()
        .astype(str)
    )

    matchdays = (
        pd.to_numeric(
            frame["matchday"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .nunique()
    )

    if fixture_count != spec.fixture_count:
        return OnboardingStageStatus(
            stage_key="fixtures",
            status="FAIL",
            summary=(
                f"Expected {spec.fixture_count} fixtures; "
                f"found {fixture_count}."
            ),
            program=program,
        )

    if duplicate_fixture_ids:
        return OnboardingStageStatus(
            stage_key="fixtures",
            status="FAIL",
            summary=(
                f"Fixture snapshot contains "
                f"{duplicate_fixture_ids} duplicate fixture IDs."
            ),
            program=program,
        )

    if len(fixture_teams) != spec.participant_count:
        return OnboardingStageStatus(
            stage_key="fixtures",
            status="FAIL",
            summary=(
                f"Expected {spec.participant_count} fixture teams; "
                f"found {len(fixture_teams)}."
            ),
            program=program,
        )

    if matchdays != spec.matchday_count:
        return OnboardingStageStatus(
            stage_key="fixtures",
            status="FAIL",
            summary=(
                f"Expected {spec.matchday_count} matchdays; "
                f"found {matchdays}."
            ),
            program=program,
        )

    return OnboardingStageStatus(
        stage_key="fixtures",
        status="PASS",
        summary=(
            f"{fixture_count}/{spec.fixture_count} fixtures, "
            f"{len(fixture_teams)}/{spec.participant_count} teams, "
            f"{matchdays}/{spec.matchday_count} matchdays."
        ),
        program=program,
    )

def evaluate_identity_alignment(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    program = (
        "research.production."
        "domestic_league_onboarding"
    )

    required_paths = {
        "target participants":
            spec.target_participants_path,
        "production repository":
            spec.club_repository_path,
        "fixture snapshot":
            spec.fixture_path,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if not path.exists()
    ]

    if missing:
        return OnboardingStageStatus(
            stage_key="identity_alignment",
            status="BLOCKED",
            summary=(
                "Identity alignment requires all upstream "
                "club artifacts. Missing: "
                + "; ".join(missing)
            ),
            program=program,
        )

    participants = pd.read_csv(
        spec.target_participants_path,
        low_memory=False,
    )

    repository = pd.read_csv(
        spec.club_repository_path,
        low_memory=False,
    )

    fixtures = pd.read_csv(
        spec.fixture_path,
        low_memory=False,
    )

    target_clubs = set(
        participants["team"]
        .dropna()
        .astype(str)
    )

    repository_clubs = set(
        repository["club"]
        .dropna()
        .astype(str)
    )

    fixture_clubs = set(
        fixtures["home_team"]
        .dropna()
        .astype(str)
    ) | set(
        fixtures["away_team"]
        .dropna()
        .astype(str)
    )

    if not (
        target_clubs
        == repository_clubs
        == fixture_clubs
    ):
        details = (
            f"target_only={sorted(target_clubs - repository_clubs)}, "
            f"repository_only={sorted(repository_clubs - target_clubs)}, "
            f"target_fixture_only={sorted(target_clubs - fixture_clubs)}, "
            f"fixture_only={sorted(fixture_clubs - target_clubs)}"
        )

        return OnboardingStageStatus(
            stage_key="identity_alignment",
            status="FAIL",
            summary=(
                "Target participants, repository clubs, and "
                f"fixture clubs do not align exactly: {details}"
            ),
            program=program,
        )

    return OnboardingStageStatus(
        stage_key="identity_alignment",
        status="PASS",
        summary=(
            f"{len(target_clubs)} clubs align exactly across "
            "target participants, production repository, "
            "and fixture snapshot."
        ),
        program=program,
    )

def evaluate_production_artifacts(
    spec: DomesticLeagueOnboardingSpec,
) -> tuple[OnboardingStageStatus, ...]:
    return (
        evaluate_club_repository(spec),
        evaluate_fixture_snapshot(spec),
        evaluate_identity_alignment(spec),
    )

def evaluate_simulation_config(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    program = "simulation.domestic_league_configs"

    config = DOMESTIC_LEAGUE_CONFIGS.get(
        spec.key
    )

    if config is None:
        return OnboardingStageStatus(
            stage_key="simulation_config",
            status="MISSING",
            summary=(
                f"No DomesticLeagueSimulationConfig is "
                f"registered for {spec.key!r}."
            ),
            program=program,
        )

    mismatches = []

    expected_values = {
        "competition_name":
            spec.competition_name,
        "season":
            spec.target_season,
        "participant_count":
            spec.participant_count,
        "matchday_count":
            spec.matchday_count,
        "fixture_count":
            spec.fixture_count,
        "repository_source":
            spec.repository_source,
    }

    for field, expected in expected_values.items():
        actual = getattr(
            config,
            field,
            None,
        )

        if actual != expected:
            mismatches.append(
                f"{field}: expected {expected!r}, "
                f"found {actual!r}"
            )

    if Path(config.fixture_path) != spec.fixture_path:
        mismatches.append(
            "fixture_path does not match onboarding spec"
        )

    if Path(config.repository_path) != spec.club_repository_path:
        mismatches.append(
            "repository_path does not match onboarding spec"
        )

    if mismatches:
        return OnboardingStageStatus(
            stage_key="simulation_config",
            status="FAIL",
            summary=(
                "Simulation configuration mismatch: "
                + "; ".join(mismatches)
            ),
            program=program,
        )

    return OnboardingStageStatus(
        stage_key="simulation_config",
        status="PASS",
        summary=(
            f"{spec.key!r} is registered for "
            f"{spec.participant_count} clubs, "
            f"{spec.matchday_count} matchdays, "
            f"{spec.fixture_count} fixtures, using "
            f"{spec.repository_source!r}."
        ),
        program=program,
    )

def evaluate_clubelo_preload(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    program = (
        "research.production."
        "preload_domestic_league_clubelo"
    )

    if not spec.club_repository_path.exists():
        return OnboardingStageStatus(
            stage_key="clubelo_preload",
            status="BLOCKED",
            summary=(
                "Production repository is required before "
                "ClubElo cache coverage can be audited."
            ),
            program=program,
        )

    frame = pd.read_csv(
        spec.club_repository_path,
        low_memory=False,
    )

    if "club" not in frame.columns:
        return OnboardingStageStatus(
            stage_key="clubelo_preload",
            status="FAIL",
            summary=(
                "Production repository has no 'club' column."
            ),
            program=program,
        )

    clubs = sorted(
        frame["club"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    repository = ClubEloRepository(
        cache_directory=CLUBELO_CACHE_DIRECTORY,
    )

    team_slugs_by_club: dict[str, str] = {}

    if spec.target_participants_path.exists():
        participants = pd.read_csv(
            spec.target_participants_path,
            low_memory=False,
        )

        if {
            "team",
            "team_slug",
        }.issubset(participants.columns):
            team_slugs_by_club = {
                str(row["team"]): str(row["team_slug"])
                for _, row in (
                    participants[
                        ["team", "team_slug"]
                    ]
                    .dropna()
                    .drop_duplicates()
                    .iterrows()
                )
            }

    cached = []
    missing = []

    for club in clubs:
        if repository.cache_path(club).exists():
            cached.append(club)
        else:
            missing.append(club)

    missing_details = []

    for club in missing:
        explicit_lookup = get_clubelo_name_override(
            competition_key=spec.key,
            production_club=club,
        )

        candidates = build_clubelo_lookup_candidates(
            production_club=club,
            explicit_lookup=explicit_lookup,
            team_slug=team_slugs_by_club.get(club),
        )

        missing_details.append(
            f"{club} [{' -> '.join(candidates)}]"
        )

    if missing:
        return OnboardingStageStatus(
            stage_key="clubelo_preload",
            status="BLOCKED",
            summary=(
                f"{len(cached)}/{len(clubs)} production-key "
                "ClubElo histories cached. Missing: "
                + "; ".join(missing_details)
                + "."
            ),
            program=program,
        )

    return OnboardingStageStatus(
        stage_key="clubelo_preload",
        status="PASS",
        summary=(
            f"{len(cached)}/{len(clubs)} production-key "
            "ClubElo histories cached."
        ),
        program=program,
    )

def evaluate_production_routing(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    program = "research.adapters.football_model_adapter"

    if (
        spec.repository_source
        not in DOMESTIC_PRODUCTION_REPOSITORY_SOURCES
    ):
        return OnboardingStageStatus(
            stage_key="production_routing",
            status="MISSING",
            summary=(
                f"{spec.repository_source!r} is not registered "
                "in the shared domestic production-model route."
            ),
            program=program,
        )

    return OnboardingStageStatus(
        stage_key="production_routing",
        status="PASS",
        summary=(
            f"{spec.repository_source!r} is registered in the "
            "shared domestic production-model route."
        ),
        program=program,
    )

def evaluate_operational_status(
    spec: DomesticLeagueOnboardingSpec,
) -> tuple[OnboardingStageStatus, ...]:
    return (
        evaluate_simulation_config(spec),
        evaluate_production_routing(spec),
        evaluate_clubelo_preload(spec),
    )

def print_onboarding_spec(
    spec: DomesticLeagueOnboardingSpec,
) -> None:
    print()
    print(
        f"{spec.competition_name} "
        f"{spec.target_season} Onboarding Specification"
    )
    print("=" * 72)

    print(
        f"Key: {spec.key}"
    )
    print(
        f"Competition ID: {spec.competition_id}"
    )
    print(
        f"Target season ID: {spec.target_season_id}"
    )

    print()
    print("Feeder")
    print("-" * 72)
    print(
        f"Key: {spec.feeder_key}"
    )
    print(
        f"Competition: {spec.feeder_competition_name}"
    )
    print(
        f"Competition ID: {spec.feeder_competition_id}"
    )
    print(
        f"Season: {spec.feeder_season}"
    )
    print(
        f"Season ID: {spec.feeder_season_id}"
    )

    print()
    print("League Structure")
    print("-" * 72)
    print(
        f"Participants: {spec.participant_count}"
    )
    print(
        f"Matchdays: {spec.matchday_count}"
    )
    print(
        f"Fixtures: {spec.fixture_count}"
    )
    print(
        f"Timezone: {spec.timezone_name}"
    )

    print()
    print("Production")
    print("-" * 72)
    print(
        f"Repository source: {spec.repository_source}"
    )
    print(
        f"Bootstrap directory: "
        f"{spec.bootstrap_directory}"
    )

def print_stage_status(
    status: OnboardingStageStatus,
) -> None:
    print(
        f"{status.stage_key}: {status.status}"
    )
    print(
        f"  {status.summary}"
    )
    print(
        f"  Program: {status.program}"
    )

def evaluate_artifact_exists(
    *,
    stage_key: str,
    path: Path,
    program: str,
    artifact_name: str,
) -> OnboardingStageStatus:
    if not path.exists():
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="MISSING",
            summary=(
                f"{artifact_name} is missing: {path}"
            ),
            program=program,
        )

    return OnboardingStageStatus(
        stage_key=stage_key,
        status="PASS",
        summary=(
            f"{artifact_name} exists: {path}"
        ),
        program=program,
    )

def evaluate_foundation_artifacts(
    spec: DomesticLeagueOnboardingSpec,
) -> tuple[OnboardingStageStatus, ...]:
    discovery_program = (
        "scripts.discover_sofascore_competition_seasons"
    )

    manifest_program = (
        "scripts.sofascore_build_competition_manifest"
    )

    return (
        evaluate_single_scope_csv(
            stage_key="discover_target_season",
            path=spec.target_season_registry_path,
            program=discovery_program,
            artifact_name="Target season registry",
            expected_values={
                "competition_key": spec.key,
                "competition_name": spec.competition_name,
                "unique_tournament_id": spec.competition_id,
                "season_start_year":
                    spec.target_season_start_year,
                "season_id": spec.target_season_id,
            },
        ),
        evaluate_single_scope_csv(
            stage_key="discover_feeder_season",
            path=spec.feeder_season_registry_path,
            program=discovery_program,
            artifact_name="Feeder season registry",
            expected_values={
                "competition_key": spec.feeder_key,
                "competition_name":
                    spec.feeder_competition_name,
                "unique_tournament_id":
                    spec.feeder_competition_id,
                "season_start_year":
                    spec.feeder_season_start_year,
                "season_id": spec.feeder_season_id,
            },
        ),
        evaluate_manifest_scope(
            stage_key="target_competition_manifest",
            path=spec.target_competition_manifest_path,
            program=manifest_program,
            artifact_name="Target competition manifest",
            expected_values={
                "competition":
                    spec.competition_name,
                "competition_id":
                    spec.competition_id,
                "season_id":
                    spec.target_season_id,
            },
        ),
        evaluate_manifest_scope(
            stage_key="feeder_competition_manifest",
            path=spec.feeder_competition_manifest_path,
            program=manifest_program,
            artifact_name="Feeder competition manifest",
            expected_values={
                "competition":
                    spec.feeder_competition_name,
                "competition_id":
                    spec.feeder_competition_id,
                "season_id":
                    spec.feeder_season_id,
            },
        ),
    )

def evaluate_single_scope_csv(
    *,
    stage_key: str,
    path: Path,
    program: str,
    artifact_name: str,
    expected_values: dict[str, object],
) -> OnboardingStageStatus:
    if not path.exists():
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="MISSING",
            summary=(
                f"{artifact_name} is missing: {path}"
            ),
            program=program,
        )

    frame = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
    )

    if len(frame) != 1:
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="FAIL",
            summary=(
                f"{artifact_name} must contain exactly 1 row; "
                f"found {len(frame)}."
            ),
            program=program,
        )

    row = frame.iloc[0]

    mismatches = []

    for column, expected in expected_values.items():
        if column not in frame.columns:
            mismatches.append(
                f"{column}=<missing column>"
            )
            continue

        actual = str(row[column]).strip()
        expected_text = str(expected).strip()

        if actual != expected_text:
            mismatches.append(
                f"{column}: expected {expected_text!r}, "
                f"found {actual!r}"
            )

    if mismatches:
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="FAIL",
            summary=(
                f"{artifact_name} identity mismatch: "
                + "; ".join(mismatches)
            ),
            program=program,
        )

    expected_summary = ", ".join(
        f"{key}={value}"
        for key, value in expected_values.items()
    )

    return OnboardingStageStatus(
        stage_key=stage_key,
        status="PASS",
        summary=(
            f"{artifact_name} matches expected scope: "
            f"{expected_summary}."
        ),
        program=program,
    )

def evaluate_manifest_scope(
    *,
    stage_key: str,
    path: Path,
    program: str,
    artifact_name: str,
    expected_values: dict[str, object],
) -> OnboardingStageStatus:
    if not path.exists():
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="MISSING",
            summary=(
                f"{artifact_name} is missing: {path}"
            ),
            program=program,
        )

    frame = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
    )

    missing_columns = sorted(
        set(expected_values) - set(frame.columns)
    )

    if missing_columns:
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="FAIL",
            summary=(
                f"{artifact_name} is missing required "
                f"columns: {missing_columns}"
            ),
            program=program,
        )

    matches = frame.copy()

    for column, expected in expected_values.items():
        matches = matches.loc[
            matches[column].astype(str).str.strip()
            == str(expected).strip()
        ]

    if len(matches) == 0:
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="FAIL",
            summary=(
                f"{artifact_name} does not contain the "
                "expected domestic competition-season scope."
            ),
            program=program,
        )

    if len(matches) > 1:
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="FAIL",
            summary=(
                f"{artifact_name} contains {len(matches)} "
                "rows for the expected domestic "
                "competition-season scope; expected exactly 1."
            ),
            program=program,
        )

    expected_summary = ", ".join(
        f"{key}={value}"
        for key, value in expected_values.items()
    )

    return OnboardingStageStatus(
        stage_key=stage_key,
        status="PASS",
        summary=(
            f"{artifact_name} contains exactly one expected "
            f"domestic scope: {expected_summary}. "
            f"Total manifest rows: {len(frame)}."
        ),
        program=program,
    )

def print_stage_statuses(
    statuses: tuple[OnboardingStageStatus, ...],
) -> None:
    for status in statuses:
        print_stage_status(status)
        print()

def build_intelligence_audit_config(
    *,
    spec: DomesticLeagueOnboardingSpec,
    ratings_path: Path,
) -> DomesticLeagueIntelligenceBackfillConfig:
    return DomesticLeagueIntelligenceBackfillConfig(
        membership_path=spec.membership_resolved_path,
        baseline_ratings_path=ratings_path,
        baseline_registry_path=spec.baseline_registry_path,
        output_directory=(
            spec.expanded_ratings_path.parent
        ),
    )

def evaluate_intelligence_status(
    spec: DomesticLeagueOnboardingSpec,
) -> tuple[OnboardingStageStatus, ...]:
    program = (
        "research.production."
        "domestic_league_intelligence_backfill"
    )

    if not spec.membership_resolved_path.exists():
        return (
            OnboardingStageStatus(
                stage_key="intelligence_coverage",
                status="BLOCKED",
                summary=(
                    "Resolved membership is required before "
                    "Player Intelligence coverage can be audited."
                ),
                program=program,
            ),
        )

    if not spec.baseline_ratings_path.exists():
        return (
            OnboardingStageStatus(
                stage_key="intelligence_coverage",
                status="MISSING",
                summary=(
                    "Frozen baseline ratings artifact is missing: "
                    f"{spec.baseline_ratings_path}"
                ),
                program=program,
            ),
        )

    baseline = audit_intelligence_coverage(
        build_intelligence_audit_config(
            spec=spec,
            ratings_path=spec.baseline_ratings_path,
        )
    )

    baseline_status = OnboardingStageStatus(
        stage_key="intelligence_coverage",
        status="PASS",
        summary=(
            f"Frozen baseline covers "
            f"{baseline.rated_players}/"
            f"{baseline.membership_players} target players "
            f"({baseline.coverage:.2%}); "
            f"{baseline.missing_players} missing."
        ),
        program=program,
    )

    if baseline.missing_players == 0:
        backfill_status = OnboardingStageStatus(
            stage_key="intelligence_backfill",
            status="SKIPPED",
            summary=(
                "Player Intelligence backfill is not required; "
                "the frozen baseline already provides 100% "
                "target-membership coverage."
            ),
            program=program,
        )

        return (
            baseline_status,
            backfill_status,
        )

    if not spec.expanded_ratings_path.exists():
        backfill_status = OnboardingStageStatus(
            stage_key="intelligence_backfill",
            status="MISSING",
            summary=(
                f"Backfill is required for "
                f"{baseline.missing_players} players, but the "
                "expanded ratings artifact is missing: "
                f"{spec.expanded_ratings_path}"
            ),
            program=program,
        )

        return (
            baseline_status,
            backfill_status,
        )

    final = audit_intelligence_coverage(
        build_intelligence_audit_config(
            spec=spec,
            ratings_path=spec.expanded_ratings_path,
        )
    )

    if final.missing_players:
        backfill_status = OnboardingStageStatus(
            stage_key="intelligence_backfill",
            status="FAIL",
            summary=(
                f"Backfill remains incomplete: "
                f"{final.rated_players}/"
                f"{final.membership_players} target players "
                f"rated ({final.coverage:.2%}); "
                f"{final.missing_players} still missing."
            ),
            program=program,
        )
    else:
        backfill_status = OnboardingStageStatus(
            stage_key="intelligence_backfill",
            status="PASS",
            summary=(
                f"Backfill was required for "
                f"{baseline.missing_players} players. "
                f"Final coverage is "
                f"{final.rated_players}/"
                f"{final.membership_players} "
                f"({final.coverage:.2%}); "
                "0 missing."
            ),
            program=program,
        )

    return (
        baseline_status,
        backfill_status,
    )

def evaluate_validation_artifact(
    *,
    stage_key: str,
    path: Path,
    program: str,
    artifact_name: str,
    expected_values: dict[str, object],
) -> OnboardingStageStatus:
    if not path.exists():
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="MISSING",
            summary=(
                f"{artifact_name} has not been recorded: "
                f"{path}"
            ),
            program=program,
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="FAIL",
            summary=(
                f"{artifact_name} could not be read as "
                f"valid JSON: {error}"
            ),
            program=program,
        )

    mismatches = []

    for key, expected in expected_values.items():
        if key not in payload:
            mismatches.append(
                f"{key}=<missing>"
            )
            continue

        actual = payload[key]

        if actual != expected:
            mismatches.append(
                f"{key}: expected {expected!r}, "
                f"found {actual!r}"
            )

    if mismatches:
        return OnboardingStageStatus(
            stage_key=stage_key,
            status="FAIL",
            summary=(
                f"{artifact_name} does not match the "
                "onboarding contract: "
                + "; ".join(mismatches)
            ),
            program=program,
        )

    return OnboardingStageStatus(
        stage_key=stage_key,
        status="PASS",
        summary=(
            f"{artifact_name} is recorded and matches "
            "the onboarding contract."
        ),
        program=program,
    )

def evaluate_structural_simulation(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    config_status = evaluate_simulation_config(spec)

    if config_status.status != "PASS":
        return OnboardingStageStatus(
            stage_key="structural_simulation",
            status="BLOCKED",
            summary=(
                "Structural simulation requires a valid "
                "DomesticLeagueSimulationConfig."
            ),
            program="scripts.run_domestic_league_simulation",
        )

    return evaluate_validation_artifact(
        stage_key="structural_simulation",
        path=spec.structural_validation_path,
        program="scripts.run_domestic_league_simulation",
        artifact_name="Structural simulation validation",
        expected_values={
            "status": "PASS",
            "validation_type": "structural",
            "competition_key": spec.key,
            "competition_name": spec.competition_name,
            "season": spec.target_season,
            "model": "structural",
            "simulations": 1,
            "participant_count": spec.participant_count,
            "expected_participant_count":
                spec.participant_count,
            "fixture_count": spec.fixture_count,
            "expected_fixture_count":
                spec.fixture_count,
            "repository_source":
                spec.repository_source,
        },
    )

def evaluate_production_simulation(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    routing_status = evaluate_production_routing(spec)
    clubelo_status = evaluate_clubelo_preload(spec)

    blockers = []

    if routing_status.status != "PASS":
        blockers.append(
            "production-model routing is incomplete"
        )

    if clubelo_status.status != "PASS":
        blockers.append(
            "ClubElo cache coverage is incomplete"
        )

    if blockers:
        return OnboardingStageStatus(
            stage_key="production_simulation",
            status="BLOCKED",
            summary=(
                "Production simulation is blocked because "
                + " and ".join(blockers)
                + "."
            ),
            program="scripts.run_domestic_league_simulation",
        )

    return evaluate_validation_artifact(
        stage_key="production_simulation",
        path=spec.production_validation_path,
        program="scripts.run_domestic_league_simulation",
        artifact_name="Production simulation validation",
        expected_values ={
            "status": "PASS",
        }
    )

def evaluate_monte_carlo(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    production_status = (
        evaluate_production_simulation(spec)
    )

    if production_status.status != "PASS":
        return OnboardingStageStatus(
            stage_key="monte_carlo",
            status="BLOCKED",
            summary=(
                "Monte Carlo validation requires a recorded "
                "successful production simulation."
            ),
            program="scripts.run_domestic_league_simulation",
        )

    return evaluate_validation_artifact(
        stage_key="monte_carlo",
        path=spec.monte_carlo_validation_path,
        program="scripts.run_domestic_league_simulation",
        artifact_name="Monte Carlo validation",
    )

def evaluate_regression_suite(
    spec: DomesticLeagueOnboardingSpec,
) -> OnboardingStageStatus:
    return evaluate_validation_artifact(
        stage_key="regression_suite",
        path=spec.regression_validation_path,
        program="pytest",
        artifact_name="Regression-suite validation",
        expected_values={
            "status": "PASS",
        }
    )

def evaluate_execution_status(
    spec: DomesticLeagueOnboardingSpec,
) -> tuple[OnboardingStageStatus, ...]:
    return (
        evaluate_structural_simulation(spec),
        evaluate_production_simulation(spec),
        evaluate_monte_carlo(spec),
        evaluate_regression_suite(spec),
    )

def evaluate_onboarding_status(
    spec: DomesticLeagueOnboardingSpec,
) -> tuple[OnboardingStageStatus, ...]:
    return (
        *evaluate_foundation_artifacts(spec),
        *evaluate_population_artifacts(spec),
        *evaluate_intelligence_status(spec),
        *evaluate_production_artifacts(spec),
        *evaluate_operational_status(spec),
        *evaluate_execution_status(spec),
    )


def print_onboarding_status(
    spec: DomesticLeagueOnboardingSpec,
) -> None:
    statuses = evaluate_onboarding_status(spec)

    print()
    print(
        f"{spec.competition_name.upper()} "
        f"{spec.target_season} ONBOARDING STATUS"
    )
    print("=" * 72)

    print()

    for status in statuses:
        print(
            f"{status.stage_key}: {status.status}"
        )
        print(
            f"  {status.summary}"
        )
        print(
            f"  Program: {status.program}"
        )
        print()

    pass_count = sum(
        status.status == "PASS"
        for status in statuses
    )

    blocked_count = sum(
        status.status == "BLOCKED"
        for status in statuses
    )

    missing_count = sum(
        status.status == "MISSING"
        for status in statuses
    )

    fail_count = sum(
        status.status == "FAIL"
        for status in statuses
    )

    skipped_count = sum(
        status.status == "SKIPPED"
        for status in statuses
    )

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(
        f"PASS:    {pass_count}"
    )
    print(
        f"BLOCKED: {blocked_count}"
    )
    print(
        f"MISSING: {missing_count}"
    )
    print(
        f"FAIL:    {fail_count}"
    )
    print(
        f"SKIPPED: {skipped_count}"
    )

def main() -> None:
    arguments = parse_args()

    if arguments.contract:
        print_onboarding_contract()

    if arguments.competition is None:
        if not arguments.contract:
            print_onboarding_contract()
        return

    spec = DOMESTIC_LEAGUE_ONBOARDING_SPECS[
        arguments.competition
    ]

    if arguments.spec:
        print_onboarding_spec(spec)

    if arguments.status:
        print_onboarding_status(spec)

    if not (
        arguments.status
        or arguments.spec
        or arguments.contract
    ):
        print_onboarding_status(spec)


if __name__ == "__main__":
    main()
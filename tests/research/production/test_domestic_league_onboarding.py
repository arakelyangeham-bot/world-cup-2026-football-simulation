#test_domestic_league_onboarding

import json
from pathlib import Path

import pandas as pd

from research.production.domestic_league_onboarding import (
    DomesticLeagueOnboardingSpec,
    evaluate_identity_alignment,
    evaluate_target_participants,
    evaluate_validation_artifact,
)

from research.production.domestic_league_onboarding import (
    evaluate_clubelo_preload,
)
from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)

def _build_spec(
    tmp_path: Path,
    *,
    target_participants_path: Path,
) -> DomesticLeagueOnboardingSpec:
    bootstrap_directory = tmp_path / "bootstrap"

    return DomesticLeagueOnboardingSpec(
        key="test_league",
        competition_name="Test League",
        competition_id=999,

        target_season="2026-27",
        target_season_start_year=2026,
        target_season_id=1001,

        feeder_key="test_feeder",
        feeder_competition_name="Test Feeder",
        feeder_competition_id=998,
        feeder_season="25/26",
        feeder_season_start_year=2025,
        feeder_season_id=1000,

        participant_count=2,
        matchday_count=2,
        fixture_count=2,
        timezone_name="UTC",

        bootstrap_directory=bootstrap_directory,
        target_season_registry_path=tmp_path / "target_registry.csv",
        feeder_season_registry_path=tmp_path / "feeder_registry.csv",
        target_competition_manifest_path=tmp_path / "target_manifest.csv",
        feeder_competition_manifest_path=tmp_path / "feeder_manifest.csv",
        target_participants_path=target_participants_path,
        feeder_teams_path=tmp_path / "feeder_teams.csv",
        feeder_players_path=tmp_path / "feeder_players.csv",
        membership_candidate_path=bootstrap_directory / "membership_candidate.csv",
        membership_resolved_path=bootstrap_directory / "membership_resolved.csv",

        baseline_registry_path=tmp_path / "baseline_registry.csv",
        baseline_ratings_path=tmp_path / "baseline_ratings.csv",

        expanded_registry_path=bootstrap_directory / "expanded_registry.csv",
        expanded_ratings_path=bootstrap_directory / "expanded_ratings.csv",
        club_repository_path=bootstrap_directory / "club_repository.csv",
        fixture_path=bootstrap_directory / "fixtures.csv",

        repository_source="test_league_production_v1",

        structural_validation_path=tmp_path / "structural_validation.json",
        production_validation_path=tmp_path / "production_validation.json",
        monte_carlo_validation_path=tmp_path / "monte_carlo_validation.json",
        regression_validation_path=tmp_path / "regression_suite.json",
    )


def test_target_participants_pass_with_complete_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "participants.csv"

    pd.DataFrame(
        [
            {
                "team_id": 1,
                "team": "Alpha",
                "team_slug": "alpha",
            },
            {
                "team_id": 2,
                "team": "Beta",
                "team_slug": "beta",
            },
        ]
    ).to_csv(
        path,
        index=False,
    )

    spec = _build_spec(
        tmp_path,
        target_participants_path=path,
    )

    result = evaluate_target_participants(spec)

    assert result.status == "PASS"


def test_target_participants_fail_when_team_slug_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "participants.csv"

    pd.DataFrame(
        [
            {
                "team_id": 1,
                "team": "Alpha",
            },
            {
                "team_id": 2,
                "team": "Beta",
            },
        ]
    ).to_csv(
        path,
        index=False,
    )

    spec = _build_spec(
        tmp_path,
        target_participants_path=path,
    )

    result = evaluate_target_participants(spec)

    assert result.status == "FAIL"
    assert "team_slug" in result.summary


def test_validation_artifact_passes_when_contract_matches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation.json"

    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "competition_key": "test_league",
                "model": "structural",
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_validation_artifact(
        stage_key="structural_simulation",
        path=path,
        program="test_program",
        artifact_name="Structural validation",
        expected_values={
            "status": "PASS",
            "competition_key": "test_league",
            "model": "structural",
        },
    )

    assert result.status == "PASS"


def test_validation_artifact_fails_when_contract_mismatches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation.json"

    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "competition_key": "wrong_league",
                "model": "structural",
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_validation_artifact(
        stage_key="structural_simulation",
        path=path,
        program="test_program",
        artifact_name="Structural validation",
        expected_values={
            "status": "PASS",
            "competition_key": "test_league",
            "model": "structural",
        },
    )

    assert result.status == "FAIL"
    assert "competition_key" in result.summary


def test_validation_artifact_is_missing_when_file_absent(
    tmp_path: Path,
) -> None:
    result = evaluate_validation_artifact(
        stage_key="structural_simulation",
        path=tmp_path / "missing.json",
        program="test_program",
        artifact_name="Structural validation",
        expected_values={
            "status": "PASS",
        },
    )

    assert result.status == "MISSING"

def test_identity_alignment_passes_for_matching_club_sets(
    tmp_path: Path,
) -> None:
    participants_path = tmp_path / "participants.csv"
    repository_path = tmp_path / "repository.csv"
    fixtures_path = tmp_path / "fixtures.csv"

    pd.DataFrame(
        [
            {"team_id": 1, "team": "Alpha", "team_slug": "alpha"},
            {"team_id": 2, "team": "Beta", "team_slug": "beta"},
        ]
    ).to_csv(participants_path, index=False)

    pd.DataFrame(
        [
            {"club": "Alpha"},
            {"club": "Beta"},
        ]
    ).to_csv(repository_path, index=False)

    pd.DataFrame(
        [
            {"home_team": "Alpha", "away_team": "Beta"},
            {"home_team": "Beta", "away_team": "Alpha"},
        ]
    ).to_csv(fixtures_path, index=False)

    spec = _build_spec(
        tmp_path,
        target_participants_path=participants_path,
    )

    object.__setattr__(
        spec,
        "club_repository_path",
        repository_path,
    )
    object.__setattr__(
        spec,
        "fixture_path",
        fixtures_path,
    )

    result = evaluate_identity_alignment(spec)

    assert result.status == "PASS"


def test_identity_alignment_fails_for_mismatched_club_sets(
    tmp_path: Path,
) -> None:
    participants_path = tmp_path / "participants.csv"
    repository_path = tmp_path / "repository.csv"
    fixtures_path = tmp_path / "fixtures.csv"

    pd.DataFrame(
        [
            {"team_id": 1, "team": "Alpha", "team_slug": "alpha"},
            {"team_id": 2, "team": "Beta", "team_slug": "beta"},
        ]
    ).to_csv(participants_path, index=False)

    pd.DataFrame(
        [
            {"club": "Alpha"},
            {"club": "Gamma"},
        ]
    ).to_csv(repository_path, index=False)

    pd.DataFrame(
        [
            {"home_team": "Alpha", "away_team": "Beta"},
            {"home_team": "Beta", "away_team": "Alpha"},
        ]
    ).to_csv(fixtures_path, index=False)

    spec = _build_spec(
        tmp_path,
        target_participants_path=participants_path,
    )

    object.__setattr__(
        spec,
        "club_repository_path",
        repository_path,
    )
    object.__setattr__(
        spec,
        "fixture_path",
        fixtures_path,
    )

    result = evaluate_identity_alignment(spec)

    assert result.status == "FAIL"
    assert "Gamma" in result.summary

def test_clubelo_preload_is_blocked_when_cache_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    participants_path = tmp_path / "participants.csv"
    repository_path = tmp_path / "repository.csv"

    pd.DataFrame(
        [
            {"team_id": 1, "team": "Alpha", "team_slug": "alpha"},
            {"team_id": 2, "team": "Beta", "team_slug": "beta"},
        ]
    ).to_csv(participants_path, index=False)

    pd.DataFrame(
        [
            {"club": "Alpha"},
            {"club": "Beta"},
        ]
    ).to_csv(repository_path, index=False)

    spec = _build_spec(
        tmp_path,
        target_participants_path=participants_path,
    )

    object.__setattr__(
        spec,
        "club_repository_path",
        repository_path,
    )

    import research.production.domestic_league_onboarding as onboarding

    monkeypatch.setattr(
        onboarding,
        "CLUBELO_CACHE_DIRECTORY",
        tmp_path / "clubelo_cache",
    )

    cache_repository = ClubEloRepository(
        cache_directory=tmp_path / "clubelo_cache",
    )

    alpha_path = cache_repository.cache_path("Alpha")
    alpha_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    alpha_path.write_text(
        "placeholder",
        encoding="utf-8",
    )

    result = evaluate_clubelo_preload(spec)

    assert result.status == "BLOCKED"
    assert "1/2" in result.summary
    assert "Beta" in result.summary

def test_clubelo_preload_passes_when_all_caches_exist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    participants_path = tmp_path / "participants.csv"
    repository_path = tmp_path / "repository.csv"

    pd.DataFrame(
        [
            {"team_id": 1, "team": "Alpha", "team_slug": "alpha"},
            {"team_id": 2, "team": "Beta", "team_slug": "beta"},
        ]
    ).to_csv(participants_path, index=False)

    pd.DataFrame(
        [
            {"club": "Alpha"},
            {"club": "Beta"},
        ]
    ).to_csv(repository_path, index=False)

    spec = _build_spec(
        tmp_path,
        target_participants_path=participants_path,
    )

    object.__setattr__(
        spec,
        "club_repository_path",
        repository_path,
    )

    import research.production.domestic_league_onboarding as onboarding

    monkeypatch.setattr(
        onboarding,
        "CLUBELO_CACHE_DIRECTORY",
        tmp_path / "clubelo_cache",
    )

    cache_repository = ClubEloRepository(
        cache_directory=tmp_path / "clubelo_cache",
    )

    for club in ["Alpha", "Beta"]:
        path = cache_repository.cache_path(club)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "placeholder",
            encoding="utf-8",
        )

    result = evaluate_clubelo_preload(spec)

    assert result.status == "PASS"
    assert "2/2" in result.summary

def test_production_simulation_is_blocked_when_clubelo_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    participants_path = tmp_path / "participants.csv"
    repository_path = tmp_path / "repository.csv"

    pd.DataFrame(
        [
            {"team_id": 1, "team": "Alpha", "team_slug": "alpha"},
            {"team_id": 2, "team": "Beta", "team_slug": "beta"},
        ]
    ).to_csv(participants_path, index=False)

    pd.DataFrame(
        [
            {"club": "Alpha"},
            {"club": "Beta"},
        ]
    ).to_csv(repository_path, index=False)

    spec = _build_spec(
        tmp_path,
        target_participants_path=participants_path,
    )

    object.__setattr__(
        spec,
        "club_repository_path",
        repository_path,
    )

    import research.production.domestic_league_onboarding as onboarding

    monkeypatch.setattr(
        onboarding,
        "CLUBELO_CACHE_DIRECTORY",
        tmp_path / "clubelo_cache",
    )

    monkeypatch.setattr(
        onboarding,
        "evaluate_production_routing",
        lambda _spec: onboarding.OnboardingStageStatus(
            stage_key="production_routing",
            status="PASS",
            summary="Production routing is registered.",
            program="test",
        ),
    )

    result = onboarding.evaluate_production_simulation(
        spec
    )

    assert result.status == "BLOCKED"
    assert "ClubElo cache coverage is incomplete" in result.summary

def test_monte_carlo_is_blocked_when_production_not_validated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    participants_path = tmp_path / "participants.csv"

    pd.DataFrame(
        [
            {"team_id": 1, "team": "Alpha", "team_slug": "alpha"},
            {"team_id": 2, "team": "Beta", "team_slug": "beta"},
        ]
    ).to_csv(participants_path, index=False)

    spec = _build_spec(
        tmp_path,
        target_participants_path=participants_path,
    )

    import research.production.domestic_league_onboarding as onboarding

    monkeypatch.setattr(
        onboarding,
        "evaluate_production_simulation",
        lambda _spec: onboarding.OnboardingStageStatus(
            stage_key="production_simulation",
            status="BLOCKED",
            summary="Production simulation is blocked.",
            program="test",
        ),
    )

    result = onboarding.evaluate_monte_carlo(
        spec
    )

    assert result.status == "BLOCKED"
    assert (
        "requires a recorded successful production simulation"
        in result.summary
    )
# test_team_feature_provider.py

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from research.football_features.team_feature_provider import (
    StaticCsvTeamFeatureProvider,
    TeamFeatureProvider,
    TeamFeatureRequest,
    TeamFeatureResult,
    normalize_prediction_date,
)


def _write_repository(
    path: Path,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    if rows is None:
        rows = [
            {
                "nation": "France",
                "att_composite": 0.81,
                "mid_composite": 0.77,
                "def_composite": 0.74,
                "gk_composite": 0.72,
                "poisson_attack_adj": 1.18,
                "poisson_defense_adj": 0.82,
            },
            {
                "nation": "Germany",
                "att_composite": 0.76,
                "mid_composite": 0.79,
                "def_composite": 0.70,
                "gk_composite": 0.75,
                "poisson_attack_adj": 1.10,
                "poisson_defense_adj": 0.88,
            },
        ]

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )

    return path


@pytest.fixture
def repository_path(
    tmp_path: Path,
) -> Path:
    return _write_repository(
        tmp_path / "team_repository.csv"
    )


@pytest.fixture
def provider(
    repository_path: Path,
) -> StaticCsvTeamFeatureProvider:
    return StaticCsvTeamFeatureProvider(
        repository_path=repository_path,
        provider_name="test_static_provider",
        representation_type="static_repository",
        aggregation_profile="legacy_static",
        repository_version="test_v1",
        repository_scope="national_teams",
    )


# ---------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "2026-06-15",
        date(2026, 6, 15),
        datetime(2026, 6, 15, 12, 30),
        pd.Timestamp("2026-06-15"),
        pd.Timestamp(
            "2026-06-15T12:30:00+02:00"
        ),
    ],
)
def test_normalize_prediction_date_returns_utc_timestamp(
    value: object,
) -> None:
    timestamp = normalize_prediction_date(
        value
    )

    assert isinstance(
        timestamp,
        pd.Timestamp,
    )

    assert timestamp.tz is not None
    assert str(timestamp.tz) == "UTC"


def test_timezone_aware_date_is_converted_to_utc() -> None:
    timestamp = normalize_prediction_date(
        "2026-06-15T02:00:00+02:00"
    )

    assert timestamp == pd.Timestamp(
        "2026-06-15T00:00:00Z"
    )


def test_invalid_prediction_date_is_rejected() -> None:
    with pytest.raises(
        Exception,
    ):
        normalize_prediction_date(
            "not-a-date"
        )


# ---------------------------------------------------------------------
# Protocol compatibility
# ---------------------------------------------------------------------


def test_static_provider_satisfies_provider_protocol(
    provider: StaticCsvTeamFeatureProvider,
) -> None:
    assert isinstance(
        provider,
        TeamFeatureProvider,
    )


# ---------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------


def test_missing_repository_file_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        StaticCsvTeamFeatureProvider(
            repository_path=(
                tmp_path / "missing.csv"
            ),
        )


def test_empty_repository_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.csv"

    pd.DataFrame().to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        (
            ValueError,
            pd.errors.EmptyDataError,
        ),
    ):
        StaticCsvTeamFeatureProvider(
            repository_path=path,
        )


def test_missing_required_columns_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_columns.csv"

    pd.DataFrame(
        [
            {
                "nation": "France",
                "att_composite": 0.8,
            }
        ]
    ).to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        StaticCsvTeamFeatureProvider(
            repository_path=path,
        )


def test_blank_provider_name_is_rejected(
    repository_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="provider_name",
    ):
        StaticCsvTeamFeatureProvider(
            repository_path=repository_path,
            provider_name="   ",
        )


def test_blank_representation_type_is_rejected(
    repository_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="representation_type",
    ):
        StaticCsvTeamFeatureProvider(
            repository_path=repository_path,
            representation_type="   ",
        )


def test_blank_aggregation_profile_is_rejected(
    repository_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="aggregation_profile",
    ):
        StaticCsvTeamFeatureProvider(
            repository_path=repository_path,
            aggregation_profile="   ",
        )


# ---------------------------------------------------------------------
# Repository validation
# ---------------------------------------------------------------------


def test_duplicate_normalized_team_names_are_rejected(
    tmp_path: Path,
) -> None:
    path = _write_repository(
        tmp_path / "duplicates.csv",
        [
            {
                "nation": "USA",
                "att_composite": 0.5,
                "mid_composite": 0.5,
                "def_composite": 0.5,
                "gk_composite": 0.5,
                "poisson_attack_adj": 1.0,
                "poisson_defense_adj": 1.0,
            },
            {
                "nation": "United States",
                "att_composite": 0.6,
                "mid_composite": 0.6,
                "def_composite": 0.6,
                "gk_composite": 0.6,
                "poisson_attack_adj": 1.1,
                "poisson_defense_adj": 0.9,
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicate normalized teams",
    ):
        StaticCsvTeamFeatureProvider(
            repository_path=path,
        )


@pytest.mark.parametrize(
    "column",
    [
        "att_composite",
        "mid_composite",
        "def_composite",
        "gk_composite",
        "poisson_attack_adj",
        "poisson_defense_adj",
    ],
)
def test_missing_numeric_values_are_rejected(
    tmp_path: Path,
    column: str,
) -> None:
    row = {
        "nation": "France",
        "att_composite": 0.81,
        "mid_composite": 0.77,
        "def_composite": 0.74,
        "gk_composite": 0.72,
        "poisson_attack_adj": 1.18,
        "poisson_defense_adj": 0.82,
    }

    row[column] = None

    path = _write_repository(
        tmp_path / f"missing_{column}.csv",
        [row],
    )

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        StaticCsvTeamFeatureProvider(
            repository_path=path,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_numeric_values_are_rejected(
    tmp_path: Path,
    bad_value: float,
) -> None:
    path = _write_repository(
        tmp_path / "non_finite.csv",
        [
            {
                "nation": "France",
                "att_composite": bad_value,
                "mid_composite": 0.77,
                "def_composite": 0.74,
                "gk_composite": 0.72,
                "poisson_attack_adj": 1.18,
                "poisson_defense_adj": 0.82,
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        StaticCsvTeamFeatureProvider(
            repository_path=path,
        )


def test_invalid_numeric_text_is_rejected(
    tmp_path: Path,
) -> None:
    path = _write_repository(
        tmp_path / "invalid_numeric.csv",
        [
            {
                "nation": "France",
                "att_composite": "not-a-number",
                "mid_composite": 0.77,
                "def_composite": 0.74,
                "gk_composite": 0.72,
                "poisson_attack_adj": 1.18,
                "poisson_defense_adj": 0.82,
            }
        ],
    )

    with pytest.raises(
        Exception,
    ):
        StaticCsvTeamFeatureProvider(
            repository_path=path,
        )


# ---------------------------------------------------------------------
# Exact feature loading
# ---------------------------------------------------------------------


def test_provider_returns_exact_repository_values(
    provider: StaticCsvTeamFeatureProvider,
) -> None:
    result = provider.get_team_features(
        TeamFeatureRequest(
            team_name="France",
            prediction_date="2026-06-15",
        )
    )

    assert result.attack == pytest.approx(
        0.81
    )

    assert result.midfield == pytest.approx(
        0.77
    )

    assert result.defense == pytest.approx(
        0.74
    )

    assert result.goalkeeper == pytest.approx(
        0.72
    )

    assert result.poisson_attack == pytest.approx(
        1.18
    )

    assert result.poisson_defense == pytest.approx(
        0.82
    )


def test_provider_returns_expected_metadata(
    provider: StaticCsvTeamFeatureProvider,
    repository_path: Path,
) -> None:
    result = provider.get_team_features(
        TeamFeatureRequest(
            team_name="France",
            prediction_date="2026-06-15",
        )
    )

    assert (
        result.requested_team_name
        == "France"
    )

    assert (
        result.canonical_team_name
        == "France"
    )

    assert (
        result.prediction_date
        == date(2026, 6, 15)
    )

    assert result.source_date is None

    assert (
        result.provider_name
        == "test_static_provider"
    )

    assert (
        result.representation_type
        == "static_repository"
    )

    assert (
        result.aggregation_profile
        == "legacy_static"
    )

    assert (
        result.repository_path
        == str(repository_path)
    )

    assert (
        result.repository_version
        == "test_v1"
    )

    assert (
        result.repository_scope
        == "national_teams"
    )

    assert result.feature_available is True

    assert (
        result.temporal_validity_pass
        is True
    )


def test_result_is_frozen(
    provider: StaticCsvTeamFeatureProvider,
) -> None:
    result = provider.get_team_features(
        TeamFeatureRequest(
            team_name="France",
            prediction_date="2026-06-15",
        )
    )

    with pytest.raises(
        (
            AttributeError,
            TypeError,
        ),
    ):
        result.attack = 1.5  # type: ignore[misc]


# ---------------------------------------------------------------------
# Team-name normalization
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "requested_name",
        "expected_canonical",
    ),
    [
        (
            "France",
            "France",
        ),
        (
            "  France  ",
            "France",
        ),
    ],
)
def test_team_name_is_normalized(
    provider: StaticCsvTeamFeatureProvider,
    requested_name: str,
    expected_canonical: str,
) -> None:
    result = provider.get_team_features(
        TeamFeatureRequest(
            team_name=requested_name,
            prediction_date="2026-06-15",
        )
    )

    assert (
        result.canonical_team_name
        == expected_canonical
    )


def test_blank_requested_team_name_is_rejected(
    provider: StaticCsvTeamFeatureProvider,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        provider.get_team_features(
            TeamFeatureRequest(
                team_name="   ",
                prediction_date="2026-06-15",
            )
        )


def test_missing_team_is_rejected(
    provider: StaticCsvTeamFeatureProvider,
) -> None:
    with pytest.raises(
        KeyError,
        match="No static team features exist",
    ):
        provider.get_team_features(
            TeamFeatureRequest(
                team_name="Spain",
                prediction_date="2026-06-15",
            )
        )


# ---------------------------------------------------------------------
# Temporal contract
# ---------------------------------------------------------------------


def test_source_date_is_preserved(
    repository_path: Path,
) -> None:
    provider = StaticCsvTeamFeatureProvider(
        repository_path=repository_path,
        source_date="2025-12-31",
    )

    result = provider.get_team_features(
        TeamFeatureRequest(
            team_name="France",
            prediction_date="2026-06-15",
        )
    )

    assert (
        result.source_date
        == date(2025, 12, 31)
    )


def test_source_date_equal_to_prediction_date_is_allowed(
    repository_path: Path,
) -> None:
    provider = StaticCsvTeamFeatureProvider(
        repository_path=repository_path,
        source_date="2026-06-15",
    )

    result = provider.get_team_features(
        TeamFeatureRequest(
            team_name="France",
            prediction_date="2026-06-15",
        )
    )

    assert result.temporal_validity_pass


def test_future_source_date_is_rejected(
    repository_path: Path,
) -> None:
    provider = StaticCsvTeamFeatureProvider(
        repository_path=repository_path,
        source_date="2026-06-16",
    )

    with pytest.raises(
        ValueError,
        match="after the requested prediction date",
    ):
        provider.get_team_features(
            TeamFeatureRequest(
                team_name="France",
                prediction_date="2026-06-15",
            )
        )


def test_static_provider_accepts_any_date_when_source_date_is_unknown(
    provider: StaticCsvTeamFeatureProvider,
) -> None:
    historical = provider.get_team_features(
        TeamFeatureRequest(
            team_name="France",
            prediction_date="2018-07-15",
        )
    )

    future = provider.get_team_features(
        TeamFeatureRequest(
            team_name="France",
            prediction_date="2030-07-15",
        )
    )

    assert (
        historical.attack
        == pytest.approx(
            future.attack
        )
    )

    assert historical.source_date is None
    assert future.source_date is None


# ---------------------------------------------------------------------
# Result validation
# ---------------------------------------------------------------------


def _valid_result(
    **overrides: object,
) -> TeamFeatureResult:
    values: dict[str, object] = {
        "requested_team_name": "France",
        "canonical_team_name": "France",
        "prediction_date": date(
            2026,
            6,
            15,
        ),
        "source_date": date(
            2026,
            1,
            1,
        ),
        "attack": 0.81,
        "midfield": 0.77,
        "defense": 0.74,
        "goalkeeper": 0.72,
        "poisson_attack": 1.18,
        "poisson_defense": 0.82,
        "provider_name": "test_provider",
        "representation_type": (
            "static_repository"
        ),
        "aggregation_profile": (
            "legacy_static"
        ),
        "feature_available": True,
        "temporal_validity_pass": True,
        "repository_path": None,
        "repository_version": None,
        "repository_scope": None,
    }

    values.update(overrides)

    return TeamFeatureResult(
        **values,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    (
        "field_name",
        "field_value",
        "message",
    ),
    [
        (
            "requested_team_name",
            "   ",
            "requested_team_name",
        ),
        (
            "canonical_team_name",
            "   ",
            "canonical_team_name",
        ),
        (
            "provider_name",
            "   ",
            "provider_name",
        ),
        (
            "representation_type",
            "   ",
            "representation_type",
        ),
        (
            "aggregation_profile",
            "   ",
            "aggregation_profile",
        ),
    ],
)
def test_result_rejects_blank_required_text(
    field_name: str,
    field_value: str,
    message: str,
) -> None:
    result = _valid_result(
        **{
            field_name: field_value,
        }
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        result.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "attack",
        "midfield",
        "defense",
        "goalkeeper",
        "poisson_attack",
        "poisson_defense",
    ],
)
def test_result_rejects_non_finite_values(
    field_name: str,
) -> None:
    result = _valid_result(
        **{
            field_name: float("inf"),
        }
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        result.validate()


def test_result_rejects_unavailable_snapshot() -> None:
    result = _valid_result(
        feature_available=False
    )

    with pytest.raises(
        ValueError,
        match="available feature snapshot",
    ):
        result.validate()


def test_result_rejects_failed_temporal_validation() -> None:
    result = _valid_result(
        temporal_validity_pass=False
    )

    with pytest.raises(
        ValueError,
        match="temporal validation",
    ):
        result.validate()


def test_result_rejects_source_date_after_prediction_date() -> None:
    result = _valid_result(
        prediction_date=date(
            2026,
            6,
            15,
        ),
        source_date=date(
            2026,
            6,
            16,
        ),
    )

    with pytest.raises(
        ValueError,
        match="source_date",
    ):
        result.validate()


def test_valid_result_passes_validation() -> None:
    result = _valid_result()

    result.validate()


# ---------------------------------------------------------------------
# Runtime repository conversion
# ---------------------------------------------------------------------


def test_to_repository_entry_returns_canonical_schema() -> None:
    result = _valid_result()

    entry = result.to_repository_entry()

    assert entry == {
        "attack": 0.81,
        "midfield": 0.77,
        "defense": 0.74,
        "gk": 0.72,
        "poisson_attack": 1.18,
        "poisson_defense": 0.82,
        "att_composite": 0.81,
        "mid_composite": 0.77,
        "def_composite": 0.74,
        "gk_composite": 0.72,
        "poisson_attack_adj": 1.18,
        "poisson_defense_adj": 0.82,
    }


def test_to_repository_entry_returns_new_dictionary_each_time() -> None:
    result = _valid_result()

    first = result.to_repository_entry()
    second = result.to_repository_entry()

    assert first == second
    assert first is not second


def test_repository_entry_mutation_does_not_change_result() -> None:
    result = _valid_result()

    entry = result.to_repository_entry()

    entry["attack"] = 999.0

    assert result.attack == pytest.approx(
        0.81
    )


def test_to_repository_entry_validates_result() -> None:
    result = _valid_result(
        attack=float("inf")
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        result.to_repository_entry()


# ---------------------------------------------------------------------
# Input immutability and caching behavior
# ---------------------------------------------------------------------


def test_source_file_is_not_modified_by_provider_lookup(
    repository_path: Path,
) -> None:
    original_bytes = repository_path.read_bytes()

    provider = StaticCsvTeamFeatureProvider(
        repository_path=repository_path,
    )

    provider.get_team_features(
        TeamFeatureRequest(
            team_name="France",
            prediction_date="2026-06-15",
        )
    )

    assert (
        repository_path.read_bytes()
        == original_bytes
    )


def test_provider_uses_loaded_repository_snapshot(
    repository_path: Path,
) -> None:
    provider = StaticCsvTeamFeatureProvider(
        repository_path=repository_path,
    )

    original = provider.get_team_features(
        TeamFeatureRequest(
            team_name="France",
            prediction_date="2026-06-15",
        )
    )

    replacement = pd.read_csv(
        repository_path
    )

    replacement.loc[
        replacement["nation"].eq(
            "France"
        ),
        "att_composite",
    ] = 99.0

    replacement.to_csv(
        repository_path,
        index=False,
    )

    second = provider.get_team_features(
        TeamFeatureRequest(
            team_name="France",
            prediction_date="2026-06-15",
        )
    )

    assert second.attack == pytest.approx(
        original.attack
    )

    assert second.attack != pytest.approx(
        99.0
    )
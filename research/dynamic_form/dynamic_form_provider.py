#dynamic_form_provider

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESIDUAL_HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_065_historical_residual_repository"
    / "team_residual_history.csv"
)

DEFAULT_WINDOW_SIZE = 8
DEFAULT_DECAY = 0.80
DEFAULT_SOURCE = "prequential_dynamic_form_v1"


@dataclass(frozen=True)
class DynamicFormRequest:
    """
    Request for a historically valid team-form estimate.
    """

    team_id: int
    team_name: str
    prediction_date: (
        str
        | date
        | datetime
        | pd.Timestamp
    )


@dataclass(frozen=True)
class DynamicFormResult:
    """
    Historically valid dynamic-form result.

    Positive attack form indicates recent scoring above
    Version 1 expectation.

    Positive defense form indicates recent conceding below
    Version 1 expectation.
    """

    requested_team_id: int
    requested_team_name: str
    resolved_team_id: int
    resolved_team_name: str

    prediction_date: date

    attack_form: float | None
    defense_form: float | None

    match_count: int
    confidence: float

    window_size: int
    decay: float
    source: str

    earliest_match_date: date | None
    latest_match_date: date | None

    form_available: bool
    temporal_validity_pass: bool


class DynamicFormProvider(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    def get_form(
        self,
        request: DynamicFormRequest,
    ) -> DynamicFormResult:
        ...


def normalize_prediction_date(
    value: (
        str
        | date
        | datetime
        | pd.Timestamp
    ),
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )
    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return timestamp


def exponential_weights(
    match_count: int,
    decay: float,
) -> np.ndarray:
    """
    Return normalized weights ordered from oldest to newest.

    The newest observation receives raw weight 1.0.
    """
    if match_count < 1:
        raise ValueError(
            "Match count must be positive."
        )

    if not 0.0 < decay <= 1.0:
        raise ValueError(
            "Decay must lie in the interval (0, 1]."
        )

    recency_positions = np.arange(
        match_count - 1,
        -1,
        -1,
        dtype=float,
    )

    raw_weights = decay ** recency_positions

    return (
        raw_weights
        / raw_weights.sum()
    )


def form_confidence(
    match_count: int,
    window_size: int,
    decay: float,
) -> float:
    """
    Measure available effective weight relative to a full
    form window.
    """
    if match_count <= 0:
        return 0.0

    effective_count = min(
        match_count,
        window_size,
    )

    available_weight = sum(
        decay ** position
        for position in range(
            effective_count
        )
    )

    full_weight = sum(
        decay ** position
        for position in range(
            window_size
        )
    )

    return float(
        available_weight / full_weight
    )


class HistoricalResidualDynamicFormProvider:
    """
    Dynamic-form provider backed by the validated Study 065
    team residual history.

    The provider never fits a model. It only aggregates
    already validated, historically generated residuals.
    """

    def __init__(
        self,
        residual_history_path: Path = (
            DEFAULT_RESIDUAL_HISTORY_PATH
        ),
        window_size: int = DEFAULT_WINDOW_SIZE,
        decay: float = DEFAULT_DECAY,
    ) -> None:
        if window_size < 1:
            raise ValueError(
                "Window size must be positive."
            )

        if not 0.0 < decay <= 1.0:
            raise ValueError(
                "Decay must lie in the interval (0, 1]."
            )

        self.residual_history_path = (
            residual_history_path
        )
        self.window_size = window_size
        self.decay = decay

        self._history = (
            self._load_history()
        )

    @property
    def provider_name(self) -> str:
        return DEFAULT_SOURCE

    def _load_history(self) -> pd.DataFrame:
        path = self.residual_history_path

        if not path.exists():
            raise FileNotFoundError(
                "Team residual history does not exist: "
                f"{path}"
            )

        dataframe = pd.read_csv(
            path,
            low_memory=False,
        )

        if dataframe.empty:
            raise ValueError(
                "Team residual history is empty."
            )

        required_columns = {
            "event_id",
            "date",
            "team_id",
            "team_name",
            "attack_residual",
            "defense_residual",
            "prediction_temporal_validity_pass",
        }

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "Team residual history is missing "
                f"columns: {sorted(missing_columns)}"
            )

        dataframe = dataframe.copy()

        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="raise",
            utc=True,
        )

        dataframe["team_id"] = pd.to_numeric(
            dataframe["team_id"],
            errors="raise",
        ).astype(int)

        for column in (
            "attack_residual",
            "defense_residual",
        ):
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise",
            )

        if dataframe[
            [
                "attack_residual",
                "defense_residual",
            ]
        ].isna().any().any():
            raise ValueError(
                "Team residual history contains missing "
                "residual values."
            )

        if not np.isfinite(
            dataframe[
                [
                    "attack_residual",
                    "defense_residual",
                ]
            ].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Team residual history contains "
                "non-finite residual values."
            )

        temporal_values = (
            dataframe[
                "prediction_temporal_validity_pass"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

        if temporal_values.isna().any():
            raise ValueError(
                "Residual history contains invalid "
                "temporal-validity values."
            )

        if not temporal_values.all():
            raise AssertionError(
                "Residual history contains temporally "
                "invalid records."
            )

        dataframe[
            "prediction_temporal_validity_pass"
        ] = temporal_values.astype(bool)

        if dataframe[
            [
                "event_id",
                "team_id",
            ]
        ].duplicated().any():
            raise ValueError(
                "Residual history contains duplicate "
                "event/team records."
            )

        return (
            dataframe
            .sort_values(
                [
                    "date",
                    "event_id",
                    "team_id",
                ]
            )
            .reset_index(drop=True)
        )

    def _resolve_team_history(
        self,
        request: DynamicFormRequest,
    ) -> pd.DataFrame:
        by_id = self._history[
            self._history["team_id"].eq(
                request.team_id
            )
        ]

        if by_id.empty:
            raise KeyError(
                "No residual history exists for team ID "
                f"{request.team_id}."
            )

        resolved_names = (
            by_id["team_name"]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(resolved_names) != 1:
            raise AssertionError(
                "One team ID maps to multiple names: "
                f"{resolved_names}"
            )

        resolved_name = resolved_names[0]

        if resolved_name != request.team_name:
            raise ValueError(
                "Requested team name does not match the "
                "residual repository identity. "
                f"Requested: {request.team_name!r}; "
                f"repository: {resolved_name!r}."
            )

        return by_id.copy()

    def get_form(
        self,
        request: DynamicFormRequest,
    ) -> DynamicFormResult:
        prediction_timestamp = (
            normalize_prediction_date(
                request.prediction_date
            )
        )

        team_history = (
            self._resolve_team_history(
                request
            )
        )

        eligible = team_history[
            team_history["date"].dt.date
            < prediction_timestamp.date()
        ].copy()

        eligible = (
            eligible
            .sort_values(
                [
                    "date",
                    "event_id",
                ]
            )
            .tail(self.window_size)
            .reset_index(drop=True)
        )

        if eligible.empty:
            return DynamicFormResult(
                requested_team_id=(
                    request.team_id
                ),
                requested_team_name=(
                    request.team_name
                ),
                resolved_team_id=(
                    request.team_id
                ),
                resolved_team_name=(
                    request.team_name
                ),
                prediction_date=(
                    prediction_timestamp.date()
                ),
                attack_form=None,
                defense_form=None,
                match_count=0,
                confidence=0.0,
                window_size=self.window_size,
                decay=self.decay,
                source=self.provider_name,
                earliest_match_date=None,
                latest_match_date=None,
                form_available=False,
                temporal_validity_pass=True,
            )

        weights = exponential_weights(
            match_count=len(eligible),
            decay=self.decay,
        )

        attack_form = float(
            np.average(
                eligible[
                    "attack_residual"
                ].to_numpy(dtype=float),
                weights=weights,
            )
        )

        defense_form = float(
            np.average(
                eligible[
                    "defense_residual"
                ].to_numpy(dtype=float),
                weights=weights,
            )
        )

        earliest_match_date = (
            eligible["date"]
            .min()
            .date()
        )

        latest_match_date = (
            eligible["date"]
            .max()
            .date()
        )

        temporal_validity_pass = (
            latest_match_date
            < prediction_timestamp.date()
        )

        if not temporal_validity_pass:
            raise AssertionError(
                "Dynamic Form included a same-day or "
                "future residual."
            )

        return DynamicFormResult(
            requested_team_id=request.team_id,
            requested_team_name=(
                request.team_name
            ),
            resolved_team_id=request.team_id,
            resolved_team_name=(
                request.team_name
            ),
            prediction_date=(
                prediction_timestamp.date()
            ),
            attack_form=attack_form,
            defense_form=defense_form,
            match_count=len(eligible),
            confidence=form_confidence(
                match_count=len(eligible),
                window_size=self.window_size,
                decay=self.decay,
            ),
            window_size=self.window_size,
            decay=self.decay,
            source=self.provider_name,
            earliest_match_date=(
                earliest_match_date
            ),
            latest_match_date=(
                latest_match_date
            ),
            form_available=True,
            temporal_validity_pass=True,
        )
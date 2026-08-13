#validate_dynamic_form_provider

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.dynamic_form.dynamic_form_provider import (
    DEFAULT_DECAY,
    DEFAULT_WINDOW_SIZE,
    DynamicFormRequest,
    HistoricalResidualDynamicFormProvider,
    exponential_weights,
    form_confidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESIDUAL_HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_065_historical_residual_repository"
    / "team_residual_history.csv"
)


def load_history() -> pd.DataFrame:
    dataframe = pd.read_csv(
        RESIDUAL_HISTORY_PATH,
        low_memory=False,
    )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    return dataframe


def select_validation_team(
    history: pd.DataFrame,
) -> tuple[int, str, pd.Timestamp]:
    team_counts = (
        history.groupby(
            [
                "team_id",
                "team_name",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    team_id, team_name = (
        team_counts.index[0]
    )

    team_history = history[
        history["team_id"].eq(
            team_id
        )
    ].sort_values("date")

    if len(team_history) < DEFAULT_WINDOW_SIZE:
        raise AssertionError(
            "Validation team does not contain a complete "
            "Dynamic Form window."
        )

    prediction_date = (
        team_history["date"].max()
        + pd.Timedelta(days=1)
    )

    return (
        int(team_id),
        str(team_name),
        prediction_date,
    )


def validate_weight_contract() -> None:
    weights = exponential_weights(
        match_count=DEFAULT_WINDOW_SIZE,
        decay=DEFAULT_DECAY,
    )

    if len(weights) != DEFAULT_WINDOW_SIZE:
        raise AssertionError(
            "Unexpected Dynamic Form weight count."
        )

    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Dynamic Form weights do not sum to one."
        )

    if not np.all(
        np.diff(weights) > 0.0
    ):
        raise AssertionError(
            "Dynamic Form weights do not increase "
            "toward the most recent match."
        )

    if not np.isclose(
        weights[-1] / weights[-2],
        1.0 / DEFAULT_DECAY,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Dynamic Form weights do not follow the "
            "configured exponential decay."
        )


def validate_confidence_contract() -> None:
    values = [
        form_confidence(
            match_count=count,
            window_size=DEFAULT_WINDOW_SIZE,
            decay=DEFAULT_DECAY,
        )
        for count in range(
            DEFAULT_WINDOW_SIZE + 1
        )
    ]

    if values[0] != 0.0:
        raise AssertionError(
            "Zero-match confidence must equal zero."
        )

    if not np.isclose(
        values[-1],
        1.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Full-window confidence must equal one."
        )

    if not np.all(
        np.diff(values) > 0.0
    ):
        raise AssertionError(
            "Dynamic Form confidence is not strictly "
            "increasing."
        )


def validate_manual_reproduction(
    provider: HistoricalResidualDynamicFormProvider,
    history: pd.DataFrame,
    team_id: int,
    team_name: str,
    prediction_date: pd.Timestamp,
) -> None:
    result = provider.get_form(
        DynamicFormRequest(
            team_id=team_id,
            team_name=team_name,
            prediction_date=prediction_date,
        )
    )

    eligible = (
        history[
            history["team_id"].eq(
                team_id
            )
            & (
                history["date"].dt.date
                < prediction_date.date()
            )
        ]
        .sort_values(
            [
                "date",
                "event_id",
            ]
        )
        .tail(DEFAULT_WINDOW_SIZE)
        .reset_index(drop=True)
    )

    weights = exponential_weights(
        match_count=len(eligible),
        decay=DEFAULT_DECAY,
    )

    expected_attack = float(
        np.average(
            eligible[
                "attack_residual"
            ].to_numpy(dtype=float),
            weights=weights,
        )
    )

    expected_defense = float(
        np.average(
            eligible[
                "defense_residual"
            ].to_numpy(dtype=float),
            weights=weights,
        )
    )

    if not np.isclose(
        result.attack_form,
        expected_attack,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Provider attack form does not reproduce "
            "the manual weighted average."
        )

    if not np.isclose(
        result.defense_form,
        expected_defense,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Provider defense form does not reproduce "
            "the manual weighted average."
        )

    if result.match_count != len(
        eligible
    ):
        raise AssertionError(
            "Provider returned an unexpected match count."
        )

    if result.match_count != (
        DEFAULT_WINDOW_SIZE
    ):
        raise AssertionError(
            "Provider did not enforce the eight-match "
            "history cap."
        )

    if not result.temporal_validity_pass:
        raise AssertionError(
            "Provider failed temporal validation."
        )

    if not (
        result.latest_match_date
        < result.prediction_date
    ):
        raise AssertionError(
            "Provider included a same-day or future "
            "residual."
        )


def validate_empty_history_result(
    provider: HistoricalResidualDynamicFormProvider,
    history: pd.DataFrame,
    team_id: int,
    team_name: str,
) -> None:
    first_team_date = (
        history.loc[
            history["team_id"].eq(
                team_id
            ),
            "date",
        ]
        .min()
    )

    result = provider.get_form(
        DynamicFormRequest(
            team_id=team_id,
            team_name=team_name,
            prediction_date=first_team_date,
        )
    )

    if result.form_available:
        raise AssertionError(
            "Form was unexpectedly available before the "
            "team's first residual match."
        )

    if result.match_count != 0:
        raise AssertionError(
            "Empty-history result has a nonzero count."
        )

    if result.confidence != 0.0:
        raise AssertionError(
            "Empty-history confidence is not zero."
        )

    if (
        result.attack_form is not None
        or result.defense_form is not None
    ):
        raise AssertionError(
            "Empty-history result contains fabricated "
            "form values."
        )


def main() -> None:
    history = load_history()

    provider = (
        HistoricalResidualDynamicFormProvider(
            residual_history_path=(
                RESIDUAL_HISTORY_PATH
            ),
            window_size=DEFAULT_WINDOW_SIZE,
            decay=DEFAULT_DECAY,
        )
    )

    (
        team_id,
        team_name,
        prediction_date,
    ) = select_validation_team(
        history
    )

    validate_weight_contract()
    validate_confidence_contract()

    validate_manual_reproduction(
        provider=provider,
        history=history,
        team_id=team_id,
        team_name=team_name,
        prediction_date=prediction_date,
    )

    validate_empty_history_result(
        provider=provider,
        history=history,
        team_id=team_id,
        team_name=team_name,
    )

    result = provider.get_form(
        DynamicFormRequest(
            team_id=team_id,
            team_name=team_name,
            prediction_date=prediction_date,
        )
    )

    print("Dynamic Form Provider Validation")
    print("=" * 76)
    print()
    print(
        f"Provider: {provider.provider_name}"
    )
    print(
        f"Team: {team_name} ({team_id})"
    )
    print(
        "Prediction date: "
        f"{result.prediction_date}"
    )
    print(
        "Residual window: "
        f"{result.match_count}/"
        f"{result.window_size}"
    )
    print(
        f"Decay: {result.decay:.2f}"
    )
    print(
        "Attack form: "
        f"{result.attack_form:.6f}"
    )
    print(
        "Defense form: "
        f"{result.defense_form:.6f}"
    )
    print(
        f"Confidence: {result.confidence:.6f}"
    )
    print(
        "Earliest included match: "
        f"{result.earliest_match_date}"
    )
    print(
        "Latest included match: "
        f"{result.latest_match_date}"
    )
    print()
    print("Residual repository loading: PASS")
    print("Team identity resolution: PASS")
    print("Eight-match history cap: PASS")
    print("Exponential weighting: PASS")
    print("Manual weighted-average reproduction: PASS")
    print("Confidence calculation: PASS")
    print("Strict prior-date filtering: PASS")
    print("Empty-history behavior: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
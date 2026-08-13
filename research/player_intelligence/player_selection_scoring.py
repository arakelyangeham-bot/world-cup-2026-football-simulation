#player_selection_scoring

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_START_RATE_COLUMN = "start_rate"
DEFAULT_MINUTES_COLUMN = "minutes_relative_to_club_max"

SELECTION_SCORE_COLUMN = "selection_score"
NORMALIZED_ROLE_RATING_COLUMN = "normalized_role_rating"
ROLE_COMPONENT_COLUMN = "role_rating_component"
START_COMPONENT_COLUMN = "start_rate_component"
MINUTES_COMPONENT_COLUMN = "minutes_component"


@dataclass(frozen=True)
class PlayerSelectionSpecification:
    """
    Immutable weighting specification for player selection.

    The specification combines:

    - role-specific football quality;
    - season-level start-rate evidence;
    - season-level relative-minutes evidence.

    The weights must be finite, non-negative, and sum to one.
    """

    name: str

    role_rating_weight: float
    start_rate_weight: float
    minutes_weight: float

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Selection specification name must not be empty."
            )

        weights = {
            "role_rating_weight": self.role_rating_weight,
            "start_rate_weight": self.start_rate_weight,
            "minutes_weight": self.minutes_weight,
        }

        for field_name, value in weights.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"{field_name} must be finite. "
                    f"Received {value!r}."
                )

            if value < 0.0:
                raise ValueError(
                    f"{field_name} must not be negative. "
                    f"Received {value!r}."
                )

        weight_sum = sum(weights.values())

        if not math.isclose(
            weight_sum,
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Selection weights must sum to one. "
                f"Received {weight_sum:.12f}."
            )

    def to_record(self) -> dict[str, object]:
        self.validate()

        return {
            "specification": self.name,
            "role_rating_weight": self.role_rating_weight,
            "start_rate_weight": self.start_rate_weight,
            "minutes_weight": self.minutes_weight,
        }


RATING_ONLY_SPECIFICATION = PlayerSelectionSpecification(
    name="rating_only",
    role_rating_weight=1.0,
    start_rate_weight=0.0,
    minutes_weight=0.0,
)


def validate_candidate_columns(
    candidates: pd.DataFrame,
    *,
    role_rating_column: str,
    start_rate_column: str = DEFAULT_START_RATE_COLUMN,
    minutes_column: str = DEFAULT_MINUTES_COLUMN,
) -> None:
    """
    Validate the candidate input contract.

    The function does not require player identity fields because
    scoring is intentionally independent of identity resolution.
    """

    required_columns = {
        role_rating_column,
        start_rate_column,
        minutes_column,
    }

    missing = required_columns - set(candidates.columns)

    if missing:
        raise ValueError(
            "Candidate dataset is missing required selection "
            f"columns: {sorted(missing)}"
        )

    if candidates.empty:
        raise ValueError(
            "Cannot score an empty candidate dataset."
        )


def _coerce_numeric_column(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    return values.astype(float)


def _validate_bounded_values(
    values: pd.Series,
    *,
    column_name: str,
) -> None:
    non_missing = values.dropna()

    if non_missing.empty:
        raise ValueError(
            f"Selection input {column_name!r} contains no "
            "usable values."
        )

    numeric_values = non_missing.to_numpy(
        dtype=float
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            f"Selection input {column_name!r} contains "
            "non-finite values."
        )

    if (
        non_missing.lt(0.0).any()
        or non_missing.gt(1.0).any()
    ):
        raise ValueError(
            f"Selection input {column_name!r} must lie "
            "between zero and one."
        )


def normalize_role_ratings(
    values: pd.Series,
) -> pd.Series:
    """
    Min-max normalize one role-rating candidate population.

    Normalization occurs within the candidate pool for one club
    and one formation role.

    If all valid candidates have the same role rating, every valid
    candidate receives a normalized value of 1.0. In that case,
    role quality does not distinguish candidates and usage evidence
    may break the tie.

    Missing role ratings remain missing.
    """

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    valid = numeric.notna()

    if not valid.any():
        raise ValueError(
            "No candidate has a valid role rating."
        )

    valid_values = numeric.loc[
        valid
    ]

    array = valid_values.to_numpy(
        dtype=float
    )

    if not np.isfinite(array).all():
        raise ValueError(
            "Role-rating candidates contain non-finite values."
        )

    minimum = float(
        valid_values.min()
    )

    maximum = float(
        valid_values.max()
    )

    normalized = pd.Series(
        np.nan,
        index=numeric.index,
        dtype=float,
    )

    if math.isclose(
        maximum,
        minimum,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        normalized.loc[valid] = 1.0

    else:
        normalized.loc[valid] = (
            valid_values
            - minimum
        ) / (
            maximum
            - minimum
        )

    return normalized


def score_candidates(
    candidates: pd.DataFrame,
    *,
    role_rating_column: str,
    specification: PlayerSelectionSpecification,
    start_rate_column: str = DEFAULT_START_RATE_COLUMN,
    minutes_column: str = DEFAULT_MINUTES_COLUMN,
) -> pd.DataFrame:
    """
    Score candidates for one club and one formation role.

    Parameters
    ----------
    candidates:
        Candidate rows for one role-selection decision.

    role_rating_column:
        Role-specific rating column, such as ``rating_ST`` or
        ``rating_CB``.

    specification:
        Immutable selection-weight specification.

    start_rate_column:
        Bounded season-level start-rate feature.

    minutes_column:
        Bounded minutes-relative-to-club-maximum feature.

    Returns
    -------
    pandas.DataFrame
        A copy of the candidate population with transparent
        scoring components and a final selection score.

    Notes
    -----
    Candidates lacking the requested role rating remain in the
    returned dataframe but receive no selection score.
    """

    specification.validate()

    validate_candidate_columns(
        candidates,
        role_rating_column=role_rating_column,
        start_rate_column=start_rate_column,
        minutes_column=minutes_column,
    )

    output = candidates.copy()

    output[role_rating_column] = (
        _coerce_numeric_column(
            output,
            role_rating_column,
        )
    )

    output[start_rate_column] = (
        _coerce_numeric_column(
            output,
            start_rate_column,
        )
    )

    output[minutes_column] = (
        _coerce_numeric_column(
            output,
            minutes_column,
        )
    )

    _validate_bounded_values(
        output[start_rate_column],
        column_name=start_rate_column,
    )

    _validate_bounded_values(
        output[minutes_column],
        column_name=minutes_column,
    )

    output[
        NORMALIZED_ROLE_RATING_COLUMN
    ] = normalize_role_ratings(
        output[role_rating_column]
    )

    role_available = output[
        NORMALIZED_ROLE_RATING_COLUMN
    ].notna()

    output[ROLE_COMPONENT_COLUMN] = np.where(
        role_available,
        output[NORMALIZED_ROLE_RATING_COLUMN]
        * specification.role_rating_weight,
        np.nan,
    )

    output[START_COMPONENT_COLUMN] = np.where(
        role_available,
        output[start_rate_column]
        * specification.start_rate_weight,
        np.nan,
    )

    output[MINUTES_COMPONENT_COLUMN] = np.where(
        role_available,
        output[minutes_column]
        * specification.minutes_weight,
        np.nan,
    )

    output[SELECTION_SCORE_COLUMN] = np.where(
        role_available,
        (
            output[ROLE_COMPONENT_COLUMN]
            + output[START_COMPONENT_COLUMN]
            + output[MINUTES_COMPONENT_COLUMN]
        ),
        np.nan,
    )

    valid_scores = output.loc[
        role_available,
        SELECTION_SCORE_COLUMN,
    ]

    if valid_scores.isna().any():
        raise ValueError(
            "Selection scoring produced missing values for "
            "role-eligible candidates."
        )

    if not np.isfinite(
        valid_scores.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Selection scoring produced non-finite values."
        )

    if (
        valid_scores.lt(0.0).any()
        or valid_scores.gt(1.0).any()
    ):
        raise AssertionError(
            "Selection scores must lie between zero and one."
        )

    return output


def rank_candidates(
    candidates: pd.DataFrame,
    *,
    role_rating_column: str,
    specification: PlayerSelectionSpecification,
    player_id_column: str = "player_id",
    start_rate_column: str = DEFAULT_START_RATE_COLUMN,
    minutes_column: str = DEFAULT_MINUTES_COLUMN,
) -> pd.DataFrame:
    """
    Score and deterministically rank role candidates.

    Tie-breaking order:

    1. final selection score;
    2. raw role rating;
    3. start rate;
    4. relative minutes;
    5. stable player ID.

    Higher football and usage values rank first. Player ID provides
    deterministic ordering only and contains no football meaning.
    """

    if player_id_column not in candidates.columns:
        raise ValueError(
            "Candidate dataset is missing deterministic "
            f"identity column {player_id_column!r}."
        )

    scored = score_candidates(
        candidates,
        role_rating_column=role_rating_column,
        specification=specification,
        start_rate_column=start_rate_column,
        minutes_column=minutes_column,
    )

    eligible = scored.loc[
        scored[
            SELECTION_SCORE_COLUMN
        ].notna()
    ].copy()

    if eligible.empty:
        raise ValueError(
            f"No candidates are eligible for role-rating "
            f"column {role_rating_column!r}."
        )

    eligible[player_id_column] = (
        eligible[player_id_column]
        .astype(str)
        .str.strip()
    )

    if eligible[
        player_id_column
    ].eq("").any():
        raise ValueError(
            "One or more candidate player IDs are empty."
        )

    eligible = (
        eligible
        .sort_values(
            [
                SELECTION_SCORE_COLUMN,
                role_rating_column,
                start_rate_column,
                minutes_column,
                player_id_column,
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    eligible.insert(
        0,
        "selection_rank",
        np.arange(
            1,
            len(eligible) + 1,
        ),
    )

    eligible.insert(
        1,
        "selection_specification",
        specification.name,
    )

    eligible.insert(
        2,
        "selection_role_rating_column",
        role_rating_column,
    )

    return eligible


def generate_weight_grid(
    *,
    role_weights: tuple[float, ...],
    start_weights: tuple[float, ...],
    minimum_minutes_weight: float = 0.0,
) -> tuple[PlayerSelectionSpecification, ...]:
    """
    Generate valid role/start/minutes specifications.

    Minutes weight is the remaining simplex mass:

    ``1 - role_weight - start_weight``

    Invalid or negative combinations are omitted.
    """

    if minimum_minutes_weight < 0.0:
        raise ValueError(
            "minimum_minutes_weight must not be negative."
        )

    specifications: list[
        PlayerSelectionSpecification
    ] = []

    seen: set[
        tuple[float, float, float]
    ] = set()

    for role_weight in role_weights:
        for start_weight in start_weights:
            minutes_weight = (
                1.0
                - role_weight
                - start_weight
            )

            if (
                minutes_weight
                < minimum_minutes_weight
                - 1e-12
            ):
                continue

            role_weight = float(
                role_weight
            )

            start_weight = float(
                start_weight
            )

            minutes_weight = float(
                max(
                    0.0,
                    minutes_weight,
                )
            )

            key = (
                round(
                    role_weight,
                    12,
                ),
                round(
                    start_weight,
                    12,
                ),
                round(
                    minutes_weight,
                    12,
                ),
            )

            if key in seen:
                continue

            specification = (
                PlayerSelectionSpecification(
                    name=(
                        f"role_{role_weight:.2f}"
                        f"_start_{start_weight:.2f}"
                        f"_minutes_{minutes_weight:.2f}"
                    ),
                    role_rating_weight=role_weight,
                    start_rate_weight=start_weight,
                    minutes_weight=minutes_weight,
                )
            )

            specification.validate()

            seen.add(key)

            specifications.append(
                specification
            )

    return tuple(
        specifications
    )
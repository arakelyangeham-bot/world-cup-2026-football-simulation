from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shared.team_name_normalizer import (
    normalize_team_name,
)


REQUIRED_COLUMNS = {
    "club",
    "att_composite",
    "mid_composite",
    "def_composite",
    "gk_composite",
    "poisson_attack_adj",
    "poisson_defense_adj",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
    "squad_quality",
    "evidence_score",
    "opta_rating",
    "rating_prior",
}


CANONICAL_NUMERIC_COLUMNS = {
    "att_composite",
    "mid_composite",
    "def_composite",
    "gk_composite",
    "poisson_attack_adj",
    "poisson_defense_adj",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
    "squad_quality",
    "evidence_score",
    "opta_rating",
    "rating_prior",
}


def load_club_team_repository(
    path: Path,
) -> dict[str, dict]:
    """
    Load a club-team repository into the canonical runtime
    football-model schema.

    The loader performs no football calculations. It exposes
    the validated team representation already stored in the
    repository.

    Canonical public schema:
        attack
        midfield
        defense
        gk
        attack_depth
        midfield_depth
        defense_depth
        squad_quality
        evidence_score
        poisson_attack
        poisson_defense
        rating_prior

    External-rating metadata:
        opta_rating
        rating_prior_source

    Temporary aliases remain available for backward
    compatibility.
    """

    repository_path = Path(path)

    if not repository_path.exists():
        raise FileNotFoundError(
            "Club team repository does not exist: "
            f"{repository_path}"
        )

    dataframe = pd.read_csv(
        repository_path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Club team repository is empty: "
            f"{repository_path}"
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required club repository columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    for column in CANONICAL_NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    if dataframe[
        list(CANONICAL_NUMERIC_COLUMNS)
    ].isna().any().any():
        raise ValueError(
            "Club repository contains missing required "
            "numeric values."
        )

    numeric_values = dataframe[
        list(CANONICAL_NUMERIC_COLUMNS)
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Club repository contains non-finite required "
            "numeric values."
        )

    repository: dict[str, dict] = {}

    for _, row in dataframe.iterrows():
        raw_club = str(row["club"]).strip()

        if not raw_club:
            raise ValueError(
                "Club name cannot be empty."
            )

        club = normalize_team_name(
            raw_club
        )

        if not club:
            raise ValueError(
                "Club name became empty after "
                "normalization."
            )

        if club in repository:
            raise ValueError(
                "Duplicate club in repository after "
                f"normalization: {club}"
            )

        attack = float(
            row["att_composite"]
        )

        midfield = float(
            row["mid_composite"]
        )

        defense = float(
            row["def_composite"]
        )

        goalkeeper = float(
            row["gk_composite"]
        )

        poisson_attack = float(
            row["poisson_attack_adj"]
        )

        poisson_defense = float(
            row["poisson_defense_adj"]
        )

        attack_depth = float(
            row["attack_depth"]
        )

        midfield_depth = float(
            row["midfield_depth"]
        )

        defense_depth = float(
            row["defense_depth"]
        )

        squad_quality = float(
            row["squad_quality"]
        )

        evidence_score = float(
            row["evidence_score"]
        )

        rating_prior = float(
            row["rating_prior"]
        )

        opta_rating = float(
            row["opta_rating"]
        )

        repository[club] = {
            # Canonical strength dimensions
            "attack": attack,
            "midfield": midfield,
            "defense": defense,
            "gk": goalkeeper,

            # Canonical depth and reliability dimensions
            "attack_depth": attack_depth,
            "midfield_depth": midfield_depth,
            "defense_depth": defense_depth,
            "squad_quality": squad_quality,
            "evidence_score": evidence_score,

            # Model-facing Poisson projections
            "poisson_attack": poisson_attack,
            "poisson_defense": poisson_defense,

            # External strength prior
            "rating_prior": rating_prior,

            # Source metadata
            "opta_rating": opta_rating,
            "rating_prior_source":
                "opta_power_rating",

            # Temporary compatibility aliases
            "att_composite": attack,
            "mid_composite": midfield,
            "def_composite": defense,
            "gk_composite": goalkeeper,
            "poisson_attack_adj":
                poisson_attack,
            "poisson_defense_adj":
                poisson_defense,
        }

    return repository

LEGACY_CLUB_REPOSITORY_COLUMNS = {
    "club",
    "att_composite",
    "mid_composite",
    "def_composite",
    "gk_composite",
    "poisson_attack_adj",
    "poisson_defense_adj",
    "opta_rating",
    "rating_prior",
}


def load_legacy_club_team_repository(
    path: Path,
) -> dict[str, dict]:
    """
    Load the historical Premier League validation repository.

    This compatibility loader preserves the original validation
    artifact contract. It must not infer or fabricate newer
    football-intelligence fields such as positional depth,
    squad quality, or evidence score.
    """

    dataframe = pd.read_csv(
        path
    )

    if dataframe.empty:
        raise ValueError(
            "Legacy club repository is empty."
        )

    missing_columns = (
        LEGACY_CLUB_REPOSITORY_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required legacy club repository "
            f"columns: {sorted(missing_columns)}"
        )

    repository: dict[
        str,
        dict,
    ] = {}

    numeric_columns = (
        "att_composite",
        "mid_composite",
        "def_composite",
        "gk_composite",
        "poisson_attack_adj",
        "poisson_defense_adj",
        "opta_rating",
        "rating_prior",
    )

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

        if dataframe[column].isna().any():
            raise ValueError(
                "Legacy club repository contains "
                f"missing values in {column!r}."
            )

    for _, row in dataframe.iterrows():
        club = normalize_team_name(
            str(row["club"])
        )

        if not club:
            raise ValueError(
                "Legacy club name cannot be empty."
            )

        if club in repository:
            raise ValueError(
                "Duplicate club in legacy repository: "
                f"{club}"
            )

        attack = float(
            row["att_composite"]
        )

        midfield = float(
            row["mid_composite"]
        )

        defense = float(
            row["def_composite"]
        )

        goalkeeper = float(
            row["gk_composite"]
        )

        poisson_attack = float(
            row["poisson_attack_adj"]
        )

        poisson_defense = float(
            row["poisson_defense_adj"]
        )

        rating_prior = float(
            row["rating_prior"]
        )

        repository[club] = {
            # Canonical runtime schema
            "team": club,
            "attack": attack,
            "midfield": midfield,
            "defense": defense,
            "gk": goalkeeper,
            "poisson_attack":
                poisson_attack,
            "poisson_defense":
                poisson_defense,
            "rating_prior":
                rating_prior,

            # Source metadata
            "opta_rating": float(
                row["opta_rating"]
            ),
            "rating_prior_source":
                "opta_power_rating",

            # Compatibility aliases
            "att_composite": attack,
            "mid_composite": midfield,
            "def_composite": defense,
            "gk_composite": goalkeeper,
            "poisson_attack_adj":
                poisson_attack,
            "poisson_defense_adj":
                poisson_defense,
        }

    return repository
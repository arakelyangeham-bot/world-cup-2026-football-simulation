#build_observation_dataset

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from research.football_observatory.representation_provider import (
    RepresentationProvider,
    RepresentationRequest,
)


REQUIRED_MATCH_COLUMNS = {
    "competition_key",
    "season_start_year",
    "event_id",
    "date",
    "home_team",
    "home_team_id",
    "away_team",
    "away_team_id",
    "home_score",
    "away_score",
    "outcome",
}


REPRESENTATION_FIELDS = (
    "attack",
    "midfield",
    "defense",
    "goalkeeper",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
    "squad_quality",
    "evidence_score",
)


@dataclass(frozen=True)
class ObservationDatasetResult:
    observations: pd.DataFrame
    exclusions: pd.DataFrame


def load_historical_matches(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Historical match dataset does not exist: {path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Historical match dataset is empty: {path}"
        )

    missing = REQUIRED_MATCH_COLUMNS - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Historical match dataset is missing required "
            f"columns: {sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe["season_start_year"] = pd.to_numeric(
        dataframe["season_start_year"],
        errors="raise",
    ).astype(int)

    dataframe["home_team_id"] = pd.to_numeric(
        dataframe["home_team_id"],
        errors="raise",
    ).astype(int)

    dataframe["away_team_id"] = pd.to_numeric(
        dataframe["away_team_id"],
        errors="raise",
    ).astype(int)

    dataframe["home_score"] = pd.to_numeric(
        dataframe["home_score"],
        errors="raise",
    ).astype(int)

    dataframe["away_score"] = pd.to_numeric(
        dataframe["away_score"],
        errors="raise",
    ).astype(int)

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
        utc=True,
    )

    if dataframe["event_id"].duplicated().any():
        duplicate_ids = (
            dataframe.loc[
                dataframe["event_id"].duplicated(
                    keep=False
                ),
                "event_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Historical match dataset contains duplicate "
            f"event IDs: {duplicate_ids[:20]}"
        )

    return dataframe.sort_values(
        ["date", "event_id"]
    ).reset_index(drop=True)


def representation_to_columns(
    prefix: str,
    result,
) -> dict[str, object]:
    representation = result.representation

    columns: dict[str, object] = {
        f"{prefix}_representation_type": (
            result.representation_type
        ),
        f"{prefix}_formation": result.formation,
        f"{prefix}_representation_competition_id": (
            result.competition_id
        ),
        f"{prefix}_representation_season_id": (
            result.season_id
        ),
        f"{prefix}_representation_source": (
            result.source
        ),
        f"{prefix}_temporal_validity_pass": (
            result.temporal_validity_pass
        ),
        f"{prefix}_representation_player_count": (
            representation.player_count
        ),
        f"{prefix}_representation_available_player_count": (
            representation.available_player_count
        ),
    }

    for field in REPRESENTATION_FIELDS:
        columns[
            f"{prefix}_{field}"
        ] = float(
            getattr(
                representation,
                field,
            )
        )

    return columns


def build_feature_differences(
    row: dict[str, object],
) -> None:
    for field in REPRESENTATION_FIELDS:
        home_column = f"home_{field}"
        away_column = f"away_{field}"

        row[f"{field}_diff"] = (
            float(row[home_column])
            - float(row[away_column])
        )


def build_target_columns(
    home_score: int,
    away_score: int,
) -> dict[str, object]:
    total_goals = (
        home_score
        + away_score
    )

    goal_difference = (
        home_score
        - away_score
    )

    return {
        "home_score": home_score,
        "away_score": away_score,
        "total_goals": total_goals,
        "goal_difference": goal_difference,
        "result": (
            "home_win"
            if home_score > away_score
            else "away_win"
            if away_score > home_score
            else "draw"
        ),
        "is_draw": int(
            home_score == away_score
        ),
        "is_home_win": int(
            home_score > away_score
        ),
        "is_away_win": int(
            away_score > home_score
        ),
        "both_teams_scored": int(
            home_score > 0
            and away_score > 0
        ),
        "is_clean_sheet": int(
            home_score == 0
            or away_score == 0
        ),
        "is_high_scoring": int(
            total_goals >= 5
        ),
        "is_blowout": int(
            abs(goal_difference) >= 3
        ),
    }


def build_observation_dataset(
    matches: pd.DataFrame,
    representation_provider: RepresentationProvider,
) -> ObservationDatasetResult:
    observations: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []

    for match in matches.itertuples(
        index=False
    ):
        prediction_year = int(
            match.season_start_year
        )

        representation_year = (
            prediction_year - 1
        )

        home_request = RepresentationRequest(
            competition_key=str(
                match.competition_key
            ),
            prediction_season_start_year=(
                prediction_year
            ),
            representation_season_start_year=(
                representation_year
            ),
            team_id=int(
                match.home_team_id
            ),
            team_name=str(
                match.home_team
            ),
        )

        away_request = RepresentationRequest(
            competition_key=str(
                match.competition_key
            ),
            prediction_season_start_year=(
                prediction_year
            ),
            representation_season_start_year=(
                representation_year
            ),
            team_id=int(
                match.away_team_id
            ),
            team_name=str(
                match.away_team
            ),
        )

        try:
            home_result = (
                representation_provider
                .get_representation(
                    home_request
                )
            )

            away_result = (
                representation_provider
                .get_representation(
                    away_request
                )
            )

        except (
            KeyError,
            ValueError,
        ) as exc:
            exclusions.append(
                {
                    "competition_key": (
                        match.competition_key
                    ),
                    "prediction_season_start_year": (
                        prediction_year
                    ),
                    "representation_season_start_year": (
                        representation_year
                    ),
                    "event_id": match.event_id,
                    "date": match.date,
                    "home_team": match.home_team,
                    "home_team_id": (
                        match.home_team_id
                    ),
                    "away_team": match.away_team,
                    "away_team_id": (
                        match.away_team_id
                    ),
                    "provider_name": (
                        representation_provider
                        .provider_name
                    ),
                    "exclusion_reason": str(exc),
                }
            )

            continue

        if not (
            home_result.temporal_validity_pass
            and away_result.temporal_validity_pass
        ):
            exclusions.append(
                {
                    "competition_key": (
                        match.competition_key
                    ),
                    "prediction_season_start_year": (
                        prediction_year
                    ),
                    "representation_season_start_year": (
                        representation_year
                    ),
                    "event_id": match.event_id,
                    "date": match.date,
                    "home_team": match.home_team,
                    "home_team_id": (
                        match.home_team_id
                    ),
                    "away_team": match.away_team,
                    "away_team_id": (
                        match.away_team_id
                    ),
                    "provider_name": (
                        representation_provider
                        .provider_name
                    ),
                    "exclusion_reason": (
                        "representation_provider_failed_"
                        "temporal_validity"
                    ),
                }
            )

            continue

        observation: dict[
            str,
            object,
        ] = {
            "competition_key": (
                match.competition_key
            ),
            "prediction_season_start_year": (
                prediction_year
            ),
            "representation_season_start_year": (
                representation_year
            ),
            "event_id": match.event_id,
            "date": match.date,
            "round": getattr(
                match,
                "round",
                None,
            ),
            "round_number": getattr(
                match,
                "round_number",
                None,
            ),
            "home_team": match.home_team,
            "home_team_id": (
                match.home_team_id
            ),
            "away_team": match.away_team,
            "away_team_id": (
                match.away_team_id
            ),
            "representation_provider": (
                representation_provider
                .provider_name
            ),
            "rating_prior_available": False,
            "rating_prior_source": (
                "unavailable"
            ),
            "home_rating_prior": float(
                "nan"
            ),
            "away_rating_prior": float(
                "nan"
            ),
            "rating_prior_diff": float(
                "nan"
            ),
        }

        observation.update(
            representation_to_columns(
                prefix="home",
                result=home_result,
            )
        )

        observation.update(
            representation_to_columns(
                prefix="away",
                result=away_result,
            )
        )

        build_feature_differences(
            observation
        )

        observation.update(
            build_target_columns(
                home_score=int(
                    match.home_score
                ),
                away_score=int(
                    match.away_score
                ),
            )
        )

        observations.append(
            observation
        )

    observation_dataframe = pd.DataFrame(
        observations
    )

    exclusion_columns = [
        "competition_key",
        "prediction_season_start_year",
        "representation_season_start_year",
        "event_id",
        "date",
        "home_team",
        "home_team_id",
        "away_team",
        "away_team_id",
        "provider_name",
        "exclusion_reason",
    ]

    exclusion_dataframe = pd.DataFrame(
        exclusions,
        columns=exclusion_columns,
    )

    if not observation_dataframe.empty:
        observation_dataframe = (
            observation_dataframe
            .sort_values(
                [
                    "date",
                    "event_id",
                ]
            )
            .reset_index(drop=True)
        )

    if not exclusion_dataframe.empty:
        exclusion_dataframe = (
            exclusion_dataframe
            .sort_values(
                [
                    "date",
                    "event_id",
                ]
            )
            .reset_index(drop=True)
        )

    return ObservationDatasetResult(
        observations=(
            observation_dataframe
        ),
        exclusions=(
            exclusion_dataframe
        ),
    )
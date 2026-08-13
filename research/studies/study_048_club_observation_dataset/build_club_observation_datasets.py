#build_club_observation_datasets

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.football_observatory.build_observation_dataset import (
    build_observation_dataset,
    load_historical_matches,
)
from research.player_intelligence.previous_season_competition_representation_provider import (
    PreviousSeasonCompetitionRepresentationProvider,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MATCH_INPUT = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "historical_matches"
    / "premier_league"
    / "premier_league_2024_completed_matches.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_048_club_observation_dataset"
)


def summarize_dataset(
    label: str,
    observations: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> dict[str, object]:
    accepted_count = len(
        observations
    )
    excluded_count = len(
        exclusions
    )

    total_count = (
        accepted_count
        + excluded_count
    )

    return {
        "representation_type": label,
        "total_match_count": total_count,
        "accepted_match_count": (
            accepted_count
        ),
        "excluded_match_count": (
            excluded_count
        ),
        "accepted_rate": (
            accepted_count / total_count
            if total_count
            else 0.0
        ),
        "unique_home_teams": (
            int(
                observations[
                    "home_team_id"
                ].nunique()
            )
            if not observations.empty
            else 0
        ),
        "unique_away_teams": (
            int(
                observations[
                    "away_team_id"
                ].nunique()
            )
            if not observations.empty
            else 0
        ),
    }


def validate_observations(
    observations: pd.DataFrame,
    representation_type: str,
) -> None:
    if observations.empty:
        raise AssertionError(
            f"{representation_type}: no observations "
            "were accepted."
        )

    if observations[
        "event_id"
    ].duplicated().any():
        raise AssertionError(
            f"{representation_type}: duplicate event IDs."
        )

    if not (
        observations[
            "representation_season_start_year"
        ]
        == (
            observations[
                "prediction_season_start_year"
            ]
            - 1
        )
    ).all():
        raise AssertionError(
            f"{representation_type}: invalid "
            "representation season relationship."
        )

    if not observations[
        "home_temporal_validity_pass"
    ].all():
        raise AssertionError(
            f"{representation_type}: home temporal "
            "validity failure."
        )

    if not observations[
        "away_temporal_validity_pass"
    ].all():
        raise AssertionError(
            f"{representation_type}: away temporal "
            "validity failure."
        )

    required_numeric = [
        "home_attack",
        "home_midfield",
        "home_defense",
        "home_goalkeeper",
        "away_attack",
        "away_midfield",
        "away_defense",
        "away_goalkeeper",
        "attack_diff",
        "midfield_diff",
        "defense_diff",
        "goalkeeper_diff",
        "home_score",
        "away_score",
    ]

    if observations[
        required_numeric
    ].isna().any().any():
        raise AssertionError(
            f"{representation_type}: required "
            "representation or target values are missing."
        )

    if observations[
        "rating_prior_available"
    ].any():
        raise AssertionError(
            f"{representation_type}: rating prior was "
            "unexpectedly marked available."
        )

    if observations[
        [
            "home_rating_prior",
            "away_rating_prior",
            "rating_prior_diff",
        ]
    ].notna().any().any():
        raise AssertionError(
            f"{representation_type}: future or "
            "unvalidated rating prior entered the dataset."
        )


def main() -> None:
    matches = load_historical_matches(
        MATCH_INPUT
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[
        dict[str, object]
    ] = []

    provider_configs = [
        {
            "label": "full_squad",
            "provider": (
                PreviousSeasonCompetitionRepresentationProvider(
                    representation_type=(
                        "full_squad"
                    ),
                )
            ),
        },
        {
            "label": (
                "expected_starting_xi"
            ),
            "provider": (
                PreviousSeasonCompetitionRepresentationProvider(
                    representation_type=(
                        "expected_starting_xi"
                    ),
                    formation="4-3-3",
                )
            ),
        },
    ]

    results: dict[str, object] = {}

    for config in provider_configs:
        label = str(
            config["label"]
        )

        provider = config[
            "provider"
        ]

        result = (
            build_observation_dataset(
                matches=matches,
                representation_provider=(
                    provider
                ),
            )
        )

        validate_observations(
            observations=(
                result.observations
            ),
            representation_type=label,
        )

        observation_path = (
            OUTPUT_DIR
            / f"{label}_observations.csv"
        )

        exclusion_path = (
            OUTPUT_DIR
            / f"{label}_exclusions.csv"
        )

        result.observations.to_csv(
            observation_path,
            index=False,
        )

        result.exclusions.to_csv(
            exclusion_path,
            index=False,
        )

        summaries.append(
            summarize_dataset(
                label=label,
                observations=(
                    result.observations
                ),
                exclusions=(
                    result.exclusions
                ),
            )
        )

        results[label] = result

    summary = pd.DataFrame(
        summaries
    )

    summary.to_csv(
        OUTPUT_DIR
        / "observation_dataset_summary.csv",
        index=False,
    )

    full_squad_events = set(
        results[
            "full_squad"
        ].observations["event_id"]
    )

    starting_xi_events = set(
        results[
            "expected_starting_xi"
        ].observations["event_id"]
    )

    if full_squad_events != starting_xi_events:
        only_full = sorted(
            full_squad_events
            - starting_xi_events
        )

        only_xi = sorted(
            starting_xi_events
            - full_squad_events
        )

        raise AssertionError(
            "The two representation datasets contain "
            "different accepted match populations. "
            f"Only full squad: {only_full[:20]}; "
            f"only expected XI: {only_xi[:20]}."
        )

    metadata = {
        "study_id": "048",
        "study_name": (
            "Leakage-Safe Club Observation Dataset"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "match_input": str(
            MATCH_INPUT.relative_to(
                PROJECT_ROOT
            )
        ),
        "prediction_season_start_year": (
            int(
                matches[
                    "season_start_year"
                ].iloc[0]
            )
        ),
        "representation_season_start_year": (
            int(
                matches[
                    "season_start_year"
                ].iloc[0]
            )
            - 1
        ),
        "rating_prior_available": False,
        "representation_types": [
            "full_squad",
            "expected_starting_xi",
        ],
        "formation": "4-3-3",
        "accepted_event_population_equal": (
            True
        ),
        "output_files": [
            "full_squad_observations.csv",
            "full_squad_exclusions.csv",
            (
                "expected_starting_xi_"
                "observations.csv"
            ),
            (
                "expected_starting_xi_"
                "exclusions.csv"
            ),
            "observation_dataset_summary.csv",
            "study_metadata.json",
        ],
    }

    with (
        OUTPUT_DIR
        / "study_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("Study 048")
    print("=" * 56)
    print()
    print(
        f"Historical matches loaded: "
        f"{len(matches)}"
    )
    print(
        "Prediction season: "
        f"{metadata['prediction_season_start_year']}"
    )
    print(
        "Representation season: "
        f"{metadata['representation_season_start_year']}"
    )
    print()
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )
    print()
    print(
        "Accepted event populations equal: PASS"
    )
    print(
        "Previous-season temporal validity: PASS"
    )
    print(
        "Historical rating-prior exclusion: PASS"
    )
    print(
        "Observation schema validation: PASS"
    )
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
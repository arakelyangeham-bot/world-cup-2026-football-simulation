#run_bundesliga_prediction_sanity_benchmark

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.production_goal_model import (
    ProductionGoalModel,
)
from research.production.scoreline_probability_calculator import (
    outcome_probabilities,
)
from research.rating_priors.clubelo_repository import (
    ClubEloRepository,
)

# Adjust this import only if the module lives elsewhere.
from simulation.live_match_observation_builder import (
    LiveMatchObservationBuilder,
    ProductionClubRepository,
)

from research.studies.study_079_bundesliga_live_observation_integration.audit_bundesliga_clubelo_cache import (
    BUNDESLIGA_CLUBELO_NAME_OVERRIDES,
    CLUBELO_CACHE_DIRECTORY,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BUNDESLIGA_REPOSITORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_078_bundesliga_production_repository"
    / "bundesliga_club_repository_v1.csv"
)

GOAL_MODEL_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "study_069_production_club_goal_model_v1"
    / "integrated_club_goal_model_v1.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "study_081_bundesliga_prediction_sanity_benchmark"
)

FIXTURE_PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "fixture_predictions.csv"
)

DISTRIBUTION_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "prediction_distribution_summary.csv"
)

TEAM_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "team_prediction_summary.csv"
)

EXTREME_CASES_PATH = (
    OUTPUT_DIRECTORY
    / "extreme_prediction_cases.csv"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "study_metadata.json"
)


PREDICTION_DATE = "2025-06-01"
EXPECTED_CLUB_COUNT = 18
EXPECTED_FIXTURE_COUNT = 18 * 17
EXTREME_CASE_COUNT = 10


def build_runtime_components() -> tuple[
    ProductionClubRepository,
    LiveMatchObservationBuilder,
    ProductionGoalModel,
]:
    club_repository = ProductionClubRepository(
        repository_path=(
            BUNDESLIGA_REPOSITORY_PATH
        )
    )

    clubelo_repository = ClubEloRepository(
        cache_directory=(
            CLUBELO_CACHE_DIRECTORY
        )
    )

    observation_builder = LiveMatchObservationBuilder(
        club_repository=club_repository,
        clubelo_repository=clubelo_repository,
        clubelo_name_overrides=(
            BUNDESLIGA_CLUBELO_NAME_OVERRIDES
        ),
    )

    goal_model = ProductionGoalModel.from_path(
        GOAL_MODEL_ARTIFACT_PATH
    )

    return (
        club_repository,
        observation_builder,
        goal_model,
    )


def build_fixture_population(
    clubs: tuple[str, ...],
) -> pd.DataFrame:
    """
    Build every ordered home-away pairing.

    With 18 clubs, this produces 18 × 17 = 306 fixtures.
    """

    if len(clubs) != EXPECTED_CLUB_COUNT:
        raise ValueError(
            "Unexpected Bundesliga club count. "
            f"Expected {EXPECTED_CLUB_COUNT}, "
            f"received {len(clubs)}."
        )

    records: list[dict[str, object]] = []

    fixture_id = 1

    for home_team in clubs:
        for away_team in clubs:
            if home_team == away_team:
                continue

            records.append(
                {
                    "fixture_id": fixture_id,
                    "prediction_date": (
                        PREDICTION_DATE
                    ),
                    "home_team": home_team,
                    "away_team": away_team,
                }
            )

            fixture_id += 1

    fixtures = pd.DataFrame(
        records
    )

    if len(fixtures) != EXPECTED_FIXTURE_COUNT:
        raise AssertionError(
            "Ordered fixture population has an unexpected "
            f"size: {len(fixtures)}."
        )

    if fixtures[
        [
            "home_team",
            "away_team",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Fixture population contains duplicate ordered "
            "matchups."
        )

    if fixtures[
        "home_team"
    ].eq(
        fixtures["away_team"]
    ).any():
        raise AssertionError(
            "Fixture population contains a self-match."
        )

    return fixtures


def predict_fixtures(
    *,
    fixtures: pd.DataFrame,
    observation_builder: LiveMatchObservationBuilder,
    goal_model: ProductionGoalModel,
) -> pd.DataFrame:
    predictions: list[
        dict[str, object]
    ] = []

    for row in fixtures.itertuples(
        index=False
    ):
        observation = observation_builder.build(
            home_team=row.home_team,
            away_team=row.away_team,
            prediction_date=row.prediction_date,
        )

        features = (
            observation.to_feature_mapping()
        )

        prediction = goal_model.predict(
            features
        )

        probabilities = outcome_probabilities(
            lambda_home=prediction.lambda_home,
            lambda_away=prediction.lambda_away,
        )

        prediction_record = {
            "fixture_id": int(row.fixture_id),
            "prediction_date": (
                str(row.prediction_date)
            ),
            "home_team": observation.home_team,
            "away_team": observation.away_team,

            "home_attack":
                observation.home_attack,
            "away_attack":
                observation.away_attack,
            "home_defense":
                observation.home_defense,
            "away_defense":
                observation.away_defense,
            "attack_depth_diff":
                observation.attack_depth_diff,
            "rating_prior_diff":
                observation.rating_prior_diff,

            "lambda_home":
                prediction.lambda_home,
            "lambda_away":
                prediction.lambda_away,
            "pred_total_goals":
                prediction.pred_total_goals,
            "pred_goal_diff":
                prediction.pred_goal_diff,

            "home_win_probability":
                probabilities.home_win,
            "draw_probability":
                probabilities.draw,
            "away_win_probability":
                probabilities.away_win,

            "home_rating_effective_from": (
                observation
                .home_rating_effective_from
                .isoformat()
            ),
            "home_rating_effective_to": (
                observation
                .home_rating_effective_to
                .isoformat()
            ),
            "away_rating_effective_from": (
                observation
                .away_rating_effective_from
                .isoformat()
            ),
            "away_rating_effective_to": (
                observation
                .away_rating_effective_to
                .isoformat()
            ),
        }

        predictions.append(
            prediction_record
        )

    dataframe = pd.DataFrame(
        predictions
    )

    validate_fixture_predictions(
        dataframe
    )

    return dataframe


def validate_fixture_predictions(
    dataframe: pd.DataFrame,
) -> None:
    if len(dataframe) != EXPECTED_FIXTURE_COUNT:
        raise AssertionError(
            "Prediction output does not preserve the "
            "complete fixture population."
        )

    if dataframe[
        "fixture_id"
    ].duplicated().any():
        raise AssertionError(
            "Prediction output contains duplicate fixture "
            "identifiers."
        )

    numeric_columns = [
        "home_attack",
        "away_attack",
        "home_defense",
        "away_defense",
        "attack_depth_diff",
        "rating_prior_diff",
        "lambda_home",
        "lambda_away",
        "pred_total_goals",
        "pred_goal_diff",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]

    numeric_values = dataframe[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise AssertionError(
            "Fixture predictions contain non-finite values."
        )

    if (
        dataframe["lambda_home"]
        .le(0.0)
        .any()
    ):
        raise AssertionError(
            "Fixture predictions contain non-positive "
            "home lambdas."
        )

    if (
        dataframe["lambda_away"]
        .le(0.0)
        .any()
    ):
        raise AssertionError(
            "Fixture predictions contain non-positive "
            "away lambdas."
        )

    probability_columns = [
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]

    probabilities = dataframe[
        probability_columns
    ]

    if (
        probabilities.lt(0.0).any().any()
        or probabilities.gt(1.0).any().any()
    ):
        raise AssertionError(
            "Fixture predictions contain probabilities "
            "outside [0, 1]."
        )

    probability_sums = (
        probabilities.sum(
            axis=1
        )
    )

    if not np.allclose(
        probability_sums.to_numpy(
            dtype=float
        ),
        np.ones(
            len(dataframe)
        ),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError(
            "Outcome probabilities do not sum to one."
        )

    expected_totals = (
        dataframe["lambda_home"]
        + dataframe["lambda_away"]
    )

    if not np.allclose(
        dataframe[
            "pred_total_goals"
        ].to_numpy(dtype=float),
        expected_totals.to_numpy(
            dtype=float
        ),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError(
            "Predicted total-goal arithmetic is invalid."
        )

    expected_differences = (
        dataframe["lambda_home"]
        - dataframe["lambda_away"]
    )

    if not np.allclose(
        dataframe[
            "pred_goal_diff"
        ].to_numpy(dtype=float),
        expected_differences.to_numpy(
            dtype=float
        ),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise AssertionError(
            "Predicted goal-difference arithmetic is "
            "invalid."
        )


def build_distribution_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    metrics = {
        "fixture_count": len(
            predictions
        ),

        "mean_lambda_home": (
            predictions[
                "lambda_home"
            ].mean()
        ),
        "mean_lambda_away": (
            predictions[
                "lambda_away"
            ].mean()
        ),
        "mean_total_goals": (
            predictions[
                "pred_total_goals"
            ].mean()
        ),
        "mean_goal_difference": (
            predictions[
                "pred_goal_diff"
            ].mean()
        ),

        "minimum_lambda_home": (
            predictions[
                "lambda_home"
            ].min()
        ),
        "maximum_lambda_home": (
            predictions[
                "lambda_home"
            ].max()
        ),
        "minimum_lambda_away": (
            predictions[
                "lambda_away"
            ].min()
        ),
        "maximum_lambda_away": (
            predictions[
                "lambda_away"
            ].max()
        ),
        "minimum_total_goals": (
            predictions[
                "pred_total_goals"
            ].min()
        ),
        "maximum_total_goals": (
            predictions[
                "pred_total_goals"
            ].max()
        ),

        "mean_home_win_probability": (
            predictions[
                "home_win_probability"
            ].mean()
        ),
        "mean_draw_probability": (
            predictions[
                "draw_probability"
            ].mean()
        ),
        "mean_away_win_probability": (
            predictions[
                "away_win_probability"
            ].mean()
        ),

        "minimum_draw_probability": (
            predictions[
                "draw_probability"
            ].min()
        ),
        "maximum_draw_probability": (
            predictions[
                "draw_probability"
            ].max()
        ),
    }

    return pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
            }
            for metric, value in (
                metrics.items()
            )
        ]
    )


def build_team_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    home = predictions[
        [
            "home_team",
            "lambda_home",
            "lambda_away",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_team": "club",
            "lambda_home":
                "expected_goals_for",
            "lambda_away":
                "expected_goals_against",
            "home_win_probability":
                "win_probability",
            "away_win_probability":
                "loss_probability",
        }
    )

    home["venue"] = "home"

    away = predictions[
        [
            "away_team",
            "lambda_away",
            "lambda_home",
            "away_win_probability",
            "draw_probability",
            "home_win_probability",
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_team": "club",
            "lambda_away":
                "expected_goals_for",
            "lambda_home":
                "expected_goals_against",
            "away_win_probability":
                "win_probability",
            "home_win_probability":
                "loss_probability",
        }
    )

    away["venue"] = "away"

    combined = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    combined[
        "expected_goal_difference"
    ] = (
        combined["expected_goals_for"]
        - combined["expected_goals_against"]
    )

    summary = (
        combined
        .groupby(
            "club",
            as_index=False,
        )
        .agg(
            appearances=(
                "club",
                "size",
            ),
            mean_expected_goals_for=(
                "expected_goals_for",
                "mean",
            ),
            mean_expected_goals_against=(
                "expected_goals_against",
                "mean",
            ),
            mean_expected_goal_difference=(
                "expected_goal_difference",
                "mean",
            ),
            mean_win_probability=(
                "win_probability",
                "mean",
            ),
            mean_draw_probability=(
                "draw_probability",
                "mean",
            ),
            mean_loss_probability=(
                "loss_probability",
                "mean",
            ),
        )
        .sort_values(
            [
                "mean_expected_goal_difference",
                "mean_win_probability",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    expected_appearances = (
        2
        * (
            EXPECTED_CLUB_COUNT
            - 1
        )
    )

    if not summary[
        "appearances"
    ].eq(
        expected_appearances
    ).all():
        raise AssertionError(
            "Team summary does not contain a balanced "
            "home-and-away population."
        )

    return summary


def select_extreme_cases(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    selections = {
        "highest_home_lambda": (
            predictions.nlargest(
                EXTREME_CASE_COUNT,
                "lambda_home",
            )
        ),
        "highest_away_lambda": (
            predictions.nlargest(
                EXTREME_CASE_COUNT,
                "lambda_away",
            )
        ),
        "highest_total_goals": (
            predictions.nlargest(
                EXTREME_CASE_COUNT,
                "pred_total_goals",
            )
        ),
        "lowest_total_goals": (
            predictions.nsmallest(
                EXTREME_CASE_COUNT,
                "pred_total_goals",
            )
        ),
        "largest_home_advantage": (
            predictions.nlargest(
                EXTREME_CASE_COUNT,
                "pred_goal_diff",
            )
        ),
        "largest_away_advantage": (
            predictions.nsmallest(
                EXTREME_CASE_COUNT,
                "pred_goal_diff",
            )
        ),
        "most_even": (
            predictions.assign(
                absolute_goal_difference=(
                    predictions[
                        "pred_goal_diff"
                    ].abs()
                )
            )
            .nsmallest(
                EXTREME_CASE_COUNT,
                "absolute_goal_difference",
            )
        ),
        "strongest_home_favorite": (
            predictions.nlargest(
                EXTREME_CASE_COUNT,
                "home_win_probability",
            )
        ),
        "strongest_away_favorite": (
            predictions.nlargest(
                EXTREME_CASE_COUNT,
                "away_win_probability",
            )
        ),
    }

    records: list[
        pd.DataFrame
    ] = []

    selected_columns = [
        "fixture_id",
        "home_team",
        "away_team",
        "lambda_home",
        "lambda_away",
        "pred_total_goals",
        "pred_goal_diff",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]

    for category, dataframe in (
        selections.items()
    ):
        output = dataframe[
            selected_columns
        ].copy()

        output.insert(
            0,
            "category",
            category,
        )

        output.insert(
            1,
            "category_rank",
            range(
                1,
                len(output) + 1,
            ),
        )

        records.append(
            output
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


def build_metadata(
    *,
    club_count: int,
    fixture_count: int,
    goal_model: ProductionGoalModel,
) -> dict[str, object]:
    return {
        "study_id": "081",
        "study_name": (
            "Bundesliga Prediction Sanity Benchmark"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "prediction_date": (
            PREDICTION_DATE
        ),
        "club_count": club_count,
        "fixture_count": fixture_count,
        "fixture_population": (
            "all_ordered_home_away_pairings"
        ),
        "goal_model_artifact_name": (
            goal_model.artifact_name
        ),
        "goal_model_artifact_version": (
            goal_model.artifact_version
        ),
        "goal_model_baseline_version": (
            goal_model.baseline_version
        ),
        "goal_model_training_end_date": (
            goal_model.training_end_date
        ),
        "status": "PASS",
        "outputs": [
            FIXTURE_PREDICTIONS_PATH.name,
            DISTRIBUTION_SUMMARY_PATH.name,
            TEAM_SUMMARY_PATH.name,
            EXTREME_CASES_PATH.name,
            METADATA_PATH.name,
        ],
    }


def main() -> None:
    print("=" * 88)
    print(
        "STUDY 081 — BUNDESLIGA PREDICTION "
        "SANITY BENCHMARK"
    )
    print("=" * 88)

    (
        club_repository,
        observation_builder,
        goal_model,
    ) = build_runtime_components()

    clubs = (
        club_repository.list_clubs()
    )

    print()
    print("Runtime configuration")
    print(
        f"  Clubs: {len(clubs)}"
    )
    print(
        f"  Prediction date: {PREDICTION_DATE}"
    )
    print(
        "  Goal model: "
        f"{goal_model.artifact_name} "
        f"v{goal_model.artifact_version}"
    )
    print(
        "  Training cutoff: "
        f"{goal_model.training_end_date}"
    )

    if (
        PREDICTION_DATE
        <= goal_model.training_end_date
    ):
        raise ValueError(
            "Study 081 prediction date must be later than "
            "the production model training cutoff."
        )

    fixtures = build_fixture_population(
        clubs
    )

    print()
    print(
        "Generating predictions for "
        f"{len(fixtures)} ordered fixtures..."
    )

    predictions = predict_fixtures(
        fixtures=fixtures,
        observation_builder=(
            observation_builder
        ),
        goal_model=goal_model,
    )

    distribution_summary = (
        build_distribution_summary(
            predictions
        )
    )

    team_summary = build_team_summary(
        predictions
    )

    extreme_cases = select_extreme_cases(
        predictions
    )

    metadata = build_metadata(
        club_count=len(clubs),
        fixture_count=len(fixtures),
        goal_model=goal_model,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        FIXTURE_PREDICTIONS_PATH,
        index=False,
    )

    distribution_summary.to_csv(
        DISTRIBUTION_SUMMARY_PATH,
        index=False,
    )

    team_summary.to_csv(
        TEAM_SUMMARY_PATH,
        index=False,
    )

    extreme_cases.to_csv(
        EXTREME_CASES_PATH,
        index=False,
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_lookup = {
        row.metric: row.value
        for row in (
            distribution_summary
            .itertuples(
                index=False
            )
        )
    }

    print()
    print("Prediction distribution")
    print("-" * 88)
    print(
        "  Mean home lambda: "
        f"{summary_lookup['mean_lambda_home']:.6f}"
    )
    print(
        "  Mean away lambda: "
        f"{summary_lookup['mean_lambda_away']:.6f}"
    )
    print(
        "  Mean total goals: "
        f"{summary_lookup['mean_total_goals']:.6f}"
    )
    print(
        "  Mean home-win probability: "
        f"{summary_lookup['mean_home_win_probability']:.6f}"
    )
    print(
        "  Mean draw probability: "
        f"{summary_lookup['mean_draw_probability']:.6f}"
    )
    print(
        "  Mean away-win probability: "
        f"{summary_lookup['mean_away_win_probability']:.6f}"
    )
    print(
        "  Lambda-home range: "
        f"{summary_lookup['minimum_lambda_home']:.6f} "
        "to "
        f"{summary_lookup['maximum_lambda_home']:.6f}"
    )
    print(
        "  Lambda-away range: "
        f"{summary_lookup['minimum_lambda_away']:.6f} "
        "to "
        f"{summary_lookup['maximum_lambda_away']:.6f}"
    )
    print(
        "  Total-goals range: "
        f"{summary_lookup['minimum_total_goals']:.6f} "
        "to "
        f"{summary_lookup['maximum_total_goals']:.6f}"
    )

    print()
    print("Highest team-strength summaries")
    print("-" * 88)
    print(
        team_summary.head(
            10
        ).to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Validation summary")
    print("  Club population: PASS")
    print("  Ordered fixture population: PASS")
    print("  Live observation construction: PASS")
    print("  Goal-model prediction: PASS")
    print("  Positive finite lambdas: PASS")
    print("  Probability normalization: PASS")
    print("  Balanced club appearances: PASS")
    print("  Distribution outputs: PASS")
    print("  Extreme-case outputs: PASS")

    print()
    print("=" * 88)
    print("OVERALL RESULT: PASS")
    print("=" * 88)
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
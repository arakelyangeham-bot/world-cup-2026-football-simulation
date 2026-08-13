#build_historical_training_dataset.py

from datetime import date
from pathlib import Path

import pandas as pd

from scripts.team_strength_loader import TEAM_STRENGTH_FILE
from shared.national_team_priors import load_fifa_points
from shared.team_name_normalizer import normalize_team_name
from shared.historical_match_catalog import (
    get_available_historical_match_datasets,
)

from research.football_features.team_feature_provider import (
    StaticCsvTeamFeatureProvider,
    TeamFeatureProvider,
    TeamFeatureRequest,
)

RAW_DIR = Path("data/raw/sofascore")
OUTPUT_DIR = Path("outputs/model_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_static_team_feature_provider(
) -> StaticCsvTeamFeatureProvider:
    """
    Construct the migration provider that reproduces the
    historical builder's existing static-repository behavior.

    This provider is not historically faithful. The same
    repository snapshot is used for every historical match.
    """

    return StaticCsvTeamFeatureProvider(
        repository_path=TEAM_STRENGTH_FILE,
        provider_name=(
            "static_csv_team_features_v1"
        ),
        representation_type=(
            "static_repository"
        ),
        aggregation_profile=(
            "legacy_static"
        ),
        repository_scope=(
            "national_teams"
        ),
    )

def resolve_provider_request_date(
    match: pd.Series,
    *,
    dataset_year: int,
) -> date:
    """
    Resolve the best date available for a provider request.

    Exact event dates are preferred. The dataset year is used
    only as a compatibility fallback while the historical input
    schemas are being audited.

    The fallback does not imply that January 1 is the actual
    match date. It exists only because the current static
    provider requires a syntactically valid date and does not
    vary its values through time.
    """

    for column in (
        "date",
        "start_timestamp",
        "match_date",
        "event_date",
    ):
        if column not in match.index:
            continue

        value = match[column]

        if pd.isna(value):
            continue

        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                "UTC"
            )
        else:
            timestamp = timestamp.tz_convert(
                "UTC"
            )

        return timestamp.date()

    return date(
        int(dataset_year),
        1,
        1,
    )

def resolve_team_features(
    *,
    provider: TeamFeatureProvider,
    team_name: str,
    prediction_date: date,
) -> dict[str, float]:
    result = provider.get_team_features(
        TeamFeatureRequest(
            team_name=team_name,
            prediction_date=prediction_date,
        )
    )

    return result.to_repository_entry()

def result_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if away_score > home_score:
        return "away_win"
    return "draw"


def main(
    team_feature_provider: (
        TeamFeatureProvider | None
    ) = None,
) -> None:
    provider = (
        team_feature_provider
        if team_feature_provider is not None
        else build_static_team_feature_provider()
    )

    fifa_points = load_fifa_points()

    rows = []
    skipped = []

    processed_datasets = 0
    skipped_datasets = 0

    for dataset in get_available_historical_match_datasets():
        path = RAW_DIR / dataset.filename
        if not path.exists():
            skipped_datasets+=1
            print(
                f"[SKIP] Missing dataset: "
                f"{dataset.dataset_id} "
                f"({path.name})"
            )
            continue

        matches = pd.read_csv(path)

        processed_datasets+=1

        year = dataset.year
        competition = dataset.competition

        for _, match in matches.iterrows():
            raw_home = match["home_team"]
            raw_away = match["away_team"]

            home = normalize_team_name(raw_home)
            away = normalize_team_name(raw_away)

            if home not in fifa_points or away not in fifa_points:
                skipped.append({
                    "competition_key": dataset.competition_key,
                    "competition": competition.display_name,
                    "dataset_id": f"{dataset.competition_key}_{dataset.year}",
                    "year": year,
                    "raw_home_team": raw_home,
                    "raw_away_team": raw_away,
                    "normalized_home_team": home,
                    "normalized_away_team": away,
                    "reason": "missing_fifa_points",
                })
                continue

            prediction_date = (
                resolve_provider_request_date(
                    match,
                    dataset_year=year,
                )
            )

            try:
                h = resolve_team_features(
                    provider=provider,
                    team_name=home,
                    prediction_date=prediction_date,
                )

                a = resolve_team_features(
                    provider=provider,
                    team_name=away,
                    prediction_date=prediction_date,
                )

            except KeyError:
                skipped.append({
                    "competition_key":
                        dataset.competition_key,
                    "competition":
                        competition.display_name,
                    "dataset_id": (
                        f"{dataset.competition_key}_"
                        f"{dataset.year}"
                    ),
                    "year":
                        year,
                    "raw_home_team":
                        raw_home,
                    "raw_away_team":
                        raw_away,
                    "normalized_home_team":
                        home,
                    "normalized_away_team":
                        away,
                    "reason":
                        "missing_team_features",
                })
                continue

            home_score = int(match["home_score"])
            away_score = int(match["away_score"])

            rows.append({
                "competition_key": dataset.competition_key,
                "competition": competition.display_name,
                "competition_category": competition.category,
                "competition_importance": competition.importance,
                "dataset_id": f"{dataset.competition_key}_{dataset.year}",
                "year": year,
                "event_id": match["event_id"],
                "home_team": home,
                "away_team": away,
                "raw_home_team": raw_home,
                "raw_away_team": raw_away,

                "home_score": home_score,
                "away_score": away_score,
                "total_goals": home_score + away_score,
                "goal_diff": home_score - away_score,
                "result": result_label(home_score, away_score),

                "home_attack": h["att_composite"],
                "home_midfield": h["mid_composite"],
                "home_defense": h["def_composite"],
                "home_gk": h["gk_composite"],

                "away_attack": a["att_composite"],
                "away_midfield": a["mid_composite"],
                "away_defense": a["def_composite"],
                "away_gk": a["gk_composite"],

                "attack_diff": h["att_composite"] - a["att_composite"],
                "midfield_diff": h["mid_composite"] - a["mid_composite"],
                "defense_diff": h["def_composite"] - a["def_composite"],
                "gk_diff": h["gk_composite"] - a["gk_composite"],

                "home_poisson_attack": h["poisson_attack_adj"],
                "home_poisson_defense": h["poisson_defense_adj"],
                "away_poisson_attack": a["poisson_attack_adj"],
                "away_poisson_defense": a["poisson_defense_adj"],

                "home_fifa_points": fifa_points[home],
                "away_fifa_points": fifa_points[away],
                "fifa_points_diff": fifa_points[home] - fifa_points[away],

                "poisson_attack_diff": h["poisson_attack_adj"] - a["poisson_attack_adj"],
                "poisson_defense_diff": h["poisson_defense_adj"] - a["poisson_defense_adj"],
            })

    dataset = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)

    dataset_file = OUTPUT_DIR / "historical_training_dataset.csv"
    skipped_file = OUTPUT_DIR / "historical_training_dataset_skipped.csv"

    dataset.to_csv(dataset_file, index=False)
    skipped_df.to_csv(skipped_file, index=False)

    print("Historical Training Dataset")
    print("---------------------------")
    print(f"Processed datasets: {processed_datasets}")
    print(f"Missing datasets:   {skipped_datasets}")
    print(f"Rows: {len(dataset)}")
    print(f"Skipped: {len(skipped_df)}")

    if len(dataset):
        print()
        print("Result distribution")
        print(dataset["result"].value_counts(normalize=True).to_string())

    print()
    print("Dataset summary")
    print("----------------")
    print(
        "Team feature provider: "
        f"{provider.provider_name}"
    )
    print(
        "Temporal representation mode: "
        "STATIC COMPATIBILITY"
    )

    print()
    print(f"Saved -> {dataset_file}")
    print(f"Saved -> {skipped_file}")



if __name__ == "__main__":
    main()
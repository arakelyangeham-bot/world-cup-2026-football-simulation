#investigate_competitive_resolution.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.football_observatory.binning import BinningStrategy
from research.football_observatory.football_population import (
    CORE_POPULATIONS,
    analyze_population,
    population_profiles_to_dataframe,
)
from research.football_observatory.observables import CORE_OBSERVABLES
from research.football_observatory.observatory_schema import (
    MatchObservation,
    match_observation_from_row,
)
from research.football_observatory.relationship import FootballRelationship
from research.football_observatory.relationship_analyzer import analyze_relationship


DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "studies"
    / "study_007_competitive_resolution"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_OUTPUT_PATH = OUTPUT_DIR / "competitive_resolution_profiles.csv"
COMPARISON_OUTPUT_PATH = OUTPUT_DIR / "draws_vs_one_goal_comparison.csv"
RELATIONSHIP_OUTPUT_PATH = OUTPUT_DIR / "competitive_resolution_relationships.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "competitive_resolution_summary.csv"


REPRESENTATIONS = {
    "fifa_points_gap": "fifa_points_gap",
    "attack_gap": "attack_gap",
    "midfield_gap": "midfield_gap",
    "defense_gap": "defense_gap",
    "gk_gap": "gk_gap",
    "poisson_attack_gap": "poisson_attack_gap",
    "poisson_defense_gap": "poisson_defense_gap",
}

OBSERVABLES = [
    "draw",
    "one_goal_match",
]


def get_population(name: str):
    for population in CORE_POPULATIONS:
        if population.name == name:
            return population

    raise ValueError(f"Unknown population: {name}")


def get_observable(name: str):
    for observable in CORE_OBSERVABLES:
        if observable.name == name:
            return observable

    raise ValueError(f"Unknown observable: {name}")


def add_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["fifa_points_gap"] = df["fifa_points_diff"].abs()
    df["attack_gap"] = df["attack_diff"].abs()
    df["midfield_gap"] = df["midfield_diff"].abs()
    df["defense_gap"] = df["defense_diff"].abs()
    df["gk_gap"] = df["gk_diff"].abs()
    df["poisson_attack_gap"] = df["poisson_attack_diff"].abs()
    df["poisson_defense_gap"] = df["poisson_defense_diff"].abs()

    return df


def load_observations() -> list[MatchObservation]:
    df = pd.read_csv(DATASET_PATH)
    df = add_gap_columns(df)

    observations = []

    for _, row in df.iterrows():
        base = match_observation_from_row(row)

        observations.append(
            MatchObservation(
                prematch=base.prematch,
                outcome=base.outcome,
                events=base.events,
                derived_prematch={
                    "fifa_points_gap": float(row["fifa_points_gap"]),
                    "attack_gap": float(row["attack_gap"]),
                    "midfield_gap": float(row["midfield_gap"]),
                    "defense_gap": float(row["defense_gap"]),
                    "gk_gap": float(row["gk_gap"]),
                    "poisson_attack_gap": float(row["poisson_attack_gap"]),
                    "poisson_defense_gap": float(row["poisson_defense_gap"]),
                },
            )
        )

    return observations


def run_population_analysis(
    observations: list[MatchObservation],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    draw_analysis = analyze_population(
        observations=observations,
        population=get_population("draws"),
    )

    one_goal_analysis = analyze_population(
        observations=observations,
        population=get_population("one_goal_matches"),
    )

    profiles = pd.concat(
        [
            population_profiles_to_dataframe(draw_analysis).iloc[[1]],
            population_profiles_to_dataframe(one_goal_analysis).iloc[[1]],
        ],
        ignore_index=True,
    )

    comparison_rows = []

    draw_profile = draw_analysis.population_profile
    one_goal_profile = one_goal_analysis.population_profile

    for metric, draw_value in draw_profile.__dict__.items():
        if metric in {"label", "matches"}:
            continue

        one_goal_value = getattr(one_goal_profile, metric)

        comparison_rows.append(
            {
                "metric": metric,
                "draws": draw_value,
                "one_goal_matches": one_goal_value,
                "difference_one_goal_minus_draws": (
                    one_goal_value - draw_value
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    return profiles, comparison


def run_relationship_analysis(
    observations: list[MatchObservation],
) -> pd.DataFrame:
    frames = []

    for representation_name, variable_name in REPRESENTATIONS.items():
        for observable_name in OBSERVABLES:
            relationship = FootballRelationship(
                name=f"{representation_name}_to_{observable_name}",
                description=(
                    f"{observable_name} response curve by "
                    f"{representation_name}."
                ),
                independent_variable=variable_name,
                observable=get_observable(observable_name),
                binning=BinningStrategy(
                    mode="quantile",
                    n_bins=10,
                ),
            )

            result = analyze_relationship(
                observations=observations,
                relationship=relationship,
            )

            result["representation"] = representation_name
            result["observable"] = observable_name
            frames.append(result)

    return pd.concat(frames, ignore_index=True)


def summarize_relationships(relationships: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (representation, observable), group in relationships.groupby(
        ["representation", "observable"]
    ):
        valid = group.dropna(subset=["observable_rate"])

        rows.append(
            {
                "representation": representation,
                "observable": observable,
                "bins": len(valid),
                "total_matches": valid["matches"].sum(),
                "min_rate": valid["observable_rate"].min(),
                "max_rate": valid["observable_rate"].max(),
                "rate_range": (
                    valid["observable_rate"].max()
                    - valid["observable_rate"].min()
                ),
                "mean_ci_width": (
                    valid["ci_upper"] - valid["ci_lower"]
                ).mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["observable", "rate_range"],
        ascending=[True, False],
    )


def main() -> None:
    observations = load_observations()

    profiles, comparison = run_population_analysis(observations)
    relationships = run_relationship_analysis(observations)
    summary = summarize_relationships(relationships)

    profiles.to_csv(PROFILE_OUTPUT_PATH, index=False)
    comparison.to_csv(COMPARISON_OUTPUT_PATH, index=False)
    relationships.to_csv(RELATIONSHIP_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("Study 007 — Competitive Resolution")
    print("----------------------------------")
    print()
    print("Population profiles")
    print(profiles.round(4).to_string(index=False))
    print()
    print("Largest draw vs one-goal differences")
    print(
        comparison
        .assign(
            abs_difference=lambda x: x[
                "difference_one_goal_minus_draws"
            ].abs()
        )
        .sort_values("abs_difference", ascending=False)
        .head(12)
        .round(4)
        .to_string(index=False)
    )
    print()
    print("Relationship summary")
    print(summary.round(4).to_string(index=False))
    print()
    print(f"Wrote profiles      -> {PROFILE_OUTPUT_PATH}")
    print(f"Wrote comparison    -> {COMPARISON_OUTPUT_PATH}")
    print(f"Wrote relationships -> {RELATIONSHIP_OUTPUT_PATH}")
    print(f"Wrote summary       -> {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
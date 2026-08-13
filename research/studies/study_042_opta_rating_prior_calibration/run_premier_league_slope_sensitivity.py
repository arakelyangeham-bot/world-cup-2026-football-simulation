#run_premier_league_slope_sensitivity

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from competition_catalog import CompetitionBuilder
from competition_catalog.competitions import PREMIER_LEAGUE
from fixture_generation import RoundRobinFixtureGenerator
from research import ExperimentCondition
from research.adapters import FootballModelAdapter
from simulation.competition import CompetitionEngine
from simulation.league_match_simulator import LeagueMatchSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_GLOBAL_CLUB_DATASET = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
    / "global_club_prior_dataset.csv"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "studies"
    / "study_042_opta_rating_prior_calibration"
    / "outputs"
    / "phase_3_premier_league_slope_sensitivity"
)

PREMIER_LEAGUE_TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]

CANDIDATE_SLOPES = [
    0.0,
    1.0,
    10.0,
    15.0,
    20.0,
    25.0,
]

REPOSITORY_SOURCE = "premier_league_validation"
BASE_SEED = 42001


def parse_slope_list(value: str) -> list[float]:
    try:
        slopes = [
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Slopes must be a comma-separated list of numbers."
        ) from exc

    if not slopes:
        raise argparse.ArgumentTypeError(
            "At least one slope is required."
        )

    if any(slope < 0.0 for slope in slopes):
        raise argparse.ArgumentTypeError(
            "Candidate slopes must be non-negative."
        )

    return slopes


def load_canonical_opta_ratings(
    path: Path,
) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Global club dataset not found: {path}"
        )

    dataframe = pd.read_csv(path)

    required_columns = {
        "club_id",
        "club",
        "opta_rating",
        "global_rank",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Global club dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["opta_rating"] = pd.to_numeric(
        dataframe["opta_rating"],
        errors="raise",
    )

    dataframe["global_rank"] = pd.to_numeric(
        dataframe["global_rank"],
        errors="raise",
    ).astype(int)

    selected_ratings: dict[str, float] = {}
    missing_teams: list[str] = []

    for team in PREMIER_LEAGUE_TEAMS:
        matches = dataframe[
            dataframe["club"].eq(team)
        ].copy()

        if matches.empty:
            missing_teams.append(team)
            continue

        # Consistent with Study 041 and the Premier League audit:
        # select the highest-ranked record when names collide.
        selected = matches.sort_values(
            ["global_rank", "opta_rating"],
            ascending=[True, False],
        ).iloc[0]

        selected_ratings[team] = float(
            selected["opta_rating"]
        )

    if missing_teams:
        raise ValueError(
            "Missing canonical Opta ratings for: "
            f"{missing_teams}"
        )

    if len(selected_ratings) != 20:
        raise ValueError(
            "Expected exactly 20 Premier League ratings, "
            f"but found {len(selected_ratings)}."
        )

    return selected_ratings


def build_football_model(
    slope: float,
):
    condition = ExperimentCondition(
        name=(
            "Premier League prior-slope sensitivity "
            f"slope={slope:g}"
        ),
        competition_format="double_round_robin",
        repository_source=REPOSITORY_SOURCE,
        match_engine="production_scoreline_first",
        simulation_count=1,
        random_seed=BASE_SEED,
        parameters={
            "competition": "Premier League",
            "study": "Study 042",
            "candidate_slope": slope,
        },
    )

    return FootballModelAdapter().from_condition(
        condition
    )


def apply_candidate_slope(
    football_model,
    opta_ratings: dict[str, float],
    slope: float,
) -> None:
    """
    Apply one experimental Opta-to-prior slope in memory.

    The canonical source dataset and repository CSV remain unchanged.
    """

    missing_repository_teams = [
        team
        for team in PREMIER_LEAGUE_TEAMS
        if team not in football_model.team_repository
    ]

    if missing_repository_teams:
        raise KeyError(
            "Football model repository is missing clubs: "
            f"{missing_repository_teams}"
        )

    for team in PREMIER_LEAGUE_TEAMS:
        opta_rating = opta_ratings[team]
        candidate_prior = slope * opta_rating

        entry = football_model.team_repository[team]

        # Canonical runtime field.
        entry["rating_prior"] = candidate_prior

        # Temporary compatibility alias.
        entry["fifa_points"] = candidate_prior

        # Preserve source information for inspection.
        entry["opta_rating"] = opta_rating
        entry["rating_prior_method"] = (
            f"linear_opta_slope_{slope:g}"
        )


def build_fixtures():
    builder = CompetitionBuilder()
    fixture_generator = RoundRobinFixtureGenerator()

    competition = builder.build(
        definition=PREMIER_LEAGUE,
        participants=PREMIER_LEAGUE_TEAMS,
    )

    fixtures = fixture_generator.generate(
        participants=competition.participants,
        double_round_robin=True,
        competition_id="premier_league",
    )

    if len(fixtures) != 380:
        raise RuntimeError(
            f"Expected 380 fixtures, found {len(fixtures)}."
        )

    return fixtures


def simulate_one_season(
    football_model,
    fixtures,
    seed: int,
) -> list[dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)

    builder = CompetitionBuilder()
    match_simulator = LeagueMatchSimulator()
    competition_engine = CompetitionEngine()

    competition = builder.build(
        definition=PREMIER_LEAGUE,
        participants=PREMIER_LEAGUE_TEAMS,
    )

    league_stage = competition.stages[0]

    league_stage.matches = (
        match_simulator.simulate_fixtures(
            fixtures=fixtures,
            football_model=football_model,
            stage_name=league_stage.name,
            competition_name=competition.name,
        )
    )

    if len(league_stage.matches) != 380:
        raise RuntimeError(
            "Season simulation did not produce 380 matches."
        )

    result = competition_engine.resolve(
        competition
    )

    if not result.stage_results:
        raise RuntimeError(
            "Competition produced no stage results."
        )

    stage_result = result.stage_results[0]

    if stage_result.standings is None:
        raise RuntimeError(
            "League stage produced no standings."
        )

    standings = stage_result.standings.as_rows()

    if len(standings) != 20:
        raise RuntimeError(
            "League standings did not contain 20 clubs."
        )

    if not all(
        row["matches_played"] == 38
        for row in standings
    ):
        raise RuntimeError(
            "One or more clubs did not play 38 matches."
        )

    return standings


def calculate_rank_correlation(
    standings: list[dict[str, Any]],
    opta_ratings: dict[str, float],
) -> float:
    """
    Pearson correlation between:

    - Opta strength rank within the 20 clubs
    - simulated final league rank

    Both rank systems use 1 as strongest/best, so a larger positive
    value means stronger alignment.
    """

    opta_order = sorted(
        PREMIER_LEAGUE_TEAMS,
        key=lambda team: opta_ratings[team],
        reverse=True,
    )

    opta_rank = {
        team: index + 1
        for index, team in enumerate(opta_order)
    }

    final_rank = {
        row["team"]: int(row["rank"])
        for row in standings
    }

    opta_values = np.array(
        [
            opta_rank[team]
            for team in PREMIER_LEAGUE_TEAMS
        ],
        dtype=float,
    )

    final_values = np.array(
        [
            final_rank[team]
            for team in PREMIER_LEAGUE_TEAMS
        ],
        dtype=float,
    )

    correlation_matrix = np.corrcoef(
        opta_values,
        final_values,
    )

    return float(correlation_matrix[0, 1])


def title_entropy(
    champion_counts: dict[str, int],
    simulation_count: int,
) -> float:
    entropy = 0.0

    for count in champion_counts.values():
        if count == 0:
            continue

        probability = count / simulation_count
        entropy -= probability * math.log(
            probability
        )

    return entropy


def run_slope_condition(
    slope: float,
    simulation_count: int,
    fixtures,
    opta_ratings: dict[str, float],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    football_model = build_football_model(
        slope=slope
    )

    apply_candidate_slope(
        football_model=football_model,
        opta_ratings=opta_ratings,
        slope=slope,
    )

    champion_counts = defaultdict(int)
    top_four_counts = defaultdict(int)
    relegation_counts = defaultdict(int)

    points_totals = defaultdict(float)
    position_totals = defaultdict(float)
    goals_for_totals = defaultdict(float)
    goals_against_totals = defaultdict(float)

    rank_correlations: list[float] = []
    first_to_last_points_spreads: list[float] = []
    goals_per_match_values: list[float] = []
    draw_rate_values: list[float] = []

    for simulation_index in range(
        simulation_count
    ):
        seed = (
            BASE_SEED
            + int(slope * 1000)
            + simulation_index
        )

        standings = simulate_one_season(
            football_model=football_model,
            fixtures=fixtures,
            seed=seed,
        )

        champion_counts[
            standings[0]["team"]
        ] += 1

        for row in standings[:4]:
            top_four_counts[
                row["team"]
            ] += 1

        for row in standings[-3:]:
            relegation_counts[
                row["team"]
            ] += 1

        for row in standings:
            team = row["team"]

            points_totals[team] += float(
                row["points"]
            )

            position_totals[team] += float(
                row["rank"]
            )

            goals_for_totals[team] += float(
                row["goals_for"]
            )

            goals_against_totals[team] += float(
                row["goals_against"]
            )

        rank_correlations.append(
            calculate_rank_correlation(
                standings=standings,
                opta_ratings=opta_ratings,
            )
        )

        first_to_last_points_spreads.append(
            float(standings[0]["points"])
            - float(standings[-1]["points"])
        )

        total_goals = sum(
            float(row["goals_for"])
            for row in standings
        )

        goals_per_match_values.append(
            total_goals / 380.0
        )

        # Every drawn match contributes one draw to each club.
        total_team_draws = sum(
            int(row["draws"])
            for row in standings
        )

        drawn_matches = total_team_draws / 2.0

        draw_rate_values.append(
            drawn_matches / 380.0
        )

    club_rows: list[dict[str, Any]] = []

    for team in PREMIER_LEAGUE_TEAMS:
        club_rows.append(
            {
                "candidate_slope": slope,
                "team": team,
                "opta_rating": opta_ratings[team],
                "rating_prior": (
                    slope * opta_ratings[team]
                ),
                "simulation_count": simulation_count,
                "champion_probability": (
                    champion_counts[team]
                    / simulation_count
                ),
                "top_four_probability": (
                    top_four_counts[team]
                    / simulation_count
                ),
                "relegation_probability": (
                    relegation_counts[team]
                    / simulation_count
                ),
                "average_points": (
                    points_totals[team]
                    / simulation_count
                ),
                "average_position": (
                    position_totals[team]
                    / simulation_count
                ),
                "average_goals_for": (
                    goals_for_totals[team]
                    / simulation_count
                ),
                "average_goals_against": (
                    goals_against_totals[team]
                    / simulation_count
                ),
            }
        )

    sorted_title_probabilities = sorted(
        [
            champion_counts[team]
            / simulation_count
            for team in PREMIER_LEAGUE_TEAMS
        ],
        reverse=True,
    )

    condition_summary = {
        "candidate_slope": slope,
        "simulation_count": simulation_count,
        "unique_champions": sum(
            1
            for probability in sorted_title_probabilities
            if probability > 0.0
        ),
        "largest_title_probability": (
            sorted_title_probabilities[0]
        ),
        "top_three_title_concentration": sum(
            sorted_title_probabilities[:3]
        ),
        "title_entropy": title_entropy(
            champion_counts=champion_counts,
            simulation_count=simulation_count,
        ),
        "mean_opta_final_rank_correlation": float(
            np.mean(rank_correlations)
        ),
        "standard_deviation_rank_correlation": float(
            np.std(
                rank_correlations,
                ddof=1,
            )
            if len(rank_correlations) > 1
            else 0.0
        ),
        "mean_first_to_last_points_spread": float(
            np.mean(
                first_to_last_points_spreads
            )
        ),
        "mean_goals_per_match": float(
            np.mean(goals_per_match_values)
        ),
        "mean_draw_rate": float(
            np.mean(draw_rate_values)
        ),
    }

    return club_rows, condition_summary


def write_rows_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError(
            f"Cannot write empty output: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(
    condition_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Study 042 – Premier League Prior-Slope Sensitivity",
        "",
        "## Condition summary",
        "",
        (
            "| Slope | Simulations | Unique champions "
            "| Largest title probability | Top-three title share "
            "| Rank correlation | Points spread "
            "| Goals per match | Draw rate |"
        ),
        (
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for row in condition_rows:
        lines.append(
            f"| {row['candidate_slope']:.1f} "
            f"| {row['simulation_count']} "
            f"| {row['unique_champions']} "
            f"| {row['largest_title_probability']:.4f} "
            f"| {row['top_three_title_concentration']:.4f} "
            f"| {row['mean_opta_final_rank_correlation']:.4f} "
            f"| {row['mean_first_to_last_points_spread']:.2f} "
            f"| {row['mean_goals_per_match']:.4f} "
            f"| {row['mean_draw_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "This experiment measures structural sensitivity to "
                "the Opta-to-prior slope. It does not establish the "
                "predictively optimal production slope."
            ),
            "",
            (
                "All non-prior team features are held fixed using the "
                "Premier League validation repository. Candidate priors "
                "are injected only in memory, and no canonical source "
                "dataset is modified."
            ),
            "",
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Study 042 Premier League "
            "prior-slope sensitivity experiment."
        )
    )

    parser.add_argument(
        "--club-dataset",
        type=Path,
        default=DEFAULT_GLOBAL_CLUB_DATASET,
        help="Path to global_club_prior_dataset.csv.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for sensitivity outputs.",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Complete league seasons per candidate slope.",
    )

    parser.add_argument(
        "--slopes",
        type=parse_slope_list,
        default=CANDIDATE_SLOPES,
        help=(
            "Comma-separated candidate slopes. "
            "Example: 0,1,10,15,20,25"
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.simulations < 1:
        raise ValueError(
            "--simulations must be at least 1."
        )

    opta_ratings = load_canonical_opta_ratings(
        arguments.club_dataset
    )

    fixtures = build_fixtures()

    all_club_rows: list[dict[str, Any]] = []
    all_condition_rows: list[dict[str, Any]] = []

    print(
        "Study 042 – Premier League "
        "Prior-Slope Sensitivity"
    )
    print(
        "============================================="
    )
    print(
        f"Simulations per slope: "
        f"{arguments.simulations}"
    )
    print(
        "Candidate slopes: "
        f"{arguments.slopes}"
    )
    print()

    for slope in arguments.slopes:
        print(
            f"Running slope {slope:g}..."
        )

        club_rows, condition_summary = (
            run_slope_condition(
                slope=slope,
                simulation_count=(
                    arguments.simulations
                ),
                fixtures=fixtures,
                opta_ratings=opta_ratings,
            )
        )

        all_club_rows.extend(club_rows)
        all_condition_rows.append(
            condition_summary
        )

        print(
            "  Largest title probability: "
            f"{condition_summary['largest_title_probability']:.3f}"
        )
        print(
            "  Mean rank correlation: "
            f"{condition_summary['mean_opta_final_rank_correlation']:.3f}"
        )
        print(
            "  Mean points spread: "
            f"{condition_summary['mean_first_to_last_points_spread']:.2f}"
        )
        print()

    output_directory = (
        arguments.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_rows_csv(
        all_club_rows,
        output_directory
        / "club_probability_summary.csv",
    )

    write_rows_csv(
        all_condition_rows,
        output_directory
        / "condition_summary.csv",
    )

    write_markdown_summary(
        condition_rows=all_condition_rows,
        output_path=(
            output_directory
            / "slope_sensitivity_summary.md"
        ),
    )

    print(f"Outputs: {output_directory}")
    print()
    print(
        "Sensitivity experiment completed. "
        "No production slope was selected."
    )


if __name__ == "__main__":
    main()
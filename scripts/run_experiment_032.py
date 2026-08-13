#run_experiment_032

import random
import numpy as np
import csv
from pathlib import Path

from research import Experiment, ExperimentCondition
from research.adapters import CompetitionAdapter, FootballModelAdapter
from research.experiment_runner import ExperimentRunner
from research.experiments import ExperimentResult
from research.metrics import (
    AverageChampionStrengthMetric,
    ChampionVarianceMetric,
    StrongestTeamChampionshipRateMetric,
    UpsetRateMetric,
)
from simulation.competition import (
    BracketBuilder,
    Competition,
    CompetitionEngine,
    MatchResult,
    Stage,
    StageType,
)


TEAMS = [
    "Argentina",
    "France",
    "Brazil",
    "England",
    "Spain",
    "Portugal",
    "Netherlands",
    "Germany",
]

REPOSITORY_SOURCES = [
    "legacy",
    "dimension_specific",
    "top_11_mean",
    "top_5_mean",
    "star_weighted",
    "starter_plus_depth",
]

SIMULATION_COUNT = 1000
BASE_SEED = 31031


def team_strength(team_data: dict) -> float:
    return (
        0.35 * team_data["attack"]
        + 0.25 * team_data["midfield"]
        + 0.25 * team_data["defense"]
        + 0.15 * team_data["gk"]
    )


def build_experiment_definition(metrics) -> Experiment:
    return Experiment(
        experiment_id="032",
        title="Repository Sensitivity Analysis",
        research_question=(
            "How sensitive are competition outcomes to the choice of "
            "team representation repository?"
        ),
        hypothesis=(
            "Changing the team representation repository will change the "
            "magnitude of competition metrics while preserving the qualitative "
            "advantage of league competitions over knockout competitions."
        ),
        fixed_variables={
            "team_set": TEAMS,
            "repository_sources_tested": REPOSITORY_SOURCES,
            "match_engine": "production_scoreline_first",
            "simulation_count": SIMULATION_COUNT,
            "random_seed_policy": f"base_seed_{BASE_SEED}_plus_run_id",
        },
        independent_variable="repository_source",
        dependent_variables=[
            "average_champion_strength",
            "strongest_team_championship_rate",
            "champion_variance",
            "upset_rate",
        ],
        metrics=metrics,
        metadata={
            "version": "3",
            "program": "competition_research",
            "status": "experiment",
        },
    )


def build_condition(format_name: str, repository_source: str) -> ExperimentCondition:
    return ExperimentCondition(
        name=f"{format_name} production condition",
        competition_format=format_name,
        repository_source=repository_source,
        match_engine="production_scoreline_first",
        simulation_count=SIMULATION_COUNT,
        random_seed=BASE_SEED,
        parameters={
            "team_count": len(TEAMS),
            "experiment": "032",
        },
    )


def simulate_league_match(model, team1: str, team2: str, stage: str, match_id: str) -> MatchResult:
    goals1, goals2 = model.simulate_match(team1, team2)

    return MatchResult(
        team1=team1,
        team2=team2,
        goals_team1=goals1,
        goals_team2=goals2,
        stage=stage,
        match_id=match_id,
        metadata={
            "experiment": "032",
            "production_model": True,
        },
    )


def simulate_knockout_match(model, team1: str, team2: str, rng: random.Random, stage: str, match_id: str) -> MatchResult:
    goals1, goals2 = model.simulate_match(team1, team2)

    if goals1 != goals2:
        return MatchResult(
            team1=team1,
            team2=team2,
            goals_team1=goals1,
            goals_team2=goals2,
            stage=stage,
            match_id=match_id,
            metadata={
                "experiment": "032",
                "production_model": True,
                "resolved_by": "regulation",
            },
        )

    strength1 = team_strength(model.team_repository[team1])
    strength2 = team_strength(model.team_repository[team2])
    p_team1 = strength1 / (strength1 + strength2)

    if rng.random() < p_team1:
        goals1 += 1
        resolved_winner = team1
    else:
        goals2 += 1
        resolved_winner = team2

    return MatchResult(
        team1=team1,
        team2=team2,
        goals_team1=goals1,
        goals_team2=goals2,
        stage=stage,
        match_id=match_id,
        metadata={
            "experiment": "032",
            "production_model": True,
            "resolved_by": "extra_time_or_penalty_proxy",
            "resolved_winner": resolved_winner,
        },
    )


def build_league_competition(model, run_id: int) -> Competition:
    matches: list[MatchResult] = []
    match_number = 1

    for i, team1 in enumerate(TEAMS):
        for j, team2 in enumerate(TEAMS):
            if i >= j:
                continue

            matches.append(
                simulate_league_match(
                    model=model,
                    team1=team1,
                    team2=team2,
                    stage="League Stage",
                    match_id=f"032_league_run_{run_id}_match_{match_number}",
                )
            )
            match_number += 1

    stage = Stage(
        name="League Stage",
        stage_type=StageType.LEAGUE,
        participants=TEAMS,
        matches=matches,
        metadata={
            "experiment": "032",
            "format": "single_round_robin_league",
            "run_id": run_id,
        },
    )

    return Competition(
        name="Experiment 032 League",
        participants=TEAMS,
        stages=[stage],
    )


def build_knockout_competition(model, run_id: int, rng: random.Random) -> Competition:
    builder = BracketBuilder()
    engine = CompetitionEngine()

    stages: list[Stage] = []
    current_teams = TEAMS[:]

    for round_name in ["Quarterfinals", "Semifinals", "Final"]:
        bracket = builder.build_high_low_bracket(
            teams=current_teams,
            name=f"{round_name} Bracket",
        )

        for tie_index, tie in enumerate(bracket.ties, start=1):
            tie.add_match_result(
                simulate_knockout_match(
                    model=model,
                    team1=tie.team1,
                    team2=tie.team2,
                    rng=rng,
                    stage=round_name,
                    match_id=f"032_knockout_run_{run_id}_{round_name}_{tie_index}",
                )
            )

        stage = Stage(
            name=round_name,
            stage_type=StageType.FINAL if round_name == "Final" else StageType.KNOCKOUT,
            participants=current_teams,
            matches=bracket.as_stage_matches(),
            metadata={
                "experiment": "032",
                "format": "seeded_knockout",
                "run_id": run_id,
            },
        )

        temp_result = engine.resolve(
            Competition(
                name=f"Experiment 032 {round_name}",
                participants=current_teams,
                stages=[stage],
            )
        )

        current_teams = temp_result.stage_results[0].qualifiers
        stages.append(stage)

    return Competition(
        name="Experiment 032 Knockout",
        participants=TEAMS,
        stages=stages,
    )


def run_condition(format_name: str, repository_source: str) -> ExperimentResult:
    condition = build_condition(format_name, repository_source)
    model = FootballModelAdapter().from_condition(condition)

    strengths = {
        team: team_strength(model.team_repository[team])
        for team in TEAMS
    }

    strongest_team = max(strengths, key=strengths.get)

    experiment_result = ExperimentResult(
        experiment_name="Repository Sensitivity Analysis",
        format_name=format_name,
        strongest_team=strongest_team,
        team_strengths=strengths,
        metadata={
            "condition": condition.summary,
        },
    )

    engine = CompetitionEngine()
    adapter = CompetitionAdapter(team_strengths=strengths)

    for run_id in range(1, SIMULATION_COUNT + 1):
        random.seed(BASE_SEED + run_id)
        np.random.seed(BASE_SEED + run_id)
        rng = random.Random(BASE_SEED + run_id)

        if format_name == "single_round_robin_league":
            competition = build_league_competition(model, run_id)
        elif format_name == "seeded_knockout":
            competition = build_knockout_competition(model, run_id, rng)
        else:
            raise ValueError(f"Unknown format: {format_name}")

        competition_result = engine.resolve(competition)

        experiment_result.add_run(
            adapter.to_experiment_run_result(
                competition_result=competition_result,
                experiment_name="Repository Sensitivity Analysis",
                format_name=format_name,
                run_id=run_id,
            )
        )

    return experiment_result


def print_report(report) -> None:
    print(f"Format: {report.format_name}")
    print(f"Runs: {report.simulation_count}")
    print()

    for row in report.as_rows():
        print(f"{row['metric_name']}: {row['metric_value']}")

    print()

def report_to_metric_dict(report) -> dict:
    return {
        row["metric_name"]: row["metric_value"]
        for row in report.as_rows()
    }


def write_results_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "repository_source",
        "format_name",
        "average_champion_strength",
        "strongest_team_championship_rate",
        "champion_variance",
        "upset_rate",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Experiment 032 Results",
        "",
        "| Repository | Format | Avg Champion Strength | Strongest Team Rate | Champion Variance | Upset Rate |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['repository_source']} | "
            f"{row['format_name']} | "
            f"{row['average_champion_strength']:.4f} | "
            f"{row['strongest_team_championship_rate']:.3f} | "
            f"{row['champion_variance']} | "
            f"{row['upset_rate']:.3f} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_metadata_json(path: Path, experiment: Experiment) -> None:
    import json

    metadata = {
        "experiment_id": experiment.experiment_id,
        "title": experiment.title,
        "research_question": experiment.research_question,
        "hypothesis": experiment.hypothesis,
        "fixed_variables": experiment.fixed_variables,
        "independent_variable": experiment.independent_variable,
        "dependent_variables": experiment.dependent_variables,
        "metrics": experiment.metric_names(),
    }

    path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

def main() -> None:
    metrics = [
        AverageChampionStrengthMetric(),
        StrongestTeamChampionshipRateMetric(),
        ChampionVarianceMetric(),
        UpsetRateMetric(),
    ]

    experiment = build_experiment_definition(metrics)
    runner = ExperimentRunner(metrics=metrics)

    print(f"Experiment {experiment.experiment_id}: {experiment.title}")
    print()
    print(f"Research question: {experiment.research_question}")
    print()
    print(f"Hypothesis: {experiment.hypothesis}")
    print()

    print("Metric Report")
    print("=============")
    print()

    output_dir = Path("outputs") / "experiment_032_repository_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "repository_sensitivity_results.csv"
    results_md = output_dir / "repository_sensitivity_results.md"
    metadata_json = output_dir / "experiment_metadata.json"

    rows = []

    for repository_source in REPOSITORY_SOURCES:
        print("=" * 70)
        print(f"Repository: {repository_source}")
        print("=" * 70)
        print()
        print("-" * (len("Repository source: ") + len(repository_source)))

        league_result = run_condition(
            format_name="single_round_robin_league",
            repository_source=repository_source,
        )

        knockout_result = run_condition(
            format_name="seeded_knockout",
            repository_source=repository_source,
        )

        league_report = runner.evaluate(league_result)
        knockout_report = runner.evaluate(knockout_result)

        print_report(league_report)
        print_report(knockout_report)

        league_metrics = report_to_metric_dict(league_report)
        knockout_metrics = report_to_metric_dict(knockout_report)

        rows.append({
            "repository_source": repository_source,
            "format_name": "single_round_robin_league",
            **league_metrics,
        })

        rows.append({
            "repository_source": repository_source,
            "format_name": "seeded_knockout",
            **knockout_metrics,
        })
    
    write_results_csv(results_csv, rows)
    write_results_markdown(results_md, rows)
    write_metadata_json(metadata_json, experiment)

    print()
    print(f"Results CSV written to: {results_csv}")
    print(f"Results Markdown written to: {results_md}")
    print(f"Metadata written to: {metadata_json}")

if __name__ == "__main__":
    main()
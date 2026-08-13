#benchmark_hierarchical_goal_samplers.py

#benchmark_hierarchical_stochastic_lambda.py

from pathlib import Path

import numpy as np
import pandas as pd

from simulation.lambda_models import expected_goals
from simulation.goal_samplers import (
    poisson_sampler,
    hierarchical_stochastic_lambda_sampler,
    draw_calibrated_hierarchical_sampler,
    draw_tempered_lambda_sampler,
    hierarchical_bivariate_poisson_sampler,
    dixon_coles_hierarchical_sampler,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv"

N = 1000
LAMBDA_MODEL = "calibrated"

SHARED_FRACTIONS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]

DRAW_STRENGTHS = [
    0.00,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
]

DIXON_COLES_RHOS = [
    0.10,
    0.20,
    0.30,
]

EXPERIMENTS = [
    {
        "label": "hierarchical",
        "mode": "hierarchical",
        "tempo_cv": 0.60,
        "team_cv": 0.10,
    }
]

DISPLAY_COLUMNS = [
    "model",
    "relative_composite_error",
    "composite_error",
    "avg_total_goals",
    "total_var",
    "draw_rate",
    "clean_sheet_rate",
    "five_plus_total_rate",
    "six_plus_total_rate",
]

for shared_fraction in SHARED_FRACTIONS:
    EXPERIMENTS.append(
        {
            "label": f"bivariate_shared={shared_fraction:.2f}",
            "mode": "hierarchical_bivariate",
            "tempo_cv": 0.60,
            "team_cv": 0.10,
            "shared_fraction": shared_fraction,
        }
    )

for strength in DRAW_STRENGTHS:
    EXPERIMENTS.append(
        {
            "label": f"draw_calibrated_draw={strength:.2f}",
            "mode": "draw_calibrated",
            "tempo_cv": 0.60,
            "team_cv": 0.10,
            "draw_strength": strength,
        }
    )

    EXPERIMENTS.append(
        {
            "label": f"draw_tempered_lambda_draw={strength:.2f}",
            "mode": "draw_tempered_lambda",
            "tempo_cv": 0.60,
            "team_cv": 0.10,
            "draw_strength": strength,
        }
    )

for rho in DIXON_COLES_RHOS:
    EXPERIMENTS.append(
        {
            "label": f"dixon_coles_rho={rho:.2f}",
            "mode": "dixon_coles",
            "tempo_cv": 0.60,
            "team_cv": 0.10,
            "rho": rho,
        }
    )

def row_to_team_dict(row, prefix):
    return {
        "attack": row[f"{prefix}_attack"],
        "defense": row[f"{prefix}_defense"],
        "poisson_attack": row[f"{prefix}_poisson_attack"],
        "poisson_defense": row[f"{prefix}_poisson_defense"],
        "fifa_points": row[f"{prefix}_fifa_points"],
    }


def summarize(label, home_goals, away_goals):
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    total_goals = home_goals + away_goals

    return {
        "model": label,
        "avg_total_goals": total_goals.mean(),
        "total_var": total_goals.var(ddof=1),
        "draw_rate": np.mean(home_goals == away_goals),
        "clean_sheet_rate": np.mean((home_goals == 0) | (away_goals == 0)),
        "five_plus_total_rate": np.mean(total_goals >= 5),
        "six_plus_total_rate": np.mean(total_goals >= 6),
    }


def simulate(df, experiment):
    home_scores = []
    away_scores = []

    mode = experiment["mode"]
    tempo_cv = experiment.get("tempo_cv")
    team_cv = experiment.get("team_cv")
    draw_strength = experiment.get("draw_strength")
    rho = experiment.get("rho")

    for _, row in df.iterrows():
        home = row_to_team_dict(row, "home")
        away = row_to_team_dict(row, "away")

        lambda_home, lambda_away = expected_goals(
            home,
            away,
            lambda_model=LAMBDA_MODEL,
        )

        for _ in range(N):
            if mode == "poisson":
                goals_home = poisson_sampler(lambda_home)
                goals_away = poisson_sampler(lambda_away)

            elif mode == "hierarchical":
                goals_home, goals_away = hierarchical_stochastic_lambda_sampler(
                    lambda_home,
                    lambda_away,
                    tempo_cv,
                    team_cv,
                )

            elif mode == "draw_calibrated":
                goals_home, goals_away = draw_calibrated_hierarchical_sampler(
                    lambda_home,
                    lambda_away,
                    tempo_cv,
                    team_cv,
                    draw_strength=draw_strength,
                )
            
            elif mode == "draw_tempered_lambda":
                goals_home, goals_away = draw_tempered_lambda_sampler(
                    lambda_home,
                    lambda_away,
                    tempo_cv,
                    team_cv,
                    draw_strength=draw_strength,
                )
            
            elif mode == "hierarchical_bivariate":
                goals_home, goals_away = hierarchical_bivariate_poisson_sampler(
                    lambda_home,
                    lambda_away,
                    tempo_cv,
                    team_cv,
                    shared_fraction=experiment["shared_fraction"],
                )
            
            elif mode == "dixon_coles":
                goals_home, goals_away = dixon_coles_hierarchical_sampler(
                    lambda_home,
                    lambda_away,
                    tempo_cv,
                    team_cv,
                    rho=rho,
                    max_goals=10,
                )

            else:
                raise ValueError(f"Unknown mode: {mode}")

            home_scores.append(goals_home)
            away_scores.append(goals_away)

    return home_scores, away_scores


def main() -> None:
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)

    rows = []

    rows.append(
        summarize(
            "historical",
            df["home_score"],
            df["away_score"],
        )
    )

    home_scores, away_scores = simulate(
        df,
        {
            "label": "poisson",
            "mode": "poisson",
        },
    )
    rows.append(summarize("poisson", home_scores, away_scores))

    for experiment in EXPERIMENTS:
        home_scores, away_scores = simulate(df, experiment)
        rows.append(
            summarize(
                experiment["label"],
                home_scores,
                away_scores,
            )
        )

    results = pd.DataFrame(rows)

    historical_row = results.loc[results["model"] == "historical"].iloc[0]

    METRIC_COLUMNS = [
        "avg_total_goals",
        "total_var",
        "draw_rate",
        "clean_sheet_rate",
        "five_plus_total_rate",
        "six_plus_total_rate",
    ]

    for metric in METRIC_COLUMNS:
        results[f"{metric}_error"] = (
            results[metric] - historical_row[metric]
        ).abs()

    EPSILON = 1e-12

    for metric in METRIC_COLUMNS:
        historical_value = historical_row[metric]

        results[f"{metric}_relative_error"] = (
            results[f"{metric}_error"] /
            max(abs(historical_value), EPSILON)
        )

    results["composite_error"] = results[
        [f"{metric}_error" for metric in METRIC_COLUMNS]
    ].sum(axis=1)

    results["relative_composite_error"] = results[
        [f"{metric}_relative_error" for metric in METRIC_COLUMNS]
    ].sum(axis=1)

    results_ranked = results.sort_values("relative_composite_error")
    
    OUTPUT_PATH = (
        PROJECT_ROOT
        / "outputs"
        / "benchmarks"
        / "hierarchical_goal_sampler_benchmark.csv"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_ranked.to_csv(OUTPUT_PATH, index=False)

    print("Hierarchical Goal Sampler Benchmark")
    print("-----------------------------------")
    print(f"Historical matches: {len(df)}")
    print(f"Samples per match: {N}")
    print()
    print(results_ranked[DISPLAY_COLUMNS].round(3).to_string(index=False))

    


if __name__ == "__main__":
    main()
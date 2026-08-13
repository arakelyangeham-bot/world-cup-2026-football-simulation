#benchmark_score_model_fitness.py

from pathlib import Path

import numpy as np
import pandas as pd

from simulation.lambda_models import expected_goals


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "historical_training_dataset.csv"
)

N = 1000

MODELS = [
    "heuristic",
    "calibrated",
]


def row_to_team_dict(row, prefix):
    return {
        "attack": row[f"{prefix}_attack"],
        "defense": row[f"{prefix}_defense"],
        "poisson_attack": row[f"{prefix}_poisson_attack"],
        "poisson_defense": row[f"{prefix}_poisson_defense"],
        "fifa_points": row[f"{prefix}_fifa_points"],
    }


def summarize_scores(home_goals, away_goals):
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)

    total_goals = home_goals + away_goals
    goal_diff = home_goals - away_goals

    return {
        "avg_home_goals": home_goals.mean(),
        "avg_away_goals": away_goals.mean(),
        "avg_total_goals": total_goals.mean(),
        "avg_goal_diff": goal_diff.mean(),
        "home_var": home_goals.var(ddof=1),
        "away_var": away_goals.var(ddof=1),
        "total_var": total_goals.var(ddof=1),
        "goal_diff_var": goal_diff.var(ddof=1),
        "draw_rate": np.mean(home_goals == away_goals),
        "home_win_rate": np.mean(home_goals > away_goals),
        "away_win_rate": np.mean(away_goals > home_goals),
        "clean_sheet_rate": np.mean((home_goals == 0) | (away_goals == 0)),
        "five_plus_total_rate": np.mean(total_goals >= 5),
        "six_plus_total_rate": np.mean(total_goals >= 6),
    }


def simulate_model(df, model_name):
    home_scores = []
    away_scores = []

    for _, row in df.iterrows():
        home = row_to_team_dict(row, "home")
        away = row_to_team_dict(row, "away")

        lambda_home, lambda_away = expected_goals(
            home,
            away,
            lambda_model=model_name,
        )

        for _ in range(N):
            home_scores.append(int(np.random.poisson(lambda_home)))
            away_scores.append(int(np.random.poisson(lambda_away)))

    return summarize_scores(home_scores, away_scores)


def normalized_absolute_error(model_value, historical_value):
    if historical_value == 0:
        return np.nan

    return abs(model_value - historical_value) / abs(historical_value)


def main() -> None:
    np.random.seed(42)

    df = pd.read_csv(DATASET_PATH)

    historical = summarize_scores(
        df["home_score"],
        df["away_score"],
    )

    rows = []

    for model_name in MODELS:
        model_summary = simulate_model(df, model_name)

        metric_errors = []

        for metric, historical_value in historical.items():
            model_value = model_summary[metric]
            error = model_value - historical_value
            abs_error = abs(error)
            norm_error = normalized_absolute_error(
                model_value,
                historical_value,
            )

            metric_errors.append(norm_error)

            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "historical": historical_value,
                    "model_value": model_value,
                    "error": error,
                    "abs_error": abs_error,
                    "normalized_abs_error": norm_error,
                    "ratio": (
                        model_value / historical_value
                        if historical_value != 0
                        else np.nan
                    ),
                }
            )

        rows.append(
            {
                "model": model_name,
                "metric": "overall_fitness",
                "historical": 0.0,
                "model_value": 0.0,
                "error": 0.0,
                "abs_error": 0.0,
                "normalized_abs_error": np.nanmean(metric_errors),
                "ratio": np.nan,
            }
        )

    results = pd.DataFrame(rows)

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "model_training"
        / "score_model_fitness.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("Score Model Fitness Benchmark")
    print("-----------------------------")
    print(f"Historical matches: {len(df)}")
    print(f"Samples per match: {N}")
    print()

    display = results[
        results["metric"].isin(
            [
                "avg_total_goals",
                "total_var",
                "goal_diff_var",
                "draw_rate",
                "clean_sheet_rate",
                "five_plus_total_rate",
                "six_plus_total_rate",
                "overall_fitness",
            ]
        )
    ]

    print(
        display[
            [
                "model",
                "metric",
                "historical",
                "model_value",
                "error",
                "normalized_abs_error",
                "ratio",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print()
    print(f"Wrote -> {output_path}")


if __name__ == "__main__":
    main()
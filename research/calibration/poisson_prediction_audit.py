#poisson_prediction_audit.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "model_training"
    / "poisson_goal_model_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scoreline_first_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_OUTPUT_PATH = OUTPUT_DIR / "poisson_prediction_audit_overall.csv"
COMPETITION_OUTPUT_PATH = OUTPUT_DIR / "poisson_prediction_audit_by_competition.csv"
FIFA_BIN_OUTPUT_PATH = OUTPUT_DIR / "poisson_prediction_audit_by_fifa_bin.csv"
PRED_DIFF_BIN_OUTPUT_PATH = OUTPUT_DIR / "poisson_prediction_audit_by_pred_diff_bin.csv"


def add_residuals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["home_goal_error"] = df["pred_home_goals"] - df["home_score"]
    df["away_goal_error"] = df["pred_away_goals"] - df["away_score"]
    df["total_goal_error"] = df["pred_total_goals"] - df["total_goals"]
    df["goal_diff_error"] = df["pred_goal_diff"] - df["goal_diff"]

    return df


def summarize(df: pd.DataFrame) -> dict:
    return {
        "matches": len(df),

        "actual_home_goals": df["home_score"].mean(),
        "pred_home_goals": df["pred_home_goals"].mean(),
        "home_goal_error": df["home_goal_error"].mean(),
        "home_goal_abs_error": df["home_goal_error"].abs().mean(),

        "actual_away_goals": df["away_score"].mean(),
        "pred_away_goals": df["pred_away_goals"].mean(),
        "away_goal_error": df["away_goal_error"].mean(),
        "away_goal_abs_error": df["away_goal_error"].abs().mean(),

        "actual_total_goals": df["total_goals"].mean(),
        "pred_total_goals": df["pred_total_goals"].mean(),
        "total_goal_error": df["total_goal_error"].mean(),
        "total_goal_abs_error": df["total_goal_error"].abs().mean(),

        "actual_goal_diff": df["goal_diff"].mean(),
        "pred_goal_diff": df["pred_goal_diff"].mean(),
        "goal_diff_error": df["goal_diff_error"].mean(),
        "goal_diff_abs_error": df["goal_diff_error"].abs().mean(),
    }


def grouped_summary(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    rows = []

    for group_value, group in df.groupby(group_col, dropna=False):
        row = {
            group_col: group_value,
        }
        row.update(summarize(group))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("matches", ascending=False)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df = add_residuals(df)

    overall = pd.DataFrame([summarize(df)])
    overall.to_csv(OVERALL_OUTPUT_PATH, index=False)

    if "competition" in df.columns:
        by_competition = grouped_summary(df, "competition")
        by_competition.to_csv(COMPETITION_OUTPUT_PATH, index=False)

    df["fifa_points_diff_bin"] = pd.cut(
        df["fifa_points_diff"],
        bins=[-1000, -300, -100, 100, 300, 1000],
        labels=[
            "away_strong_300_plus",
            "away_strong_100_300",
            "balanced_minus100_100",
            "home_strong_100_300",
            "home_strong_300_plus",
        ],
    )

    by_fifa_bin = grouped_summary(df, "fifa_points_diff_bin")
    by_fifa_bin.to_csv(FIFA_BIN_OUTPUT_PATH, index=False)

    df["pred_goal_diff_bin"] = pd.cut(
        df["pred_goal_diff"],
        bins=[-10, -2, -1, 1, 2, 10],
        labels=[
            "away_by_2_plus",
            "away_by_1_to_2",
            "balanced_pred",
            "home_by_1_to_2",
            "home_by_2_plus",
        ],
    )

    by_pred_diff_bin = grouped_summary(df, "pred_goal_diff_bin")
    by_pred_diff_bin.to_csv(PRED_DIFF_BIN_OUTPUT_PATH, index=False)

    print("Poisson Prediction Audit")
    print("------------------------")
    print()
    print("Overall")
    print(overall.round(6).to_string(index=False))
    print()
    print(f"Wrote overall       -> {OVERALL_OUTPUT_PATH}")
    print(f"Wrote competition   -> {COMPETITION_OUTPUT_PATH}")
    print(f"Wrote FIFA bins     -> {FIFA_BIN_OUTPUT_PATH}")
    print(f"Wrote pred diff bins -> {PRED_DIFF_BIN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
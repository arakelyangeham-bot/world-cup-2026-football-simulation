#historical_match_engine_validation.py

from pathlib import Path
import math

import pandas as pd

from match_engine import poisson_expected_goals
from team_strength_loader import load_poisson_team_strengths
from team_name_normalizer import normalize_team_name


RAW_DIR = Path("data/raw/sofascore")
OUTPUT_DIR = Path("outputs/match_engine")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2010, 2014, 2018, 2022]


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def outcome_probabilities(lambda_a: float, lambda_b: float, max_goals: int = 10) -> dict:
    a_win = 0.0
    draw = 0.0
    b_win = 0.0

    for goals_a in range(max_goals + 1):
        p_a = poisson_pmf(goals_a, lambda_a)

        for goals_b in range(max_goals + 1):
            p_b = poisson_pmf(goals_b, lambda_b)
            p = p_a * p_b

            if goals_a > goals_b:
                a_win += p
            elif goals_b > goals_a:
                b_win += p
            else:
                draw += p

    total = a_win + draw + b_win

    return {
        "team_a_win": a_win / total,
        "draw": draw / total,
        "team_b_win": b_win / total,
    }


def actual_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "team_a_win"
    if away_score > home_score:
        return "team_b_win"
    return "draw"


def main():
    strengths = load_poisson_team_strengths()
    rows = []
    skipped = []

    for year in YEARS:
        path = RAW_DIR / f"wc_{year}_match_results.csv"
        df = pd.read_csv(path)

        for _, match in df.iterrows():
            raw_home = match["home_team"]
            raw_away = match["away_team"]

            home = normalize_team_name(raw_home)
            away = normalize_team_name(raw_away)

            if home not in strengths or away not in strengths:
                skipped.append({
                    "year": year,
                    "raw_home_team": raw_home,
                    "raw_away_team": raw_away,
                    "normalized_home_team": home,
                    "normalized_away_team": away,
                    "reason": "missing_team_strength",
                })
                continue

            lambda_home, lambda_away = poisson_expected_goals(
                strengths[home],
                strengths[away],
            )

            probs = outcome_probabilities(lambda_home, lambda_away)
            result = actual_result(
                int(match["home_score"]),
                int(match["away_score"]),
            )

            predicted_prob_actual = max(probs[result], 1e-12)

            rows.append({
                "year": year,
                "event_id": match["event_id"],
                "home_team": home,
                "away_team": away,
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "p_home_win": probs["team_a_win"],
                "p_draw": probs["draw"],
                "p_away_win": probs["team_b_win"],
                "actual_result": result,
                "predicted_prob_actual": predicted_prob_actual,
                "log_loss": -math.log(predicted_prob_actual),
                "brier_score": (
                    (probs["team_a_win"] - (1 if result == "team_a_win" else 0)) ** 2
                    + (probs["draw"] - (1 if result == "draw" else 0)) ** 2
                    + (probs["team_b_win"] - (1 if result == "team_b_win" else 0)) ** 2
                ),
            })

    results = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)

    results_file = OUTPUT_DIR / "historical_match_engine_validation.csv"
    skipped_file = OUTPUT_DIR / "historical_match_engine_skipped.csv"
    summary_file = OUTPUT_DIR / "historical_match_engine_validation_summary.csv"

    results.to_csv(results_file, index=False)
    skipped_df.to_csv(skipped_file, index=False)

    summary = pd.DataFrame([{
        "matches_evaluated": len(results),
        "matches_skipped": len(skipped_df),
        "avg_log_loss": results["log_loss"].mean() if len(results) else None,
        "avg_brier_score": results["brier_score"].mean() if len(results) else None,
        "avg_predicted_home_lambda": results["lambda_home"].mean() if len(results) else None,
        "avg_predicted_away_lambda": results["lambda_away"].mean() if len(results) else None,
        "avg_actual_home_goals": results["home_score"].mean() if len(results) else None,
        "avg_actual_away_goals": results["away_score"].mean() if len(results) else None,
    }])

    summary.to_csv(summary_file, index=False)

    print("Historical Match Engine Validation")
    print("----------------------------------")
    print(summary.to_string(index=False))

    print()
    print(f"Saved -> {results_file}")
    print(f"Saved -> {summary_file}")
    print(f"Saved -> {skipped_file}")


if __name__ == "__main__":
    main()
# historical_wc_analysis.py

from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw/sofascore")
OUTPUT_DIR = Path("outputs/historical_world_cups")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOURNAMENTS = [2010, 2014, 2018, 2022]


def classify_stage(stage: str) -> str:
    stage = str(stage).lower()

    if "group" in stage:
        return "group"
    return "knockout"


def summarize_tournament(year: int, df: pd.DataFrame) -> dict:
    df = df.copy()

    df["total_goals"] = df["home_score"] + df["away_score"]
    df["is_draw"] = df["home_score"] == df["away_score"]
    df["stage_type"] = df["stage"].apply(classify_stage)

    group_df = df[df["stage_type"] == "group"]
    knockout_df = df[df["stage_type"] == "knockout"]

    return {
        "year": year,
        "matches": len(df),
        "total_goals": int(df["total_goals"].sum()),
        "goals_per_match": df["total_goals"].mean(),
        "group_matches": len(group_df),
        "group_goals_per_match": group_df["total_goals"].mean(),
        "group_draw_rate": group_df["is_draw"].mean(),
        "knockout_matches": len(knockout_df),
        "knockout_goals_per_match": knockout_df["total_goals"].mean(),
        "zero_zero_rate": ((df["home_score"] == 0) & (df["away_score"] == 0)).mean(),
        "three_plus_goal_rate": (df["total_goals"] >= 3).mean(),
        "avg_margin": (df["home_score"] - df["away_score"]).abs().mean(),
    }


def main():
    rows = []

    for year in TOURNAMENTS:
        path = RAW_DIR / f"wc_{year}_match_results.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)

        if len(df) != 64:
            raise ValueError(f"{year}: expected 64 matches, got {len(df)}")

        rows.append(summarize_tournament(year, df))

    summary = pd.DataFrame(rows)

    average_row = {
        "year": "average",
        "matches": summary["matches"].mean(),
        "total_goals": summary["total_goals"].mean(),
        "goals_per_match": summary["goals_per_match"].mean(),
        "group_matches": summary["group_matches"].mean(),
        "group_goals_per_match": summary["group_goals_per_match"].mean(),
        "group_draw_rate": summary["group_draw_rate"].mean(),
        "knockout_matches": summary["knockout_matches"].mean(),
        "knockout_goals_per_match": summary["knockout_goals_per_match"].mean(),
        "zero_zero_rate": summary["zero_zero_rate"].mean(),
        "three_plus_goal_rate": summary["three_plus_goal_rate"].mean(),
        "avg_margin": summary["avg_margin"].mean(),
    }

    summary_with_avg = pd.concat(
        [summary, pd.DataFrame([average_row])],
        ignore_index=True,
    )

    out_file = OUTPUT_DIR / "historical_world_cup_summary.csv"
    summary_with_avg.to_csv(out_file, index=False)

    print("Historical World Cup Summary")
    print("----------------------------")
    print(summary_with_avg.to_string(index=False))

    print()
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
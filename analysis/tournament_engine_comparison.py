# tournament_engine_comparison.py

import json
from pathlib import Path

import pandas as pd

from shared.config import PROJECT_ROOT


POISSON_DIR = PROJECT_ROOT / "outputs" / "monte_carlo_poisson"
ML_DIR = PROJECT_ROOT / "outputs" / "monte_carlo_ml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparison"

STAGES = [
    "champion",
    "runner_up",
    "semifinal",
    "quarterfinal",
    "round_of_16",
]


def load_probability_table(output_dir, stage):
    path = output_dir / f"{stage}_probabilities.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def compare_stage(stage):
    poisson = load_probability_table(POISSON_DIR, stage)
    ml = load_probability_table(ML_DIR, stage)

    poisson = poisson.rename(
        columns={
            "count": "poisson_count",
            "probability": "poisson_probability",
        }
    )

    ml = ml.rename(
        columns={
            "count": "ml_count",
            "probability": "ml_probability",
        }
    )

    merged = poisson.merge(
        ml,
        on="team",
        how="outer",
    ).fillna(0)

    merged["stage"] = stage
    merged["probability_delta"] = (
        merged["ml_probability"] - merged["poisson_probability"]
    )
    merged["abs_probability_delta"] = merged["probability_delta"].abs()

    return merged.sort_values(
        "abs_probability_delta",
        ascending=False,
    )


def load_statistics(output_dir):
    path = output_dir / "simulation_statistics.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path).iloc[0]


def compare_statistics():
    poisson = load_statistics(POISSON_DIR)
    ml = load_statistics(ML_DIR)

    rows = []

    for metric in poisson.index:
        poisson_value = poisson[metric]
        ml_value = ml[metric]

        try:
            delta = ml_value - poisson_value
        except TypeError:
            delta = None

        rows.append(
            {
                "metric": metric,
                "poisson": poisson_value,
                "ml": ml_value,
                "delta": delta,
            }
        )

    return pd.DataFrame(rows)


def summarize_biggest_changes(stage_df, stage, n=10):
    stage_only = stage_df[stage_df["stage"] == stage].copy()

    biggest_gainers = stage_only.sort_values(
        "probability_delta",
        ascending=False,
    ).head(n)

    biggest_losers = stage_only.sort_values(
        "probability_delta",
        ascending=True,
    ).head(n)

    return biggest_gainers, biggest_losers


def write_markdown(all_stage_df, stats_df):
    champion_gainers, champion_losers = summarize_biggest_changes(
        all_stage_df,
        "champion",
        n=10,
    )

    semifinal_gainers, semifinal_losers = summarize_biggest_changes(
        all_stage_df,
        "semifinal",
        n=10,
    )

    quarterfinal_gainers, quarterfinal_losers = summarize_biggest_changes(
        all_stage_df,
        "quarterfinal",
        n=10,
    )

    top_champion_changes = all_stage_df[
        all_stage_df["stage"] == "champion"
    ].head(20)

    markdown = f"""# Tournament Engine Comparison

## Purpose

This report compares the original Poisson-based tournament simulator against the machine-learning-guided simulator.

The ML simulator uses the calibrated LightGBM production model to guide match outcome probabilities while preserving Poisson scoreline generation.

## Champion Probability: Largest Changes

{top_champion_changes[[
    "team",
    "poisson_probability",
    "ml_probability",
    "probability_delta",
]].to_markdown(index=False)}

## Biggest Championship Probability Gainers

{champion_gainers[[
    "team",
    "poisson_probability",
    "ml_probability",
    "probability_delta",
]].to_markdown(index=False)}

## Biggest Championship Probability Losers

{champion_losers[[
    "team",
    "poisson_probability",
    "ml_probability",
    "probability_delta",
]].to_markdown(index=False)}

## Biggest Semifinal Probability Gainers

{semifinal_gainers[[
    "team",
    "poisson_probability",
    "ml_probability",
    "probability_delta",
]].to_markdown(index=False)}

## Biggest Semifinal Probability Losers

{semifinal_losers[[
    "team",
    "poisson_probability",
    "ml_probability",
    "probability_delta",
]].to_markdown(index=False)}

## Biggest Quarterfinal Probability Gainers

{quarterfinal_gainers[[
    "team",
    "poisson_probability",
    "ml_probability",
    "probability_delta",
]].to_markdown(index=False)}

## Biggest Quarterfinal Probability Losers

{quarterfinal_losers[[
    "team",
    "poisson_probability",
    "ml_probability",
    "probability_delta",
]].to_markdown(index=False)}

## Tournament Statistics Comparison

{stats_df.to_markdown(index=False)}

## Interpretation

The comparison highlights how replacing the original Poisson-only outcome engine with the calibrated ML-guided engine changes tournament-level forecasts.

Positive deltas indicate teams that became more likely under the ML-guided simulator. Negative deltas indicate teams that became less likely.

The tournament statistics table should be used to confirm that the ML-guided simulator still produces plausible tournament-wide behavior, especially goals per match, extra-time frequency, and penalty-shootout frequency.

## Next Step

If the ML-guided simulator produces realistic aggregate statistics and meaningful but explainable probability shifts, it should be treated as the preferred tournament engine.

The original Poisson simulator should remain available as a benchmark and fallback engine.
"""

    output_path = OUTPUT_DIR / "tournament_engine_comparison.md"
    output_path.write_text(markdown, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage_frames = []

    for stage in STAGES:
        stage_frames.append(compare_stage(stage))

    all_stage_df = pd.concat(stage_frames, ignore_index=True)

    stats_df = compare_statistics()

    stage_csv = OUTPUT_DIR / "tournament_engine_stage_comparison.csv"
    stats_csv = OUTPUT_DIR / "tournament_engine_statistics_comparison.csv"
    json_path = OUTPUT_DIR / "tournament_engine_comparison.json"

    all_stage_df.to_csv(stage_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)

    payload = {
        "stage_comparison": all_stage_df.to_dict(orient="records"),
        "statistics_comparison": stats_df.to_dict(orient="records"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    write_markdown(all_stage_df, stats_df)

    print("Tournament engine comparison complete.")
    print(f"Stage comparison CSV: {stage_csv}")
    print(f"Statistics comparison CSV: {stats_csv}")
    print(f"Markdown: {OUTPUT_DIR / 'tournament_engine_comparison.md'}")
    print(f"JSON: {json_path}")

    print()
    print("Top champion probability changes:")
    print(
        all_stage_df[all_stage_df["stage"] == "champion"][
            [
                "team",
                "poisson_probability",
                "ml_probability",
                "probability_delta",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
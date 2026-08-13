#football_gap_report.py

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_SCORELINE_FIRST = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "match_generation"
    / "scoreline_first_match_engine_benchmark.csv"
)

INPUT_SCORELINE_TVD = (
    PROJECT_ROOT
    / "outputs"
    / "benchmarks"
    / "scoreline_distribution_benchmark.csv"
)

INPUT_SCORELINE_FREQ = (
    PROJECT_ROOT
    / "outputs"
    / "benchmarks"
    / "scoreline_frequency_comparison.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research" / "football_gap_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "football_gap_report.md"
SUMMARY_PATH = OUTPUT_DIR / "football_gap_summary.csv"


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scoreline_first = pd.read_csv(INPUT_SCORELINE_FIRST)
    scoreline_tvd = pd.read_csv(INPUT_SCORELINE_TVD)
    scoreline_freq = pd.read_csv(INPUT_SCORELINE_FREQ)

    return scoreline_first, scoreline_tvd, scoreline_freq


def get_row(df: pd.DataFrame, model: str) -> pd.Series:
    rows = df[df["model"] == model]

    if rows.empty:
        raise ValueError(f"Missing model row: {model}")

    return rows.iloc[0]


def largest_frequency_gaps(freq: pd.DataFrame, model: str, n: int = 10) -> pd.DataFrame:
    error_col = f"{model}_error"
    abs_col = f"{model}_abs_error"

    if error_col not in freq.columns or abs_col not in freq.columns:
        raise ValueError(f"Missing frequency columns for model: {model}")

    cols = ["scoreline", "historical", model, error_col, abs_col]

    return (
        freq[cols]
        .sort_values(abs_col, ascending=False)
        .head(n)
        .copy()
    )


def write_report(
    scoreline_first: pd.DataFrame,
    scoreline_tvd: pd.DataFrame,
    scoreline_freq: pd.DataFrame,
) -> None:
    historical = get_row(scoreline_first, "historical")
    production = get_row(scoreline_first, "production_hybrid")
    scoreline_model = get_row(scoreline_first, "scoreline_first")

    best_tvd = (
        scoreline_tvd
        .sort_values("total_variation_distance")
        .iloc[0]
    )

    production_gaps = largest_frequency_gaps(
        scoreline_freq,
        model="dixon_coles_rho=0.30",
        n=10,
    )

    largest_gap = production_gaps.iloc[0]

    summary_rows = [
        {
            "metric": "production_scoreline_tvd",
            "value": production["scoreline_tvd"],
        },
        {
            "metric": "scoreline_first_scoreline_tvd",
            "value": scoreline_model["scoreline_tvd"],
        },
        {
            "metric": "historical_draw_rate",
            "value": historical["draw_rate"],
        },
        {
            "metric": "production_draw_rate",
            "value": production["draw_rate"],
        },
        {
            "metric": "scoreline_first_draw_rate",
            "value": scoreline_model["draw_rate"],
        },
        {
            "metric": "largest_scoreline_gap",
            "value": largest_gap["scoreline"],
        },
        {
            "metric": "largest_scoreline_gap_error",
            "value": largest_gap["dixon_coles_rho=0.30_error"],
        },
    ]

    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)

    lines = []

    lines.append("# Football Gap Report v1")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- Best scoreline TVD model in existing benchmark: "
        f"`{best_tvd['model']}` "
        f"({best_tvd['total_variation_distance']:.4f})."
    )
    lines.append(
        f"- Production hybrid scoreline TVD: "
        f"{production['scoreline_tvd']:.4f}."
    )
    lines.append(
        f"- Scoreline-first engine scoreline TVD: "
        f"{scoreline_model['scoreline_tvd']:.4f}."
    )
    lines.append(
        f"- Largest production scoreline frequency gap: "
        f"`{largest_gap['scoreline']}` "
        f"with error {largest_gap['dixon_coles_rho=0.30_error']:.4f}."
    )
    lines.append("")
    lines.append("## Scoreline Realism")
    lines.append("")
    lines.append(scoreline_first.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## Draw Behavior")
    lines.append("")
    lines.append(
        f"- Historical draw rate: {pct(historical['draw_rate'])}"
    )
    lines.append(
        f"- Production hybrid draw rate: {pct(production['draw_rate'])}"
    )
    lines.append(
        f"- Scoreline-first draw rate: {pct(scoreline_model['draw_rate'])}"
    )
    lines.append(
        f"- Historical 0-0 rate: {pct(historical['zero_zero_rate'])}"
    )
    lines.append(
        f"- Production 0-0 rate: {pct(production['zero_zero_rate'])}"
    )
    lines.append(
        f"- Scoreline-first 0-0 rate: {pct(scoreline_model['zero_zero_rate'])}"
    )
    lines.append(
        f"- Historical 1-1 rate: {pct(historical['one_one_rate'])}"
    )
    lines.append(
        f"- Production 1-1 rate: {pct(production['one_one_rate'])}"
    )
    lines.append(
        f"- Scoreline-first 1-1 rate: {pct(scoreline_model['one_one_rate'])}"
    )
    lines.append("")
    lines.append("## Largest Production Scoreline Frequency Gaps")
    lines.append("")
    lines.append(production_gaps.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## Tail Behavior")
    lines.append("")
    lines.append(
        f"- Historical five-plus total goals rate: "
        f"{pct(historical['five_plus_total_rate'])}"
    )
    lines.append(
        f"- Production five-plus total goals rate: "
        f"{pct(production['five_plus_total_rate'])}"
    )
    lines.append(
        f"- Scoreline-first five-plus total goals rate: "
        f"{pct(scoreline_model['five_plus_total_rate'])}"
    )
    lines.append("")
    lines.append("## Research Priority Recommendation")
    lines.append("")
    lines.append(
        "The next research project should focus on the largest scoreline "
        "frequency gaps identified above, especially if the same gaps appear "
        "across both the production hybrid and scoreline-first engine."
    )
    lines.append("")
    lines.append(
        "Candidate explanations to evaluate next include low-score dependence, "
        "draw behavior, match tempo, and goal-state effects."
    )
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote report  -> {REPORT_PATH}")
    print(f"Wrote summary -> {SUMMARY_PATH}")


def main() -> None:
    scoreline_first, scoreline_tvd, scoreline_freq = load_inputs()
    write_report(scoreline_first, scoreline_tvd, scoreline_freq)


if __name__ == "__main__":
    main()
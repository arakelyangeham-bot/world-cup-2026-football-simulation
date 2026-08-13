#compare_team_repositories.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

LEGACY_PATH = PROJECT_ROOT / "data" / "processed" / "wc_2026_team_strength.csv"

PLAYER_INTELLIGENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "player_intelligence"
    / "player_intelligence_team_repository.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "player_intelligence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "team_repository_comparison.csv"
TEAM_OVERLAP_OUTPUT_PATH = OUTPUT_DIR / "team_repository_overlap.csv"


FEATURES = [
    "att_composite",
    "mid_composite",
    "def_composite",
    "gk_composite",
    "poisson_attack_adj",
    "poisson_defense_adj",
]


def summarize_feature(df: pd.DataFrame, source: str, feature: str) -> dict:
    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    return {
        "source": source,
        "feature": feature,
        "count": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "p25": values.quantile(0.25),
        "median": values.quantile(0.50),
        "p75": values.quantile(0.75),
        "max": values.max(),
        "range": values.max() - values.min(),
    }


def main() -> None:
    legacy = pd.read_csv(LEGACY_PATH)
    player_intelligence = pd.read_csv(PLAYER_INTELLIGENCE_PATH)

    legacy["source"] = "legacy"
    player_intelligence["source"] = "player_intelligence"

    legacy_teams = set(legacy["nation"].dropna().astype(str))
    pi_teams = set(player_intelligence["nation"].dropna().astype(str))

    overlap_rows = []

    for team in sorted(legacy_teams | pi_teams):
        overlap_rows.append(
            {
                "team": team,
                "in_legacy": team in legacy_teams,
                "in_player_intelligence": team in pi_teams,
            }
        )

    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(TEAM_OVERLAP_OUTPUT_PATH, index=False)

    summary_rows = []

    for feature in FEATURES:
        if feature in legacy.columns:
            summary_rows.append(
                summarize_feature(
                    legacy,
                    "legacy",
                    feature,
                )
            )

        if feature in player_intelligence.columns:
            summary_rows.append(
                summarize_feature(
                    player_intelligence,
                    "player_intelligence",
                    feature,
                )
            )

    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("Team Repository Comparison")
    print("--------------------------")
    print(f"Legacy teams: {len(legacy_teams)}")
    print(f"Player Intelligence teams: {len(pi_teams)}")
    print(f"Team overlap: {len(legacy_teams & pi_teams)}")
    print(f"Only legacy: {len(legacy_teams - pi_teams)}")
    print(f"Only player intelligence: {len(pi_teams - legacy_teams)}")
    print()
    print(comparison.round(6).to_string(index=False))
    print()
    print(f"Wrote summary -> {SUMMARY_OUTPUT_PATH}")
    print(f"Wrote overlap -> {TEAM_OVERLAP_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
#run_player_intelligence_validation_suite.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPOSITORY_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_011_team_representation_calibration"
    / "repositories"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_011_team_representation_calibration"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "validation_suite_repository_summary.csv"

FEATURES = [
    "att_composite",
    "mid_composite",
    "def_composite",
    "gk_composite",
    "poisson_attack_adj",
    "poisson_defense_adj",
]


def summarize_feature(
    df: pd.DataFrame,
    strategy: str,
    feature: str,
) -> dict:
    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    return {
        "strategy": strategy,
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


def strategy_name_from_path(path: Path) -> str:
    return path.stem.replace("_team_repository", "")


def main() -> None:
    repository_paths = sorted(REPOSITORY_DIR.glob("*_team_repository.csv"))

    if not repository_paths:
        raise FileNotFoundError(
            f"No strategy repositories found in {REPOSITORY_DIR}"
        )

    rows = []

    print("Player Intelligence Validation Suite")
    print("------------------------------------")

    for path in repository_paths:
        strategy = strategy_name_from_path(path)
        df = pd.read_csv(path)

        print(f"Auditing {strategy}: {len(df)} teams")

        for feature in FEATURES:
            if feature not in df.columns:
                continue

            rows.append(
                summarize_feature(
                    df=df,
                    strategy=strategy,
                    feature=feature,
                )
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_PATH, index=False)

    print()
    print(summary.round(6).to_string(index=False))
    print()
    print(f"Wrote validation summary -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
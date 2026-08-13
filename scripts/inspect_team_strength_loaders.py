# inspect_team_strength_loaders.py

from pathlib import Path

import pandas as pd

from team_strength_loader import load_poisson_team_strengths


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_FILES = [
    PROJECT_ROOT / "data" / "processed" / "wc_2026_team_strength.csv",
    PROJECT_ROOT / "data" / "processed" / "wc_2026_team_model_params.csv",
    PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv",
]


def inspect_csv(path):
    print()
    print("=" * 80)
    print(path)
    print("=" * 80)

    if not path.exists():
        print("Missing")
        return

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print()
    print("Columns:")
    for col in df.columns:
        print(" -", col)

    print()
    print("Head:")
    print(df.head().to_string(index=False))


def inspect_poisson_loader():
    print()
    print("=" * 80)
    print("load_poisson_team_strengths()")
    print("=" * 80)

    strengths = load_poisson_team_strengths()

    print("Teams loaded:", len(strengths))

    sample_items = list(strengths.items())[:10]

    for team, values in sample_items:
        print(team, values)


def main():
    inspect_poisson_loader()

    for path in CANDIDATE_FILES:
        inspect_csv(path)


if __name__ == "__main__":
    main()
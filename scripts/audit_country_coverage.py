#audit_country_coverage.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "profiles": PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_profiles.csv",
    "player_dataset": PROJECT_ROOT / "data" / "processed" / "wc_2026_player_dataset.csv",
    "model_features": PROJECT_ROOT / "data" / "processed" / "wc_2026_model_features.csv",
    "player_ratings": PROJECT_ROOT / "data" / "processed" / "player_ratings.csv",
}

for name, path in FILES.items():
    print("=" * 60)
    print(name)
    print(path)

    df = pd.read_csv(path)

    print(f"Rows: {len(df):,}")

    if "country" not in df.columns:
        print("No country column found.")
        continue

    missing = df["country"].isna().sum()
    print(f"Missing country: {missing:,}")
    print(f"Missing country %: {missing / len(df):.2%}")

    print("\nTop countries:")
    print(df["country"].value_counts(dropna=False).head(25))

    if missing > 0:
        print("\nSample missing-country players:")
        cols = [c for c in ["player_id", "player", "team", "current_team", "source_competitions"] if c in df.columns]
        print(df[df["country"].isna()][cols].head(25))
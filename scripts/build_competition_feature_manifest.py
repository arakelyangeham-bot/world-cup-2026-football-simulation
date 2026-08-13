#build_competition_feature_manifest.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COVERAGE_FILE = PROJECT_ROOT / "outputs" / "competition_stat_coverage.csv"
OUT_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "competition_feature_manifest.csv"

MIN_COVERAGE = 0.50

coverage = pd.read_csv(COVERAGE_FILE)

rows = []

id_cols = ["competition", "season_year", "rows", "players"]

coverage_cols = [
    col for col in coverage.columns
    if col.endswith("_coverage")
]

for _, row in coverage.iterrows():
    for col in coverage_cols:
        feature = col.replace("_coverage", "")
        coverage_value = row[col]

        rows.append({
            "competition": row["competition"],
            "season_year": row["season_year"],
            "feature": feature,
            "coverage": coverage_value,
            "available": coverage_value >= MIN_COVERAGE,
        })

manifest = pd.DataFrame(rows)

manifest.to_csv(OUT_FILE, index=False)

print(manifest.head(50))
print(f"Wrote: {OUT_FILE}")
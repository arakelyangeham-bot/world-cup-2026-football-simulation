from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wc_2026_model_features.csv"
)

EDA_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "eda"
)

EDA_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

IS_TEST_DATASET = len(df) < 20

HAS_POSITION = "position" in df.columns
HAS_NATION = "nation" in df.columns
HAS_GROUP = "group" in df.columns
HAS_MINUTES = "minutesPlayed" in df.columns

if IS_TEST_DATASET:
    print("Small test dataset detected. Running schema checks only.")

if not HAS_POSITION:
    print("No position column found. Skipping position-based EDA.")

if not HAS_NATION:
    print("No nation column found. Skipping nation-based EDA.")

if not HAS_GROUP:
    print("No group column found. Skipping group-based EDA.")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print("\nPOSITION COUNTS")
if HAS_POSITION:
    print("\nPOSITION COUNTS")
    print(df["position"].value_counts(dropna=False))

print("\nGROUP COUNTS")
if HAS_GROUP:
    print("\nGROUP COUNTS")
    print(df["group"].value_counts(dropna=False))

print("\nPLAYERS WITH STATS")
if HAS_MINUTES:
    print("\nPLAYERS WITH STATS")
    print(df["minutesPlayed"].notna().sum())

print("\nMISSINGNESS")
missing = (
    df.isna()
      .mean()
      .sort_values(ascending=False)
)

missing.to_csv(
    EDA_DIR / "missingness.csv"
)

print(missing.head(25))

print("\nPER90 FEATURES")

per90_cols = [
    col for col in df.columns
    if col.endswith("_per90")
]

summary = (
    df[per90_cols]
    .describe()
    .T
)

summary.to_csv(
    EDA_DIR / "per90_summary.csv"
)

print(summary.head())

if HAS_POSITION:
    print("\nPOSITION AVERAGES")

    numeric_cols = df.select_dtypes(include="number").columns

    position_summary = (
        df.groupby("position")[numeric_cols]
          .mean()
    )

    position_summary.to_csv(
        EDA_DIR / "position_averages.csv"
    )

    print(position_summary.head())
else:
    print("\nSkipping position averages.")

numeric_cols = df.select_dtypes(include="number").columns

print("\nDone.")
#analyze_scoreline_errors.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "benchmarks" / "scoreline_frequency_comparison.csv"

df = pd.read_csv(INPUT_PATH)

print()
print("Largest remaining hierarchical errors")
print("-------------------------------------")
print(
    df.sort_values("hierarchical_abs_error", ascending=False)
    .head(20)
    .round(4)
    .to_string(index=False)
)

print()
print("Largest improvements over Poisson")
print("---------------------------------")
print(
    df.sort_values("improvement", ascending=False)
    .head(20)
    .round(4)
    .to_string(index=False)
)

print()
print("Largest regressions versus Poisson")
print("----------------------------------")
print(
    df.sort_values("improvement", ascending=True)
    .head(20)
    .round(4)
    .to_string(index=False)
)
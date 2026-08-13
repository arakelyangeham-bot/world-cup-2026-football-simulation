from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "outputs" / "model_training" / "historical_training_dataset.csv"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("Columns")
    print("-------")
    for col in df.columns:
        print(col)

    print()
    print("First 5 rows")
    print("------------")
    print(df.head().to_string())

    possible_goal_cols = [
        col for col in df.columns
        if "goal" in col.lower()
        or "score" in col.lower()
    ]

    print()
    print("Possible goal/score columns")
    print("---------------------------")
    for col in possible_goal_cols:
        print(col)


if __name__ == "__main__":
    main()
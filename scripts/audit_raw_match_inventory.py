from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"


def inspect_csv(path: Path) -> dict:
    try:
        df = pd.read_csv(path)
        columns = list(df.columns)
        rows = len(df)
    except Exception as exc:
        columns = []
        rows = None
        error = str(exc)
    else:
        error = ""

    return {
        "path": path,
        "rows": rows,
        "columns": columns,
        "error": error,
    }


def main() -> None:
    csv_files = sorted(RAW_ROOT.rglob("*.csv"))

    print(f"Raw root: {RAW_ROOT}")
    print(f"CSV files found: {len(csv_files)}")

    total_rows = 0

    for path in csv_files:
        info = inspect_csv(path)
        rel = path.relative_to(PROJECT_ROOT)

        print()
        print(rel)
        print("-" * len(str(rel)))

        if info["error"]:
            print(f"ERROR: {info['error']}")
            continue

        print(f"Rows: {info['rows']}")
        print("Columns:")
        for col in info["columns"]:
            print(f"  - {col}")

        total_rows += info["rows"]

    print()
    print("Summary")
    print("-------")
    print(f"Files: {len(csv_files)}")
    print(f"Total rows: {total_rows}")


if __name__ == "__main__":
    main()
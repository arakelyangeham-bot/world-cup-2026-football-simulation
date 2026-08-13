from pathlib import Path
import csv

OUTPUT_DIR = Path("outputs/monte_carlo")

FILES = [
    "champion_probabilities.csv",
    "runner_up_probabilities.csv",
    "semifinal_probabilities.csv",
    "quarterfinal_probabilities.csv",
    "round_of_16_probabilities.csv",
]

def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def audit_probability_file(filename: str) -> None:
    path = OUTPUT_DIR / filename
    rows = read_rows(path)

    total_count = sum(int(row["count"]) for row in rows)
    total_probability = sum(float(row["probability"]) for row in rows)

    print()
    print(filename)
    print("-" * len(filename))
    print(f"Rows: {len(rows)}")
    print(f"Total count: {total_count}")
    print(f"Total probability: {total_probability:.6f}")

    print("Top 5:")
    for row in rows[:5]:
        print(
            f"  {row['team']:<25} "
            f"{int(row['count']):>6} "
            f"{float(row['probability']):.3f}"
        )

def main() -> None:
    for filename in FILES:
        audit_probability_file(filename)

if __name__ == "__main__":
    main()
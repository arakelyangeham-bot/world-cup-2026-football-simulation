#compare_to_baseline.py

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from scripts.benchmarks.comparison.schema import compare_schema
from scripts.benchmarks.comparison.scoreline import compare_scoreline
from scripts.benchmarks.comparison.calibration import compare_calibration

SUITES = {
    "benchmark": {
        "scoreline_distribution_benchmark.csv",
        "scoreline_frequency_comparison.csv",
    },
    "calibration": {
        "probability_calibration_summary.csv",
    },
    "monte_carlo": {
        "champion_probabilities.csv",
        "runner_up_probabilities.csv",
        "semifinal_probabilities.csv",
        "quarterfinal_probabilities.csv",
        "round_of_16_probabilities.csv",
        "simulation_statistics.csv",
    },
    "full": None,
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare candidate benchmark artifacts against a frozen baseline."
    )
    parser.add_argument(
        "--baseline-dir",
        default="outputs/baselines/v5_dixon_coles_hierarchical",
        help="Directory containing frozen baseline artifacts.",
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Directory containing candidate artifacts to compare.",
    )
    parser.add_argument(
        "--output",
        default="outputs/baseline_comparison_report.csv",
        help="Output CSV report path.",
    )

    parser.add_argument(
        "--suite",
        choices=sorted(SUITES.keys()),
        default="full",
        help="Artifact suite to compare.",
    )

    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    output_path = Path(args.output)

    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory does not exist: {baseline_dir}")

    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate directory does not exist: {candidate_dir}")

    all_baseline_files = sorted(
        f for f in baseline_dir.iterdir()
        if f.is_file()
        and f.name != "baseline_manifest.csv"
        and not f.name.startswith("self_comparison_report")
    )

    selected_suite = SUITES[args.suite]

    if selected_suite is None:
        baseline_files = all_baseline_files
    else:
        baseline_files = [
            f for f in all_baseline_files
            if f.name in selected_suite
        ]

    results = compare_schema(
        baseline_files=baseline_files,
        candidate_dir=candidate_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in results for key in row.keys()})

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    fail_count = sum(row["status"] == "FAIL" for row in results)
    warn_count = sum(row["status"] == "WARN" for row in results)
    pass_count = sum(row["status"] == "PASS" for row in results)

    if args.suite in {"benchmark", "full"}:
        scoreline_fail_count, scoreline_warn_count = compare_scoreline(
            baseline_dir=baseline_dir,
            candidate_dir=candidate_dir,
            output_path=output_path,
        )

        fail_count += scoreline_fail_count
        warn_count += scoreline_warn_count
    
    if args.suite in {"calibration", "full"}:
        calibration_fail_count, calibration_warn_count = compare_calibration(
            baseline_dir=baseline_dir,
            candidate_dir=candidate_dir,
            output_path=output_path,
        )

        fail_count += calibration_fail_count
        warn_count += calibration_warn_count

    print(f"Wrote comparison report: {output_path}")
    print(f"PASS: {pass_count}")
    print(f"WARN: {warn_count}")
    print(f"FAIL: {fail_count}")

    if fail_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
#calibration.py

from __future__ import annotations

import csv
from pathlib import Path


def calibration_metrics(path: Path) -> dict[str, float]:
    import pandas as pd

    df = pd.read_csv(path)

    if len(df) != 1:
        raise ValueError(
            f"Expected exactly one row in calibration summary: {path}"
        )

    row = df.iloc[0]

    return {
        "brier": float(row["multiclass_brier_score"]),
        "log_loss": float(row["multiclass_log_loss"]),
        "ece": float(row["ece_mean"]),
    }


def compare_calibration(
    baseline_dir: Path,
    candidate_dir: Path,
    output_path: Path,
) -> tuple[int, int]:
    calibration_file = "probability_calibration_summary.csv"
    baseline_path = baseline_dir / calibration_file
    candidate_path = candidate_dir / calibration_file

    if not baseline_path.exists() or not candidate_path.exists():
        return 0, 0

    baseline_metrics = calibration_metrics(baseline_path)
    candidate_metrics = calibration_metrics(candidate_path)

    calibration_report_path = output_path.with_name(
        output_path.stem + "_calibration.csv"
    )

    rows = []

    for metric, baseline_value in baseline_metrics.items():
        candidate_value = candidate_metrics[metric]
        delta = candidate_value - baseline_value

        if abs(delta) < 1e-9:
            status = "WARN"
            notes = "Candidate matches baseline"
        elif delta < 0:
            status = "PASS"
            notes = "Candidate improves metric"
        else:
            status = "FAIL"
            notes = "Candidate regresses metric"

        rows.append(
            {
                "metric": metric,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "delta": delta,
                "percent_change": (
                    delta / baseline_value if baseline_value != 0 else None
                ),
                "status": status,
                "notes": notes,
            }
        )

    with calibration_report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "baseline_value",
                "candidate_value",
                "delta",
                "percent_change",
                "status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote calibration report: {calibration_report_path}")

    for row in rows:
        print(
            f"{row['metric']}: "
            f"baseline={row['baseline_value']:.6f}, "
            f"candidate={row['candidate_value']:.6f}, "
            f"status={row['status']}"
        )

    fail_count = sum(row["status"] == "FAIL" for row in rows)
    warn_count = sum(row["status"] == "WARN" for row in rows)

    return fail_count, warn_count
#scoreline.py

from __future__ import annotations

import csv
from pathlib import Path


def production_scoreline_tvd(path: Path) -> tuple[str, float]:
    import pandas as pd

    df = pd.read_csv(path)

    if "model" not in df.columns or "total_variation_distance" not in df.columns:
        raise ValueError(f"Missing required TVD columns in {path}")

    if "is_production" in df.columns:
        production_rows = df[df["is_production"].astype(str).str.lower() == "true"]

        if len(production_rows) != 1:
            raise ValueError(
                f"Expected exactly one production row in {path}, found {len(production_rows)}"
            )

        row = production_rows.iloc[0]
    else:
        non_historical = df[df["model"] != "historical"]

        if non_historical.empty:
            raise ValueError(f"No non-historical model rows found in {path}")

        row = non_historical.sort_values("total_variation_distance").iloc[0]

    return str(row["model"]), float(row["total_variation_distance"])


def compare_scoreline(
    baseline_dir: Path,
    candidate_dir: Path,
    output_path: Path,
) -> tuple[int, int]:
    scoreline_file = "scoreline_distribution_benchmark.csv"
    baseline_path = baseline_dir / scoreline_file
    candidate_path = candidate_dir / scoreline_file

    if not baseline_path.exists() or not candidate_path.exists():
        return 0, 0

    baseline_model, baseline_tvd = production_scoreline_tvd(baseline_path)
    candidate_model, candidate_tvd = production_scoreline_tvd(candidate_path)

    tvd_delta = candidate_tvd - baseline_tvd

    if abs(tvd_delta) < 1e-9:
        status = "WARN"
        notes = "Candidate TVD matches baseline"
    elif tvd_delta < 0:
        status = "PASS"
        notes = "Candidate improves TVD"
    else:
        status = "FAIL"
        notes = "Candidate regresses TVD"

    tvd_report_path = output_path.with_name(
        output_path.stem + "_scoreline_tvd.csv"
    )

    with tvd_report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "baseline_model",
                "candidate_model",
                "baseline_value",
                "candidate_value",
                "delta",
                "percent_change",
                "status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "metric": "scoreline_total_variation_distance",
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "baseline_value": baseline_tvd,
                "candidate_value": candidate_tvd,
                "delta": tvd_delta,
                "percent_change": (
                    tvd_delta / baseline_tvd if baseline_tvd != 0 else None
                ),
                "status": status,
                "notes": notes,
            }
        )

    print(f"Wrote TVD report: {tvd_report_path}")
    print(
        f"Scoreline TVD: baseline={baseline_tvd:.6f}, "
        f"candidate={candidate_tvd:.6f}, status={status}"
    )

    fail_count = 1 if status == "FAIL" else 0
    warn_count = 1 if status == "WARN" else 0

    return fail_count, warn_count
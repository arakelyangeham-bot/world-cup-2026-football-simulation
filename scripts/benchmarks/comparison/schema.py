#schema.py

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def csv_shape(path: Path) -> tuple[int, int, list[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return 0, 0, []

    header = rows[0]
    return max(len(rows) - 1, 0), len(header), header


def compare_file(
    baseline_file: Path,
    candidate_file: Path,
) -> dict[str, object]:
    result: dict[str, object] = {
        "filename": baseline_file.name,
        "candidate_exists": candidate_file.exists(),
        "baseline_size_bytes": baseline_file.stat().st_size,
        "candidate_size_bytes": None,
        "baseline_sha256": sha256(baseline_file),
        "candidate_sha256": None,
        "csv_rows_match": None,
        "csv_columns_match": None,
        "csv_header_match": None,
        "status": "PASS",
        "notes": "",
    }

    if not candidate_file.exists():
        result["status"] = "FAIL"
        result["notes"] = "Missing candidate file"
        return result

    result["candidate_size_bytes"] = candidate_file.stat().st_size
    result["candidate_sha256"] = sha256(candidate_file)

    if baseline_file.suffix.lower() == ".csv":
        b_rows, b_cols, b_header = csv_shape(baseline_file)
        c_rows, c_cols, c_header = csv_shape(candidate_file)

        result["baseline_rows"] = b_rows
        result["candidate_rows"] = c_rows
        result["baseline_columns"] = b_cols
        result["candidate_columns"] = c_cols

        result["csv_rows_match"] = b_rows == c_rows
        result["csv_columns_match"] = b_cols == c_cols
        result["csv_header_match"] = b_header == c_header

        if not result["csv_columns_match"] or not result["csv_header_match"]:
            result["status"] = "FAIL"
            result["notes"] = "CSV schema mismatch"
        elif not result["csv_rows_match"]:
            result["status"] = "WARN"
            result["notes"] = "CSV row count differs"

    return result


def compare_schema(
    baseline_files: list[Path],
    candidate_dir: Path,
) -> list[dict[str, object]]:
    return [
        compare_file(
            baseline_file=baseline_file,
            candidate_file=candidate_dir / baseline_file.name,
        )
        for baseline_file in baseline_files
    ]
#promote_baseline.py

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


SUITES = {
    "benchmark": {
        "scoreline_distribution_benchmark.csv",
        "scoreline_frequency_comparison.csv",
    },
    "monte_carlo": {
        "champion_probabilities.csv",
        "runner_up_probabilities.csv",
        "semifinal_probabilities.csv",
        "quarterfinal_probabilities.csv",
        "round_of_16_probabilities.csv",
        "simulation_statistics.csv",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(baseline_dir: Path) -> None:
    rows = []

    for file in sorted(baseline_dir.glob("*")):
        if not file.is_file():
            continue
        if file.name == "baseline_manifest.csv":
            continue

        rows.append(
            {
                "filename": file.name,
                "size_bytes": file.stat().st_size,
                "sha256": sha256(file),
            }
        )

    output_path = baseline_dir / "baseline_manifest.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote benchmark or simulation outputs into a versioned baseline."
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES.keys()),
        required=True,
    )
    parser.add_argument("--parent", default=None)
    parser.add_argument("--reason", required=True)

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    baseline_dir = Path("outputs") / "baselines" / f"{args.version}_{args.name}"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if baseline_dir.exists():
        raise FileExistsError(
            f"Baseline already exists: {baseline_dir}. "
            "Refusing to overwrite an existing baseline."
        )

    baseline_dir.mkdir(parents=True)

    copied = []

    for filename in sorted(SUITES[args.suite]):
        source_file = source_dir / filename

        if not source_file.exists():
            raise FileNotFoundError(f"Missing required source artifact: {source_file}")

        target_file = baseline_dir / filename
        shutil.copy2(source_file, target_file)
        copied.append(filename)

    metadata = {
        "baseline_version": args.version,
        "baseline_name": args.name,
        "suite": args.suite,
        "parent": args.parent,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": args.reason,
        "copied_artifacts": copied,
    }

    with (baseline_dir / "baseline_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme = f"""# Baseline {args.version}: {args.name}

## Suite

{args.suite}

## Parent

{args.parent}

## Reason

{args.reason}

## Artifacts

{chr(10).join(f"- {filename}" for filename in copied)}
"""

    with (baseline_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(readme)

    write_manifest(baseline_dir)

    print(f"Created baseline: {baseline_dir}")
    print(f"Copied {len(copied)} artifacts")
    print("Wrote README.md, baseline_metadata.json, and baseline_manifest.csv")


if __name__ == "__main__":
    main()
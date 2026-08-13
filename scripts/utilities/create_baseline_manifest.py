#create_baseline_manifest.py

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

BASELINE_DIR = Path("outputs/baselines/v5_dixon_coles_hierarchical")
OUTPUT_FILE = BASELINE_DIR / "baseline_manifest.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


rows = []

for file in sorted(BASELINE_DIR.glob("*")):

    if not file.is_file():
        continue

    if file.name == OUTPUT_FILE.name:
        continue

    rows.append({
        "filename": file.name,
        "size_bytes": file.stat().st_size,
        "sha256": sha256(file),
    })


with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "size_bytes",
            "sha256",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} entries to {OUTPUT_FILE}")
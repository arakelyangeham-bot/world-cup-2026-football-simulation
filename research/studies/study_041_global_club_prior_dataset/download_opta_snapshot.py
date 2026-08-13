from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SOURCE_URL = (
    "https://dataviz.theanalyst.com/"
    "opta-power-rankings/pr-reference.json"
)

RAW_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "raw"
    / "opta_power_rankings"
)


def download_snapshot(
    source_url: str,
    timeout_seconds: int = 30,
) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": (
            "https://dataviz.theanalyst.com/"
            "opta-power-rankings/"
        ),
    }

    response = requests.get(
        source_url,
        headers=headers,
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise RuntimeError(
            "The Opta endpoint did not return valid JSON."
        ) from exc

    if not isinstance(payload, list):
        raise TypeError(
            "Expected the Opta snapshot root to be a JSON list, "
            f"but received {type(payload).__name__}."
        )

    if not payload:
        raise RuntimeError(
            "The Opta endpoint returned an empty snapshot."
        )

    return payload


def validate_minimum_structure(
    snapshot: list[dict[str, Any]],
) -> None:
    required_fields = {
        "contestantId",
        "currentRating",
        "rank",
        "contestantName",
    }

    for row_number, record in enumerate(snapshot, start=1):
        if not isinstance(record, dict):
            raise TypeError(
                f"Snapshot row {row_number} is not a JSON object."
            )

        missing = required_fields - set(record)

        if missing:
            raise ValueError(
                f"Snapshot row {row_number} is missing fields: "
                f"{sorted(missing)}"
            )


def write_raw_snapshot(
    snapshot: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            snapshot,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def collect_snapshot(
    source_url: str,
    snapshot_date: str | None = None,
) -> Path:
    now = datetime.now(timezone.utc)

    effective_date = (
        snapshot_date
        if snapshot_date is not None
        else now.date().isoformat()
    )

    filename_date = effective_date.replace("-", "_")

    output_path = (
        RAW_OUTPUT_DIRECTORY
        / f"opta_power_rankings_{filename_date}.json"
    )

    snapshot = download_snapshot(source_url)
    validate_minimum_structure(snapshot)
    write_raw_snapshot(snapshot, output_path)

    print("Opta JSON Snapshot Collection")
    print("=============================")
    print(f"Records collected: {len(snapshot)}")
    print(f"Snapshot date: {effective_date}")
    print(f"Source: {source_url}")
    print(f"Output: {output_path}")
    print()
    print("Raw JSON snapshot written successfully.")

    return output_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a dated raw JSON snapshot of "
            "the Opta Power Rankings."
        )
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_SOURCE_URL,
        help="Direct Opta Power Rankings JSON endpoint.",
    )

    parser.add_argument(
        "--snapshot-date",
        default=None,
        help=(
            "Snapshot date in YYYY-MM-DD format. "
            "Defaults to today's UTC date."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    collect_snapshot(
        source_url=arguments.url,
        snapshot_date=arguments.snapshot_date,
    )


if __name__ == "__main__":
    main()
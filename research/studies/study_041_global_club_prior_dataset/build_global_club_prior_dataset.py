#build_global_club_prior_dataset

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "raw"
    / "opta_power_rankings"
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "research"
    / "data"
    / "processed"
)

DEFAULT_INPUT_PATH = (
    RAW_DIRECTORY
    / "opta_power_rankings_2026_07_18.json"
)

DEFAULT_OUTPUT_PATH = (
    PROCESSED_DIRECTORY
    / "global_club_prior_dataset.csv"
)

SOURCE_URL = (
    "https://dataviz.theanalyst.com/"
    "opta-power-rankings/pr-reference.json"
)

OUTPUT_COLUMNS = [
    "club_id",
    "opta_id",
    "club",
    "club_short",
    "club_full",
    "club_code",
    "opta_rating",
    "global_rank",
    "competition_id",
    "rating_prior",
    "rating_prior_method",
    "snapshot_date",
    "rating_source",
    "source_url",
]


def load_raw_snapshot(
    input_path: Path,
) -> list[dict[str, Any]]:
    """
    Load the unmodified Opta JSON snapshot.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Opta snapshot not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise TypeError(
            "Expected the raw Opta snapshot to contain a JSON list."
        )

    if not payload:
        raise ValueError(
            "The raw Opta snapshot is empty."
        )

    return payload


def validate_raw_record(
    record: dict[str, Any],
    row_number: int,
) -> None:
    required_fields = {
        "contestantId",
        "currentRating",
        "rank",
        "contestantName",
        "contestantShortName",
        "contestantClubName",
        "contestantCode",
    }

    missing_fields = required_fields - set(record)

    if missing_fields:
        raise ValueError(
            f"Raw row {row_number} is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    if not str(record["contestantId"]).strip():
        raise ValueError(
            f"Raw row {row_number} has an empty contestantId."
        )

    if not str(record["contestantName"]).strip():
        raise ValueError(
            f"Raw row {row_number} has an empty contestantName."
        )

    try:
        rating = float(record["currentRating"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Raw row {row_number} has an invalid currentRating."
        ) from exc

    if not 0.0 <= rating <= 100.0:
        raise ValueError(
            f"Raw row {row_number} has an out-of-range rating: "
            f"{rating}"
        )

    try:
        global_rank = int(record["rank"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Raw row {row_number} has an invalid rank."
        ) from exc

    if global_rank < 1:
        raise ValueError(
            f"Raw row {row_number} has a non-positive rank."
        )


def transform_record(
    record: dict[str, Any],
    snapshot_date: str,
) -> dict[str, Any]:
    """
    Convert one raw Opta record into the canonical project schema.

    rating_prior deliberately remains empty until a calibration method
    has been selected and validated.
    """

    opta_id = record.get("optaId")
    competition_id = record.get("tmcl")

    return {
        "club_id": str(record["contestantId"]).strip(),
        "opta_id": (
            ""
            if opta_id is None
            else int(float(opta_id))
        ),
        "club": str(record["contestantName"]).strip(),
        "club_short": str(
            record["contestantShortName"]
        ).strip(),
        "club_full": str(
            record["contestantClubName"]
        ).strip(),
        "club_code": str(
            record["contestantCode"]
        ).strip(),
        "opta_rating": float(record["currentRating"]),
        "global_rank": int(record["rank"]),
        "competition_id": (
            ""
            if competition_id is None
            else str(competition_id).strip()
        ),

        # Reserved for the calibration stage.
        "rating_prior": "",
        "rating_prior_method": "unassigned",

        "snapshot_date": snapshot_date,
        "rating_source": "Opta Power Rankings",
        "source_url": SOURCE_URL,
    }


def build_canonical_dataset(
    raw_snapshot: list[dict[str, Any]],
    snapshot_date: str,
) -> list[dict[str, Any]]:
    canonical_rows: list[dict[str, Any]] = []

    seen_club_ids: set[str] = set()
    seen_global_ranks: set[int] = set()

    for row_number, record in enumerate(
        raw_snapshot,
        start=1,
    ):
        if not isinstance(record, dict):
            raise TypeError(
                f"Raw row {row_number} is not a JSON object."
            )

        validate_raw_record(
            record=record,
            row_number=row_number,
        )

        canonical_record = transform_record(
            record=record,
            snapshot_date=snapshot_date,
        )

        club_id = canonical_record["club_id"]
        global_rank = canonical_record["global_rank"]

        if club_id in seen_club_ids:
            raise ValueError(
                f"Duplicate club_id found: {club_id}"
            )

        if global_rank in seen_global_ranks:
            raise ValueError(
                f"Duplicate global rank found: {global_rank}"
            )

        seen_club_ids.add(club_id)
        seen_global_ranks.add(global_rank)
        canonical_rows.append(canonical_record)

    canonical_rows.sort(
        key=lambda row: row["global_rank"]
    )

    return canonical_rows


def write_canonical_dataset(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError(
            "Cannot write an empty canonical club dataset."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_snapshot_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Snapshot date must use YYYY-MM-DD format."
        ) from exc

    return value


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical Global Club Prior Dataset "
            "from a raw Opta JSON snapshot."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the raw Opta JSON snapshot.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the canonical CSV dataset.",
    )

    parser.add_argument(
        "--snapshot-date",
        type=validate_snapshot_date,
        default="2026-07-18",
        help="Source snapshot date in YYYY-MM-DD format.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    raw_snapshot = load_raw_snapshot(
        input_path=arguments.input,
    )

    canonical_rows = build_canonical_dataset(
        raw_snapshot=raw_snapshot,
        snapshot_date=arguments.snapshot_date,
    )

    write_canonical_dataset(
        rows=canonical_rows,
        output_path=arguments.output,
    )

    highest_rated = canonical_rows[0]
    lowest_rated = canonical_rows[-1]

    print("Global Club Prior Dataset Build")
    print("===============================")
    print(f"Raw records: {len(raw_snapshot)}")
    print(f"Canonical records: {len(canonical_rows)}")
    print(f"Snapshot date: {arguments.snapshot_date}")
    print(f"Output: {arguments.output}")
    print()
    print(
        "Highest-rated club: "
        f"{highest_rated['club']} "
        f"({highest_rated['opta_rating']:.4f})"
    )
    print(
        "Lowest-rated club: "
        f"{lowest_rated['club']} "
        f"({lowest_rated['opta_rating']:.4f})"
    )
    print()
    print(
        "rating_prior remains unassigned pending calibration."
    )
    print("Canonical dataset written successfully.")


if __name__ == "__main__":
    main()
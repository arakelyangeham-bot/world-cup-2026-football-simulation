# generate_third_place_assignments.py

from pathlib import Path
import re
from itertools import combinations

try:
    from pypdf import PdfReader
except ImportError as exc:
    raise ImportError("Install pypdf first: pip install pypdf") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_FILE = PROJECT_ROOT / "data" / "raw" / "FWC2026_regulations_EN.pdf"
OUT_FILE = PROJECT_ROOT / "scripts" / "wc2026_third_place_assignments.py"

COLUMN_ORDER = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]


def extract_rows_from_pdf(pdf_file: Path) -> dict[tuple[str, ...], list[str]]:
    reader = PdfReader(pdf_file)

    assignments = {}

    row_pattern = re.compile(
        r"^\s*(\d{1,3})\s+"
        r"(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+"
        r"(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s*$"
    )

    for page in reader.pages:
        text = page.extract_text() or ""

        for line in text.splitlines():
            match = row_pattern.match(line)
            if not match:
                continue

            option = int(match.group(1))
            slots = list(match.groups()[1:])

            groups = tuple(sorted(slot[1] for slot in slots))

            if groups in assignments:
                raise ValueError(f"Duplicate third-place combo found: {groups}")

            assignments[groups] = slots

    return assignments


def validate_assignments(assignments: dict[tuple[str, ...], list[str]]) -> None:
    expected_combos = {
        tuple(combo)
        for combo in combinations("ABCDEFGHIJKL", 8)
    }

    actual_combos = set(assignments)

    missing = expected_combos - actual_combos
    extra = actual_combos - expected_combos

    if missing:
        raise ValueError(f"Missing combos: {sorted(missing)[:10]}")

    if extra:
        raise ValueError(f"Unexpected combos: {sorted(extra)[:10]}")

    if len(assignments) != 495:
        raise ValueError(f"Expected 495 assignments, got {len(assignments)}")

    for combo, slots in assignments.items():
        if len(slots) != 8:
            raise ValueError(f"{combo}: expected 8 slots")

        slot_groups = sorted(slot[1] for slot in slots)

        if slot_groups != sorted(combo):
            raise ValueError(f"{combo}: slot groups do not match combo")


def write_python_module(assignments: dict[tuple[str, ...], list[str]]) -> None:
    lines = [
        "# Auto-generated from FIFA World Cup 2026 Regulations, Annex C.",
        "# Do not edit by hand.",
        "",
        f"THIRD_PLACE_COLUMN_ORDER = {COLUMN_ORDER!r}",
        "",
        "THIRD_PLACE_ASSIGNMENTS = {",
    ]

    for combo in sorted(assignments):
        lines.append(f"    {combo!r}: {assignments[combo]!r},")

    lines.append("}")
    lines.append("")

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    assignments = extract_rows_from_pdf(PDF_FILE)
    validate_assignments(assignments)
    write_python_module(assignments)

    print(f"Generated {len(assignments)} third-place assignments")
    print(f"Wrote: {OUT_FILE}")


if __name__ == "__main__":
    main()
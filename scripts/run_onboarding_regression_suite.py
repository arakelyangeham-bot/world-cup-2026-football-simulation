#run_onboarding_regression_suite

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "onboarding_validation"
    / "regression_suite.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical project regression suite and "
            "record a PASS artifact only when pytest succeeds."
        )
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Validation JSON output path. Defaults to "
            "outputs/onboarding_validation/regression_suite.json."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
    ]

    print(
        "Running:",
        " ".join(command),
    )
    print()

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise SystemExit(
            completed.returncode
        )

    payload = {
        "status": "PASS",
        "validation_type": "regression_suite",
        "command": command,
        "exit_code": completed.returncode,
        "recorded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    output_path = arguments.output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Regression validation: {output_path}"
    )


if __name__ == "__main__":
    main()
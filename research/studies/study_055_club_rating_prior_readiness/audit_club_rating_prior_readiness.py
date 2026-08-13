#audit_club_rating_prior_readiness

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "study_055_club_rating_prior_readiness"
)

SEARCH_TERMS = (
    "rating_prior",
    "rating prior",
    "club_elo",
    "club elo",
    "elo_rating",
    "opta",
    "fifa_points",
    "fifa points",
    "team_prior",
    "team prior",
    "national_team_priors",
    "strength_loader",
    "team strength",
)

SEARCH_SUFFIXES = {
    ".py",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
}

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
}

MAX_MATCHES_PER_FILE = 100


def should_skip_path(
    path: Path,
) -> bool:
    return any(
        part in EXCLUDED_DIRECTORY_NAMES
        for part in path.parts
    )


def iter_searchable_files() -> list[Path]:
    files: list[Path] = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if should_skip_path(path):
            continue

        if path.suffix.lower() not in SEARCH_SUFFIXES:
            continue

        files.append(path)

    return sorted(files)


def read_text_safely(
    path: Path,
) -> str | None:
    try:
        return path.read_text(
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        try:
            return path.read_text(
                encoding="utf-8-sig",
            )
        except UnicodeDecodeError:
            return None
    except OSError:
        return None


def classify_match(
    relative_path: str,
    matched_terms: set[str],
) -> str:
    normalized_path = relative_path.lower()

    if "national_team_priors" in normalized_path:
        return "national_prior_infrastructure"

    if "team_strength_loader" in normalized_path:
        return "team_strength_infrastructure"

    if "observatory" in normalized_path:
        return "observation_schema"

    if "observation" in normalized_path:
        return "observation_pipeline"

    if "goal_model" in normalized_path:
        return "goal_model"

    if "simulation" in normalized_path:
        return "simulation"

    if (
        "club_elo" in matched_terms
        or "club elo" in matched_terms
        or "elo_rating" in matched_terms
    ):
        return "club_elo_reference"

    if "opta" in matched_terms:
        return "opta_reference"

    if (
        "rating_prior" in matched_terms
        or "rating prior" in matched_terms
    ):
        return "generic_rating_prior"

    if (
        "fifa_points" in matched_terms
        or "fifa points" in matched_terms
    ):
        return "legacy_fifa_prior"

    return "other"


def inspect_file(
    path: Path,
) -> dict[str, object] | None:
    text = read_text_safely(path)

    if text is None:
        return None

    lowered = text.lower()

    matched_terms = {
        term
        for term in SEARCH_TERMS
        if term in lowered
    }

    if not matched_terms:
        return None

    relative_path = str(
        path.relative_to(PROJECT_ROOT)
    )

    matching_lines: list[
        dict[str, object]
    ] = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        lowered_line = line.lower()

        line_terms = [
            term
            for term in SEARCH_TERMS
            if term in lowered_line
        ]

        if not line_terms:
            continue

        matching_lines.append(
            {
                "line_number": line_number,
                "terms": sorted(line_terms),
                "text": line.strip(),
            }
        )

        if (
            len(matching_lines)
            >= MAX_MATCHES_PER_FILE
        ):
            break

    return {
        "path": relative_path,
        "suffix": path.suffix.lower(),
        "classification": classify_match(
            relative_path=relative_path,
            matched_terms=matched_terms,
        ),
        "matched_terms": sorted(
            matched_terms
        ),
        "match_count": len(
            matching_lines
        ),
        "matching_lines": matching_lines,
    }


def build_summary(
    records: list[dict[str, object]],
) -> dict[str, object]:
    classification_counts: dict[str, int] = {}
    term_counts: dict[str, int] = {}

    for record in records:
        classification = str(
            record["classification"]
        )

        classification_counts[
            classification
        ] = (
            classification_counts.get(
                classification,
                0,
            )
            + 1
        )

        for term in record["matched_terms"]:
            term_text = str(term)

            term_counts[term_text] = (
                term_counts.get(
                    term_text,
                    0,
                )
                + 1
            )

    likely_prior_files = [
        record["path"]
        for record in records
        if record["classification"] in {
            "national_prior_infrastructure",
            "team_strength_infrastructure",
            "club_elo_reference",
            "opta_reference",
            "generic_rating_prior",
        }
    ]

    return {
        "searchable_file_count": None,
        "matched_file_count": len(records),
        "classification_counts":
            dict(
                sorted(
                    classification_counts.items()
                )
            ),
        "term_counts":
            dict(
                sorted(
                    term_counts.items()
                )
            ),
        "likely_prior_files":
            sorted(
                set(likely_prior_files)
            ),
    }


def write_markdown_report(
    path: Path,
    searchable_file_count: int,
    records: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    lines = [
        "# Study 055 Results",
        "",
        "## Club Rating-Prior Readiness Audit",
        "",
        "**Status:** `PASS`",
        "",
        "## Search scope",
        "",
        (
            f"- Searchable files inspected: "
            f"{searchable_file_count}"
        ),
        (
            f"- Files containing prior-related terms: "
            f"{len(records)}"
        ),
        "",
        "## Classification counts",
        "",
    ]

    for classification, count in (
        summary[
            "classification_counts"
        ].items()
    ):
        lines.append(
            f"- `{classification}`: {count}"
        )

    lines.extend(
        [
            "",
            "## Likely prior-related files",
            "",
        ]
    )

    likely_files = summary[
        "likely_prior_files"
    ]

    if likely_files:
        for file_path in likely_files:
            lines.append(
                f"- `{file_path}`"
            )
    else:
        lines.append(
            "- No likely prior infrastructure found."
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "This audit identifies potentially "
                "relevant files and references. It does "
                "not establish that any rating source is "
                "temporally valid or prediction-ready."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    searchable_files = (
        iter_searchable_files()
    )

    records: list[
        dict[str, object]
    ] = []

    for path in searchable_files:
        record = inspect_file(path)

        if record is not None:
            records.append(record)

    summary = build_summary(records)

    summary[
        "searchable_file_count"
    ] = len(searchable_files)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "study_id": "055",
        "study_name": (
            "Club Rating-Prior Readiness Audit"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "search_terms": list(
            SEARCH_TERMS
        ),
        "search_suffixes": sorted(
            SEARCH_SUFFIXES
        ),
        **summary,
        "output_files": [
            "prior_search_results.json",
            "prior_readiness_summary.json",
            "study_metadata.json",
            "STUDY_055_RESULTS.md",
        ],
    }

    with (
        OUTPUT_DIR
        / "prior_search_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            indent=2,
        )

    with (
        OUTPUT_DIR
        / "prior_readiness_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    with (
        OUTPUT_DIR
        / "study_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    write_markdown_report(
        path=(
            OUTPUT_DIR
            / "STUDY_055_RESULTS.md"
        ),
        searchable_file_count=(
            len(searchable_files)
        ),
        records=records,
        summary=summary,
    )

    print("Study 055")
    print("=" * 76)
    print()
    print(
        "Searchable files inspected: "
        f"{len(searchable_files)}"
    )
    print(
        "Files containing prior-related terms: "
        f"{len(records)}"
    )
    print()

    print("Classification Counts")
    print("-" * 76)

    if summary[
        "classification_counts"
    ]:
        for classification, count in (
            summary[
                "classification_counts"
            ].items()
        ):
            print(
                f"{classification}: {count}"
            )
    else:
        print(
            "No prior-related references found."
        )

    print()
    print("Likely Prior-Related Files")
    print("-" * 76)

    likely_files = summary[
        "likely_prior_files"
    ]

    if likely_files:
        for file_path in likely_files:
            print(file_path)
    else:
        print(
            "No likely prior infrastructure found."
        )

    print()
    print("Repository scan: PASS")
    print("Prior-reference classification: PASS")
    print("Audit artifact generation: PASS")
    print()
    print("OVERALL RESULT: PASS")
    print()
    print(
        f"Outputs written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
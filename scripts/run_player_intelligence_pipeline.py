#run_player_intelligence_pipeline

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    description: str


STAGES = [
    Stage(
        name="resolve_evidence",
        module="scripts.resolve_player_evidence",
        description=(
            "Resolve source competition-season evidence "
            "to canonical player identities."
        ),
    ),
    Stage(
        name="feature_engineering",
        module="scripts.sofascore_feature_engineering",
        description=(
            "Build per-90 features while preserving "
            "canonical competition-season grain."
        ),
    ),
    Stage(
        name="canonical_registry",
        module="scripts.build_canonical_player_registry",
        description=(
            "Build one authoritative identity row "
            "per canonical player from current profiles."
        ),
    ),
    Stage(
        name="weighted_features",
        module="scripts.build_weighted_player_features",
        description=(
            "Aggregate canonical competition-season "
            "features into weighted player representations."
        ),
    ),
    Stage(
        name="attributes",
        module="scripts.score_player_attributes",
        description=(
            "Transform weighted player features and "
            "construct Player Intelligence attributes."
        ),
    ),
    Stage(
        name="ratings",
        module="scripts.build_player_ratings_v4",
        description=(
            "Construct role-specific player ratings."
        ),
    ),
]


def run_stage(stage: Stage) -> None:
    command = [
        sys.executable,
        "-m",
        stage.module,
    ]

    print()
    print("=" * 88)
    print(f"STAGE: {stage.name}")
    print(f"MODULE: {stage.module}")
    print(f"INFO: {stage.description}")
    print("=" * 88)

    start = time.time()

    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    elapsed = time.time() - start

    if process.returncode != 0:
        raise SystemExit(
            f"Stage {stage.name!r} failed with "
            f"exit code {process.returncode}."
        )

    print(
        f"Finished {stage.name} "
        f"in {elapsed:.1f}s"
    )


def main() -> None:
    print("WC 2026 Player Intelligence pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    print("Pipeline stages:")

    for index, stage in enumerate(
        STAGES,
        start=1,
    ):
        print(
            f"  {index}. "
            f"{stage.name:<20} "
            f"{stage.module}"
        )

    pipeline_start = time.time()

    for stage in STAGES:
        run_stage(stage)

    elapsed = (
        time.time()
        - pipeline_start
    )

    print()
    print("=" * 88)
    print("PLAYER INTELLIGENCE PIPELINE COMPLETE")
    print("=" * 88)
    print(
        f"Completed stages: {len(STAGES)}"
    )
    print(
        f"Total elapsed: {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
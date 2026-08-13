# run_sofascore_pipeline.py
#
# LEGACY SOFASCORE ORCHESTRATOR
#
# This runner reflects an earlier, smaller version of the project and is
# retained for historical/reference purposes and older workflows.
#
# It is NOT the authoritative production runner for the modern Player
# Intelligence architecture.
#
# Current Player Intelligence production entry point:
#
#     python -m scripts.run_player_intelligence_pipeline
#
# The modern Player Intelligence path is:
#
#     resolve_player_evidence
#         -> sofascore_feature_engineering
#         -> build_canonical_player_registry
#         -> build_weighted_player_features
#         -> score_player_attributes
#         -> build_player_ratings_v4
#
# Do not extend this legacy orchestrator with new Player Intelligence
# stages unless the project architecture is deliberately reconsidered.

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOG_DIR = PROJECT_ROOT / "outputs" / "pipeline_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    required: bool = True
    description: str = ""


STAGES = {
    "player_stats": Stage(
        name="player_stats",
        script="sofascore_wc_scraper.py",
        required=False,
        description="Refresh WC player stats from Sofascore. Slow and most block-prone.",
    ),
    "match_scraper": Stage(
        name="match_scraper",
        script="sofascore_match_scraper.py",
        description="Refresh fixtures, results, and match team stats.",
    ),
    "merge_roster": Stage(
        name="merge_roster",
        script="sofascore_merge_roster.py",
        description="Merge roster with scraped player stats.",
    ),
    "feature_engineering": Stage(
        name="feature_engineering",
        script="sofascore_feature_engineering.py",
        description="Create per-90 model feature table.",
    ),
    "team_aggregator": Stage(
        name="team_aggregator",
        script="sofascore_team_aggregator.py",
        description="Aggregate player stats into team strength metrics.",
    ),
    "match_prediction": Stage(
        name="match_prediction",
        script="sofascore_match_prediction.py",
        description="Build model params, predictions, score matrices, and metadata.",
    ),
    "backtest": Stage(
        name="backtest",
        script="sofascore_backtest.py",
        description="Run time-respecting backtests.",
    ),
    "optimizer": Stage(
        name="optimizer",
        script="sofascore_model_optimizer.py",
        description="Search model hyperparameters.",
    ),
    "optimizer_apply": Stage(
        name="optimizer_apply",
        script="sofascore_model_optimizer.py",
        description="Search model hyperparameters and write best params to model_config.json.",
    ),
    "simulator": Stage(
        name="simulator",
        script="sofascore_tournament_simulator.py",
        description="Run Monte Carlo tournament simulator.",
    ),
    "eda": Stage(
        name="eda",
        script="sofascore_eda.py",
        required=False,
        description="Basic dataset EDA outputs.",
    ),
    "country_position": Stage(
        name="country_position",
        script="sofascore_country_position_analysis.py",
        required=False,
        description="Country-position heatmaps and grouped stats.",
    ),
    "correlation": Stage(
        name="correlation",
        script="sofascore_correlation_analysis.py",
        required=False,
        description="Correlation matrices and scatter-pair audit.",
    ),
    "plots": Stage(
        name="plots",
        script="plot_sofascore_wc_2026.py",
        required=False,
        description="Interactive player scatterplots.",
    ),
}


MODE_STAGE_NAMES = {
    # Normal matchday update during the tournament.
    "matchday": [
        "match_scraper",
        "merge_roster",
        "feature_engineering",
        "team_aggregator",
        "match_prediction",
        "backtest",
        "optimizer_apply",
        "match_prediction",
        "backtest",
        "simulator",
        "eda",
        "country_position",
        "correlation",
        "plots",
    ],
    # Full refresh, including player stats.
    "full": [
        "player_stats",
        "match_scraper",
        "merge_roster",
        "feature_engineering",
        "team_aggregator",
        "match_prediction",
        "backtest",
        "optimizer_apply",
        "match_prediction",
        "backtest",
        "simulator",
        "eda",
        "country_position",
        "correlation",
        "plots",
    ],
    # No scraping; rebuild everything from existing raw data.
    "modeling": [
        "merge_roster",
        "feature_engineering",
        "team_aggregator",
        "match_prediction",
        "backtest",
        "optimizer_apply",
        "match_prediction",
        "backtest",
        "simulator",
    ],
    # Only regenerate analysis/plot artifacts.
    "analysis": [
        "eda",
        "country_position",
        "correlation",
        "plots",
    ],
}


def parse_csv_list(text: str | None) -> set[str]:
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def resolve_stages(mode: str, only: set[str], skip: set[str]) -> list[Stage]:
    if mode not in MODE_STAGE_NAMES:
        raise ValueError(f"Unknown mode: {mode}")

    stage_names = MODE_STAGE_NAMES[mode]

    if only:
        unknown = only - set(STAGES)
        if unknown:
            raise ValueError(f"Unknown --only stage(s): {sorted(unknown)}")
        stage_names = [name for name in stage_names if name in only]

    if skip:
        unknown = skip - set(STAGES)
        if unknown:
            raise ValueError(f"Unknown --skip stage(s): {sorted(unknown)}")
        stage_names = [name for name in stage_names if name not in skip]

    return [STAGES[name] for name in stage_names]


def script_path(stage: Stage) -> Path:
    return SCRIPTS_DIR / stage.script


def check_scripts_exist(stages: Iterable[Stage]) -> None:
    missing = [stage.script for stage in stages if not script_path(stage).exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "The following pipeline scripts were not found in scripts/:\n"
            f"  - {joined}\n\n"
            "Run this orchestrator from a normal project layout, or place it inside scripts/."
        )


def run_stage(stage: Stage, continue_on_optional_failure: bool, dry_run: bool) -> bool:
    path = script_path(stage)
    command = [sys.executable, str(path)]
    if stage.name == "optimizer_apply":
        command.append("--write-config")

    print("\n" + "=" * 90)
    print(f"STAGE: {stage.name}")
    print(f"SCRIPT: {stage.script}")
    if stage.description:
        print(f"INFO:   {stage.description}")
    print("=" * 90)

    if dry_run:
        print("DRY RUN:", " ".join(command))
        return True

    start = time.time()
    proc = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.time() - start

    if proc.returncode == 0:
        print(f"✓ Finished {stage.name} in {elapsed:.1f}s")
        return True

    print(f"✗ Failed {stage.name} after {elapsed:.1f}s with exit code {proc.returncode}")

    if stage.required or not continue_on_optional_failure:
        raise SystemExit(proc.returncode)

    print(f"Continuing because {stage.name} is optional and --continue-on-optional-failure is enabled.")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the WC 2026 Sofascore pipeline end to end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_STAGE_NAMES),
        default="matchday",
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated subset of stages to run within the selected mode.",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated stage names to skip.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print available modes/stages and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without running them.",
    )
    parser.add_argument(
        "--continue-on-optional-failure",
        action="store_true",
        help="Continue if optional analysis/plot/player-stat stages fail.",
    )
    return parser.parse_args()


def print_stage_listing() -> None:
    print("Available modes:")
    for mode, names in MODE_STAGE_NAMES.items():
        print(f"\n  {mode}:")
        for name in names:
            stage = STAGES[name]
            req = "required" if stage.required else "optional"
            print(f"    - {name:<20} {stage.script:<38} {req}")

    print("\nExcluded by design:")
    print("  - sofascore_test_scraper.py  diagnostic only")
    print("  - sofascore_utils.py         imported helper")


def main() -> None:
    args = parse_args()

    if args.list_stages:
        print_stage_listing()
        return

    only = parse_csv_list(args.only)
    skip = parse_csv_list(args.skip)
    stages = resolve_stages(args.mode, only=only, skip=skip)

    if not stages:
        raise SystemExit("No stages selected.")

    check_scripts_exist(stages)

    print("WC 2026 Sofascore pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Mode: {args.mode}")
    print("Selected stages:")
    for i, stage in enumerate(stages, 1):
        req = "required" if stage.required else "optional"
        print(f"  {i:02d}. {stage.name:<20} {stage.script:<38} {req}")

    start = time.time()
    completed = []
    failed_optional = []

    for stage in stages:
        ok = run_stage(
            stage,
            continue_on_optional_failure=args.continue_on_optional_failure,
            dry_run=args.dry_run,
        )
        if ok:
            completed.append(stage.name)
        else:
            failed_optional.append(stage.name)

    elapsed = time.time() - start
    print("\n" + "=" * 90)
    print("PIPELINE COMPLETE")
    print("=" * 90)
    print(f"Completed stages: {', '.join(completed) if completed else 'none'}")
    if failed_optional:
        print(f"Optional failures: {', '.join(failed_optional)}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

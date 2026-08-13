# sofascore_config_loader.py
#
# Shared helper: loads model_config.json and validates it.
# All pipeline scripts that use model hyperparameters import from here
# instead of hardcoding values.
#
# Usage in any pipeline script:
#
#   from sofascore_config_loader import load_config
#   cfg = load_config(PROJECT_ROOT)
#   PRIOR_WEIGHT      = cfg["prior_weight"]
#   HOME_ADV_CAP      = cfg["home_adv_cap"]
#   RHO               = cfg["rho"]
#   TOURNAMENT_AVG_XG = cfg["tournament_avg_xg"]
#
# If model_config.json is missing, defaults are returned silently so
# the pipeline works on a fresh repo without running the optimizer first.

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CONFIG_FILENAME = "model_config.json"
HISTORY_FILENAME = "model_config_history.jsonl"

# Defaults: used when no config file exists and as validation bounds.
DEFAULTS = {
    "prior_weight":       2.0,
    "home_adv_cap":       1.15,
    "rho":               -0.13,
    "tournament_avg_xg":  1.294,
}

BOUNDS = {
    "prior_weight":       (0.0,  20.0),
    "home_adv_cap":       (1.0,   2.0),
    "rho":               (-1.0,   1.0),
    "tournament_avg_xg":  (0.1,   5.0),
}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_config(project_root: Path, verbose: bool = True) -> dict:
    """
    Load model_config.json from project_root.
    Falls back to DEFAULTS for any missing or invalid key.
    Returns a flat dict of parameter values (no _meta).
    """
    config_path = project_root / CONFIG_FILENAME

    if not config_path.exists():
        if verbose:
            print(
                f"[config] {CONFIG_FILENAME} not found — using defaults. "
                f"Run sofascore_model_optimizer.py --write-config to create it.",
                file=sys.stderr,
            )
        return dict(DEFAULTS)

    with open(config_path) as f:
        raw = json.load(f)

    params = {}
    for key, default in DEFAULTS.items():
        val = raw.get(key, default)
        lo, hi = BOUNDS[key]
        if not (lo <= float(val) <= hi):
            print(
                f"[config] WARNING: {key}={val} is outside valid range [{lo}, {hi}]. "
                f"Falling back to default {default}.",
                file=sys.stderr,
            )
            val = default
        params[key] = float(val)

    if verbose:
        meta = raw.get("_meta", {})
        updated = meta.get("last_updated") or "never"
        metric  = meta.get("optimized_for_metric") or "n/a"
        n       = meta.get("n_matches_used") or "n/a"
        print(
            f"[config] Loaded {CONFIG_FILENAME}  "
            f"(updated: {updated}, optimized for: {metric}, n_matches: {n})"
        )

    return params


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_config(
    project_root: Path,
    params: dict,
    metric_name: str,
    metric_value: float,
    n_matches: int,
    fold_weights: dict | None = None,
    updated_by: str = "sofascore_model_optimizer.py",
) -> None:
    """
    Write params + metadata to model_config.json and append a snapshot
    to model_config_history.jsonl for full audit trail.
    """
    config_path  = project_root / CONFIG_FILENAME
    history_path = project_root / HISTORY_FILENAME

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    config = {
        **{k: params[k] for k in DEFAULTS if k in params},
        "_meta": {
            "last_updated":          now,
            "optimized_for_metric":  metric_name,
            "metric_value":          round(float(metric_value), 6),
            "n_matches_used":        n_matches,
            "fold_weights":          fold_weights,
            "updated_by":            updated_by,
        },
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Append to history log (one JSON object per line)
    with open(history_path, "a") as f:
        f.write(json.dumps(config) + "\n")

    print(f"[config] Wrote {CONFIG_FILENAME}  ({metric_name}={metric_value:.5f}, n={n_matches})")
    print(f"[config] Appended to {HISTORY_FILENAME}")


# ---------------------------------------------------------------------------
# Print current config (standalone usage)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Print current model config.")
    p.add_argument("--project-root", default=".", help="Path to project root.")
    args = p.parse_args()

    root = Path(args.project_root).resolve()
    cfg  = load_config(root, verbose=False)
    print(json.dumps(cfg, indent=2))

    history_path = root / HISTORY_FILENAME
    if history_path.exists():
        with open(history_path) as f:
            lines = f.readlines()
        print(f"\nHistory: {len(lines)} update(s) logged in {HISTORY_FILENAME}")
        if lines:
            last = json.loads(lines[-1])
            print(f"Last update: {last.get('_meta', {}).get('last_updated', 'unknown')}")

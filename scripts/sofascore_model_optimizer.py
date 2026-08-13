# sofascore_model_optimizer.py  —  Stage 5 of the WC 2026 prediction pipeline
#
# Hyperparameter optimizer for the Sofascore Poisson match predictor.
#
# It reuses sofascore_backtest.py directly, runs the same time-respecting folds,
# and scores many combinations of:
#   - prior_weight
#   - home_adv_cap
#   - Dixon-Coles rho
#
# Outputs:
#   data/processed/wc_2026_model_optimization_results.csv
#       One row per parameter combination, using OVERALL model metrics.
#
#   data/processed/wc_2026_model_optimization_fold_metrics.csv
#       Fold-level metrics for every parameter combination.
#
#   data/processed/wc_2026_model_optimization_best_params.csv
#       Best parameter set by each scoring metric.
#
#   outputs/predictions/optimizer_<metric>_by_<parameter>.png
#       One-factor response plots.
#
#   outputs/predictions/optimizer_<metric>_prior_vs_homeadv_best_rho.png
#       Response surface for prior_weight × home_adv_cap at the best rho.

from pathlib import Path
from itertools import product
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sofascore_backtest as bt
from sofascore_config_loader import write_config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "processed"
CHART_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_OUT = OUT_DIR / "wc_2026_model_optimization_results.csv"
FOLD_METRICS_OUT = OUT_DIR / "wc_2026_model_optimization_fold_metrics.csv"
BEST_PARAMS_OUT = OUT_DIR / "wc_2026_model_optimization_best_params.csv"

# Conservative default grid: useful but fast. Expand from the command line.
DEFAULT_PRIOR_WEIGHTS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
DEFAULT_HOME_ADV_CAPS = [1.00, 1.05, 1.10, 1.15, 1.20]
DEFAULT_RHOS = [-0.20, -0.15, -0.13, -0.10, -0.05, 0.00]

METRICS_LOWER_IS_BETTER = ["brier", "brier_sum", "rps", "log_loss"]
METRICS_HIGHER_IS_BETTER = ["accuracy", "mean_actual_prob"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_float_list(text):
    """Parse comma-separated floats from CLI args."""
    if text is None or str(text).strip() == "":
        return []
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def safe_num(x, ndigits=5):
    if pd.isna(x):
        return np.nan
    return round(float(x), ndigits)


def run_backtest_for_params(stats, results, strength, prior_weight, home_adv_cap, rho):
    """Run all backtest folds for one hyperparameter combination."""
    fold_frames = []

    for fold in bt.FOLDS:
        df = bt.run_fold(
            fold,
            stats,
            results,
            strength,
            prior_weight=prior_weight,
            home_adv_cap=home_adv_cap,
            rho=rho,
        )
        if not df.empty:
            fold_frames.append(df)

    if not fold_frames:
        raise RuntimeError("No predictions generated for parameter combination.")

    pred_df = pd.concat(fold_frames, ignore_index=True)
    metric_cols = pred_df.apply(bt.evaluate_prediction, axis=1)
    pred_df = pd.concat([pred_df, metric_cols], axis=1)

    metrics_df = bt.metric_summary(pred_df)
    model_metrics = metrics_df[metrics_df["model"] == "model"].copy()

    for col, val in [
        ("prior_weight", prior_weight),
        ("home_adv_cap", home_adv_cap),
        ("rho", rho),
    ]:
        model_metrics[col] = val

    overall = model_metrics[model_metrics["fold"] == "OVERALL"].copy()
    if overall.empty:
        raise RuntimeError("No OVERALL metrics returned for parameter combination.")

    row = overall.iloc[0].to_dict()
    row.update({
        "prior_weight": prior_weight,
        "home_adv_cap": home_adv_cap,
        "rho": rho,
    })
    return row, model_metrics


def build_best_params_table(results_df):
    rows = []

    for metric in METRICS_LOWER_IS_BETTER:
        if metric not in results_df.columns:
            continue
        best = results_df.sort_values(metric, ascending=True).iloc[0]
        rows.append({
            "selection_metric": metric,
            "direction": "minimize",
            "best_value": safe_num(best[metric]),
            "prior_weight": best["prior_weight"],
            "home_adv_cap": best["home_adv_cap"],
            "rho": best["rho"],
            "brier": best["brier"],
            "rps": best["rps"],
            "log_loss": best["log_loss"],
            "accuracy": best["accuracy"],
        })

    for metric in METRICS_HIGHER_IS_BETTER:
        if metric not in results_df.columns:
            continue
        best = results_df.sort_values(metric, ascending=False).iloc[0]
        rows.append({
            "selection_metric": metric,
            "direction": "maximize",
            "best_value": safe_num(best[metric]),
            "prior_weight": best["prior_weight"],
            "home_adv_cap": best["home_adv_cap"],
            "rho": best["rho"],
            "brier": best["brier"],
            "rps": best["rps"],
            "log_loss": best["log_loss"],
            "accuracy": best["accuracy"],
        })

    return pd.DataFrame(rows)


def plot_one_factor(results_df, parameter, metric):
    """Plot mean metric by one parameter, averaging over other parameters."""
    if parameter not in results_df.columns or metric not in results_df.columns:
        return

    g = (
        results_df.groupby(parameter, as_index=False)[metric]
        .mean()
        .sort_values(parameter)
    )
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(g[parameter], g[metric], marker="o")
    ax.set_xlabel(parameter)
    ax.set_ylabel(f"Mean {metric}")
    direction = "lower is better" if metric in METRICS_LOWER_IS_BETTER else "higher is better"
    ax.set_title(f"Optimizer response: {metric} by {parameter} ({direction})")
    fig.tight_layout()
    out = CHART_DIR / f"optimizer_{metric}_by_{parameter}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prior_home_surface(results_df, metric, best_rho):
    """Heatmap of prior_weight × home_adv_cap at a selected rho."""
    sub = results_df[np.isclose(results_df["rho"].astype(float), float(best_rho))].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="prior_weight",
        columns="home_adv_cap",
        values=metric,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("home_adv_cap")
    ax.set_ylabel("prior_weight")
    ax.set_title(f"{metric} response surface at rho={best_rho}")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    out = CHART_DIR / f"optimizer_{metric}_prior_vs_homeadv_best_rho.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize WC 2026 Sofascore Poisson model hyperparameters.")
    parser.add_argument(
        "--prior-weights",
        default=",".join(str(x) for x in DEFAULT_PRIOR_WEIGHTS),
        help="Comma-separated prior_weight values, e.g. 0.5,1,2,3,4,5",
    )
    parser.add_argument(
        "--home-adv-caps",
        default=",".join(str(x) for x in DEFAULT_HOME_ADV_CAPS),
        help="Comma-separated home_adv_cap values, e.g. 1.0,1.05,1.1,1.15,1.2",
    )
    parser.add_argument(
        "--rhos",
        default=",".join(str(x) for x in DEFAULT_RHOS),
        help="Comma-separated Dixon-Coles rho values, e.g. -0.2,-0.15,-0.13,-0.1,-0.05,0",
    )
    parser.add_argument(
        "--sort-metric",
        default="log_loss",
        choices=METRICS_LOWER_IS_BETTER + METRICS_HIGHER_IS_BETTER,
        help="Metric used for the printed leaderboard.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of leaderboard rows to print.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help=(
            "Write the best parameter set to model_config.json after optimization. "
            "The metric used for selection is --sort-metric. "
            "A full history entry is also appended to model_config_history.jsonl."
        ),
    )
    parser.add_argument(
        "--fold-weight-recent",
        type=float,
        default=1.0,
        help=(
            "Weight multiplier applied to the most recent fold when averaging metrics "
            "across folds. 1.0 = equal weight. 2.0 = latest fold counts twice. "
            "Useful when the latest fold has few matches and you want to downweight it "
            "(use <1.0) or upweight it (use >1.0)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    prior_weights = parse_float_list(args.prior_weights)
    home_adv_caps = parse_float_list(args.home_adv_caps)
    rhos = parse_float_list(args.rhos)

    if not prior_weights or not home_adv_caps or not rhos:
        raise ValueError("At least one value is required for each parameter grid.")

    combos = list(product(prior_weights, home_adv_caps, rhos))
    print("Loading backtest inputs...")
    stats, results, strength = bt.load_inputs()

    print(f"Grid size: {len(combos):,} combinations")
    print(f"  prior_weight : {prior_weights}")
    print(f"  home_adv_cap : {home_adv_caps}")
    print(f"  rho          : {rhos}\n")

    start = time.time()
    result_rows = []
    fold_metric_frames = []

    for i, (prior_weight, home_adv_cap, rho) in enumerate(combos, start=1):
        print(
            f"[{i:>3}/{len(combos):<3}] "
            f"prior={prior_weight:g}, home_adv_cap={home_adv_cap:g}, rho={rho:g}"
        )
        try:
            overall_row, fold_metrics = run_backtest_for_params(
                stats,
                results,
                strength,
                prior_weight=prior_weight,
                home_adv_cap=home_adv_cap,
                rho=rho,
            )
        except Exception as exc:
            print(f"  WARN: failed combination: {exc}")
            continue

        result_rows.append(overall_row)
        fold_metric_frames.append(fold_metrics)

    if not result_rows:
        raise RuntimeError("No optimizer results were generated.")

    results_df = pd.DataFrame(result_rows)
    fold_metrics_df = pd.concat(fold_metric_frames, ignore_index=True)

    # Move parameter columns to the front.
    front = ["prior_weight", "home_adv_cap", "rho"]
    ordered_cols = front + [c for c in results_df.columns if c not in front]
    results_df = results_df[ordered_cols]

    sort_ascending = args.sort_metric in METRICS_LOWER_IS_BETTER
    results_df = results_df.sort_values(args.sort_metric, ascending=sort_ascending).reset_index(drop=True)
    fold_metrics_df = fold_metrics_df.sort_values(
        ["prior_weight", "home_adv_cap", "rho", "fold"]
    ).reset_index(drop=True)

    best_params_df = build_best_params_table(results_df)

    results_df.to_csv(RESULTS_OUT, index=False)
    fold_metrics_df.to_csv(FOLD_METRICS_OUT, index=False)
    best_params_df.to_csv(BEST_PARAMS_OUT, index=False)

    # Plots
    for metric in ["log_loss", "brier", "rps", "accuracy"]:
        for parameter in ["prior_weight", "home_adv_cap", "rho"]:
            plot_one_factor(results_df, parameter, metric)

    if args.sort_metric in results_df.columns:
        best_rho = results_df.iloc[0]["rho"]
        plot_prior_home_surface(results_df, args.sort_metric, best_rho)

    elapsed = time.time() - start

    print("\n" + "=" * 90)
    print(f"MODEL OPTIMIZATION LEADERBOARD — sorted by {args.sort_metric}")
    print("=" * 90)
    show_cols = [
        "prior_weight", "home_adv_cap", "rho", "n_matches",
        "brier", "rps", "log_loss", "accuracy", "mean_actual_prob",
    ]
    show_cols = [c for c in show_cols if c in results_df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(results_df[show_cols].head(args.top_n).to_string(index=False))

    print("\nBest parameter set by metric:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(best_params_df.to_string(index=False))

    print("\nOutputs written:")
    print(f"  Overall grid results -> {RESULTS_OUT}")
    print(f"  Fold-level metrics   -> {FOLD_METRICS_OUT}")
    print(f"  Best params          -> {BEST_PARAMS_OUT}")
    print(f"  Charts               -> {CHART_DIR / 'optimizer_*.png'}")
    # --write-config: persist best params to model_config.json
    if args.write_config:
        best_row = results_df.iloc[0]
        best_params = {
            "prior_weight":      float(best_row["prior_weight"]),
            "home_adv_cap":      float(best_row["home_adv_cap"]),
            "rho":               float(best_row["rho"]),
            "tournament_avg_xg": 1.294,   # updated by match scraper, not optimizer
        }
        n_matches = int(best_row["n_matches"]) if "n_matches" in best_row else None
        metric_val = float(best_row[args.sort_metric]) if args.sort_metric in best_row else None

        print(f"\n--write-config active. Writing best params to model_config.json...")
        print(f"  prior_weight : {best_params['prior_weight']}")
        print(f"  home_adv_cap : {best_params['home_adv_cap']}")
        print(f"  rho          : {best_params['rho']}")
        print(f"  metric       : {args.sort_metric} = {metric_val}")

        # Warn if fold 2 had very few test matches (noisy optimisation signal)
        fold2_rows = fold_metrics_df[
            (fold_metrics_df["fold"] != "OVERALL") &
            (fold_metrics_df["prior_weight"] == best_params["prior_weight"]) &
            (fold_metrics_df["home_adv_cap"] == best_params["home_adv_cap"]) &
            (fold_metrics_df["rho"] == best_params["rho"])
        ]
        if not fold2_rows.empty and "n_matches" in fold2_rows.columns:
            min_fold_n = fold2_rows["n_matches"].min()
            if min_fold_n < 12:
                print(
                    f"\n  WARNING: smallest fold has only {min_fold_n} test matches. "
                    f"Optimised parameters may reflect noise rather than signal. "
                    f"Consider waiting for more matches before writing config."
                )

        write_config(
            project_root=PROJECT_ROOT,
            params=best_params,
            metric_name=args.sort_metric,
            metric_value=metric_val,
            n_matches=n_matches,
            fold_weights={"recent_fold_weight": args.fold_weight_recent},
        )
    else:
        print(
            "\n(Run with --write-config to persist these parameters to model_config.json)"
        )

    print(f"\nElapsed: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
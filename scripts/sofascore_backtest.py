# sofascore_backtest.py  —  Stage 3b of the WC 2026 prediction pipeline
#
# Backtests the current regularised Poisson match predictor on completed
# group-stage matches using time-respecting folds:
#   Fold R1_to_R2  : train on round 1 xG, predict completed round 2 matches
#   Fold R12_to_R3 : train on rounds 1+2 xG, predict completed round 3 matches
#
# Outputs:
#   data/processed/wc_2026_backtest_predictions.csv
#   data/processed/wc_2026_backtest_metrics.csv
#   data/processed/wc_2026_backtest_calibration.csv
#   outputs/predictions/backtest_reliability_*.png

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATCH_STATS_FILE   = PROJECT_ROOT / "data" / "raw" / "sofascore" / "wc_2026_match_team_stats.csv"
MATCH_RESULTS_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "wc_2026_match_results.csv"
TEAM_STRENGTH_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_team_strength.csv"

PRED_OUT = PROJECT_ROOT / "data" / "processed" / "wc_2026_backtest_predictions.csv"
METRICS_OUT = PROJECT_ROOT / "data" / "processed" / "wc_2026_backtest_metrics.csv"
CALIBRATION_OUT = PROJECT_ROOT / "data" / "processed" / "wc_2026_backtest_calibration.csv"
CHART_DIR = PROJECT_ROOT / "outputs" / "predictions"

PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Defaults mirror sofascore_match_prediction.py
PRIOR_WEIGHT = 3.0
MAX_GOALS = 7
MIN_DEFENSE_WEAKNESS = 0.05
HOME_ADV_CAP = 1.15
DIXON_COLES_RHO = -0.13
EPS = 1e-15

NAME_ALIASES = {
    "Curaçao": "Curacao",
    "Curacao": "Curacao",
    "Cabo Verde": "Cape Verde",
    "Cape Verde": "Cape Verde",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Bosnia & Herzegovina": "Bosnia & Herzegovina",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Turkey": "Turkey",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Cote d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",
}

FOLDS = [
    {
        "fold": "R1_to_R2",
        "train_rounds": ["1"],
        "test_rounds": ["2"],
        "description": "Train on round 1 xG, predict completed round 2 matches",
    },
    {
        "fold": "R12_to_R3",
        "train_rounds": ["1", "2"],
        "test_rounds": ["3"],
        "description": "Train on rounds 1+2 xG, predict completed round 3 matches",
    },
]

# ---------------------------------------------------------------------------
# Helpers copied/adapted from sofascore_match_prediction.py
# ---------------------------------------------------------------------------

def normalise(name):
    if pd.isna(name):
        return name
    return NAME_ALIASES.get(str(name).strip(), str(name).strip())


def build_match_xg_table(stats):
    home = (
        stats[stats["side"] == "home"]
        [["event_id", "team", "expectedGoals"]]
        .rename(columns={"team": "home_team", "expectedGoals": "home_xg"})
    )
    away = (
        stats[stats["side"] == "away"]
        [["event_id", "team", "expectedGoals"]]
        .rename(columns={"team": "away_team", "expectedGoals": "away_xg"})
    )
    paired = home.merge(away, on="event_id")

    rows = []
    for _, r in paired.iterrows():
        rows.append({
            "event_id": r["event_id"],
            "team": normalise(r["home_team"]),
            "xg_for": r["home_xg"],
            "xg_against": r["away_xg"],
            "is_home": True,
        })
        rows.append({
            "event_id": r["event_id"],
            "team": normalise(r["away_team"]),
            "xg_for": r["away_xg"],
            "xg_against": r["home_xg"],
            "is_home": False,
        })
    return pd.DataFrame(rows)


def compute_team_xg_stats(match_xg):
    if match_xg.empty:
        return pd.DataFrame(columns=["n_matches", "avg_xg_for", "avg_xg_against"])
    return (
        match_xg
        .groupby("team")
        .agg(
            n_matches=("xg_for", "count"),
            avg_xg_for=("xg_for", "mean"),
            avg_xg_against=("xg_against", "mean"),
        )
        .round(4)
    )


def build_model_params(xg_stats, team_strength, tournament_avg_xg, prior_weight):
    ts = team_strength.copy()
    ts.index = ts.index.map(normalise)

    rows = []
    all_teams = sorted(set(xg_stats.index) | set(ts.index))
    for team in all_teams:
        has_match = team in xg_stats.index
        has_player = team in ts.index

        n = int(xg_stats.loc[team, "n_matches"]) if has_match else 0
        m_att = xg_stats.loc[team, "avg_xg_for"] if has_match else np.nan
        m_def = xg_stats.loc[team, "avg_xg_against"] if has_match else np.nan

        if pd.notna(m_att):
            blended_att = (n * m_att + prior_weight * tournament_avg_xg) / (n + prior_weight)
        else:
            blended_att = tournament_avg_xg

        if pd.notna(m_def):
            blended_def = (n * m_def + prior_weight * tournament_avg_xg) / (n + prior_weight)
        else:
            blended_def = tournament_avg_xg

        p_att_comp = ts.loc[team, "att_composite"] if has_player and "att_composite" in ts.columns else np.nan
        p_def_comp = ts.loc[team, "def_composite"] if has_player and "def_composite" in ts.columns else np.nan

        rows.append({
            "team": team,
            "n_matches": n,
            "match_xg_for": round(m_att, 4) if pd.notna(m_att) else np.nan,
            "match_xg_against": round(m_def, 4) if pd.notna(m_def) else np.nan,
            "blended_attack": round(blended_att, 4),
            "blended_defense": round(blended_def, 4),
            "att_composite": round(p_att_comp, 4) if pd.notna(p_att_comp) else np.nan,
            "def_composite": round(p_def_comp, 4) if pd.notna(p_def_comp) else np.nan,
        })

    params = pd.DataFrame(rows).set_index("team")
    params["attack_strength"] = (params["blended_attack"] / tournament_avg_xg).round(4)
    params["defense_weakness"] = (params["blended_defense"] / tournament_avg_xg).round(4)
    params["defense_weakness"] = params["defense_weakness"].clip(lower=MIN_DEFENSE_WEAKNESS)
    return params


def dc_tau(i, j, mu, nu, rho):
    if i == 0 and j == 0:
        return 1 - mu * nu * rho
    if i == 0 and j == 1:
        return 1 + mu * rho
    if i == 1 and j == 0:
        return 1 + nu * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def predict_match(home, away, params, tournament_avg_xg, home_adv, rho):
    h = normalise(home)
    a = normalise(away)
    if h not in params.index or a not in params.index:
        missing = [x for x in [h, a] if x not in params.index]
        raise KeyError(f"Missing params for {missing}")

    lambda_home = (
        params.loc[h, "attack_strength"]
        * params.loc[a, "defense_weakness"]
        * tournament_avg_xg
        * home_adv
    )
    lambda_away = (
        params.loc[a, "attack_strength"]
        * params.loc[h, "defense_weakness"]
        * tournament_avg_xg
        / home_adv
    )

    goals = np.arange(MAX_GOALS + 1)
    p_home = poisson.pmf(goals, lambda_home)
    p_away = poisson.pmf(goals, lambda_away)
    score_matrix = np.outer(p_home, p_away)

    for i in range(2):
        for j in range(2):
            score_matrix[i, j] *= dc_tau(i, j, lambda_home, lambda_away, rho)
    score_matrix /= score_matrix.sum()

    p_home_win = score_matrix[np.tril_indices(MAX_GOALS + 1, k=-1)].sum()
    p_draw = np.trace(score_matrix)
    p_away_win = score_matrix[np.triu_indices(MAX_GOALS + 1, k=1)].sum()
    flat_idx = np.unravel_index(score_matrix.argmax(), score_matrix.shape)

    return {
        "lambda_home": float(lambda_home),
        "lambda_away": float(lambda_away),
        "p_home_win": float(p_home_win),
        "p_draw": float(p_draw),
        "p_away_win": float(p_away_win),
        "most_likely_score": f"{flat_idx[0]}-{flat_idx[1]}",
        "most_likely_score_prob": float(score_matrix[flat_idx]),
    }

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def actual_outcome(row):
    hs = int(row["home_score"])
    aw = int(row["away_score"])
    if hs > aw:
        return "H", np.array([1.0, 0.0, 0.0])
    if hs == aw:
        return "D", np.array([0.0, 1.0, 0.0])
    return "A", np.array([0.0, 0.0, 1.0])


def brier_score(probs, actual):
    return float(np.mean((probs - actual) ** 2))


def multiclass_brier_sum(probs, actual):
    return float(np.sum((probs - actual) ** 2))


def log_loss(probs, actual):
    idx = int(np.argmax(actual))
    return float(-np.log(max(probs[idx], EPS)))


def ranked_probability_score(probs, actual):
    return float(np.mean((np.cumsum(probs) - np.cumsum(actual)) ** 2))


def evaluate_prediction(row):
    probs = np.array([row["p_home_win"], row["p_draw"], row["p_away_win"]], dtype=float)
    actual = np.array([row["actual_home_win"], row["actual_draw"], row["actual_away_win"]], dtype=float)
    return pd.Series({
        "brier": brier_score(probs, actual),
        "brier_sum": multiclass_brier_sum(probs, actual),
        "log_loss": log_loss(probs, actual),
        "rps": ranked_probability_score(probs, actual),
        "correct": int(row["predicted_outcome"] == row["actual_outcome"]),
        "actual_prob": float(probs[int(np.argmax(actual))]),
    })


def metric_summary(pred_df):
    rows = []
    baselines = {
        "model": None,
        "uniform_baseline": np.array([1 / 3, 1 / 3, 1 / 3]),
        "wc_prior_baseline": np.array([0.45, 0.27, 0.28]),
    }

    for fold, g in pred_df.groupby("fold", sort=False):
        actuals = g[["actual_home_win", "actual_draw", "actual_away_win"]].to_numpy(dtype=float)
        for model_name, fixed_probs in baselines.items():
            if fixed_probs is None:
                probs = g[["p_home_win", "p_draw", "p_away_win"]].to_numpy(dtype=float)
                pred_outcomes = g["predicted_outcome"].to_numpy()
            else:
                probs = np.tile(fixed_probs, (len(g), 1))
                labels = np.array(["H", "D", "A"])
                pred_outcomes = np.repeat(labels[np.argmax(fixed_probs)], len(g))

            brier = np.mean(np.mean((probs - actuals) ** 2, axis=1))
            brier_sum = np.mean(np.sum((probs - actuals) ** 2, axis=1))
            ll = np.mean([-np.log(max(probs[i, np.argmax(actuals[i])], EPS)) for i in range(len(g))])
            rps = np.mean(np.mean((np.cumsum(probs, axis=1) - np.cumsum(actuals, axis=1)) ** 2, axis=1))
            accuracy = np.mean(pred_outcomes == g["actual_outcome"].to_numpy())

            rows.append({
                "fold": fold,
                "model": model_name,
                "n_matches": len(g),
                "brier": round(float(brier), 5),
                "brier_sum": round(float(brier_sum), 5),
                "rps": round(float(rps), 5),
                "log_loss": round(float(ll), 5),
                "accuracy": round(float(accuracy), 5),
                "mean_actual_prob": round(float(np.mean([probs[i, np.argmax(actuals[i])] for i in range(len(g))])), 5),
            })

    # Overall rows
    all_df = pred_df.copy()
    for model_name, fixed_probs in baselines.items():
        actuals = all_df[["actual_home_win", "actual_draw", "actual_away_win"]].to_numpy(dtype=float)
        if fixed_probs is None:
            probs = all_df[["p_home_win", "p_draw", "p_away_win"]].to_numpy(dtype=float)
            pred_outcomes = all_df["predicted_outcome"].to_numpy()
        else:
            probs = np.tile(fixed_probs, (len(all_df), 1))
            labels = np.array(["H", "D", "A"])
            pred_outcomes = np.repeat(labels[np.argmax(fixed_probs)], len(all_df))
        rows.append({
            "fold": "OVERALL",
            "model": model_name,
            "n_matches": len(all_df),
            "brier": round(float(np.mean(np.mean((probs - actuals) ** 2, axis=1))), 5),
            "brier_sum": round(float(np.mean(np.sum((probs - actuals) ** 2, axis=1))), 5),
            "rps": round(float(np.mean(np.mean((np.cumsum(probs, axis=1) - np.cumsum(actuals, axis=1)) ** 2, axis=1))), 5),
            "log_loss": round(float(np.mean([-np.log(max(probs[i, np.argmax(actuals[i])], EPS)) for i in range(len(all_df))])), 5),
            "accuracy": round(float(np.mean(pred_outcomes == all_df["actual_outcome"].to_numpy())), 5),
            "mean_actual_prob": round(float(np.mean([probs[i, np.argmax(actuals[i])] for i in range(len(all_df))])), 5),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calibration(pred_df, n_bins=5):
    rows = []
    events = [
        ("home_win", "p_home_win", "actual_home_win"),
        ("draw", "p_draw", "actual_draw"),
        ("away_win", "p_away_win", "actual_away_win"),
    ]
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for fold_name, fold_df in list(pred_df.groupby("fold", sort=False)) + [("OVERALL", pred_df)]:
        for event_name, prob_col, actual_col in events:
            tmp = fold_df[[prob_col, actual_col]].copy()
            tmp["bin"] = pd.cut(tmp[prob_col], bins=bins, include_lowest=True)
            for interval, b in tmp.groupby("bin", observed=False):
                if len(b) == 0:
                    continue
                rows.append({
                    "fold": fold_name,
                    "event": event_name,
                    "bin": str(interval),
                    "n": len(b),
                    "mean_predicted_prob": round(float(b[prob_col].mean()), 5),
                    "observed_rate": round(float(b[actual_col].mean()), 5),
                    "calibration_error": round(float(b[prob_col].mean() - b[actual_col].mean()), 5),
                })
    return pd.DataFrame(rows)


def plot_reliability(cal_df, fold):
    sub = cal_df[cal_df["fold"] == fold]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for event_name, g in sub.groupby("event"):
        ax.plot(g["mean_predicted_prob"], g["observed_rate"], marker="o", label=event_name)
    ax.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Backtest reliability curve — {fold}")
    ax.legend()
    fig.tight_layout()
    out = CHART_DIR / f"backtest_reliability_{fold}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def load_inputs():
    stats = pd.read_csv(MATCH_STATS_FILE)
    results = pd.read_csv(MATCH_RESULTS_FILE)
    strength = pd.read_csv(TEAM_STRENGTH_FILE, index_col="nation")

    for col in ["team", "opponent", "home_team", "away_team"]:
        if col in stats.columns:
            stats[col] = stats[col].map(normalise)
    for col in ["home_team", "away_team", "winner"]:
        if col in results.columns:
            results[col] = results[col].map(lambda x: normalise(x) if pd.notna(x) else x)
    strength.index = strength.index.map(normalise)

    stats["round"] = stats["round"].astype(str)
    results["round"] = results["round"].astype(str)
    return stats, results, strength


def run_fold(fold, stats, results, strength, prior_weight, home_adv_cap, rho):
    train_rounds = [str(x) for x in fold["train_rounds"]]
    test_rounds = [str(x) for x in fold["test_rounds"]]

    completed_results = results[results["status_code"] == 100].copy()
    train_event_ids = completed_results[completed_results["round"].isin(train_rounds)]["event_id"].unique()
    train_stats = stats[stats["event_id"].isin(train_event_ids)].copy()

    match_xg = build_match_xg_table(train_stats)
    xg_stats = compute_team_xg_stats(match_xg)

    # The tournament average is fit only on the training fold to avoid leakage.
    if xg_stats.empty:
        raise ValueError(f"No training xG data for fold {fold['fold']}")
    tournament_avg_xg = float(xg_stats["avg_xg_for"].mean())

    home_xg = match_xg[match_xg["is_home"]]["xg_for"].mean()
    away_xg = match_xg[~match_xg["is_home"]]["xg_for"].mean()
    raw_home_adv = float(home_xg / away_xg) if away_xg and away_xg > 0 else 1.0
    home_adv = min(raw_home_adv, home_adv_cap) if home_adv_cap is not None else raw_home_adv

    params = build_model_params(xg_stats, strength, tournament_avg_xg, prior_weight)

    test = completed_results[completed_results["round"].isin(test_rounds)].copy()
    test = test[test["home_score"].notna() & test["away_score"].notna()].copy()

    rows = []
    for _, match in test.iterrows():
        try:
            pred = predict_match(
                match["home_team"],
                match["away_team"],
                params,
                tournament_avg_xg,
                home_adv,
                rho,
            )
        except KeyError as exc:
            print(f"WARN: skipping {match['home_team']} vs {match['away_team']}: {exc}")
            continue

        outcome_label, actual = actual_outcome(match)
        probs = np.array([pred["p_home_win"], pred["p_draw"], pred["p_away_win"]])
        pred_label = ["H", "D", "A"][int(np.argmax(probs))]

        rows.append({
            "fold": fold["fold"],
            "description": fold["description"],
            "event_id": match["event_id"],
            "date": match["date"],
            "stage": match.get("stage"),
            "round": match["round"],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "home_score": int(match["home_score"]),
            "away_score": int(match["away_score"]),
            "actual_outcome": outcome_label,
            "actual_home_win": actual[0],
            "actual_draw": actual[1],
            "actual_away_win": actual[2],
            "p_home_win": round(pred["p_home_win"], 5),
            "p_draw": round(pred["p_draw"], 5),
            "p_away_win": round(pred["p_away_win"], 5),
            "predicted_outcome": pred_label,
            "predicted_label": {"H": "home_win", "D": "draw", "A": "away_win"}[pred_label],
            "lambda_home": round(pred["lambda_home"], 5),
            "lambda_away": round(pred["lambda_away"], 5),
            "most_likely_score": pred["most_likely_score"],
            "most_likely_score_prob": round(pred["most_likely_score_prob"], 5),
            "train_rounds": "+".join(train_rounds),
            "test_rounds": "+".join(test_rounds),
            "train_matches": int(len(train_event_ids)),
            "prior_weight": prior_weight,
            "rho": rho,
            "raw_home_adv": round(raw_home_adv, 5),
            "home_adv_used": round(home_adv, 5),
            "tournament_avg_xg_train": round(tournament_avg_xg, 5),
        })

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest the WC 2026 Sofascore Poisson match predictor.")
    parser.add_argument("--prior-weight", type=float, default=PRIOR_WEIGHT)
    parser.add_argument("--rho", type=float, default=DIXON_COLES_RHO)
    parser.add_argument("--home-adv-cap", type=float, default=HOME_ADV_CAP)
    parser.add_argument("--calibration-bins", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    stats, results, strength = load_inputs()

    fold_frames = []
    for fold in FOLDS:
        print(f"\nRunning fold {fold['fold']}: {fold['description']}")
        df = run_fold(
            fold,
            stats,
            results,
            strength,
            prior_weight=args.prior_weight,
            home_adv_cap=args.home_adv_cap,
            rho=args.rho,
        )
        print(f"  Predictions scored: {len(df)}")
        if not df.empty:
            fold_frames.append(df)

    if not fold_frames:
        raise RuntimeError("No backtest predictions were generated. Check completed matches and round labels.")

    pred_df = pd.concat(fold_frames, ignore_index=True)
    metric_cols = pred_df.apply(evaluate_prediction, axis=1)
    pred_df = pd.concat([pred_df, metric_cols], axis=1)

    metrics_df = metric_summary(pred_df)
    cal_df = build_calibration(pred_df, n_bins=args.calibration_bins)

    pred_df.to_csv(PRED_OUT, index=False)
    metrics_df.to_csv(METRICS_OUT, index=False)
    cal_df.to_csv(CALIBRATION_OUT, index=False)

    for fold in list(pred_df["fold"].unique()) + ["OVERALL"]:
        plot_reliability(cal_df, fold)

    print("\n" + "=" * 80)
    print("WC 2026 BACKTEST METRICS")
    print("=" * 80)
    with pd.option_context("display.max_columns", None, "display.width", 150):
        print(metrics_df.to_string(index=False))

    print("\nLargest model misses by log loss:")
    miss_cols = [
        "fold", "home_team", "away_team", "home_score", "away_score",
        "actual_outcome", "p_home_win", "p_draw", "p_away_win", "predicted_outcome", "log_loss"
    ]
    with pd.option_context("display.max_columns", None, "display.width", 150):
        print(pred_df.sort_values("log_loss", ascending=False)[miss_cols].head(10).to_string(index=False))

    print("\nOutputs written:")
    print(f"  Predictions  -> {PRED_OUT}")
    print(f"  Metrics      -> {METRICS_OUT}")
    print(f"  Calibration  -> {CALIBRATION_OUT}")
    print(f"  Charts       -> {CHART_DIR / 'backtest_reliability_*.png'}")


if __name__ == "__main__":
    main()

# sofascore_match_prediction.py  —  Stage 3 of the prediction pipeline
#
# Fits a regularised Poisson goals model and generates win/draw/loss
# probabilities plus scoreline distributions for all upcoming WC 2026 matches.
#
# Model overview
# ──────────────
# For a match between team A (home) and team B (away):
#
#   λ_A = attack_A × (defense_B / μ) × home_adv
#   λ_B = attack_B × (defense_A / μ)
#
# where μ is the tournament average xG per team per match.
#
# attack_i and defense_i are Bayesian blends of:
#   • match-derived estimate  (xG for/against from completed games)
#   • player-derived prior    (poisson_attack_adj / poisson_defense_adj
#                              from sofascore_team_aggregator.py)
#
# The blend weight (PRIOR_WEIGHT) represents how many additional "phantom"
# matches the player prior is worth. At 2–3 real matches per team this
# keeps the prior influential while letting observed data dominate.
# Increase after round 3 when every team has 3 data points.
#
# Outputs
# ───────
#   data/processed/wc_2026_team_model_params.csv
#       Per-team attack/defense estimates and blended model parameters.
#
#   data/processed/wc_2026_match_predictions.csv
#       One row per upcoming match: W/D/L probabilities, expected goals,
#       most likely scoreline.
#
#   outputs/predictions/scorelines_<home>_vs_<away>.csv  (optional)
#       Full P(i–j) matrix up to MAX_GOALS per match.

from pathlib import Path
from itertools import product
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATCH_STATS_FILE   = PROJECT_ROOT / "data" / "raw" / "sofascore" / "wc_2026_match_team_stats.csv"
MATCH_RESULTS_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "wc_2026_match_results.csv"
TEAM_STRENGTH_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_team_strength.csv"

PARAMS_OUT   = PROJECT_ROOT / "data" / "processed" / "wc_2026_team_model_params.csv"
PRED_OUT     = PROJECT_ROOT / "data" / "processed" / "wc_2026_match_predictions.csv"
MODEL_META_OUT = PROJECT_ROOT / "data" / "processed" / "wc_2026_model_metadata.json"
SCORE_DIR    = PROJECT_ROOT / "outputs" / "predictions"

PARAMS_OUT.parent.mkdir(parents=True, exist_ok=True)
SCORE_DIR.mkdir(parents=True, exist_ok=True)

# How many phantom matches the player-level prior is worth.
# With 2–3 real matches per team, 3 keeps the prior meaningful.
# Reduce toward 1 once every team has 4+ matches.
PRIOR_WEIGHT = 2.0

# Scoreline computation cap (P(i goals) for i in 0..MAX_GOALS)
MAX_GOALS = 7

# Floor on defense_weakness. A negative value arises when a team concedes
# 0 goals and the player-level prior (which subtracts defensive stat terms)
# goes below zero — e.g. Spain and Argentina after a perfect group stage.
# A team can't have a negative defensive target; floor at 0.05, meaning the
# model treats even the best defense as allowing 5% of average xG.
MIN_DEFENSE_WEAKNESS = 0.05

# The raw xG-based home advantage sits at ~1.60, inflated by early group
# stage mismatches (Germany 7-1 Curacao, Canada 6-0 Qatar etc.).
# Capping at 1.15 — close to the long-run WC home advantage in the
# literature — keeps lambdas reasonable without removing the signal.
# Set to None to use the raw computed value.
HOME_ADV_CAP = 1.15
RHO = 0.0

# Sofascore uses accented/alternative names in the fixture list that differ
# from what the player scraper stored. Map them to a canonical form here.
NAME_ALIASES = {
    "Curaçao":       "Curacao",
    "Cabo Verde":    "Cape Verde",
    "Cote d'Ivoire": "Côte d'Ivoire",   # just in case
}

# ---------------------------------------------------------------------------
# Team name normalisation
# ---------------------------------------------------------------------------

def normalise(name: str) -> str:
    if pd.isna(name):
        return name
    return NAME_ALIASES.get(str(name).strip(), str(name).strip())

# ---------------------------------------------------------------------------
# Step 1 — per-team xG estimates from completed matches
# ---------------------------------------------------------------------------

def build_match_xg_table(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Pair home and away rows for each event so we can compute
    xG-for and xG-against per team per match.
    """
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
            "team":       normalise(r["home_team"]),
            "xg_for":     r["home_xg"],
            "xg_against": r["away_xg"],
            "is_home":    True,
        })
        rows.append({
            "team":       normalise(r["away_team"]),
            "xg_for":     r["away_xg"],
            "xg_against": r["home_xg"],
            "is_home":    False,
        })

    return pd.DataFrame(rows)


def compute_team_xg_stats(match_xg: pd.DataFrame) -> pd.DataFrame:
    return (
        match_xg
        .groupby("team")
        .agg(
            n_matches     = ("xg_for",    "count"),
            avg_xg_for    = ("xg_for",    "mean"),
            avg_xg_against= ("xg_against","mean"),
        )
        .round(4)
    )

# ---------------------------------------------------------------------------
# Step 2 — blend match estimates with player prior
# ---------------------------------------------------------------------------

def blend(match_val, n_matches, prior_val, prior_weight):
    """
    Weighted average: n real observations + prior_weight phantom observations.
    If either value is NaN, fall back to the other.
    """
    if pd.isna(match_val) and pd.isna(prior_val):
        return np.nan
    if pd.isna(match_val):
        return float(prior_val)
    if pd.isna(prior_val):
        return float(match_val)
    return (n_matches * match_val + prior_weight * prior_val) / (n_matches + prior_weight)


def build_model_params(
    xg_stats: pd.DataFrame,
    team_strength: pd.DataFrame,
    tournament_avg_xg: float,
    prior_weight: float,
) -> pd.DataFrame:
    """
    Returns one row per team with blended attack and defense estimates
    normalised so the tournament mean = 1.0 (multiplicative form).

    Prior design note
    ─────────────────
    The player-level stats from Stage 1 (poisson_attack_adj / defense_adj)
    are weighted AVERAGES of individual player per-90 rates — i.e. roughly
    one player's contribution (~0.3 xG/90), not the team's collective total
    (~1.3 xG/match). Blending them directly with match xG creates a severe
    scale mismatch that suppresses all lambdas by ~3×.

    Instead we use SHRINKAGE TOWARD THE TOURNAMENT MEAN: each team's
    estimate is pulled toward tournament_avg_xg by prior_weight phantom
    matches. This is well-calibrated, scale-correct, and equivalent to a
    Bayesian prior centred on "average team".

    Player composite scores (from Stage 1) still inform predictions in
    Stage 4 (tournament simulation) where they can modulate survival
    probability adjustments. They're preserved in the output CSV.
    """
    ts = team_strength.copy()
    ts.index = ts.index.map(normalise)

    rows = []
    all_teams = sorted(set(xg_stats.index) | set(ts.index))

    for team in all_teams:
        has_match  = team in xg_stats.index
        has_player = team in ts.index

        n     = int(xg_stats.loc[team, "n_matches"])    if has_match  else 0
        m_att = xg_stats.loc[team, "avg_xg_for"]        if has_match  else np.nan
        m_def = xg_stats.loc[team, "avg_xg_against"]    if has_match  else np.nan

        # Shrink toward tournament mean (prior_weight phantom average matches)
        if pd.notna(m_att):
            blended_att = (n * m_att + prior_weight * tournament_avg_xg) / (n + prior_weight)
        else:
            blended_att = tournament_avg_xg   # no data: assume average

        if pd.notna(m_def):
            blended_def = (n * m_def + prior_weight * tournament_avg_xg) / (n + prior_weight)
        else:
            blended_def = tournament_avg_xg

        # Preserve player composite scores for reference / Stage 4
        p_att_comp = ts.loc[team, "att_composite"]  if has_player else np.nan
        p_def_comp = ts.loc[team, "def_composite"]  if has_player else np.nan

        rows.append({
            "team":             team,
            "n_matches":        n,
            "match_xg_for":     round(m_att, 4) if pd.notna(m_att) else np.nan,
            "match_xg_against": round(m_def, 4) if pd.notna(m_def) else np.nan,
            "blended_attack":   round(blended_att, 4),
            "blended_defense":  round(blended_def, 4),
            "att_composite":    round(p_att_comp, 4) if pd.notna(p_att_comp) else np.nan,
            "def_composite":    round(p_def_comp, 4) if pd.notna(p_def_comp) else np.nan,
        })

    params = pd.DataFrame(rows).set_index("team")

    # Normalise to tournament mean: strength/weakness = 1.0 means average
    params["attack_strength"]  = (params["blended_attack"]  / tournament_avg_xg).round(4)
    params["defense_weakness"] = (params["blended_defense"] / tournament_avg_xg).round(4)

    # Floor: even the best defense gets a small positive target
    params["defense_weakness"] = params["defense_weakness"].clip(lower=MIN_DEFENSE_WEAKNESS)

    return params

# ---------------------------------------------------------------------------
# Step 3 — match prediction
# ---------------------------------------------------------------------------

def predict_match(
    home: str,
    away: str,
    params: pd.DataFrame,
    tournament_avg_xg: float,
    home_adv: float,
    neutral: bool = False,
) -> dict:
    """
    Predict a single match. Returns probabilities and scoreline info.

    neutral=True disables the home advantage multiplier (knockouts
    played at pre-assigned neutral venues).
    """
    adv = 1.0 if neutral else home_adv

    h = normalise(home)
    a = normalise(away)

    if h not in params.index or a not in params.index:
        missing = [x for x in [h, a] if x not in params.index]
        return {"home_team": home, "away_team": away, "error": f"Missing params: {missing}"}

    # Expected goals each team will generate
    lambda_home = params.loc[h, "attack_strength"] * params.loc[a, "defense_weakness"] * tournament_avg_xg * adv
    lambda_away = params.loc[a, "attack_strength"] * params.loc[h, "defense_weakness"] * tournament_avg_xg / adv

    # Scoreline probability matrix P(home_goals=i, away_goals=j)
    goals = np.arange(MAX_GOALS + 1)
    p_home = poisson.pmf(goals, lambda_home)
    p_away = poisson.pmf(goals, lambda_away)
    score_matrix = np.outer(p_home, p_away)

    # Dixon-Coles low-score correction
    # Adjusts the slight over-independence between teams when scores are low.
    for i in range(2):
        for j in range(2):
            tau = _dc_tau(i, j, lambda_home, lambda_away, RHO)
            score_matrix[i, j] *= tau

    # Renormalise after correction
    score_matrix /= score_matrix.sum()

    # Win/draw/loss from lower/upper triangles
    p_home_win = score_matrix[np.tril_indices(MAX_GOALS + 1, k=-1)].sum()
    p_draw     = np.trace(score_matrix)
    p_away_win = score_matrix[np.triu_indices(MAX_GOALS + 1, k=1)].sum()

    # Most likely scoreline
    flat_idx   = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
    ml_score   = f"{flat_idx[0]}–{flat_idx[1]}"
    ml_prob    = score_matrix[flat_idx]

    return {
        "home_team":       home,
        "away_team":       away,
        "lambda_home":     round(lambda_home, 3),
        "lambda_away":     round(lambda_away, 3),
        "p_home_win":      round(p_home_win,  3),
        "p_draw":          round(p_draw,       3),
        "p_away_win":      round(p_away_win,   3),
        "most_likely_score": ml_score,
        "most_likely_score_prob": round(float(ml_prob), 3),
        "score_matrix":    score_matrix,   # kept for Stage 4 simulator
    }


def _dc_tau(i, j, mu, nu, rho):
    """Dixon-Coles correction factor for low-scoring cells."""
    if   i == 0 and j == 0: return 1 - mu * nu * rho
    elif i == 0 and j == 1: return 1 + mu * rho
    elif i == 1 and j == 0: return 1 + nu * rho
    elif i == 1 and j == 1: return 1 - rho
    return 1.0


def save_scoreline_matrix(score_matrix: np.ndarray, home: str, away: str):
    """Save full P(i–j) table as CSV for inspection."""
    df = pd.DataFrame(
        score_matrix,
        index   = [f"{home} scores {i}" for i in range(MAX_GOALS + 1)],
        columns = [f"{away} scores {j}" for j in range(MAX_GOALS + 1)],
    ).round(4)
    safe_name = f"{home.replace(' ', '_')}_vs_{away.replace(' ', '_')}"
    df.to_csv(SCORE_DIR / f"scorelines_{safe_name}.csv")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stats   = pd.read_csv(MATCH_STATS_FILE)
    results = pd.read_csv(MATCH_RESULTS_FILE)
    team_strength = pd.read_csv(TEAM_STRENGTH_FILE, index_col="nation")

    # Normalise names across all sources
    stats["team"]     = stats["team"].map(normalise)
    stats["opponent"] = stats["opponent"].map(normalise)

    # ── Tournament-level constants ──────────────────────────────────────────

    match_xg   = build_match_xg_table(stats)
    xg_stats   = compute_team_xg_stats(match_xg)

    tournament_avg_xg = xg_stats["avg_xg_for"].mean()
    print(f"Tournament avg xG per team per match: {tournament_avg_xg:.3f}")

    # Home advantage: ratio of mean home xG to mean away xG
    home_xg = match_xg[match_xg["is_home"]]["xg_for"].mean()
    away_xg = match_xg[~match_xg["is_home"]]["xg_for"].mean()
    home_adv_raw = home_xg / away_xg
    if HOME_ADV_CAP is not None:
        home_adv = min(home_adv_raw, HOME_ADV_CAP)
        print(f"Home advantage factor (xG ratio): {home_adv_raw:.3f}, capped at {home_adv:.3f}")
    else:
        home_adv = home_adv_raw
        print(f"Home advantage factor (xG ratio): {home_adv:.3f} (uncapped)")

    # ── Build team parameters ───────────────────────────────────────────────

    params = build_model_params(xg_stats, team_strength, tournament_avg_xg, PRIOR_WEIGHT)
    params.to_csv(PARAMS_OUT)
    print(f"\nModel params saved → {PARAMS_OUT}")

    model_metadata = {
        "tournament_avg_xg": round(float(tournament_avg_xg), 6),
        "home_adv_raw": round(float(home_adv_raw), 6),
        "home_adv": round(float(home_adv), 6),
        "home_adv_cap": HOME_ADV_CAP,
        "prior_weight": PRIOR_WEIGHT,
        "rho": RHO,
        "max_goals": MAX_GOALS,
        "min_defense_weakness": MIN_DEFENSE_WEAKNESS,
        "source_match_stats_file": str(MATCH_STATS_FILE),
        "source_match_results_file": str(MATCH_RESULTS_FILE),
        "source_team_strength_file": str(TEAM_STRENGTH_FILE),
    }
    MODEL_META_OUT.write_text(json.dumps(model_metadata, indent=2))
    print(f"Model metadata saved → {MODEL_META_OUT}")

    # Quick sanity check: top 10 by attack and defense
    print("\n── Top 10 attack strength ──")
    print(params["attack_strength"].sort_values(ascending=False).head(10).to_string())
    print("\n── Top 10 defense (lowest weakness = best) ──")
    print(params["defense_weakness"].sort_values().head(10).to_string())

    # ── Predict upcoming matches ────────────────────────────────────────────

    upcoming = results[results["status_code"] != 100].copy()

    # Skip matches where team names are still bracket placeholders
    real_upcoming = upcoming[
        ~upcoming["home_team"].str.match(r"^(W|L)\d+$|^\d+[A-Z]$|^[A-Z]\d+$", na=False) &
        ~upcoming["away_team"].str.match(r"^(W|L)\d+$|^\d+[A-Z]$|^[A-Z]\d+$", na=False) &
        ~upcoming["home_team"].str.contains(r"3[A-Z]", na=False) &
        ~upcoming["away_team"].str.contains(r"3[A-Z]", na=False)
    ]

    print(f"\nUpcoming matches with known teams: {len(real_upcoming)}")
    print(f"Bracket placeholders (skipped): {len(upcoming) - len(real_upcoming)}")

    pred_rows = []
    for _, match in real_upcoming.iterrows():
        # Group stage: apply home advantage (venues assigned as home/away)
        # Knockout stages at neutral venues: set neutral=True
        is_knockout = str(match.get("round", "")).lower() not in {"1", "2", "3", "nan"}

        result = predict_match(
            home=match["home_team"],
            away=match["away_team"],
            params=params,
            tournament_avg_xg=tournament_avg_xg,
            home_adv=home_adv,
            neutral=is_knockout,
        )

        if "error" in result:
            print(f"  WARN: {result['error']}")
            continue

        # Save scoreline matrix
        save_scoreline_matrix(
            result.pop("score_matrix"),
            match["home_team"],
            match["away_team"],
        )

        result["date"]  = match["date"]
        result["round"] = match["round"]
        result["stage"] = match.get("stage")
        pred_rows.append(result)

    pred_df = pd.DataFrame(pred_rows).sort_values("date")
    pred_df.to_csv(PRED_OUT, index=False)
    print(f"\nPredictions saved → {PRED_OUT}")

    # ── Print predictions ───────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"  WC 2026 MATCH PREDICTIONS")
    print(f"{'='*80}")
    print(f"{'Date':<18} {'Match':<42} {'Home W':>7} {'Draw':>7} {'Away W':>7} {'Score':>6}")
    print(f"{'-'*18} {'-'*42} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    for _, p in pred_df.iterrows():
        matchup = f"{p['home_team']} vs {p['away_team']}"
        print(
            f"{str(p['date'])[:16]:<18} {matchup:<42} "
            f"{p['p_home_win']:>6.1%} {p['p_draw']:>6.1%} {p['p_away_win']:>6.1%} "
            f"  {p['most_likely_score']}"
        )


if __name__ == "__main__":
    main()
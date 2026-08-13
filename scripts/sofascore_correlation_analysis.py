# sofascore_correlation_analysis.py
#
# Generates per-position correlation matrices for all per-90 stats and audits
# the scatter pairs defined in plot_sofascore_wc_2026.py against them.
#
# Outputs:
#   outputs/correlation/<POSITION>_corr_matrix.csv     — full correlation matrix
#   outputs/correlation/<POSITION>_corr_heatmap.png    — visual heatmap
#   outputs/correlation/scatter_pair_audit.csv         — r-value for every
#                                                         existing scatter pair
#   outputs/correlation/top_pairs_per_position.csv     — highest-r pairs
#                                                         per position (new ideas)

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

# ---------------------------------------------------------------------------
# Configuration — keep in sync with the other pipeline scripts
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE  = PROJECT_ROOT / "data" / "processed" / "wc_2026_model_features.csv"
OUT_DIR     = PROJECT_ROOT / "outputs" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match the threshold in sofascore_country_position_analysis.py
MIN_MINUTES = 30

# Minimum number of players with data in both columns before we bother
# computing a correlation. Below this the r-value is essentially noise.
MIN_OBSERVATIONS = 10

# How many top pairs to surface per position (for the discovery CSV)
TOP_N_PAIRS = 15

# Absolute r threshold below which a scatter pair is flagged as weak
WEAK_R_THRESHOLD = 0.35

POSITION_ORDER = ["GK", "CB", "FB", "DM", "CM", "WM", "AM", "W", "ST"]

# Existing scatter pairs from plot_sofascore_wc_2026.py — audited below.
SCATTER_PAIRS_BY_POSITION = {
    "GK": [
        ("saves_per90", "goalsPrevented"),
        ("saves_per90", "cleanSheet_per90"),
        ("highClaims_per90", "successfulRunsOut_per90"),
        ("savesCaught_per90", "savesParried_per90"),
    ],
    "CB": [
        ("aerialDuelsWon_per90", "aerialDuelsWonPercentage"),
        ("clearances_per90", "outfielderBlocks_per90"),
        ("clearances_per90", "tackles_per90"),
        ("accurateLongBalls_per90", "clearances_per90"),
    ],
    "FB": [
        ("tackles_per90", "interceptions_per90"),
        ("accurateCrosses_per90", "keyPasses_per90"),
        ("ballRecovery_per90", "successfulDribbles_per90"),
    ],
    "DM": [
        ("ballRecovery_per90", "interceptions_per90"),
        ("tacklesWon_per90", "ballRecovery_per90"),
        ("groundDuelsWon_per90", "interceptions_per90"),
        ("possessionLost_per90", "errorLeadToShot_per90"),
        ("totalLongBalls_per90", "keyPasses_per90"),
    ],
    "CM": [
        ("accuratePasses_per90", "accurateFinalThirdPasses_per90"),
        ("accurateOppositionHalfPasses_per90", "keyPasses_per90"),
        ("interceptions_per90", "keyPasses_per90"),
        ("ballRecovery_per90", "totalPasses_per90"),
    ],
    "AM": [
        ("keyPasses_per90", "expectedAssists_per90"),
        ("successfulDribbles_per90", "bigChancesCreated_per90"),
        ("accurateFinalThirdPasses_per90", "expectedAssists_per90"),
    ],
    "WM": [
        ("accurateCrosses_per90", "keyPasses_per90"),
        ("successfulDribbles_per90", "wasFouled_per90"),
        ("ballRecovery_per90", "accurateOppositionHalfPasses_per90"),
        ("dribbledPast_per90", "fouls_per90"),
    ],
    "W": [
        ("successfulDribbles_per90", "expectedAssists_per90"),
        ("expectedAssists_per90", "assists_per90"),
        ("possessionWonAttThird_per90", "expectedAssists_per90"),
        ("accurateCrosses_per90", "expectedAssists_per90"),
    ],
    "ST": [
        ("expectedGoals_per90", "goals_per90"),
        ("shotsOnTarget_per90", "expectedGoals_per90"),
        ("possessionWonAttThird_per90", "expectedGoals_per90"),
        ("successfulDribbles_per90", "expectedGoals_per90"),
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_position(position_string, target_position):
    if pd.isna(position_string):
        return False
    parts = (
        str(position_string)
        .replace(",", " ")
        .replace("/", " ")
        .split()
    )
    return target_position in parts


def compute_corr_matrix(pos_df, per90_cols, min_obs):
    """
    Pearson correlation on per90 columns, requiring at least min_obs
    non-null paired observations. Pairs below that threshold are set to NaN
    rather than returned as spurious r-values from tiny samples.
    """
    cols_present = [c for c in per90_cols if c in pos_df.columns]
    if len(cols_present) < 2:
        return pd.DataFrame()

    subset = pos_df[cols_present].copy()

    # Compute r-matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = subset.corr(method="pearson", min_periods=min_obs)

    return corr


def plot_corr_heatmap(corr, position, n_players, out_path):
    """Triangular heatmap of the correlation matrix."""
    if corr.empty or corr.shape[0] < 2:
        return

    n = corr.shape[0]
    fig_size = max(10, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Upper triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    data = corr.values.copy()
    data[mask] = np.nan

    im = ax.imshow(data, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [c.replace("_per90", "").replace("_", " ") for c in corr.columns],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [c.replace("_per90", "").replace("_", " ") for c in corr.index],
        fontsize=7,
    )

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            if np.isnan(val):
                continue
            txt_col = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=txt_col)

    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.6)
    ax.set_title(
        f"WC 2026 — {position} per-90 correlations  "
        f"(n={n_players}, min {MIN_MINUTES} min played)",
        fontsize=11, pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_pair_r(corr, col_a, col_b):
    """Return the r-value for a pair, or NaN if either col is missing."""
    if col_a in corr.index and col_b in corr.columns:
        return corr.loc[col_a, col_b]
    if col_b in corr.index and col_a in corr.columns:
        return corr.loc[col_b, col_a]
    return np.nan


def top_pairs_from_corr(corr, n):
    """
    Extract the N highest absolute-r pairs from the lower triangle,
    excluding self-correlations and NaN entries.
    """
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i):
            r = corr.iloc[i, j]
            if pd.notna(r):
                pairs.append((cols[i], cols[j], r))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_FILE)

    df_qualified = df[
        df["minutesPlayed"].notna() & (df["minutesPlayed"] >= MIN_MINUTES)
    ].copy()

    per90_cols = [c for c in df.columns if c.endswith("_per90")]

    print(f"Players meeting MIN_MINUTES ({MIN_MINUTES}): {len(df_qualified):,}")
    print(f"Per90 columns: {len(per90_cols)}")
    print(f"MIN_OBSERVATIONS for valid r: {MIN_OBSERVATIONS}\n")

    audit_rows   = []   # one row per existing scatter pair
    top_pair_rows = []  # one row per top-N discovery pair

    for position in POSITION_ORDER:
        pos_df = df_qualified[
            df_qualified["position"].apply(lambda x: has_position(x, position))
        ]

        n_players = len(pos_df)
        print(f"{'=' * 60}")
        print(f"  {position}  ({n_players} players)")
        print(f"{'=' * 60}")

        if n_players < 3:
            print("  Too few players — skipping.\n")
            continue

        corr = compute_corr_matrix(pos_df, per90_cols, MIN_OBSERVATIONS)

        if corr.empty:
            print("  No per90 columns with enough data — skipping.\n")
            continue

        # Save matrix CSV
        csv_path = OUT_DIR / f"{position}_corr_matrix.csv"
        corr.to_csv(csv_path)

        # Save heatmap PNG
        png_path = OUT_DIR / f"{position}_corr_heatmap.png"
        plot_corr_heatmap(corr, position, n_players, png_path)
        print(f"  Heatmap → {png_path}")

        # --- Audit existing scatter pairs ---
        existing_pairs = SCATTER_PAIRS_BY_POSITION.get(position, [])
        if existing_pairs:
            print(f"\n  Existing scatter pair audit:")
            print(f"  {'Column A':<45} {'Column B':<45} {'r':>6}  Status")
            print(f"  {'-'*45} {'-'*45} {'------':>6}  ------")

        for col_a, col_b in existing_pairs:
            r = get_pair_r(corr, col_a, col_b)
            if pd.isna(r):
                status = "⚠ missing data"
            elif abs(r) < WEAK_R_THRESHOLD:
                status = "✗ weak"
            elif abs(r) < 0.60:
                status = "~ moderate"
            else:
                status = "✓ strong"

            r_str = f"{r:.3f}" if pd.notna(r) else "  NaN"
            print(f"  {col_a:<45} {col_b:<45} {r_str:>6}  {status}")

            audit_rows.append({
                "position": position,
                "col_a": col_a,
                "col_b": col_b,
                "r": round(r, 4) if pd.notna(r) else np.nan,
                "abs_r": round(abs(r), 4) if pd.notna(r) else np.nan,
                "status": status,
                "n_players": n_players,
            })

        # --- Top N pairs discovery ---
        top = top_pairs_from_corr(corr, TOP_N_PAIRS)
        if top:
            print(f"\n  Top {TOP_N_PAIRS} pairs by |r| (new ideas):")
            print(f"  {'Column A':<45} {'Column B':<45} {'r':>6}")
            print(f"  {'-'*45} {'-'*45} {'------':>6}")
            for col_a, col_b, r in top:
                marker = "  ★" if (col_a, col_b) in existing_pairs or \
                                   (col_b, col_a) in existing_pairs else ""
                print(f"  {col_a:<45} {col_b:<45} {r:>6.3f}{marker}")
                top_pair_rows.append({
                    "position": position,
                    "col_a": col_a,
                    "col_b": col_b,
                    "r": round(r, 4),
                    "abs_r": round(abs(r), 4),
                    "already_plotted": marker.strip() == "★",
                    "n_players": n_players,
                })

        print()

    # --- Summary CSVs ---
    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        audit_path = OUT_DIR / "scatter_pair_audit.csv"
        audit_df.sort_values(["position", "abs_r"], ascending=[True, False]) \
                .to_csv(audit_path, index=False)
        print(f"Audit CSV → {audit_path}")

        # Print overall summary
        total   = len(audit_df)
        strong  = (audit_df["abs_r"] >= 0.60).sum()
        mod     = ((audit_df["abs_r"] >= WEAK_R_THRESHOLD) & (audit_df["abs_r"] < 0.60)).sum()
        weak    = (audit_df["abs_r"] < WEAK_R_THRESHOLD).sum()
        missing = audit_df["r"].isna().sum()

        print(f"\n{'=' * 60}")
        print(f"  SCATTER PAIR AUDIT SUMMARY  ({total} existing pairs)")
        print(f"{'=' * 60}")
        print(f"  Strong   (|r| ≥ 0.60) : {strong:>3}  ({strong/total*100:.0f}%)")
        print(f"  Moderate (|r| ≥ 0.35) : {mod:>3}  ({mod/total*100:.0f}%)")
        print(f"  Weak     (|r| < 0.35) : {weak:>3}  ({weak/total*100:.0f}%)  ← consider removing")
        print(f"  Missing data          : {missing:>3}")

    top_df = pd.DataFrame(top_pair_rows)
    if not top_df.empty:
        top_path = OUT_DIR / "top_pairs_per_position.csv"
        top_df.to_csv(top_path, index=False)
        print(f"Top pairs CSV → {top_path}")

    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
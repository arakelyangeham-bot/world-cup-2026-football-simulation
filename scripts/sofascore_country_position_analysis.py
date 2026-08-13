# sofascore_country_position_analysis.py
#
# Groups WC 2026 player stats by nation + position and produces:
#   1. outputs/eda/country_position_per90_averages.csv  — full grouped table
#   2. outputs/eda/country_position_<position>.csv       — one CSV per position
#   3. Printed summary tables per position
#   4. Heatmap charts per position saved to outputs/charts/

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_model_features.csv"
EDA_DIR    = PROJECT_ROOT / "outputs" / "eda"
CHART_DIR  = PROJECT_ROOT / "outputs" / "charts"

EDA_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Position ordering and display labels
# ---------------------------------------------------------------------------

POSITION_ORDER = ["GK", "CB", "FB", "DM", "CM", "WM", "AM", "W", "ST"]

# Stat subsets that are actually meaningful for each position group.
# Per90 columns are built dynamically from these base names.
POSITION_STATS = {
    "GK": [
        "saves", "cleanSheet", "goalsConceded", "goalsPrevented",
        "savedShotsFromInsideTheBox", "savedShotsFromOutsideTheBox",
        "penaltySave", "runsOut", "successfulRunsOut", "highClaims",
        "punches", "goalsConcededInsideTheBox", "goalsConcededOutsideTheBox",
    ],
    "CB": [
        "clearances", "interceptions", "aerialDuelsWon", "tackles",
        "blockShots", "errorLeadToGoal", "errorLeadToShot",
        "groundDuelsWon", "totalDuelsWon", "outfielderBlocks",
        "accurateLongBalls", "ballRecovery", "dribbledPast",
    ],
    "FB": [
        "tackles", "interceptions", "clearances", "accurateCrosses",
        "successfulDribbles", "aerialDuelsWon", "groundDuelsWon",
        "keyPasses", "assists", "ballRecovery", "wasFouled",
        "possessionWonAttThird",
    ],
    "DM": [
        "tackles", "interceptions", "groundDuelsWon", "aerialDuelsWon",
        "ballRecovery", "accuratePasses", "accurateLongBalls",
        "clearances", "fouls", "possessionLost", "keyPasses",
        "outfielderBlocks",
    ],
    "CM": [
        "keyPasses", "assists", "accuratePasses", "totalPasses",
        "accurateFinalThirdPasses", "successfulDribbles", "goals",
        "groundDuelsWon", "tackles", "interceptions", "passToAssist",
        "bigChancesCreated", "possessionWonAttThird",
    ],
    "WM": [
        "keyPasses", "assists", "successfulDribbles", "accurateCrosses",
        "goals", "bigChancesCreated", "passToAssist", "totalShots",
        "shotsOnTarget", "wasFouled", "dispossessed",
        "accurateFinalThirdPasses",
    ],
    "AM": [
        "keyPasses", "assists", "goals", "bigChancesCreated",
        "successfulDribbles", "passToAssist", "expectedAssists",
        "totalShots", "shotsOnTarget", "expectedGoals",
        "accurateFinalThirdPasses", "wasFouled",
    ],
    "W": [
        "goals", "assists", "successfulDribbles", "totalShots",
        "shotsOnTarget", "expectedGoals", "bigChancesCreated",
        "wasFouled", "accurateCrosses", "keyPasses", "dispossessed",
        "passToAssist",
    ],
    "ST": [
        "goals", "assists", "totalShots", "shotsOnTarget",
        "expectedGoals", "bigChancesCreated", "bigChancesMissed",
        "headedGoals", "goalsFromInsideTheBox", "goalsFromOutsideTheBox",
        "aerialDuelsWon", "offsides", "wasFouled",
    ],
}

# Nicer axis labels for charts (base stat name → readable string)
READABLE = {
    "saves":                      "Saves",
    "cleanSheet":                 "Clean sheets",
    "goalsConceded":              "Goals conceded",
    "goalsPrevented":             "Goals prevented",
    "savedShotsFromInsideTheBox": "Saves inside box",
    "savedShotsFromOutsideTheBox":"Saves outside box",
    "penaltySave":                "Penalty saves",
    "runsOut":                    "Runs out",
    "successfulRunsOut":          "Successful runs out",
    "highClaims":                 "High claims",
    "punches":                    "Punches",
    "goalsConcededInsideTheBox":  "Goals conceded inside box",
    "goalsConcededOutsideTheBox": "Goals conceded outside box",
    "clearances":                 "Clearances",
    "interceptions":              "Interceptions",
    "aerialDuelsWon":             "Aerial duels won",
    "tackles":                    "Tackles",
    "blockShots":                 "Blocked shots",
    "errorLeadToGoal":            "Errors → goal",
    "errorLeadToShot":            "Errors → shot",
    "groundDuelsWon":             "Ground duels won",
    "totalDuelsWon":              "Total duels won",
    "outfielderBlocks":           "Outfielder blocks",
    "accurateLongBalls":          "Accurate long balls",
    "ballRecovery":               "Ball recoveries",
    "dribbledPast":               "Dribbled past",
    "accurateCrosses":            "Accurate crosses",
    "successfulDribbles":         "Successful dribbles",
    "keyPasses":                  "Key passes",
    "assists":                    "Assists",
    "wasFouled":                  "Was fouled",
    "possessionWonAttThird":      "Poss won att. third",
    "accuratePasses":             "Accurate passes",
    "accurateLongBalls":          "Accurate long balls",
    "fouls":                      "Fouls",
    "possessionLost":             "Possession lost",
    "totalPasses":                "Total passes",
    "accurateFinalThirdPasses":   "Acc. final-third passes",
    "goals":                      "Goals",
    "passToAssist":               "Pre-assist passes",
    "bigChancesCreated":          "Big chances created",
    "bigChancesMissed":           "Big chances missed",
    "dispossessed":               "Dispossessed",
    "totalShots":                 "Total shots",
    "shotsOnTarget":              "Shots on target",
    "expectedGoals":              "xG",
    "expectedAssists":            "xA",
    "goalsFromInsideTheBox":      "Goals inside box",
    "goalsFromOutsideTheBox":     "Goals outside box",
    "headedGoals":                "Headed goals",
    "offsides":                   "Offsides",
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum minutes a player must have played to be included in country/position
# averages. Raise this as the tournament progresses and more minutes accumulate.
# At group stage start: 30 is lenient but filters pure cameos.
# Suggested progression:
#   Group stage start : 30
#   After round 2    : 60
#   Knockouts        : 90+
MIN_MINUTES = 30

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

missing = [c for c in ["nation", "position"] if c not in df.columns]
if missing:
    raise ValueError(f"Required columns not found: {missing}")

# Split into qualified and excluded pools so we can report what was dropped
df_all = df[df["minutesPlayed"].notna() & (df["minutesPlayed"] > 0)].copy()
df = df_all[df_all["minutesPlayed"] >= MIN_MINUTES].copy()
dropped = df_all[df_all["minutesPlayed"] < MIN_MINUTES]

print(f"MIN_MINUTES threshold : {MIN_MINUTES}")
print(f"Players with minutes  : {len(df_all):,}")
print(f"Players included      : {len(df):,}")
if len(dropped):
    print(f"Players excluded      : {len(dropped):,}  "
          f"(max {dropped['minutesPlayed'].max():.0f} mins — raise MIN_MINUTES to include)")

per90_cols = [c for c in df.columns if c.endswith("_per90")]

print(f"Nations              : {df['nation'].nunique()}")
print(f"Per90 columns        : {len(per90_cols)}")

# ---------------------------------------------------------------------------
# Full grouped table (all positions, all per90 stats)
# ---------------------------------------------------------------------------

grouped_all = (
    df.groupby(["nation", "position"])[per90_cols]
    .mean()
    .round(3)
)

grouped_all.to_csv(EDA_DIR / "country_position_per90_averages.csv")
print(f"\nWrote: {EDA_DIR / 'country_position_per90_averages.csv'}")

# ---------------------------------------------------------------------------
# Per-position CSVs + printed tables + heatmap charts
# ---------------------------------------------------------------------------

def build_per90_col_list(base_stats, available_cols):
    """Return only _per90 columns that exist in the dataframe."""
    wanted = [f"{s}_per90" for s in base_stats]
    return [c for c in wanted if c in available_cols]


def readable_col(col):
    """Strip _per90 suffix and map to human label."""
    base = col.replace("_per90", "")
    return READABLE.get(base, base)


def plot_heatmap(pivot, position, out_path):
    """
    One row per nation, one column per stat.
    Values are z-scored per column so different-scale stats are comparable.
    Green = above average for that stat, red = below.
    """
    if pivot.empty or pivot.shape[0] < 2:
        print(f"  [skip chart] not enough nations for {position}")
        return

    # z-score per column (stat) so scale differences don't dominate colour
    z = pivot.copy()
    for col in z.columns:
        std = z[col].std()
        if std > 0:
            z[col] = (z[col] - z[col].mean()) / std
        else:
            z[col] = 0.0

    n_nations, n_stats = z.shape
    fig_w = max(12, n_stats * 1.1)
    fig_h = max(6,  n_nations * 0.45)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        z.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-2.5,
        vmax=2.5,
    )

    # Axis labels
    ax.set_xticks(range(n_stats))
    ax.set_xticklabels(
        [readable_col(c) for c in z.columns],
        rotation=40,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(n_nations))
    ax.set_yticklabels(z.index, fontsize=8)

    # Annotate cells with the raw (non-z-scored) value
    raw = pivot.reindex(index=z.index, columns=z.columns)
    for i in range(n_nations):
        for j in range(n_stats):
            val = raw.iloc[i, j]
            txt = f"{val:.2f}" if pd.notna(val) else "—"
            # pick contrasting text colour
            bg = z.iloc[i, j]
            text_color = "white" if abs(bg) > 1.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6.5, color=text_color)

    plt.colorbar(im, ax=ax, label="z-score vs. tournament average", shrink=0.6)
    ax.set_title(
        f"WC 2026 — {position}: per-90 averages by nation (z-scored)",
        fontsize=12,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart → {out_path}")


for position in POSITION_ORDER:
    pos_df = df[df["position"] == position]

    if pos_df.empty:
        print(f"\n[{position}] No players found — skipping.")
        continue

    stat_cols = build_per90_col_list(POSITION_STATS[position], per90_cols)

    if not stat_cols:
        print(f"\n[{position}] No matching per90 columns — skipping.")
        continue

    # Group by nation, average the relevant per90 stats
    pos_grouped = (
        pos_df.groupby("nation")[stat_cols]
        .mean()
        .round(3)
        .sort_index()
    )

    # Also add player count per nation for context
    player_counts = pos_df.groupby("nation").size().rename("n_players")
    pos_grouped = pos_grouped.join(player_counts)

    # Write CSV
    csv_path = EDA_DIR / f"country_position_{position}.csv"
    pos_grouped.to_csv(csv_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  {position}  ({len(pos_df)} players, {pos_grouped.shape[0]} nations)")
    print(f"{'=' * 60}")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", "{:.3f}".format,
    ):
        # Show top 10 nations by number of players for brevity in terminal
        print(pos_grouped.sort_values("n_players", ascending=False).head(10).to_string())
    print(f"  CSV  → {csv_path}")

    # Heatmap (drop n_players column before plotting)
    chart_path = CHART_DIR / f"heatmap_{position}.png"
    plot_heatmap(
        pivot=pos_grouped.drop(columns="n_players"),
        position=position,
        out_path=chart_path,
    )

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

print(f"\n{'=' * 60}")
print("All outputs written.")
print(f"  CSVs   : {EDA_DIR}")
print(f"  Charts : {CHART_DIR}")
print(f"{'=' * 60}")
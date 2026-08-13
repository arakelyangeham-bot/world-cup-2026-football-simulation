# sofascore_team_aggregator.py  —  Stage 1 of the prediction pipeline
#
# Collapses player-level per-90 stats into team-level strength metrics,
# weighted by minutes played and split by position group.
#
# Position groups:
#   GK      : GK
#   DEFENSE : CB, FB
#   MIDFIELD: DM, CM, WM
#   ATTACK  : AM, W, ST
#
# Output: data/processed/wc_2026_team_strength.csv
#   One row per nation. Columns are prefixed by group (gk_, def_, mid_, att_)
#   plus a set of derived composite scores used directly by the Poisson model.
#
# Also writes: outputs/eda/team_strength_coverage.csv
#   Minutes of data available per nation per position group — useful for
#   flagging nations where thin coverage makes the estimates unreliable.

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wc_2026_model_features.csv"
OUT_FILE   = PROJECT_ROOT / "data" / "processed" / "wc_2026_team_strength.csv"
COV_FILE   = PROJECT_ROOT / "outputs" / "eda" / "team_strength_coverage.csv"

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
COV_FILE.parent.mkdir(parents=True, exist_ok=True)

# Match threshold in the other pipeline scripts.
MIN_MINUTES = 30

# Position group membership
POSITION_GROUPS = {
    "GK":      ["GK"],
    "DEFENSE": ["CB", "FB"],
    "MIDFIELD":["DM", "CM", "WM"],
    "ATTACK":  ["AM", "W", "ST"],
}

# Per-90 stats to aggregate for each group.
# Each entry is (column, aggregation_intent) where intent is one of:
#   "higher_better"  — more is stronger (xG, saves, key passes …)
#   "lower_better"   — less is stronger (goals conceded, errors …)
#   "neutral"        — informational, not directionally scored
#
# Only stats from TOTAL_STATS in feature_engineering.py are guaranteed
# to have _per90 variants; the rest are included only if present.

GK_STATS = [
    ("saves_per90",                       "higher_better"),
    ("cleanSheet_per90",                  "higher_better"),
    ("goalsConceded_per90",               "lower_better"),
    ("goalsPrevented_per90",              "higher_better"),
    ("savedShotsFromInsideTheBox_per90",  "higher_better"),
    ("savedShotsFromOutsideTheBox_per90", "higher_better"),
    ("penaltySave_per90",                 "higher_better"),
    ("runsOut_per90",                     "neutral"),
    ("successfulRunsOut_per90",           "higher_better"),
    ("highClaims_per90",                  "higher_better"),
    ("savesCaught_per90",                 "higher_better"),
    ("savesParried_per90",                "neutral"),
]

DEFENSE_STATS = [
    ("clearances_per90",          "higher_better"),
    ("aerialDuelsWon_per90",      "higher_better"),
    ("interceptions_per90",       "higher_better"),
    ("tackles_per90",             "higher_better"),
    ("tacklesWon_per90",          "higher_better"),
    ("outfielderBlocks_per90",    "higher_better"),
    ("ballRecovery_per90",        "higher_better"),
    ("accurateLongBalls_per90",   "neutral"),
    ("dribbledPast_per90",        "lower_better"),
    ("errorLeadToGoal_per90",     "lower_better"),
    ("errorLeadToShot_per90",     "lower_better"),
    ("groundDuelsWon_per90",      "higher_better"),
    ("totalDuelsWon_per90",       "higher_better"),
    ("fouls_per90",               "lower_better"),
    ("possessionWonAttThird_per90","higher_better"),
]

MIDFIELD_STATS = [
    ("accuratePasses_per90",              "higher_better"),
    ("accurateFinalThirdPasses_per90",    "higher_better"),
    ("accurateOppositionHalfPasses_per90","higher_better"),
    ("keyPasses_per90",                   "higher_better"),
    ("bigChancesCreated_per90",           "higher_better"),
    ("expectedAssists_per90",             "higher_better"),
    ("assists_per90",                     "higher_better"),
    ("passToAssist_per90",                "higher_better"),
    ("ballRecovery_per90",                "higher_better"),
    ("interceptions_per90",               "higher_better"),
    ("tackles_per90",                     "higher_better"),
    ("successfulDribbles_per90",          "higher_better"),
    ("possessionLost_per90",              "lower_better"),
    ("dispossessed_per90",                "lower_better"),
]

ATTACK_STATS = [
    ("goals_per90",                "higher_better"),
    ("expectedGoals_per90",        "higher_better"),
    ("shotsOnTarget_per90",        "higher_better"),
    ("totalShots_per90",           "higher_better"),
    ("shotsFromInsideTheBox_per90","higher_better"),
    ("bigChancesCreated_per90",    "higher_better"),
    ("bigChancesMissed_per90",     "lower_better"),
    ("assists_per90",              "higher_better"),
    ("expectedAssists_per90",      "higher_better"),
    ("keyPasses_per90",            "higher_better"),
    ("successfulDribbles_per90",   "higher_better"),
    ("wasFouled_per90",            "higher_better"),
    ("aerialDuelsWon_per90",       "higher_better"),
    ("headedGoals_per90",          "neutral"),
    ("penaltyGoals_per90",         "neutral"),
]

GROUP_STATS = {
    "GK":      GK_STATS,
    "DEFENSE": DEFENSE_STATS,
    "MIDFIELD":MIDFIELD_STATS,
    "ATTACK":  ATTACK_STATS,
}

# Prefix used in the output column names
GROUP_PREFIX = {
    "GK":      "gk",
    "DEFENSE": "def",
    "MIDFIELD":"mid",
    "ATTACK":  "att",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_position(position_string, targets):
    """Return True if any of `targets` appears in a player's position field."""
    if pd.isna(position_string):
        return False
    parts = (
        str(position_string)
        .replace(",", " ")
        .replace("/", " ")
        .split()
    )
    return any(t in parts for t in targets)


def weighted_mean(series, weights):
    """Minutes-weighted average, ignoring NaN rows."""
    mask = series.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(series[mask], weights=weights[mask])


def aggregate_group(group_df, stats, prefix):
    """
    Compute the minutes-weighted mean of each stat for a single
    (nation, position_group) slice. Returns a dict of prefixed columns.
    """
    row = {}
    w = group_df["minutesPlayed"]

    for col, _ in stats:
        if col not in group_df.columns:
            row[f"{prefix}_{col}"] = np.nan
        else:
            row[f"{prefix}_{col}"] = weighted_mean(group_df[col], w)

    return row

# ---------------------------------------------------------------------------
# Composite scores
# ---------------------------------------------------------------------------
# These are single numbers summarising each group's contribution to
# attacking threat and defensive solidity. They're what the Poisson model
# will use directly; the per-stat columns are available for deeper analysis.
#
# Methodology: z-score each constituent stat across all nations, then average
# the z-scores for "higher_better" stats and subtract z-scores for
# "lower_better" stats, giving a single signed composite where higher = better.

def build_composites(df, stats, prefix, group_name):
    """
    Add a single composite score column per position group.
    Returns the modified dataframe.
    """
    higher = [f"{prefix}_{c}" for c, d in stats if d == "higher_better"
              and f"{prefix}_{c}" in df.columns]
    lower  = [f"{prefix}_{c}" for c, d in stats if d == "lower_better"
              and f"{prefix}_{c}" in df.columns]

    z_scores = pd.DataFrame(index=df.index)

    for col in higher:
        std = df[col].std()
        if std > 0:
            z_scores[col] = (df[col] - df[col].mean()) / std
        else:
            z_scores[col] = 0.0

    for col in lower:
        std = df[col].std()
        if std > 0:
            # Invert: being lower than average is good, so negate the z-score
            z_scores[col] = -((df[col] - df[col].mean()) / std)
        else:
            z_scores[col] = 0.0

    all_z_cols = list(z_scores.columns)
    if all_z_cols:
        df[f"{prefix}_composite"] = z_scores[all_z_cols].mean(axis=1)
    else:
        df[f"{prefix}_composite"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_FILE)

    # Filter to players with meaningful minutes
    df = df[df["minutesPlayed"].notna() & (df["minutesPlayed"] >= MIN_MINUTES)].copy()
    print(f"Players with >= {MIN_MINUTES} minutes: {len(df):,}")
    print(f"Nations: {df['nation'].nunique()}")

    nations = sorted(df["nation"].dropna().unique())
    team_rows = []
    coverage_rows = []

    for nation in nations:
        nation_df = df[df["nation"] == nation]
        row = {"nation": nation}
        cov = {"nation": nation}

        for group_name, positions in POSITION_GROUPS.items():
            prefix = GROUP_PREFIX[group_name]
            stats  = GROUP_STATS[group_name]

            group_df = nation_df[
                nation_df["position"].apply(lambda x: has_position(x, positions))
            ]

            # Coverage: total minutes available for this group
            total_mins   = group_df["minutesPlayed"].sum()
            n_players    = len(group_df)
            cov[f"{group_name}_minutes"] = total_mins
            cov[f"{group_name}_players"] = n_players

            if group_df.empty:
                # Fill with NaN so the nation still appears in the output
                for col, _ in stats:
                    row[f"{prefix}_{col}"] = np.nan
            else:
                row.update(aggregate_group(group_df, stats, prefix))

        team_rows.append(row)
        coverage_rows.append(cov)

    team_df = pd.DataFrame(team_rows).set_index("nation")
    cov_df  = pd.DataFrame(coverage_rows).set_index("nation")

    # ---------------------------------------------------------------------------
    # Composite scores
    # ---------------------------------------------------------------------------
    for group_name, stats in GROUP_STATS.items():
        prefix = GROUP_PREFIX[group_name]
        team_df = build_composites(team_df, stats, prefix, group_name)

    # ---------------------------------------------------------------------------
    # Poisson model inputs
    # ---------------------------------------------------------------------------
    # Four numbers per team that Stage 3 will use directly.
    # They're derived here rather than inside the model so the aggregator
    # is the single source of truth for what "team strength" means.

    # Attack rate: expected goals generated per 90 by attacking players.
    # Primary signal for how many goals a team scores.
    team_df["poisson_attack"] = team_df["att_expectedGoals_per90"].fillna(
        team_df["att_goals_per90"]   # fallback if xG missing
    )

    # Defense rate: goals conceded per 90 by the GK (directly observed).
    # Primary signal for how many goals a team lets in.
    team_df["poisson_defense"] = team_df["gk_goalsConceded_per90"]

    # Attack quality adjustment: shots on target rate boosts attack signal.
    # Captures teams that generate high-quality chances vs shot-happy teams.
    team_df["poisson_attack_adj"] = (
        team_df["poisson_attack"] * 0.7
        + team_df["att_shotsOnTarget_per90"].fillna(0) * 0.1
        + team_df["mid_expectedAssists_per90"].fillna(0) * 0.2
    )

    # Defense quality adjustment: incorporates GK save quality and
    # outfield defensive pressure.
    team_df["poisson_defense_adj"] = (
        team_df["poisson_defense"] * 0.6
        + team_df["gk_goalsPrevented_per90"].fillna(0) * (-0.2)   # negative = fewer conceded
        + team_df["def_interceptions_per90"].fillna(0) * (-0.1)
        + team_df["def_clearances_per90"].fillna(0) * (-0.05)
        + team_df["def_errorLeadToGoal_per90"].fillna(0) * 0.25   # errors inflate conceded
    )

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    team_df = team_df.round(4)
    team_df.to_csv(OUT_FILE)
    cov_df.to_csv(COV_FILE)

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  TEAM STRENGTH SUMMARY")
    print(f"{'=' * 60}")

    summary_cols = [
        "att_composite",
        "mid_composite",
        "def_composite",
        "gk_composite",
        "poisson_attack_adj",
        "poisson_defense_adj",
    ]
    present = [c for c in summary_cols if c in team_df.columns]

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", "{:.3f}".format,
    ):
        print(
            team_df[present]
            .sort_values("att_composite", ascending=False)
            .to_string()
        )

    print(f"\nRows    : {len(team_df)}")
    print(f"Columns : {len(team_df.columns)}")
    print(f"\nTeam strength → {OUT_FILE}")
    print(f"Coverage      → {COV_FILE}")

    # Warn on thin coverage
    print(f"\n{'=' * 60}")
    print("  COVERAGE WARNINGS  (< 90 mins in a position group)")
    print(f"{'=' * 60}")
    warned = False
    for group_name in POSITION_GROUPS:
        col = f"{group_name}_minutes"
        thin = cov_df[cov_df[col] < 90]
        if not thin.empty:
            print(f"\n  {group_name}: {len(thin)} nations below 90 mins")
            print(f"  {', '.join(thin.index.tolist())}")
            warned = True
    if not warned:
        print("  None — all nations have >= 90 mins per group.")


if __name__ == "__main__":
    main()
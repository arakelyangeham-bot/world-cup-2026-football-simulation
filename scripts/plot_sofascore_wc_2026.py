import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import plotly.express as px

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Players below this threshold are plotted in a distinct colour as a warning
# that their per-90 stats are unreliable (small sample). Raise in tandem with
# MIN_MINUTES in sofascore_country_position_analysis.py as the tournament goes on.
#   Group stage start : 30
#   After round 2    : 60
#   Knockouts        : 90+
LOW_MINUTES_THRESHOLD = 30

MINUTES_COLOUR_MAP = {
    "Sufficient minutes": "#1f77b4",                          # blue
    f"Low minutes (<{LOW_MINUTES_THRESHOLD})": "#d62728",     # red
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE    = PROCESSED_DIR / "wc_2026_model_features.csv"

EDA_DIR = PROJECT_ROOT / "reports" / "figures"
EDA_DIR.mkdir(parents=True, exist_ok=True)

INTERACTIVE_DIR = PROJECT_ROOT / "reports" / "interactive"
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scatter pairs per position
# ---------------------------------------------------------------------------

SCATTER_PAIRS_BY_POSITION = {
    "GK": [
        ("saves_per90", "goalsPrevented"),
        ("saves_per90", "cleanSheet_per90"),
        ("highClaims_per90", "successfulRunsOut_per90"),
        ("savesCaught_per90", "savesParried_per90")
    ],
    "CB": [
        ("aerialDuelsWon_per90", "longBalls_per90"),
        ("clearances_per90", "totalPasses_per90"),
        ("interceptions_per90", "totalPasses_per90")
    ],
    "FB": [
        ("tackles_per90", "interceptions_per90"),
        ("accurateCrosses_per90", "keyPasses_per90"),
        ("ballRecovery_per90", "successfulDribbles_per90")
    ],
    "DM": [
        ("ballRecovery_per90", "totalPasses_per90"),
        ("tacklesWon_per90", "longBalls_per90"),
        ("interceptions_per90", "keyPasses_per90")
    ],
    "CM": [
        ("accurateFinalThirdPasses_per90", "keyPasses_per90"),
        ("interceptions_per90", "keyPasses_per90"),
        ("ballRecovery_per90", "totalPasses_per90")
    ],
    "AM": [
        ("keyPasses_per90", "expectedAssists_per90"),
        ("successfulDribbles_per90", "expectedAssists_per90"),
        ("accurateFinalThirdPasses_per90", "expectedAssists_per90"),
    ],
    "WM": [
        ("accurateCrosses_per90", "keyPasses_per90"),
        ("successfulDribbles_per90", "wasFouled_per90"),
        ("ballRecovery_per90", "accurateOppositionHalfPasses_per90"),
        ("dribbledPast_per90", "fouls_per90")
    ],
    "W": [
        ("successfulDribbles_per90", "expectedAssists_per90"),
        ("expectedAssists_per90", "assists_per90"),
        ("possessionWonAttThird_per90", "expectedAssists_per90"),
        ("accurateCrosses_per90", "expectedAssists_per90")
    ],
    "ST": [
        ("expectedGoals_per90", "goals_per90"),
        ("shotsOnTarget_per90", "expectedGoals_per90"),
        ("touches_per90", "expectedGoals_per90"),
        ("successfulDribbles_per90", "expectedGoals_per90")
    ]
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_position(position_string, target_position):
    if pd.isna(position_string):
        return False
    positions = (
        str(position_string)
        .replace(",", " ")
        .replace("/", " ")
        .split()
    )
    return target_position in positions


def make_minutes_flag(minutes_series, threshold):
    """Return a Series with human-readable flag labels for colouring."""
    low_label = f"Low minutes (<{threshold})"
    return minutes_series.apply(
        lambda m: "Sufficient minutes"
        if pd.notna(m) and m >= threshold
        else low_label
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_FILE)

    # Attach a minutes-flag column once; it travels with every subset.
    df["minutes_flag"] = make_minutes_flag(
        df["minutesPlayed"], LOW_MINUTES_THRESHOLD
    )

    low_label = f"Low minutes (<{LOW_MINUTES_THRESHOLD})"
    n_flagged = (df["minutes_flag"] == low_label).sum()
    print(f"Players flagged as low-minutes (<{LOW_MINUTES_THRESHOLD} min): {n_flagged}")

    for position, pairs in SCATTER_PAIRS_BY_POSITION.items():
        position_df = df[df["position"].apply(lambda x: has_position(x, position))]

        for x_col, y_col in pairs:
            if x_col not in df.columns or y_col not in df.columns:
                print(f"Skipping {position}: missing {x_col} or {y_col}")
                continue

            scatter_df = position_df.dropna(subset=[x_col, y_col])

            if scatter_df.empty:
                continue

            hover_fields = [
                c for c in [
                    "player",
                    "country",
                    "position",
                    "minutesPlayed",
                    "minutes_flag",
                ]
                if c in scatter_df.columns
            ]

            # Ensure both category labels always appear in the legend, even
            # if the filtered subset happens to contain only one of them.
            present_flags  = set(scatter_df["minutes_flag"].unique())
            all_flags      = set(MINUTES_COLOUR_MAP.keys())
            missing_flags  = all_flags - present_flags
            category_order = list(MINUTES_COLOUR_MAP.keys())

            if missing_flags:
                # Append a phantom row off-screen so the legend entry exists.
                phantom_rows = []
                for flag in missing_flags:
                    phantom = {col: None for col in scatter_df.columns}
                    phantom[x_col] = None
                    phantom[y_col] = None
                    phantom["minutes_flag"] = flag
                    phantom_rows.append(phantom)
                scatter_df = pd.concat(
                    [scatter_df, pd.DataFrame(phantom_rows)],
                    ignore_index=True,
                )

            n_low = (scatter_df["minutes_flag"] == low_label).sum()
            print(
                f"{position}: {x_col} vs {y_col} "
                f"({len(scatter_df)} players, {n_low} flagged)"
            )

            fig = px.scatter(
                scatter_df,
                x=x_col,
                y=y_col,
                color="minutes_flag",
                color_discrete_map=MINUTES_COLOUR_MAP,
                category_orders={"minutes_flag": category_order},
                hover_data=hover_fields,
                title=(
                    f"{position}: {x_col} vs {y_col}"
                    f"<br><sup>Red = fewer than {LOW_MINUTES_THRESHOLD} minutes played"
                    f" — treat per-90 values with caution</sup>"
                ),
            )

            # Make flagged points slightly transparent and smaller so they
            # don't dominate the chart but are still clearly visible.
            for trace in fig.data:
                if trace.name == low_label:
                    trace.marker.opacity = 0.55
                    trace.marker.size    = 7
                    trace.marker.symbol  = "diamond"
                else:
                    trace.marker.opacity = 0.85
                    trace.marker.size    = 9

            fig.write_html(
                INTERACTIVE_DIR / f"{position.lower()}_{x_col}_vs_{y_col}.html"
            )


if __name__ == "__main__":
    main()
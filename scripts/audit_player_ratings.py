#audit_player_ratings.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "data" / "processed" / "player_ratings.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "player_ratings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_FILE)

ROLE_COLUMNS = [
    "rating_GK",
    "rating_CB",
    "rating_FB",
    "rating_DM",
    "rating_CM",
    "rating_AM",
    "rating_WM",
    "rating_W",
    "rating_ST",
]

for rating_col in ROLE_COLUMNS:
    if rating_col not in df.columns:
        continue

    role = rating_col.replace("rating_", "")

    top = (
        df.dropna(subset=[rating_col])
        .sort_values(rating_col, ascending=False)
        .head(25)
    )

    cols = [
        "player",
        "team",
        "current_team",
        "country",
        "eligible_roles",
        "minutesPlayed",
        "rating",
        rating_col,
    ]

    cols = [c for c in cols if c in top.columns]

    top[cols].to_csv(
        OUT_DIR / f"top_{role}.csv",
        index=False,
    )

    print(f"\nTop {role}")
    print(top[cols].head(10))

summary = df[ROLE_COLUMNS].describe().T
summary.to_csv(OUT_DIR / "rating_summary.csv")

print("\nRating summary")
print(summary)

print(f"\nWrote audits to {OUT_DIR}")
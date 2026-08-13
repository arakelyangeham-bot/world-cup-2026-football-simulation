#build_role_feature_manifest.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "role_feature_manifest.csv"

ROLE_FEATURES = {
    "GK": [
        ("rating", 1.5, False),
        ("saves_per90", 2.0, True),
        ("goalsPrevented", 2.0, False),
        ("cleanSheet_per90", 1.0, True),
    ],
    "CB": [
        ("rating", 1.5, False),
        ("clearances_per90", 1.5, True),
        ("aerialDuelsWon_per90", 1.5, True),
        ("interceptions_per90", 1.5, True),
    ],
    "FB": [
        ("rating", 1.5, False),
        ("tackles_per90", 1.3, True),
        ("interceptions_per90", 1.2, True),
        ("accurateCrosses_per90", 1.2, True),
        ("keyPasses_per90", 1.0, True),
    ],
    "DM": [
        ("rating", 1.5, False),
        ("ballRecovery_per90", 1.5, True),
        ("tacklesWon_per90", 1.4, True),
        ("interceptions_per90", 1.4, True),
        ("totalPasses_per90", 1.0, True),
    ],
    "CM": [
        ("rating", 1.5, False),
        ("totalPasses_per90", 1.2, True),
        ("keyPasses_per90", 1.2, True),
        ("accurateFinalThirdPasses_per90", 1.3, True),
    ],
    "AM": [
        ("rating", 1.5, False),
        ("keyPasses_per90", 1.5, True),
        ("expectedAssists_per90", 1.6, False),
        ("successfulDribbles_per90", 1.1, True),
    ],
    "WM": [
        ("rating", 1.5, False),
        ("accurateCrosses_per90", 1.5, True),
        ("keyPasses_per90", 1.3, True),
        ("ballRecovery_per90", 0.8, True),
    ],
    "W": [
        ("rating", 1.5, False),
        ("expectedAssists_per90", 1.4, False),
        ("successfulDribbles_per90", 1.4, True),
        ("goals_per90", 1.2, True),
    ],
    "ST": [
        ("rating", 1.5, False),
        ("expectedGoals_per90", 1.8, False),
        ("goals_per90", 1.8, True),
        ("shotsOnTarget_per90", 1.3, True),
    ],
}

rows = []

for role, features in ROLE_FEATURES.items():
    for feature, weight, required in features:
        rows.append({
            "role": role,
            "feature": feature,
            "weight": weight,
            "required": required,
            "minimum_coverage": 0.0,
        })

df = pd.DataFrame(rows)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_FILE, index=False)

print(df)
print(f"Wrote: {OUT_FILE}")
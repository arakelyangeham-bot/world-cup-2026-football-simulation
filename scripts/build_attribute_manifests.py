#build_attribute_manifests.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "sofascore"

FEATURE_ATTRIBUTE_FILE = OUT_DIR / "feature_attribute_manifest.csv"
ROLE_ATTRIBUTE_FILE = OUT_DIR / "role_attribute_manifest.csv"

FEATURE_ATTRIBUTES = [
    ("goals_per90", "finishing", 1.8),
    ("shotsOnTarget_per90", "finishing", 1.3),
    ("expectedGoals_per90", "finishing", 1.8),

    ("assists_per90", "chance_creation", 1.4),
    ("keyPasses_per90", "chance_creation", 1.5),
    ("expectedAssists_per90", "chance_creation", 1.6),
    ("accurateCrosses_per90", "chance_creation", 1.1),

    ("successfulDribbles_per90", "ball_carrying", 1.4),
    ("wasFouled_per90", "ball_carrying", 0.8),

    ("totalPasses_per90", "passing", 1.2),
    ("accurateFinalThirdPasses_per90", "passing", 1.3),
    ("accurateLongBalls_per90", "passing", 0.9),

    ("tackles_per90", "defending", 1.2),
    ("tacklesWon_per90", "defending", 1.3),
    ("interceptions_per90", "defending", 1.4),
    ("ballRecovery_per90", "defending", 1.2),
    ("clearances_per90", "defending", 1.0),

    ("aerialDuelsWon_per90", "aerial", 1.4),

    ("saves_per90", "goalkeeping", 1.8),
    ("goalsPrevented", "goalkeeping", 2.0),
    ("cleanSheet_per90", "goalkeeping", 1.0),

    ("rating", "overall", 1.5),
]

ROLE_ATTRIBUTES = [
    ("GK", "goalkeeping", 2.0),
    ("GK", "passing", 0.5),
    ("GK", "overall", 0.8),

    ("CB", "defending", 2.0),
    ("CB", "aerial", 1.3),
    ("CB", "passing", 0.7),
    ("CB", "overall", 0.8),

    ("FB", "defending", 1.4),
    ("FB", "chance_creation", 1.0),
    ("FB", "ball_carrying", 0.8),
    ("FB", "overall", 0.8),

    ("DM", "defending", 1.7),
    ("DM", "passing", 1.2),
    ("DM", "overall", 0.8),

    ("CM", "passing", 1.6),
    ("CM", "chance_creation", 0.9),
    ("CM", "defending", 0.8),
    ("CM", "overall", 0.8),

    ("AM", "chance_creation", 1.8),
    ("AM", "ball_carrying", 1.0),
    ("AM", "finishing", 0.8),
    ("AM", "overall", 0.8),

    ("WM", "chance_creation", 1.5),
    ("WM", "defending", 0.9),
    ("WM", "ball_carrying", 0.8),
    ("WM", "overall", 0.8),

    ("W", "chance_creation", 1.5),
    ("W", "ball_carrying", 1.4),
    ("W", "finishing", 1.0),
    ("W", "overall", 0.8),

    ("ST", "finishing", 2.0),
    ("ST", "chance_creation", 0.7),
    ("ST", "ball_carrying", 0.5),
    ("ST", "overall", 0.8),
]

feature_df = pd.DataFrame(
    FEATURE_ATTRIBUTES,
    columns=["feature", "attribute", "weight"],
)

role_df = pd.DataFrame(
    ROLE_ATTRIBUTES,
    columns=["role", "attribute", "weight"],
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

feature_df.to_csv(FEATURE_ATTRIBUTE_FILE, index=False)
role_df.to_csv(ROLE_ATTRIBUTE_FILE, index=False)

print(feature_df)
print(role_df)
print(f"Wrote: {FEATURE_ATTRIBUTE_FILE}")
print(f"Wrote: {ROLE_ATTRIBUTE_FILE}")
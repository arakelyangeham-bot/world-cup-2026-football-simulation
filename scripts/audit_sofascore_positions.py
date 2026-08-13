#audit_sofascore_positions.py

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILES_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "sofascore_player_profiles.csv"
OUT_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "position_audit.csv"

df = pd.read_csv(PROFILES_FILE)

audit = (
    df.groupby(["position", "positions_detailed"], dropna=False)
    .size()
    .reset_index(name="player_count")
    .sort_values("player_count", ascending=False)
)

print(audit.head(50))

audit.to_csv(OUT_FILE, index=False)

print(f"Wrote: {OUT_FILE}")
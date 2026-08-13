#check_processed_player_pipeline

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_IDS = {
    26768,
    980418,
    1129940,
    1130647,
    1142248,
}

FILES = {
    "player_registry": PROJECT_ROOT / "data" / "processed" / "player_registry.csv",
    "player_attribute_scores": PROJECT_ROOT / "data" / "processed" / "player_attribute_scores.csv",
    "player_ratings": PROJECT_ROOT / "data" / "processed" / "player_ratings.csv",
}

for name, path in FILES.items():
    print("=" * 60)
    print(name)
    print("=" * 60)

    df = pd.read_csv(path)

    ids = pd.to_numeric(
        df["player_id"],
        errors="coerce",
    ).astype("Int64")

    matches = df.loc[
        ids.isin(TARGET_IDS),
        [c for c in df.columns if c in ("player_id", "player")]
    ]

    print(matches)

    found = set(
        ids[ids.isin(TARGET_IDS)]
        .dropna()
        .astype(int)
        .tolist()
    )

    print("Found:", sorted(found))
    print("Missing:", sorted(TARGET_IDS - found))
    print()
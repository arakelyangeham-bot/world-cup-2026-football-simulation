#check_bayern_player_ids

import pandas as pd


TARGET_IDS = {
    1142248,
    1129940,
    1130647,
    980418,
    26768,
}

ratings = pd.read_csv(
    "data/processed/player_ratings.csv"
)

ratings["normalized_player_id"] = (
    pd.to_numeric(
        ratings["player_id"],
        errors="coerce",
    )
    .astype("Int64")
)

matches = ratings[
    ratings["normalized_player_id"].isin(
        TARGET_IDS
    )
]

print("player_id dtype:")
print(ratings["player_id"].dtype)
print()

print("Matching player-rating rows:")
print(
    matches[
        [
            "player_id",
            "normalized_player_id",
            "player",
        ]
    ].to_string(index=False)
)

print()
print(
    "Matched IDs:",
    sorted(
        matches[
            "normalized_player_id"
        ]
        .dropna()
        .astype(int)
        .tolist()
    ),
)

print(
    "Missing IDs:",
    sorted(
        TARGET_IDS
        - set(
            matches[
                "normalized_player_id"
            ]
            .dropna()
            .astype(int)
            .tolist()
        )
    ),
)
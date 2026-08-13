from pathlib import Path

import pandas as pd

from research.player_intelligence.starting_xi_builder import (
    StartingXIBuilder,
    parse_roles,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_FILE = PROJECT_ROOT / "data" / "processed" / "player_ratings.csv"
FORMATION_FILE = PROJECT_ROOT / "data" / "raw" / "sofascore" / "formation_manifest.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "expected_lineups.csv"

FORMATION = "4-3-3"


if __name__ == "__main__":
    ratings = pd.read_csv(RATINGS_FILE)
    formations = pd.read_csv(FORMATION_FILE)

    formation_df = formations[formations["formation"] == FORMATION]

    ratings["eligible_roles_list"] = ratings["eligible_roles"].apply(parse_roles)

    builder = StartingXIBuilder(formation=FORMATION)

    all_lineups = []

    for country, country_df in ratings.groupby("country"):
        print(f"Building lineup for {country}")

        lineup = builder.build_for_team(
            team_df=country_df,
            formation_df=formation_df,
        )

        for row in lineup:
            row["country"] = country
            row["formation"] = FORMATION

        all_lineups.extend(lineup)

    out = pd.DataFrame(all_lineups)

    out.to_csv(OUT_FILE, index=False)

    print(out.head(30))
    print(f"Wrote: {OUT_FILE}")
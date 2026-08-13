#scoreline_distribution_analysis.py

from pathlib import Path
from collections import Counter

import pandas as pd

RAW_DIR = Path("data/raw/sofascore")
YEARS = [2010, 2014, 2018, 2022]


def canonical_score(a: int, b: int) -> str:
    """
    Treat 2-1 and 1-2 as the same scoreline.
    """
    high = max(a, b)
    low = min(a, b)
    return f"{high}-{low}"


def main():

    counter = Counter()
    total_matches = 0

    for year in YEARS:
        df = pd.read_csv(RAW_DIR / f"wc_{year}_match_results.csv")

        for _, row in df.iterrows():
            score = canonical_score(
                int(row.home_score),
                int(row.away_score),
            )

            counter[score] += 1
            total_matches += 1

    distribution = (
        pd.DataFrame(
            [
                {
                    "scoreline": score,
                    "count": count,
                    "probability": count / total_matches,
                }
                for score, count in counter.items()
            ]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    print(distribution.head(25))

    distribution.to_csv(
        "outputs/historical_world_cups/historical_scoreline_distribution.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
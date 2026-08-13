#audit_team_name_matching.py

from pathlib import Path

import pandas as pd

from team_strength_loader import load_poisson_team_strengths
from team_name_normalizer import normalize_team_name


RAW_DIR = Path("data/raw/sofascore")
YEARS = [2010, 2014, 2018, 2022]


def main():
    strengths = load_poisson_team_strengths()
    strength_names = set(strengths.keys())

    historical_raw_names = set()
    historical_normalized_names = set()

    for year in YEARS:
        path = RAW_DIR / f"wc_{year}_match_results.csv"
        df = pd.read_csv(path)

        historical_raw_names.update(df["home_team"].dropna().astype(str))
        historical_raw_names.update(df["away_team"].dropna().astype(str))

        historical_normalized_names.update(
            df["home_team"].dropna().astype(str).map(normalize_team_name)
        )
        historical_normalized_names.update(
            df["away_team"].dropna().astype(str).map(normalize_team_name)
        )

    missing_after_normalization = sorted(
        historical_normalized_names - strength_names
    )

    strength_only = sorted(
        strength_names - historical_normalized_names
    )

    print("Team Name Matching Audit")
    print("------------------------")
    print(f"Strength teams: {len(strength_names)}")
    print(f"Historical raw teams: {len(historical_raw_names)}")
    print(f"Historical normalized teams: {len(historical_normalized_names)}")

    print()
    print("Historical teams missing from strengths after normalization")
    print("-----------------------------------------------------------")
    for name in missing_after_normalization:
        print(name)

    print()
    print("Strength teams not present in historical World Cups")
    print("---------------------------------------------------")
    for name in strength_only:
        print(name)

    out_dir = Path("outputs/team_name_matching")
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"missing_after_normalization": missing_after_normalization}).to_csv(
        out_dir / "missing_after_normalization.csv",
        index=False,
    )

    pd.DataFrame({"strength_only": strength_only}).to_csv(
        out_dir / "strength_only.csv",
        index=False,
    )

    print()
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
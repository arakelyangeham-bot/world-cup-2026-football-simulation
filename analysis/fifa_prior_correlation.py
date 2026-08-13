#fifa_prior_correlation.py

import pandas as pd

from shared.config import PROJECT_ROOT
from shared.national_team_priors import load_national_team_priors


TEAM_STRENGTH_PATH = PROJECT_ROOT / "data" / "processed" / "wc_2026_team_strength.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "team_strength_analysis"
OUTPUT_CSV = OUTPUT_DIR / "fifa_prior_correlation.csv"


FEATURE_COLUMNS = [
    "gk_composite",
    "def_composite",
    "mid_composite",
    "att_composite",
    "poisson_attack",
    "poisson_defense",
    "poisson_attack_adj",
    "poisson_defense_adj",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strengths = pd.read_csv(TEAM_STRENGTH_PATH)
    priors = load_national_team_priors()

    rows = []

    for _, row in strengths.iterrows():
        team = row["nation"]

        if team not in priors:
            print(f"Missing FIFA prior: {team}")
            continue

        merged = {
            "team": team,
            "fifa_rank": priors[team]["fifa_rank"],
            "fifa_points": priors[team]["fifa_points"],
        }

        for col in FEATURE_COLUMNS:
            merged[col] = row[col]

        rows.append(merged)

    df = pd.DataFrame(rows)

    correlation = df[
        ["fifa_rank", "fifa_points"] + FEATURE_COLUMNS
    ].corr(numeric_only=True)

    fifa_corr = correlation.loc[
        ["fifa_rank", "fifa_points"],
        FEATURE_COLUMNS,
    ].T.reset_index()

    fifa_corr = fifa_corr.rename(columns={"index": "feature"})

    fifa_corr.to_csv(OUTPUT_CSV, index=False)

    print("FIFA prior correlation complete.")
    print(fifa_corr.to_string(index=False))
    print()
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
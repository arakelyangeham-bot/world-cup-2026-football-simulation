#build_match_feature_table.py

from itertools import permutations
from pathlib import Path

import pandas as pd

from match_engine import poisson_expected_goals
from team_strength_loader import TEAM_STRENGTH_FILE


OUTPUT_DIR = Path("outputs/match_engine")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(TEAM_STRENGTH_FILE)

    required = {
        "nation",
        "att_composite",
        "mid_composite",
        "def_composite",
        "gk_composite",
        "poisson_attack_adj",
        "poisson_defense_adj",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    teams = df.set_index("nation").to_dict("index")

    rows = []

    for team_a, team_b in permutations(sorted(teams), 2):
        a = teams[team_a]
        b = teams[team_b]

        lambda_a, lambda_b = poisson_expected_goals(
            {
                "attack": a["poisson_attack_adj"],
                "defense": a["poisson_defense_adj"],
            },
            {
                "attack": b["poisson_attack_adj"],
                "defense": b["poisson_defense_adj"],
            },
        )

        rows.append({
            "team_a": team_a,
            "team_b": team_b,

            "team_a_attack": a["att_composite"],
            "team_a_midfield": a["mid_composite"],
            "team_a_defense": a["def_composite"],
            "team_a_gk": a["gk_composite"],

            "team_b_attack": b["att_composite"],
            "team_b_midfield": b["mid_composite"],
            "team_b_defense": b["def_composite"],
            "team_b_gk": b["gk_composite"],

            "attack_diff": a["att_composite"] - b["att_composite"],
            "midfield_diff": a["mid_composite"] - b["mid_composite"],
            "defense_diff": a["def_composite"] - b["def_composite"],
            "gk_diff": a["gk_composite"] - b["gk_composite"],

            "team_a_poisson_attack": a["poisson_attack_adj"],
            "team_a_poisson_defense": a["poisson_defense_adj"],
            "team_b_poisson_attack": b["poisson_attack_adj"],
            "team_b_poisson_defense": b["poisson_defense_adj"],

            "poisson_lambda_a": lambda_a,
            "poisson_lambda_b": lambda_b,
            "poisson_lambda_diff": lambda_a - lambda_b,
            "poisson_total_lambda": lambda_a + lambda_b,
        })

    feature_table = pd.DataFrame(rows)

    out_file = OUTPUT_DIR / "match_feature_table.csv"
    feature_table.to_csv(out_file, index=False)

    print("Match Feature Table")
    print("-------------------")
    print(f"Teams: {len(teams)}")
    print(f"Rows: {len(feature_table)}")
    print()
    print(feature_table.head(10).to_string(index=False))
    print()
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
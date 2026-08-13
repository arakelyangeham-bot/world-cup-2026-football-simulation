#wc2026_data.py

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROSTER_PATH = PROJECT_ROOT / "data" / "roster" / "world_cup_2026_roster_with_sofascore_ids.csv"


def load_groups_from_roster() -> dict[str, list[str]]:
    roster = pd.read_csv(ROSTER_PATH)

    required_columns = {"nation", "group"}
    missing_columns = required_columns - set(roster.columns)

    if missing_columns:
        raise ValueError(f"Roster file missing columns: {sorted(missing_columns)}")

    teams = (
        roster[["nation", "group"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["group", "nation"])
    )

    groups = {
        group: group_df["nation"].tolist()
        for group, group_df in teams.groupby("group", sort=True)
    }

    return groups


GROUPS = load_groups_from_roster()


TEAM_TO_GROUP = {
    team: group
    for group, teams in GROUPS.items()
    for team in teams
}


ALL_TEAMS = [
    team
    for teams in GROUPS.values()
    for team in teams
]


def validate_world_cup_data() -> None:
    if len(GROUPS) != 12:
        raise ValueError(f"Expected 12 groups, got {len(GROUPS)}")

    for group, teams in GROUPS.items():
        if len(teams) != 4:
            raise ValueError(f"Group {group} must have 4 teams, got {len(teams)}")

    if len(ALL_TEAMS) != 48:
        raise ValueError(f"Expected 48 teams, got {len(ALL_TEAMS)}")

    if len(set(ALL_TEAMS)) != 48:
        raise ValueError("Duplicate team names found.")

    if len(TEAM_TO_GROUP) != 48:
        raise ValueError("TEAM_TO_GROUP mapping is incomplete.")


if __name__ == "__main__":
    validate_world_cup_data()

    print("World Cup data smoke test passed.")
    print("Groups:", len(GROUPS))
    print("Teams:", len(ALL_TEAMS))

    for group, teams in GROUPS.items():
        print(group, teams)
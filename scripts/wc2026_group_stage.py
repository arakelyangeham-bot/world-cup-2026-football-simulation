#wc2026_group_stage.py

import random
from itertools import combinations
from scripts.wc2026_data import GROUPS
from dataclasses import dataclass, asdict
from scripts.team_strength_loader import load_team_repository
from shared.team_name_normalizer import normalize_team_name

@dataclass
class GroupStanding:
    team: str
    group: str
    finish: int
    points: int
    goals_for: int
    goals_against: int
    goal_difference: int
    qualification_type: str | None = None

@dataclass
class GroupMatchResult:
    group: str
    team1: str
    team2: str
    goals_team1: int
    goals_team2: int

def simulate_fake_group_stage() -> dict[str, list[str]]:
    """
    Temporary deterministic group-stage output.

    Returns each group ordered from 1st to 4th.
    """
    return {
        group: teams[:]
        for group, teams in GROUPS.items()
    }


def extract_fake_qualifiers(
    group_standings: dict[str, list[str]],
) -> list[str]:
    """
    Temporary 2026-style qualifier extractor.

    Takes:
    - top 2 from each group = 24 teams
    - best 8 third-place teams = 8 teams

    Returns 32 teams.
    """
    qualifiers = []

    for group in sorted(group_standings):
        qualifiers.extend(group_standings[group][:2])


    third_place_teams = [
        group_standings[group][2]
        for group in sorted(group_standings)
    ]

    qualifiers.extend(third_place_teams[:8])

    if len(qualifiers) != 32:
        raise ValueError(f"Expected 32 qualifiers, got {len(qualifiers)}")

    if len(set(qualifiers)) != 32:
        raise ValueError("Duplicate qualifiers found.")

    return qualifiers


def simulate_group_match(
    team_a: str,
    team_b: str,
    team_strengths: dict[str, dict[str, float]],
) -> tuple[int, int]:
    
    from simulation.match_engine_adapter import simulate_match_score
    from simulation.simulation_config import MATCH_ENGINE_MODE

    normalized_team_a = normalize_team_name(team_a)
    normalized_team_b = normalize_team_name(team_b)

    if normalized_team_a not in team_strengths:
        raise ValueError(
            f"Missing team strength for group-stage team: "
            f"{team_a} -> {normalized_team_a}"
        )

    if normalized_team_b not in team_strengths:
        raise ValueError(
            f"Missing team strength for group-stage team: "
            f"{team_b} -> {normalized_team_b}"
        )

    team_a_strength = team_strengths[normalized_team_a]
    team_b_strength = team_strengths[normalized_team_b]

    return simulate_match_score(
        team_a_strength,
        team_b_strength,
        mode=MATCH_ENGINE_MODE,
    )


def simulate_group(
    group_name: str,
    teams: list[str],
    team_strengths: dict[str, float] | None = None,
) -> list[GroupStanding]:
    
    match_results = []

    table = {
        team: {
            "team": team,
            "group": group_name,
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
        }
        for team in teams
    }

    if team_strengths is None:
        team_strengths = {
            team: {
                "attack": 1.0,
                "defense": 1.0,
            }
            for team in teams
        }

    for team_a, team_b in combinations(teams, 2):
        goals_a, goals_b = simulate_group_match(team_a, team_b, team_strengths)

        match_results.append(
            GroupMatchResult(
                group=group_name,
                team1=team_a,
                team2=team_b,
                goals_team1=goals_a,
                goals_team2=goals_b,
            )
)

        table[team_a]["goals_for"] += goals_a
        table[team_a]["goals_against"] += goals_b
        table[team_b]["goals_for"] += goals_b
        table[team_b]["goals_against"] += goals_a

        if goals_a > goals_b:
            table[team_a]["points"] += 3
        elif goals_b > goals_a:
            table[team_b]["points"] += 3
        else:
            table[team_a]["points"] += 1
            table[team_b]["points"] += 1

    for row in table.values():
        row["goal_difference"] = row["goals_for"] - row["goals_against"]

    ranked_rows = sorted(
        table.values(),
        key=lambda row: (
            row["points"],
            row["goal_difference"],
            row["goals_for"],
            row["team"],
        ),
        reverse=True,
    )

    standings = [
        GroupStanding(
            team=row["team"],
            group=row["group"],
            finish=i + 1,
            points=row["points"],
            goals_for=row["goals_for"],
            goals_against=row["goals_against"],
            goal_difference=row["goal_difference"],
        )
        for i, row in enumerate(ranked_rows)
    ]

    return standings, match_results

def simulate_group_stage_with_matches(
    team_ratings: dict[str, float] | None = None,
) -> tuple[dict[str, list[GroupStanding]], list[GroupMatchResult]]:
    standings = {}
    all_matches = []

    for group_name, teams in GROUPS.items():
        group_standings, group_matches = simulate_group(
            group_name,
            teams,
            team_ratings,
        )

        standings[group_name] = group_standings
        all_matches.extend(group_matches)

    return standings, all_matches

def simulate_group_stage(
    team_ratings: dict[str, float] | None = None,
) -> dict[str, list[GroupStanding]]:
    standings, _ = simulate_group_stage_with_matches(team_ratings)
    return standings

def standings_to_team_names(
    group_standings: dict[str, list[GroupStanding]],
) -> dict[str, list[str]]:
    return {
        group: [row.team for row in rows]
        for group, rows in group_standings.items()
    }

def extract_qualifiers(
    group_standings: dict[str, list[str]],
) -> list[str]:
    qualifiers = []

    for group in sorted(group_standings):
        qualifiers.extend(group_standings[group][:2])

    third_place_teams = [
        group_standings[group][2]
        for group in sorted(group_standings)
    ]

    qualifiers.extend(third_place_teams[:8])

    if len(qualifiers) != 32:
        raise ValueError(f"Expected 32 qualifiers, got {len(qualifiers)}")

    if len(set(qualifiers)) != 32:
        raise ValueError("Duplicate qualifiers found.")

    return qualifiers

def extract_qualifiers_from_standings(
    group_standings: dict[str, list[GroupStanding]],
) -> list[GroupStanding]:
    qualifiers: list[GroupStanding] = []

    for group in sorted(group_standings):
        rows = group_standings[group]

        for row in rows[:2]:
            row.qualification_type = "top_two"
            qualifiers.append(row)

    third_place_rows = [
        group_standings[group][2]
        for group in sorted(group_standings)
    ]

    best_third_place_rows = sorted(
        third_place_rows,
        key=lambda row: (
            row.points,
            row.goal_difference,
            row.goals_for,
            row.team,
        ),
        reverse=True,
    )[:8]

    for row in best_third_place_rows:
        row.qualification_type = "best_third"
        qualifiers.append(row)

    if len(qualifiers) != 32:
        raise ValueError(f"Expected 32 qualifiers, got {len(qualifiers)}")

    if len({q.team for q in qualifiers}) != 32:
        raise ValueError("Duplicate qualified teams found.")

    return qualifiers

def qualified_teams_to_names(
    qualifiers: list[GroupStanding],
) -> list[str]:
    return [q.team for q in qualifiers]

if __name__ == "__main__":
    standings = simulate_fake_group_stage()
    qualifiers = extract_fake_qualifiers(standings)

    print("Fake group-stage smoke test")
    print("Groups:", len(standings))
    print("Qualifiers:", len(qualifiers))
    print("First qualifier:", qualifiers[0])
    print("Last qualifier:", qualifiers[-1])

    print()
    print("Simulated group-stage smoke test")


    team_strengths = load_team_repository()
    print("Loaded team strengths:", len(team_strengths))

    simulated_standings = simulate_group_stage(team_strengths)

    simulated_qualifiers = extract_qualifiers_from_standings(simulated_standings)

    print("Groups:", len(simulated_standings))
    print("Qualifiers:", len(simulated_qualifiers))
    print("Group A standings:")

    for row in simulated_standings["A"]:
        print(asdict(row))

    print()
    print("Qualified teams sample:")

    for row in simulated_qualifiers[:10]:
        print(asdict(row))
    
        print()
    
    print("Qualified teams sample:")

    for row in simulated_qualifiers[:10]:
        print(asdict(row))
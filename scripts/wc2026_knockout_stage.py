#wc2026_knockout_stage.py

import random
from dataclasses import dataclass

from simulation.match_engine_adapter import simulate_match_score
from simulation.simulation_config import MATCH_ENGINE_MODE
from scripts.wc2026_group_stage import GroupStanding
from scripts.wc2026_knockout_mapping import KnockoutMatch
from shared.team_name_normalizer import normalize_team_name


@dataclass
class KnockoutResult:
    match_number: int
    team1: GroupStanding
    team2: GroupStanding
    goals_team1: int
    goals_team2: int
    went_to_extra_time: bool
    went_to_penalties: bool
    winner: GroupStanding

ROUND_OF_16_PAIRINGS = [
    (73, 75),
    (77, 78),
    (79, 80),
    (81, 83),
    (84, 86),
    (85, 87),
    (76, 74),
    (82, 88),
]

QUARTERFINAL_PAIRINGS = [
    (89, 90),
    (91, 92),
    (93, 94),
    (95, 96),
]

SEMIFINAL_PAIRINGS = [
    (97, 98),
    (99, 100),
]

THIRD_PLACE_PLAYOFF_PAIRING = [
    (101, 102),
]

FINAL_PAIRING = [
    (101, 102),
]


def simulate_knockout_match(
    match: KnockoutMatch,
    team_repository,
) -> KnockoutResult:
    
    from scripts.match_engine import simulate_poisson_score

    team1_name = normalize_team_name(match.team1.team)
    team2_name = normalize_team_name(match.team2.team)

    if team1_name not in team_repository:
        raise ValueError(
            f"Missing team strength for knockout team: "
            f"{match.team1.team} -> {team1_name}"
        )

    if team2_name not in team_repository:
        raise ValueError(
            f"Missing team strength for knockout team: "
            f"{match.team2.team} -> {team2_name}"
        )

    team1_data = team_repository[team1_name]
    team2_data = team_repository[team2_name]

    goals1, goals2 = simulate_match_score(
        team1_data,
        team2_data,
        mode=MATCH_ENGINE_MODE,
    )

    went_to_extra_time = False
    went_to_penalties = False

    if goals1 > goals2:
        winner = match.team1
    elif goals2 > goals1:
        winner = match.team2
    else:
        went_to_extra_time = True

        et_goals1, et_goals2 = simulate_poisson_score(
            extra_time_team_features(team1_data),
            extra_time_team_features(team2_data),
        )

        goals1 += et_goals1
        goals2 += et_goals2

        if goals1 > goals2:
            winner = match.team1
        elif goals2 > goals1:
            winner = match.team2
        else:
            went_to_penalties = True
            winner = random.choice([match.team1, match.team2])

    return KnockoutResult(
        match_number=match.match_number,
        team1=match.team1,
        team2=match.team2,
        goals_team1=goals1,
        goals_team2=goals2,
        went_to_extra_time=went_to_extra_time,
        went_to_penalties=went_to_penalties,
        winner=winner,
    )

def build_next_round(
    previous_results: list[KnockoutResult],
    pairings: list[tuple[int, int]],
    starting_match_number: int,
) -> list[KnockoutMatch]:
    winner_by_match = {
        result.match_number: result.winner
        for result in previous_results
    }

    next_matches = []

    for i, (match_a, match_b) in enumerate(pairings):
        if match_a not in winner_by_match:
            raise ValueError(f"Missing winner for match {match_a}")

        if match_b not in winner_by_match:
            raise ValueError(f"Missing winner for match {match_b}")

        next_matches.append(
            KnockoutMatch(
                match_number=starting_match_number + i,
                team1=winner_by_match[match_a],
                team2=winner_by_match[match_b],
            )
        )

    return next_matches

def build_third_place_playoff(
    semifinal_results: list[KnockoutResult],
) -> list[KnockoutMatch]:
    if len(semifinal_results) != 2:
        raise ValueError("Expected exactly 2 semifinal results")

    losers = []

    for result in semifinal_results:
        if result.winner.team == result.team1.team:
            losers.append(result.team2)
        else:
            losers.append(result.team1)

    return [
        KnockoutMatch(
            match_number=103,
            team1=losers[0],
            team2=losers[1],
        )
    ]

def extra_time_team_features(team_data, attack_scale=0.35):
    return {
        "attack": team_data["attack"] * attack_scale,
        "defense": team_data["defense"],

        "poisson_attack":
            team_data["poisson_attack"] * attack_scale,
        "poisson_defense":
            team_data["poisson_defense"],

        # Generic calibrated lambda-model prior
        "rating_prior": team_data.get(
            "rating_prior",
            team_data["fifa_points"],
        ),

        # Temporary compatibility alias
        "fifa_points": team_data["fifa_points"],
    }
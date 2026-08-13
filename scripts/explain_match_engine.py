#explain_match_engine.py

import argparse

from team_strength_loader import load_poisson_team_strengths


MEAN_ATTACK = 0.295
MEAN_DEFENSE = 0.636
BASE_GOALS = 1.35
ATTACK_WEIGHT = 1.2
DEFENSE_WEIGHT = 0.8
MIN_LAMBDA = 0.2


def explain_expected_goals(team_a: str, team_b: str) -> dict:
    strengths = load_poisson_team_strengths()

    if team_a not in strengths:
        raise ValueError(f"Missing team strength for {team_a}")

    if team_b not in strengths:
        raise ValueError(f"Missing team strength for {team_b}")

    a = strengths[team_a]
    b = strengths[team_b]

    attack_a = a["attack"] - MEAN_ATTACK
    attack_b = b["attack"] - MEAN_ATTACK

    defense_a = MEAN_DEFENSE - a["defense"]
    defense_b = MEAN_DEFENSE - b["defense"]

    raw_lambda_a = BASE_GOALS + ATTACK_WEIGHT * attack_a - DEFENSE_WEIGHT * defense_b
    raw_lambda_b = BASE_GOALS + ATTACK_WEIGHT * attack_b - DEFENSE_WEIGHT * defense_a

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_raw": a,
        "team_b_raw": b,
        "team_a_attack_deviation": attack_a,
        "team_b_attack_deviation": attack_b,
        "team_a_defense_adjustment": defense_a,
        "team_b_defense_adjustment": defense_b,
        "team_a_raw_lambda": raw_lambda_a,
        "team_b_raw_lambda": raw_lambda_b,
        "team_a_lambda": max(MIN_LAMBDA, raw_lambda_a),
        "team_b_lambda": max(MIN_LAMBDA, raw_lambda_b),
    }


def print_team_block(label: str, raw: dict, attack_dev: float, opp_def_adj: float, raw_lambda: float, final_lambda: float) -> None:
    print(label)
    print("-" * len(label))
    print(f"Raw attack:              {raw['attack']:.3f}")
    print(f"Attack deviation:        {attack_dev:.3f}")
    print(f"Opponent def adjustment: {opp_def_adj:.3f}")
    print(f"Raw lambda:              {raw_lambda:.3f}")
    print(f"Final lambda:            {final_lambda:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("team_a")
    parser.add_argument("team_b")

    args = parser.parse_args()

    result = explain_expected_goals(args.team_a, args.team_b)

    print()
    print(f"{result['team_a']} vs {result['team_b']}")
    print("=" * (len(result["team_a"]) + len(result["team_b"]) + 4))
    print()
    print("Formula")
    print("-------")
    print("lambda = base_goals + attack_weight * attack_deviation - defense_weight * opponent_defense_adjustment")
    print()
    print(f"base_goals     = {BASE_GOALS}")
    print(f"attack_weight  = {ATTACK_WEIGHT}")
    print(f"defense_weight = {DEFENSE_WEIGHT}")
    print(f"minimum lambda = {MIN_LAMBDA}")
    print()

    print_team_block(
        result["team_a"],
        result["team_a_raw"],
        result["team_a_attack_deviation"],
        result["team_b_defense_adjustment"],
        result["team_a_raw_lambda"],
        result["team_a_lambda"],
    )

    print_team_block(
        result["team_b"],
        result["team_b_raw"],
        result["team_b_attack_deviation"],
        result["team_a_defense_adjustment"],
        result["team_b_raw_lambda"],
        result["team_b_lambda"],
    )


if __name__ == "__main__":
    main()
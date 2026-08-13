import random
from collections import Counter
from typing import Callable

from wc2026_bracket import KNOCKOUT_BRACKET
from wc2026_seed_assignment import (assign_ranked_qualifiers_to_r32_slots, build_fake_ranked_qualifiers)
from wc2026_group_stage import (
    simulate_group_stage,
    extract_qualifiers_from_standings,
    qualified_teams_to_names,
)

PlayMatchFn = Callable[[str, str], str]


def play_match_random(team_a: str, team_b: str) -> str:
    return random.choice([team_a, team_b])


def play_match_always_a(team_a: str, team_b: str) -> str:
    return team_a


def run_knockout_bracket(
    slot_to_team: dict[str, str],
    play_match: PlayMatchFn = play_match_random,
) -> dict[str, str]:
    resolved_slots = dict(slot_to_team)

    for match in KNOCKOUT_BRACKET:
        team_a = resolved_slots.get(match.slot_a)
        team_b = resolved_slots.get(match.slot_b)

        if team_a is None:
            raise ValueError(f"{match.match_id} missing slot_a: {match.slot_a}")

        if team_b is None:
            raise ValueError(f"{match.match_id} missing slot_b: {match.slot_b}")

        winner = play_match(team_a, team_b)

        if winner not in (team_a, team_b):
            raise ValueError(
                f"{match.match_id} returned invalid winner: {winner}. "
                f"Expected {team_a} or {team_b}."
            )

        loser = team_b if winner == team_a else team_a

        resolved_slots[match.winner_slot] = winner

        if match.loser_slot:
            resolved_slots[match.loser_slot] = loser

    return resolved_slots



def run_monte_carlo_smoke_test(n_sims: int = 10_000) -> Counter:
    champion_counts = Counter()

    for _ in range(n_sims):
        ranked_qualifiers = build_fake_ranked_qualifiers()
        slot_to_team = assign_ranked_qualifiers_to_r32_slots(ranked_qualifiers)

        result = run_knockout_bracket(
            slot_to_team=slot_to_team,
            play_match=play_match_random,
        )

        champion_counts[result["CHAMPION"]] += 1

    return champion_counts

def run_full_fake_tournament(
    play_match: PlayMatchFn = play_match_random,
) -> dict[str, str]:
    group_standings = simulate_group_stage()
    qualifiers = extract_qualifiers_from_standings(group_standings)

    slot_to_team = assign_ranked_qualifiers_to_r32_slots(qualifiers)

    return run_knockout_bracket(
        slot_to_team=slot_to_team,
        play_match=play_match,
    )

if __name__ == "__main__":
    print("Seed assignment smoke test")

    ranked_qualifiers = build_fake_ranked_qualifiers()
    slot_to_team = assign_ranked_qualifiers_to_r32_slots(ranked_qualifiers)

    deterministic_result = run_knockout_bracket(
        slot_to_team=slot_to_team,
        play_match=play_match_always_a,
    )

    print("R32 slots filled:", len(slot_to_team))
    print("Champion:", deterministic_result["CHAMPION"])
    print("Runner-up:", deterministic_result["RUNNER_UP"])
    print("Third place:", deterministic_result["THIRD_PLACE_WINNER"])

    print()
    print("Full fake tournament smoke test")

    full_result = run_full_fake_tournament(play_match=play_match_random)

    print("Champion:", full_result["CHAMPION"])
    print("Runner-up:", full_result["RUNNER_UP"])
    print("Third place:", full_result["THIRD_PLACE_WINNER"])

    print()
    print("Monte Carlo smoke test")

    n_sims = 10_000
    counts = run_monte_carlo_smoke_test(n_sims=n_sims)

    for team, count in counts.most_common(10):
        print(f"{team}: {count / n_sims:.2%}")
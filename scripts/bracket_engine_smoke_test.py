import random
from wc2026_bracket import KNOCKOUT_BRACKET, R32_INPUT_SLOTS


def play_match_random(team_a: str, team_b: str) -> str:
    return random.choice([team_a, team_b])


def run_knockout_bracket(slot_to_team: dict[str, str]) -> dict[str, str]:
    resolved_slots = dict(slot_to_team)

    for match in KNOCKOUT_BRACKET:
        team_a = resolved_slots.get(match.slot_a)
        team_b = resolved_slots.get(match.slot_b)

        if team_a is None:
            raise ValueError(f"{match.match_id} missing slot_a: {match.slot_a}")

        if team_b is None:
            raise ValueError(f"{match.match_id} missing slot_b: {match.slot_b}")

        winner = play_match_random(team_a, team_b)
        loser = team_b if winner == team_a else team_a

        resolved_slots[match.winner_slot] = winner

        if match.loser_slot:
            resolved_slots[match.loser_slot] = loser

    return resolved_slots


def build_fake_r32_slots() -> dict[str, str]:
    return {
        slot: f"Team_{i + 1:02d}"
        for i, slot in enumerate(R32_INPUT_SLOTS)
    }


if __name__ == "__main__":
    slot_to_team = build_fake_r32_slots()
    result = run_knockout_bracket(slot_to_team)

    print("Champion:", result["CHAMPION"])
    print("Runner-up:", result["RUNNER_UP"])
    print("Third place:", result["THIRD_PLACE_WINNER"])
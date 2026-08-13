# sofascore_bracket_audit.py

from wc2026_bracket import (
    KNOCKOUT_BRACKET,
    R32_INPUT_SLOTS,
    validate_bracket_structure,
)
from wc2026_seed_assignment import (
    assign_ranked_qualifiers_to_r32_slots,
    build_fake_ranked_qualifiers,
)
from sofascore_tournament_simulator import (
    run_knockout_bracket,
    play_match_random,
)

from sofascore_tournament_simulator import run_full_fake_tournament

def audit_bracket_structure() -> None:
    validate_bracket_structure()
    print("Bracket structure audit passed.")


def audit_seed_assignment() -> None:
    ranked_qualifiers = build_fake_ranked_qualifiers()
    slot_to_team = assign_ranked_qualifiers_to_r32_slots(ranked_qualifiers)

    missing_slots = [slot for slot in R32_INPUT_SLOTS if slot not in slot_to_team]

    if missing_slots:
        raise ValueError(f"Missing R32 slots: {missing_slots}")

    if len(slot_to_team) != 32:
        raise ValueError(f"Expected 32 R32 slots, got {len(slot_to_team)}")

    if len(set(slot_to_team.values())) != 32:
        raise ValueError("Duplicate teams found in R32 slot assignment.")

    print("Seed assignment audit passed.")


def audit_knockout_resolution() -> None:
    ranked_qualifiers = build_fake_ranked_qualifiers()
    slot_to_team = assign_ranked_qualifiers_to_r32_slots(ranked_qualifiers)

    result = run_knockout_bracket(
        slot_to_team=slot_to_team,
        play_match=play_match_random,
    )

    required_terminal_slots = [
        "CHAMPION",
        "RUNNER_UP",
        "THIRD_PLACE_WINNER",
    ]

    for slot in required_terminal_slots:
        if slot not in result:
            raise ValueError(f"Missing terminal slot: {slot}")

    if result["CHAMPION"] == result["RUNNER_UP"]:
        raise ValueError("Champion and runner-up are the same team.")

    if result["THIRD_PLACE_WINNER"] in {
        result["CHAMPION"],
        result["RUNNER_UP"],
    }:
        raise ValueError("Third-place winner overlaps with finalist.")

    print("Knockout resolution audit passed.")


def audit_every_match_resolves() -> None:
    ranked_qualifiers = build_fake_ranked_qualifiers()
    slot_to_team = assign_ranked_qualifiers_to_r32_slots(ranked_qualifiers)

    result = run_knockout_bracket(
        slot_to_team=slot_to_team,
        play_match=play_match_random,
    )

    for match in KNOCKOUT_BRACKET:
        if match.winner_slot not in result:
            raise ValueError(f"{match.match_id} missing winner slot.")

        if match.loser_slot and match.loser_slot not in result:
            raise ValueError(f"{match.match_id} missing loser slot.")

    print("Every match resolution audit passed.")

def audit_full_fake_tournament() -> None:
    result = run_full_fake_tournament(play_match=play_match_random)

    required_terminal_slots = [
        "CHAMPION",
        "RUNNER_UP",
        "THIRD_PLACE_WINNER",
    ]

    for slot in required_terminal_slots:
        if slot not in result:
            raise ValueError(f"Full tournament missing terminal slot: {slot}")

    if result["CHAMPION"] == result["RUNNER_UP"]:
        raise ValueError("Full tournament champion and runner-up are the same team.")

    if result["THIRD_PLACE_WINNER"] in {
        result["CHAMPION"],
        result["RUNNER_UP"],
    }:
        raise ValueError("Full tournament third-place winner overlaps with finalist.")

    print("Full fake tournament audit passed.")

def run_all_audits() -> None:
    audit_bracket_structure()
    audit_seed_assignment()
    audit_knockout_resolution()
    audit_every_match_resolves()
    audit_full_fake_tournament()

    print()
    print("All bracket audits passed.")


if __name__ == "__main__":
    run_all_audits()
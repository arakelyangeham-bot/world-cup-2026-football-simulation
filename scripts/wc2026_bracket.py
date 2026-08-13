# wc2026_bracket.py
"""
Seed-only 2026 World Cup knockout bracket.

This file must stay structural only:
- no team names
- no probabilities
- no simulation logic
- no CSV loading
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Match:
    match_id: str
    round_name: str
    slot_a: str
    slot_b: str
    winner_slot: str
    loser_slot: Optional[str] = None


ROUND_OF_32 = [
    Match("R32_01", "Round of 32", "R32_01_A", "R32_01_B", "W_R32_01"),
    Match("R32_02", "Round of 32", "R32_02_A", "R32_02_B", "W_R32_02"),
    Match("R32_03", "Round of 32", "R32_03_A", "R32_03_B", "W_R32_03"),
    Match("R32_04", "Round of 32", "R32_04_A", "R32_04_B", "W_R32_04"),
    Match("R32_05", "Round of 32", "R32_05_A", "R32_05_B", "W_R32_05"),
    Match("R32_06", "Round of 32", "R32_06_A", "R32_06_B", "W_R32_06"),
    Match("R32_07", "Round of 32", "R32_07_A", "R32_07_B", "W_R32_07"),
    Match("R32_08", "Round of 32", "R32_08_A", "R32_08_B", "W_R32_08"),
    Match("R32_09", "Round of 32", "R32_09_A", "R32_09_B", "W_R32_09"),
    Match("R32_10", "Round of 32", "R32_10_A", "R32_10_B", "W_R32_10"),
    Match("R32_11", "Round of 32", "R32_11_A", "R32_11_B", "W_R32_11"),
    Match("R32_12", "Round of 32", "R32_12_A", "R32_12_B", "W_R32_12"),
    Match("R32_13", "Round of 32", "R32_13_A", "R32_13_B", "W_R32_13"),
    Match("R32_14", "Round of 32", "R32_14_A", "R32_14_B", "W_R32_14"),
    Match("R32_15", "Round of 32", "R32_15_A", "R32_15_B", "W_R32_15"),
    Match("R32_16", "Round of 32", "R32_16_A", "R32_16_B", "W_R32_16"),
]


ROUND_OF_16 = [
    Match("R16_01", "Round of 16", "W_R32_01", "W_R32_02", "W_R16_01"),
    Match("R16_02", "Round of 16", "W_R32_03", "W_R32_04", "W_R16_02"),
    Match("R16_03", "Round of 16", "W_R32_05", "W_R32_06", "W_R16_03"),
    Match("R16_04", "Round of 16", "W_R32_07", "W_R32_08", "W_R16_04"),
    Match("R16_05", "Round of 16", "W_R32_09", "W_R32_10", "W_R16_05"),
    Match("R16_06", "Round of 16", "W_R32_11", "W_R32_12", "W_R16_06"),
    Match("R16_07", "Round of 16", "W_R32_13", "W_R32_14", "W_R16_07"),
    Match("R16_08", "Round of 16", "W_R32_15", "W_R32_16", "W_R16_08"),
]


QUARTERFINALS = [
    Match("QF_01", "Quarterfinal", "W_R16_01", "W_R16_02", "W_QF_01"),
    Match("QF_02", "Quarterfinal", "W_R16_03", "W_R16_04", "W_QF_02"),
    Match("QF_03", "Quarterfinal", "W_R16_05", "W_R16_06", "W_QF_03"),
    Match("QF_04", "Quarterfinal", "W_R16_07", "W_R16_08", "W_QF_04"),
]


SEMIFINALS = [
    Match("SF_01", "Semifinal", "W_QF_01", "W_QF_02", "W_SF_01", "L_SF_01"),
    Match("SF_02", "Semifinal", "W_QF_03", "W_QF_04", "W_SF_02", "L_SF_02"),
]


THIRD_PLACE = [
    Match("THIRD_PLACE", "Third Place", "L_SF_01", "L_SF_02", "THIRD_PLACE_WINNER"),
]


FINAL = [
    Match("FINAL", "Final", "W_SF_01", "W_SF_02", "CHAMPION", "RUNNER_UP"),
]


KNOCKOUT_BRACKET = (
    ROUND_OF_32
    + ROUND_OF_16
    + QUARTERFINALS
    + SEMIFINALS
    + THIRD_PLACE
    + FINAL
)


R32_INPUT_SLOTS = [
    slot
    for match in ROUND_OF_32
    for slot in (match.slot_a, match.slot_b)
]


def get_matches_by_round(round_name: str) -> list[Match]:
    return [m for m in KNOCKOUT_BRACKET if m.round_name == round_name]


def validate_bracket_structure() -> None:
    match_ids = [m.match_id for m in KNOCKOUT_BRACKET]
    if len(match_ids) != len(set(match_ids)):
        raise ValueError("Duplicate match_id found in bracket.")

    winner_slots = [m.winner_slot for m in KNOCKOUT_BRACKET]
    if len(winner_slots) != len(set(winner_slots)):
        raise ValueError("Duplicate winner_slot found in bracket.")

    produced_slots = set(winner_slots)
    produced_slots.update(m.loser_slot for m in KNOCKOUT_BRACKET if m.loser_slot)

    for match in KNOCKOUT_BRACKET:
        for slot in (match.slot_a, match.slot_b):
            if slot.startswith(("W_", "L_")) and slot not in produced_slots:
                raise ValueError(
                    f"{match.match_id} references unresolved input slot: {slot}"
                )

    if len(R32_INPUT_SLOTS) != 32:
        raise ValueError("Round of 32 must have exactly 32 input slots.")


if __name__ == "__main__":
    validate_bracket_structure()
    print("Bracket structure is valid.")
# wc2026_knockout_mapping.py

from scripts.wc2026_third_place_assignments import THIRD_PLACE_ASSIGNMENTS
from dataclasses import dataclass
from scripts.wc2026_group_stage import GroupStanding


@dataclass
class KnockoutMatch:
    match_number: int
    team1: GroupStanding
    team2: GroupStanding

ROUND_OF_32_FIXED_MATCHES = {
    73: ("2A", "2B"),
    74: ("1E", None),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", None),
    78: ("2E", "2I"),
    79: ("1A", None),
    80: ("1L", None),
    81: ("1D", None),
    82: ("1G", None),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", None),
    86: ("1J", "2H"),
    87: ("1K", None),
    88: ("2D", "2G"),
}

THIRD_PLACE_MATCH_ORDER = [74, 77, 79, 80, 81, 82, 85, 87]

# Initial table from the currently still-possible combinations shown in the Wikipedia table.
# Key = sorted groups whose third-place teams qualified.
# Value = third-place slot assigned to matches 74,77,79,80,81,82,85,87.

def build_round_of_32(
    qualifiers: list[GroupStanding],
) -> list[KnockoutMatch]:
    slot_to_team = {}

    for row in qualifiers:
        if row.finish in (1, 2):
            slot = f"{row.finish}{row.group}"
        elif row.finish == 3:
            slot = f"3{row.group}"
        else:
            raise ValueError(f"Unexpected qualifier finish: {row}")

        if slot in slot_to_team:
            raise ValueError(f"Duplicate slot found: {slot}")

        slot_to_team[slot] = row

    third_groups = tuple(
        sorted(row.group for row in qualifiers if row.finish == 3)
    )

    if len(third_groups) != 8:
        raise ValueError(f"Expected 8 third-place qualifiers, got {len(third_groups)}")

    if third_groups not in THIRD_PLACE_ASSIGNMENTS:
        raise ValueError(
            f"No official Annex C mapping found for third-place groups {third_groups}"
        )

    third_assignments = THIRD_PLACE_ASSIGNMENTS[third_groups]

    matches = dict(ROUND_OF_32_FIXED_MATCHES)

    for match_no, third_slot in zip(THIRD_PLACE_MATCH_ORDER, third_assignments):
        first_slot, second_slot = matches[match_no]

        if second_slot is not None:
            raise ValueError(f"Match {match_no} already has second slot")

        matches[match_no] = (first_slot, third_slot)

    knockout_matches = []

    for match_no in sorted(matches):
        slot1, slot2 = matches[match_no]

        if slot1 not in slot_to_team:
            raise ValueError(f"Missing team for slot {slot1}")

        if slot2 not in slot_to_team:
            raise ValueError(f"Missing team for slot {slot2}")

        knockout_matches.append(
            KnockoutMatch(
                match_number=match_no,
                team1=slot_to_team[slot1],
                team2=slot_to_team[slot2],
            )
        )

    if len(knockout_matches) != 16:
        raise ValueError(f"Expected 16 matches, got {len(knockout_matches)}")

    used_teams = [
        match.team1.team
        for match in knockout_matches
    ] + [
        match.team2.team
        for match in knockout_matches
    ]

    if len(set(used_teams)) != 32:
        raise ValueError("Duplicate teams in Round of 32")

    return knockout_matches
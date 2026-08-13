from wc2026_bracket import R32_INPUT_SLOTS
from wc2026_group_stage import GroupStanding

def assign_ranked_qualifiers_to_r32_slots(
    qualifiers: list[GroupStanding],
) -> dict[str, str]:
    """
    Temporary seed assignment layer.

    Input:
        ranked_qualifiers:
            A 32-team list already ordered according to bracket placement.

    Output:
        {
            "R32_01_A": "Team_01",
            "R32_01_B": "Team_02",
            ...
        }

    Later this module will be expanded to:
    - read group standings
    - select top 2 from each group
    - rank best third-place teams
    - apply official 2026 third-place routing
    - fill the R32 bracket slots
    """
    if len(qualifiers) != 32:
        raise ValueError(
            f"Expected 32 ranked qualifiers, got {len(qualifiers)}"
        )

    if len({q.team for q in qualifiers}) != 32:
        raise ValueError("Qualified teams must be unique.")
    
    return {
        slot: qualifier.team
        for slot, qualifier in zip(R32_INPUT_SLOTS, qualifiers)
    }


def build_fake_ranked_qualifiers() -> list[str]:
    return [f"Team_{i + 1:02d}" for i in range(32)]


if __name__ == "__main__":
    from wc2026_group_stage import (
        simulate_group_stage,
        extract_qualifiers_from_standings,
    )

    group_standings = simulate_group_stage()
    qualifiers = extract_qualifiers_from_standings(group_standings)
    slot_to_team = assign_ranked_qualifiers_to_r32_slots(qualifiers)

    print("Seed assignment module smoke test")
    print("Ranked qualifiers:", len(qualifiers))
    print("R32 slots filled:", len(slot_to_team))
    print("First slot:", R32_INPUT_SLOTS[0], "=", slot_to_team[R32_INPUT_SLOTS[0]])
    print("Last slot:", R32_INPUT_SLOTS[-1], "=", slot_to_team[R32_INPUT_SLOTS[-1]])
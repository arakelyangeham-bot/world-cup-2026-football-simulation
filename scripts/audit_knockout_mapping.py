# audit_knockout_mapping.py

from wc2026_knockout_mapping import (
    ROUND_OF_32_FIXED_MATCHES,
    THIRD_PLACE_ASSIGNMENTS,
    THIRD_PLACE_MATCH_ORDER,
)


def main():
    print("Round of 32 fixed matches:", len(ROUND_OF_32_FIXED_MATCHES))
    print("Third-place assignment rows:", len(THIRD_PLACE_ASSIGNMENTS))
    print("Third-place match slots:", THIRD_PLACE_MATCH_ORDER)

    if len(ROUND_OF_32_FIXED_MATCHES) != 16:
        raise ValueError("Expected 16 Round of 32 matches")

    if len(THIRD_PLACE_MATCH_ORDER) != 8:
        raise ValueError("Expected 8 third-place match slots")

    for combo, assignments in THIRD_PLACE_ASSIGNMENTS.items():
        print()
        print("Checking combo:", "".join(combo))

        if len(combo) != 8:
            raise ValueError(f"{combo}: expected 8 qualifying third-place groups")

        if len(assignments) != 8:
            raise ValueError(f"{combo}: expected 8 third-place assignments")

        assigned_groups = sorted(slot[1] for slot in assignments)

        if assigned_groups != sorted(combo):
            raise ValueError(
                f"{combo}: assignment groups {assigned_groups} "
                f"do not match combo {sorted(combo)}"
            )

        matches = dict(ROUND_OF_32_FIXED_MATCHES)

        for match_no, third_slot in zip(THIRD_PLACE_MATCH_ORDER, assignments):
            first, second = matches[match_no]

            if second is not None:
                raise ValueError(f"Match {match_no} already has second team: {second}")

            matches[match_no] = (first, third_slot)

        all_slots = []

        for match_no in sorted(matches):
            first, second = matches[match_no]

            if first is None or second is None:
                raise ValueError(f"Match {match_no} has an empty slot")

            all_slots.extend([first, second])

        if len(all_slots) != 32:
            raise ValueError(f"{combo}: expected 32 slots, got {len(all_slots)}")

        if len(set(all_slots)) != 32:
            raise ValueError(f"{combo}: duplicate slots found")

        print("Passed")

    print()
    print("Knockout mapping audit passed.")


if __name__ == "__main__":
    main()
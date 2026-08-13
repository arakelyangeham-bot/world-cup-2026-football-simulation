#position_normalizer.py

from __future__ import annotations


POSITION_ALIASES = {
    "GK": "GK",
    "GOALKEEPER": "GK",

    "CB": "CB",
    "CENTER BACK": "CB",
    "CENTRE BACK": "CB",

    "FB": "FB",
    "FULLBACK": "FB",
    "FULL BACK": "FB",
    "LB": "FB",
    "RB": "FB",

    "DM": "DM",
    "CDM": "DM",

    "CM": "CM",
    "CENTRAL MIDFIELD": "CM",

    "WM": "WM",
    "WIDE MIDFIELD": "WM",
    "LM": "WM",
    "RM": "WM",

    "AM": "AM",
    "CAM": "AM",

    "W": "W",
    "LW": "W",
    "RW": "W",
    "WINGER": "W",

    "ST": "ST",
    "CF": "ST",
    "STRIKER": "ST",
}


def normalize_position(position: str | None) -> str | None:
    if position is None:
        return None

    cleaned = str(position).strip().upper()

    if not cleaned:
        return None

    return POSITION_ALIASES.get(cleaned, cleaned)


def parse_position_string(position_string: str | None) -> tuple[str, ...]:
    if position_string is None:
        return ()

    raw = (
        str(position_string)
        .replace("/", ",")
        .replace(";", ",")
        .split(",")
    )

    positions = []

    for part in raw:
        position = normalize_position(part)
        if position is not None and position not in positions:
            positions.append(position)

    return tuple(positions)


def primary_position(position_string: str | None) -> str | None:
    positions = parse_position_string(position_string)

    if not positions:
        return None

    return positions[0]


def secondary_positions(position_string: str | None) -> tuple[str, ...]:
    positions = parse_position_string(position_string)

    if len(positions) <= 1:
        return ()

    return positions[1:]
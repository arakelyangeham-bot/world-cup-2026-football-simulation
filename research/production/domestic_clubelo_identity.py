#domestic_clubelo_identity

from __future__ import annotations


LA_LIGA_CLUBELO_NAME_OVERRIDES = {
    "Athletic Club": "Bilbao",
    "Atlético Madrid": "Atletico",
    "Celta Vigo": "Celta",
    "Deportivo Alavés": "Alaves",
    "Deportivo de A Coruña": "Depor",
    "FC Barcelona": "Barcelona",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "Rayo Vallecano": "RayoVallecano",
    "Real Betis": "Betis",
    "Real Madrid": "RealMadrid",
    "Real Racing Club": "Santander",
    "Real Sociedad": "Sociedad",
}


SERIE_A_2026_27_CLUBELO_NAME_OVERRIDES = {
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "SSC Napoli": "Napoli",
}

CLUBELO_NAME_OVERRIDES_BY_COMPETITION = {
    "la_liga": LA_LIGA_CLUBELO_NAME_OVERRIDES,
    "serie_a": SERIE_A_2026_27_CLUBELO_NAME_OVERRIDES,
}

def deduplicate_lookup_candidates(
    candidates: list[str],
) -> list[str]:
    deduplicated = []
    seen = set()

    for candidate in candidates:
        normalized = candidate.strip().casefold()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        deduplicated.append(
            candidate.strip()
        )

    return deduplicated


def build_clubelo_lookup_candidates(
    *,
    production_club: str,
    explicit_lookup: str | None = None,
    team_slug: str | None = None,
) -> list[str]:
    candidates = []

    if explicit_lookup:
        candidates.append(
            explicit_lookup
        )

    candidates.append(
        production_club
    )

    if team_slug:
        candidates.append(
            team_slug
        )

    return deduplicate_lookup_candidates(
        candidates
    )

def get_clubelo_name_override(
    *,
    competition_key: str,
    production_club: str,
) -> str | None:
    overrides = (
        CLUBELO_NAME_OVERRIDES_BY_COMPETITION.get(
            competition_key,
            {},
        )
    )

    return overrides.get(
        production_club
    )
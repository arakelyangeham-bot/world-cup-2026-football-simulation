#competition_registry.py

from dataclasses import dataclass


@dataclass(frozen=True)
class Competition:
    key: str
    display_name: str
    category: str
    importance: float
    filename_pattern: str


COMPETITIONS: list[Competition] = [
    Competition(
        key="world_cup",
        display_name="FIFA World Cup",
        category="global_final_tournament",
        importance=1.00,
        filename_pattern="wc_{year}_match_results.csv",
    ),
    Competition(
        key="world_cup_qualifiers",
        display_name="FIFA World Cup Qualifiers",
        category="global_qualification",
        importance=0.75,
        filename_pattern="wcq_{year}_match_results.csv",
    ),
    Competition(
        key="euro",
        display_name="UEFA European Championship",
        category="continental_final_tournament",
        importance=0.95,
        filename_pattern="euro_{year}_match_results.csv",
    ),
    Competition(
        key="euro_qualifiers",
        display_name="UEFA Euro Qualifiers",
        category="continental_qualification",
        importance=0.70,
        filename_pattern="euroq_{year}_match_results.csv",
    ),
    Competition(
        key="copa_america",
        display_name="Copa América",
        category="continental_final_tournament",
        importance=0.95,
        filename_pattern="copa_america_{year}_match_results.csv",
    ),
    Competition(
        key="afcon",
        display_name="Africa Cup of Nations",
        category="continental_final_tournament",
        importance=0.85,
        filename_pattern="afcon_{year}_match_results.csv",
    ),
    Competition(
        key="asian_cup",
        display_name="AFC Asian Cup",
        category="continental_final_tournament",
        importance=0.85,
        filename_pattern="asian_cup_{year}_match_results.csv",
    ),
    Competition(
        key="concacaf_gold_cup",
        display_name="CONCACAF Gold Cup",
        category="continental_final_tournament",
        importance=0.80,
        filename_pattern="gold_cup_{year}_match_results.csv",
    ),
    Competition(
        key="nations_league",
        display_name="UEFA Nations League",
        category="nations_league",
        importance=0.65,
        filename_pattern="nations_league_{year}_match_results.csv",
    ),
    Competition(
        key="friendly",
        display_name="International Friendly",
        category="friendly",
        importance=0.40,
        filename_pattern="friendly_{year}_match_results.csv",
    ),
        Competition(
        key="premier_league",
        display_name="Premier League",
        category="domestic_league",
        importance=0.90,
        filename_pattern=(
            "premier_league_{year}_match_results.csv"
        ),
    ),
    Competition(
        key="la_liga",
        display_name="La Liga",
        category="domestic_league",
        importance=0.90,
        filename_pattern=(
            "la_liga_{year}_match_results.csv"
        ),
    ),
    Competition(
        key="serie_a",
        display_name="Serie A",
        category="domestic_league",
        importance=0.90,
        filename_pattern=(
            "serie_a_{year}_match_results.csv"
        ),
    ),
    Competition(
        key="bundesliga",
        display_name="Bundesliga",
        category="domestic_league",
        importance=0.90,
        filename_pattern=(
            "bundesliga_{year}_match_results.csv"
        ),
    ),
    Competition(
        key="ligue_1",
        display_name="Ligue 1",
        category="domestic_league",
        importance=0.90,
        filename_pattern=(
            "ligue_1_{year}_match_results.csv"
        ),
    ),
]


COMPETITION_BY_KEY: dict[str, Competition] = {
    competition.key: competition
    for competition in COMPETITIONS
}


def get_competition(key: str) -> Competition:
    if key not in COMPETITION_BY_KEY:
        known = ", ".join(sorted(COMPETITION_BY_KEY))
        raise KeyError(f"Unknown competition key: {key}. Known: {known}")

    return COMPETITION_BY_KEY[key]
# team_name_normalizer.py

import unicodedata


TEAM_NAME_ALIASES = {
    # Canonical: USA
    "United States": "USA",
    "United States of America": "USA",
    "US": "USA",

    # Canonical: Iran
    "IR Iran": "Iran",
    "Iran (Islamic Republic of)": "Iran",

    # Canonical: South Korea
    "Korea Republic": "South Korea",
    "Korea Rep.": "South Korea",

    # Canonical: DR Congo
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",

    # Canonical: Cape Verde
    "Cabo Verde": "Cape Verde",

    # Canonical: Bosnia & Herzegovina
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",

    # Canonical: Türkiye
    "Turkey": "Türkiye",
    "Turkiye": "Türkiye",

    # Canonical: Curacao
    "Curaçao": "Curacao",
}

def normalize_team_name(name):
    if name is None:
        return name

    normalized = unicodedata.normalize("NFKC", str(name))
    normalized = " ".join(normalized.strip().split())

    return TEAM_NAME_ALIASES.get(normalized, normalized)
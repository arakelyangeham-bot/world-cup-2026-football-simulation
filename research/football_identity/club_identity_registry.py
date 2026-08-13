#club_identity_registry

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClubIdentity:
    """
    Canonical identity record for one football club.

    observation_name:
        Name currently used by the observation dataset.

    canonical_name:
        Stable project-level display name.

    clubelo_lookup_name:
        Source-specific identifier accepted by the
        ClubElo history endpoint.
    """

    observation_name: str
    canonical_name: str
    clubelo_lookup_name: str
    expected_country: str = "ENG"


CLUB_IDENTITIES: dict[str, ClubIdentity] = {
    "Arsenal": ClubIdentity(
        observation_name="Arsenal",
        canonical_name="Arsenal",
        clubelo_lookup_name="Arsenal",
    ),

    "Aston Villa": ClubIdentity(
        observation_name="Aston Villa",
        canonical_name="Aston Villa",
        clubelo_lookup_name="AstonVilla",
    ),

    "Bournemouth": ClubIdentity(
        observation_name="Bournemouth",
        canonical_name="AFC Bournemouth",
        clubelo_lookup_name="Bournemouth",
    ),

    "Brentford": ClubIdentity(
        observation_name="Brentford",
        canonical_name="Brentford",
        clubelo_lookup_name="Brentford",
    ),

    "Brighton & Hove Albion": ClubIdentity(
        observation_name="Brighton & Hove Albion",
        canonical_name="Brighton & Hove Albion",
        clubelo_lookup_name="Brighton",
    ),

    "Chelsea": ClubIdentity(
        observation_name="Chelsea",
        canonical_name="Chelsea",
        clubelo_lookup_name="Chelsea",
    ),

    "Crystal Palace": ClubIdentity(
        observation_name="Crystal Palace",
        canonical_name="Crystal Palace",
        clubelo_lookup_name="CrystalPalace",
    ),

    "Everton": ClubIdentity(
        observation_name="Everton",
        canonical_name="Everton",
        clubelo_lookup_name="Everton",
    ),

    "Fulham": ClubIdentity(
        observation_name="Fulham",
        canonical_name="Fulham",
        clubelo_lookup_name="Fulham",
    ),

    "Liverpool FC": ClubIdentity(
        observation_name="Liverpool FC",
        canonical_name="Liverpool",
        clubelo_lookup_name="Liverpool",
    ),

    "Manchester City": ClubIdentity(
        observation_name="Manchester City",
        canonical_name="Manchester City",
        clubelo_lookup_name="ManCity",
    ),

    "Manchester United": ClubIdentity(
        observation_name="Manchester United",
        canonical_name="Manchester United",
        clubelo_lookup_name="ManUnited",
    ),

    "Newcastle United": ClubIdentity(
        observation_name="Newcastle United",
        canonical_name="Newcastle United",
        clubelo_lookup_name="Newcastle",
    ),

    "Nottingham Forest": ClubIdentity(
        observation_name="Nottingham Forest",
        canonical_name="Nottingham Forest",
        clubelo_lookup_name="Forest",
    ),

    "Tottenham Hotspur": ClubIdentity(
        observation_name="Tottenham Hotspur",
        canonical_name="Tottenham Hotspur",
        clubelo_lookup_name="Tottenham",
    ),

    "West Ham United": ClubIdentity(
        observation_name="West Ham United",
        canonical_name="West Ham United",
        clubelo_lookup_name="WestHam",
    ),

    "Wolverhampton": ClubIdentity(
        observation_name="Wolverhampton",
        canonical_name="Wolverhampton Wanderers",
        clubelo_lookup_name="Wolves",
    ),
}


def normalize_identity_key(
    value: str,
) -> str:
    normalized = value.strip().casefold()

    if not normalized:
        raise ValueError(
            "Club identity value cannot be empty."
        )

    return normalized


def _build_casefold_lookup() -> dict[str, ClubIdentity]:
    lookup: dict[str, ClubIdentity] = {}

    for identity in CLUB_IDENTITIES.values():
        candidate_names = {
            identity.observation_name,
            identity.canonical_name,
        }

        for candidate_name in candidate_names:
            key = normalize_identity_key(
                candidate_name
            )

            existing = lookup.get(key)

            if (
                existing is not None
                and existing != identity
            ):
                raise ValueError(
                    "Ambiguous club identity name: "
                    f"{candidate_name!r}"
                )

            lookup[key] = identity

    return lookup


_CASEFOLD_LOOKUP = _build_casefold_lookup()


def get_club_identity(
    club_name: str,
) -> ClubIdentity:
    """
    Resolve an observation or canonical name to its
    registered club identity.
    """
    key = normalize_identity_key(
        club_name
    )

    try:
        return _CASEFOLD_LOOKUP[key]
    except KeyError as error:
        available = ", ".join(
            sorted(CLUB_IDENTITIES)
        )

        raise KeyError(
            f"Unknown club identity: {club_name!r}. "
            f"Registered observation names: {available}"
        ) from error


def get_clubelo_lookup_name(
    club_name: str,
) -> str:
    return get_club_identity(
        club_name
    ).clubelo_lookup_name


def list_registered_observation_names() -> tuple[
    str,
    ...,
]:
    return tuple(
        sorted(CLUB_IDENTITIES)
    )
#test_domestic_clubelo_identity

from research.production.domestic_clubelo_identity import (
    build_clubelo_lookup_candidates,
    deduplicate_lookup_candidates,
    get_clubelo_name_override,
)


def test_deduplicate_lookup_candidates_preserves_order() -> None:
    candidates = [
        "Milan",
        "AC Milan",
        "milan",
    ]

    result = deduplicate_lookup_candidates(
        candidates
    )

    assert result == [
        "Milan",
        "AC Milan",
    ]


def test_deduplicate_lookup_candidates_is_case_insensitive() -> None:
    candidates = [
        "Atalanta",
        "atalanta",
        "ATALANTA",
    ]

    result = deduplicate_lookup_candidates(
        candidates
    )

    assert result == [
        "Atalanta",
    ]


def test_deduplicate_lookup_candidates_ignores_blank_values() -> None:
    candidates = [
        " ",
        "Inter",
        "",
    ]

    result = deduplicate_lookup_candidates(
        candidates
    )

    assert result == [
        "Inter",
    ]


def test_lookup_candidates_prioritize_explicit_override() -> None:
    result = build_clubelo_lookup_candidates(
        production_club="AC Milan",
        explicit_lookup="Milan",
        team_slug="milan",
    )

    assert result == [
        "Milan",
        "AC Milan",
    ]


def test_lookup_candidates_use_production_name_without_override() -> None:
    result = build_clubelo_lookup_candidates(
        production_club="Atalanta",
        team_slug="atalanta",
    )

    assert result == [
        "Atalanta",
    ]


def test_lookup_candidates_keep_genuinely_distinct_slug() -> None:
    result = build_clubelo_lookup_candidates(
        production_club="Example United FC",
        team_slug="example-united",
    )

    assert result == [
        "Example United FC",
        "example-united",
    ]


def test_serie_a_override_is_registered() -> None:
    result = get_clubelo_name_override(
        competition_key="serie_a",
        production_club="SSC Napoli",
    )

    assert result == "Napoli"


def test_unknown_club_has_no_override() -> None:
    result = get_clubelo_name_override(
        competition_key="serie_a",
        production_club="Atalanta",
    )

    assert result is None


def test_unknown_competition_has_no_override() -> None:
    result = get_clubelo_name_override(
        competition_key="unknown_league",
        production_club="Example FC",
    )

    assert result is None
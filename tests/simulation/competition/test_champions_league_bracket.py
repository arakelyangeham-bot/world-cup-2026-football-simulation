#test_champions_league_bracket

import pytest

from simulation.competition.champions_league_bracket import (
    CHAMPIONS_LEAGUE_PLAYOFF_BANDS,
    CHAMPIONS_LEAGUE_ROUND_OF_16_SEEDED_PAIRS,
    CHAMPIONS_LEAGUE_ROUND_OF_16_REGIONS,
    CHAMPIONS_LEAGUE_QUARTERFINAL_REGIONS,
    ChampionsLeagueBracketOccupant,
    advance_bracket_occupant,
    ChampionsLeagueQuarterfinalTie,
    build_champions_league_quarterfinal_ties,
    ChampionsLeagueSemifinalTie,
    build_champions_league_semifinal_tie,
    ChampionsLeagueFinal,
    build_champions_league_final,
    build_champions_league_playoff_ties,
    build_champions_league_round_of_16_ties,
    _playoff_winners_by_band,
    quarterfinal_to_tie,
    league_phase_rank_by_team,
    CHAMPIONS_LEAGUE_SEMIFINAL_REGIONS,
    CHAMPIONS_LEAGUE_ROUND_OF_16_SLOTS,
    _round_of_16_slot_winners,

)

from simulation.competition.match_result import MatchResult
from simulation.competition.stage import Stage, StageType
from simulation.competition.stage_resolver import StageResolver


from random import Random

from simulation.competition.standings import StandingRow


def _ranked_rows() -> list[StandingRow]:
    return [
        StandingRow(team=f"Team {index:02d}")
        for index in range(1, 37)
    ]


def test_playoff_builder_creates_eight_ties():
    ties = build_champions_league_playoff_ties(
        _ranked_rows(),
        Random(202627),
    )

    assert len(ties) == 8


def test_playoff_builder_uses_only_allowed_ranking_bands():
    ties = build_champions_league_playoff_ties(
        _ranked_rows(),
        Random(202627),
    )

    allowed = {
        "Team 09": {"Team 23", "Team 24"},
        "Team 10": {"Team 23", "Team 24"},
        "Team 11": {"Team 21", "Team 22"},
        "Team 12": {"Team 21", "Team 22"},
        "Team 13": {"Team 19", "Team 20"},
        "Team 14": {"Team 19", "Team 20"},
        "Team 15": {"Team 17", "Team 18"},
        "Team 16": {"Team 17", "Team 18"},
    }

    for tie in ties:
        assert tie.unseeded_team in allowed[tie.seeded_team]


def test_seeded_team_hosts_second_leg():
    ties = build_champions_league_playoff_ties(
        _ranked_rows(),
        Random(202627),
    )

    for tie in ties:
        assert tie.first_leg_home == tie.unseeded_team
        assert tie.second_leg_home == tie.seeded_team


def test_playoff_builder_is_reproducible_from_seed():
    first = build_champions_league_playoff_ties(
        _ranked_rows(),
        Random(202627),
    )

    second = build_champions_league_playoff_ties(
        _ranked_rows(),
        Random(202627),
    )

    assert first == second

def test_playoff_pairing_bands_match_uefa_structure():
    actual = [
        (
            (band.seeded.first, band.seeded.second),
            (band.unseeded.first, band.unseeded.second),
        )
        for band in CHAMPIONS_LEAGUE_PLAYOFF_BANDS
    ]

    assert actual == [
        ((9, 10), (23, 24)),
        ((11, 12), (21, 22)),
        ((13, 14), (19, 20)),
        ((15, 16), (17, 18)),
    ]


def test_round_of_16_seeded_pairs_match_uefa_structure():
    actual = [
        (pair.first, pair.second)
        for pair in CHAMPIONS_LEAGUE_ROUND_OF_16_SEEDED_PAIRS
    ]

    assert actual == [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
    ]


def test_playoff_bands_cover_positions_9_through_24_once():
    positions = []

    for band in CHAMPIONS_LEAGUE_PLAYOFF_BANDS:
        positions.extend(
            [
                band.seeded.first,
                band.seeded.second,
                band.unseeded.first,
                band.unseeded.second,
            ]
        )

    assert sorted(positions) == list(range(9, 25))
    assert len(set(positions)) == 16

def test_round_of_16_regions_link_to_correct_playoff_bands():
    actual = [
        (
            region.region_id,
            (
                region.direct_seed_pair.first,
                region.direct_seed_pair.second,
            ),
            region.playoff_band_id,
        )
        for region in CHAMPIONS_LEAGUE_ROUND_OF_16_REGIONS
    ]

    assert actual == [
        ("A", (1, 2), "IV"),
        ("B", (3, 4), "III"),
        ("C", (5, 6), "II"),
        ("D", (7, 8), "I"),
    ]

def test_round_of_16_topology_covers_all_regions_and_bands_once():
    region_ids = [
        region.region_id
        for region in CHAMPIONS_LEAGUE_ROUND_OF_16_REGIONS
    ]

    playoff_band_ids = [
        region.playoff_band_id
        for region in CHAMPIONS_LEAGUE_ROUND_OF_16_REGIONS
    ]

    assert set(region_ids) == {"A", "B", "C", "D"}
    assert set(playoff_band_ids) == {"I", "II", "III", "IV"}

    assert len(region_ids) == len(set(region_ids))
    assert len(playoff_band_ids) == len(
        set(playoff_band_ids)
    )

def test_playoff_builder_preserves_two_ties_per_band():
    ties = build_champions_league_playoff_ties(
        _ranked_rows(),
        Random(202627),
    )

    counts = {
        band_id: sum(
            tie.band_id == band_id
            for tie in ties
        )
        for band_id in {"I", "II", "III", "IV"}
    }

    assert counts == {
        "I": 2,
        "II": 2,
        "III": 2,
        "IV": 2,
    }

def test_round_of_16_builder_respects_slots():
    ties = build_champions_league_round_of_16_ties(
        _ranked_rows(),
        _playoff_winners_by_band(),
        Random(202627),
    )

    allowed = {
        "R16-1": {
            "seeded": {"Team 01", "Team 02"},
            "playoff": {"Team 15", "Team 16"},
        },
        "R16-2": {
            "seeded": {"Team 07", "Team 08"},
            "playoff": {"Team 09", "Team 10"},
        },
        "R16-3": {
            "seeded": {"Team 05", "Team 06"},
            "playoff": {"Team 11", "Team 12"},
        },
        "R16-4": {
            "seeded": {"Team 03", "Team 04"},
            "playoff": {"Team 13", "Team 14"},
        },
        "R16-5": {
            "seeded": {"Team 03", "Team 04"},
            "playoff": {"Team 13", "Team 14"},
        },
        "R16-6": {
            "seeded": {"Team 05", "Team 06"},
            "playoff": {"Team 11", "Team 12"},
        },
        "R16-7": {
            "seeded": {"Team 07", "Team 08"},
            "playoff": {"Team 09", "Team 10"},
        },
        "R16-8": {
            "seeded": {"Team 01", "Team 02"},
            "playoff": {"Team 15", "Team 16"},
        },
    }

    assert len(ties) == 8

    for tie in ties:
        assert tie.seeded_team in (
            allowed[tie.slot_id]["seeded"]
        )
        assert tie.playoff_winner in (
            allowed[tie.slot_id]["playoff"]
        )

def test_advance_bracket_occupant_preserves_top_seed_when_seeded_team_wins():
    first = ChampionsLeagueBracketOccupant(
        team="Team 01",
        source_rank=1,
        inherited_seed_rank=1,
        bracket_region="A",
    )
    second = ChampionsLeagueBracketOccupant(
        team="Team 15",
        source_rank=15,
        inherited_seed_rank=15,
        bracket_region="A",
    )

    result = advance_bracket_occupant(
        winner="Team 01",
        first=first,
        second=second,
    )

    assert result.team == "Team 01"
    assert result.source_rank == 1
    assert result.inherited_seed_rank == 1
    assert result.bracket_region == "A"


def test_advance_bracket_occupant_transfers_seed_when_lower_ranked_team_wins():
    first = ChampionsLeagueBracketOccupant(
        team="Team 01",
        source_rank=1,
        inherited_seed_rank=1,
        bracket_region="A",
    )
    second = ChampionsLeagueBracketOccupant(
        team="Team 15",
        source_rank=15,
        inherited_seed_rank=15,
        bracket_region="A",
    )

    result = advance_bracket_occupant(
        winner="Team 15",
        first=first,
        second=second,
    )

    assert result.team == "Team 15"
    assert result.source_rank == 15
    assert result.inherited_seed_rank == 1
    assert result.bracket_region == "A"


def test_advance_bracket_occupant_preserves_real_source_rank():
    first = ChampionsLeagueBracketOccupant(
        team="Team 03",
        source_rank=3,
        inherited_seed_rank=3,
        bracket_region="B",
    )
    second = ChampionsLeagueBracketOccupant(
        team="Team 13",
        source_rank=13,
        inherited_seed_rank=13,
        bracket_region="B",
    )

    result = advance_bracket_occupant(
        winner="Team 13",
        first=first,
        second=second,
    )

    assert result.source_rank == 13
    assert result.inherited_seed_rank == 3


def test_advance_bracket_occupant_rejects_unknown_winner():
    first = ChampionsLeagueBracketOccupant(
        team="Team 01",
        source_rank=1,
        inherited_seed_rank=1,
        bracket_region="A",
    )
    second = ChampionsLeagueBracketOccupant(
        team="Team 15",
        source_rank=15,
        inherited_seed_rank=15,
        bracket_region="A",
    )

    with pytest.raises(
        ValueError,
        match="Winner must be one of the bracket occupants",
    ):
        advance_bracket_occupant(
            winner="Team 99",
            first=first,
            second=second,
        )

def test_quarterfinal_builder_uses_predetermined_slots():
    ties = build_champions_league_quarterfinal_ties(
        _round_of_16_slot_winners()
    )

    assert len(ties) == 4

    actual = [
        (
            tie.quarterfinal_id,
            {
                tie.team1.bracket_region,
                tie.team2.bracket_region,
            },
        )
        for tie in ties
    ]

    assert actual == [
        ("1", {"R16-1", "R16-2"}),
        ("2", {"R16-3", "R16-4"}),
        ("3", {"R16-5", "R16-6"}),
        ("4", {"R16-7", "R16-8"}),
    ]

def test_quarterfinal_better_inherited_seed_hosts_second_leg():
    winners = _round_of_16_slot_winners()

    winners["R16-1"] = ChampionsLeagueBracketOccupant(
        team="Rank 15 Upset Winner",
        source_rank=15,
        inherited_seed_rank=1,
        bracket_region="R16-1",
    )

    winners["R16-2"] = ChampionsLeagueBracketOccupant(
        team="Rank 7 Winner",
        source_rank=7,
        inherited_seed_rank=7,
        bracket_region="R16-2",
    )

    ties = build_champions_league_quarterfinal_ties(
        winners
    )

    quarterfinal_1 = ties[0]

    assert (
        quarterfinal_1.second_leg_home
        == "Rank 15 Upset Winner"
    )
    assert (
        quarterfinal_1.first_leg_home
        == "Rank 7 Winner"
    )

def test_quarterfinal_builder_requires_all_eight_r16_slots():
    winners = _round_of_16_slot_winners()
    del winners["R16-8"]

    with pytest.raises(
        ValueError,
        match="R16-1 through R16-8",
    ):
        build_champions_league_quarterfinal_ties(
            winners
        )

def test_advance_bracket_occupant_can_move_to_new_region():
    first = ChampionsLeagueBracketOccupant(
        team="Team A Winner",
        source_rank=15,
        inherited_seed_rank=1,
        bracket_region="A",
    )
    second = ChampionsLeagueBracketOccupant(
        team="Team D Winner",
        source_rank=7,
        inherited_seed_rank=7,
        bracket_region="D",
    )

    result = advance_bracket_occupant(
        winner="Team D Winner",
        first=first,
        second=second,
        destination_region="1",
    )

    assert result.team == "Team D Winner"
    assert result.source_rank == 7

    # Team D inherits the superior bracket seed carried
    # by its defeated opponent.
    assert result.inherited_seed_rank == 1

    # But its location advances from R16 region D
    # into quarterfinal region 1.
    assert result.bracket_region == "1"

def test_advance_bracket_occupant_preserves_region_by_default():
    first = ChampionsLeagueBracketOccupant(
        team="Team 01",
        source_rank=1,
        inherited_seed_rank=1,
        bracket_region="A",
    )
    second = ChampionsLeagueBracketOccupant(
        team="Team 15",
        source_rank=15,
        inherited_seed_rank=15,
        bracket_region="A",
    )

    result = advance_bracket_occupant(
        winner="Team 15",
        first=first,
        second=second,
    )

    assert result.bracket_region == "A"

def test_semifinal_builder_uses_quarterfinal_winners():
    quarterfinal_winners = {
        "1": ChampionsLeagueBracketOccupant(
            team="QF1 Winner",
            source_rank=15,
            inherited_seed_rank=1,
            bracket_region="1",
        ),
        "2": ChampionsLeagueBracketOccupant(
            team="QF2 Winner",
            source_rank=4,
            inherited_seed_rank=3,
            bracket_region="2",
        ),
    }

    tie = build_champions_league_semifinal_tie(
        quarterfinal_winners
    )

    assert tie.team1.team == "QF1 Winner"
    assert tie.team2.team == "QF2 Winner"


def test_semifinal_better_inherited_seed_hosts_second_leg():
    quarterfinal_winners = {
        "1": ChampionsLeagueBracketOccupant(
            team="Upset Winner",
            source_rank=15,
            inherited_seed_rank=1,
            bracket_region="1",
        ),
        "2": ChampionsLeagueBracketOccupant(
            team="Higher Original Rank",
            source_rank=3,
            inherited_seed_rank=3,
            bracket_region="2",
        ),
    }

    tie = build_champions_league_semifinal_tie(
        quarterfinal_winners
    )

    assert tie.first_leg_home == "Higher Original Rank"
    assert tie.second_leg_home == "Upset Winner"


def test_semifinal_builder_requires_both_quarterfinal_regions():
    quarterfinal_winners = {
        "1": ChampionsLeagueBracketOccupant(
            team="QF1 Winner",
            source_rank=1,
            inherited_seed_rank=1,
            bracket_region="1",
        ),
    }

    with pytest.raises(
        ValueError,
        match="quarterfinals 1 and 2",
    ):
        build_champions_league_semifinal_tie(
            quarterfinal_winners
        )

def test_final_builder_uses_both_semifinal_winners():
    semifinal_winners = {
        "SF1": ChampionsLeagueBracketOccupant(
            team="Semifinal 1 Winner",
            source_rank=15,
            inherited_seed_rank=1,
            bracket_region="SF1",
        ),
        "SF2": ChampionsLeagueBracketOccupant(
            team="Semifinal 2 Winner",
            source_rank=4,
            inherited_seed_rank=3,
            bracket_region="SF2",
        ),
    }

    final = build_champions_league_final(
        semifinal_winners
    )

    assert final.team1.team == "Semifinal 1 Winner"
    assert final.team2.team == "Semifinal 2 Winner"


def test_final_is_neutral_site():
    semifinal_winners = {
        "SF1": ChampionsLeagueBracketOccupant(
            team="Team A",
            source_rank=1,
            inherited_seed_rank=1,
            bracket_region="SF1",
        ),
        "SF2": ChampionsLeagueBracketOccupant(
            team="Team B",
            source_rank=2,
            inherited_seed_rank=2,
            bracket_region="SF2",
        ),
    }

    final = build_champions_league_final(
        semifinal_winners
    )

    assert final.neutral_site is True


def test_final_builder_requires_both_semifinal_winners():
    semifinal_winners = {
        "SF1": ChampionsLeagueBracketOccupant(
            team="Team A",
            source_rank=1,
            inherited_seed_rank=1,
            bracket_region="SF1",
        ),
    }

    with pytest.raises(
        ValueError,
        match="semifinals SF1 and SF2",
    ):
        build_champions_league_final(
            semifinal_winners
        )

def test_quarterfinal_to_tie_integrates_with_stage_resolver():
    quarterfinal = ChampionsLeagueQuarterfinalTie(
        quarterfinal_id="1",
        team1=ChampionsLeagueBracketOccupant(
            team="Team A",
            source_rank=1,
            inherited_seed_rank=1,
            bracket_region="1",
        ),
        team2=ChampionsLeagueBracketOccupant(
            team="Team B",
            source_rank=7,
            inherited_seed_rank=7,
            bracket_region="1",
        ),
        first_leg_home="Team B",
        second_leg_home="Team A",
    )

    tie = quarterfinal_to_tie(
        quarterfinal,
        [
            MatchResult(
                team1="Team B",
                team2="Team A",
                goals_team1=1,
                goals_team2=1,
            ),
            MatchResult(
                team1="Team A",
                team2="Team B",
                goals_team1=2,
                goals_team2=0,
            ),
        ],
    )

    stage = Stage(
        name="Quarterfinal",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=["Team A", "Team B"],
        matches=[tie],
    )

    result = StageResolver().resolve(stage)

    assert result.qualifiers == ["Team A"]
    assert result.eliminated == ["Team B"]
    assert result.match_results[0].winner == "Team A"

def test_league_phase_rank_by_team_maps_all_36_positions():
    ranks = league_phase_rank_by_team(
        _ranked_rows()
    )

    assert len(ranks) == 36
    assert ranks["Team 01"] == 1
    assert ranks["Team 08"] == 8
    assert ranks["Team 24"] == 24
    assert ranks["Team 36"] == 36

def test_league_phase_rank_by_team_rejects_wrong_team_count():
    with pytest.raises(
        ValueError,
        match="exactly 36 teams",
    ):
        league_phase_rank_by_team(
            _ranked_rows()[:-1]
        )

def test_round_of_16_slots_cover_all_eight_bracket_positions():
    assert [
        slot.slot_id
        for slot in CHAMPIONS_LEAGUE_ROUND_OF_16_SLOTS
    ] == [
        "R16-1",
        "R16-2",
        "R16-3",
        "R16-4",
        "R16-5",
        "R16-6",
        "R16-7",
        "R16-8",
    ]


def test_quarterfinals_receive_two_r16_slots_each():
    actual = [
        (
            quarterfinal.quarterfinal_id,
            quarterfinal.round_of_16_slots,
        )
        for quarterfinal
        in CHAMPIONS_LEAGUE_QUARTERFINAL_REGIONS
    ]

    assert actual == [
        ("1", ("R16-1", "R16-2")),
        ("2", ("R16-3", "R16-4")),
        ("3", ("R16-5", "R16-6")),
        ("4", ("R16-7", "R16-8")),
    ]


def test_semifinal_topology_uses_four_quarterfinals():
    actual = [
        (
            semifinal.semifinal_id,
            semifinal.quarterfinal_regions,
        )
        for semifinal
        in CHAMPIONS_LEAGUE_SEMIFINAL_REGIONS
    ]

    assert actual == [
        ("SF1", ("1", "2")),
        ("SF2", ("3", "4")),
    ]

def test_round_of_16_builder_assigns_all_eight_slots():
    ties = build_champions_league_round_of_16_ties(
        _ranked_rows(),
        _playoff_winners_by_band(),
        Random(202627),
    )

    assert len(ties) == 8

    assert {
        tie.slot_id
        for tie in ties
    } == {
        f"R16-{index}"
        for index in range(1, 9)
    }

def test_round_of_16_builder_uses_every_direct_seed_once():
    ties = build_champions_league_round_of_16_ties(
        _ranked_rows(),
        _playoff_winners_by_band(),
        Random(202627),
    )

    assert {
        tie.seeded_team
        for tie in ties
    } == {
        f"Team {index:02d}"
        for index in range(1, 9)
    }

    assert len(
        [
            tie.seeded_team
            for tie in ties
        ]
    ) == 8

def test_round_of_16_builder_uses_every_playoff_winner_once():
    ties = build_champions_league_round_of_16_ties(
        _ranked_rows(),
        _playoff_winners_by_band(),
        Random(202627),
    )

    actual = [
        tie.playoff_winner
        for tie in ties
    ]

    assert set(actual) == {
        f"Team {index:02d}"
        for index in range(9, 17)
    }

    assert len(actual) == len(set(actual)) == 8
#test_champions_league_knockout_integration

from __future__ import annotations

from random import Random

from simulation.competition.champions_league_bracket import (
    build_champions_league_playoff_ties,
    build_champions_league_round_of_16_ties,
)
from simulation.competition.match_result import MatchResult
from simulation.competition.stage import Stage, StageType
from simulation.competition.stage_resolver import StageResolver
from simulation.competition.standings import StandingRow
from simulation.competition.tie import Tie

from simulation.competition.champions_league_bracket import (
    ChampionsLeagueBracketOccupant,
    ChampionsLeagueQuarterfinalTie,
    advance_bracket_occupant,
    build_champions_league_quarterfinal_ties,
    build_champions_league_round_of_16_ties,
    league_phase_rank_by_team,
    build_champions_league_final,
    build_champions_league_semifinal_tie,
)


def _ranked_rows() -> list[StandingRow]:
    return [
        StandingRow(
            team=f"Team {rank:02d}",
            points=100 - rank,
        )
        for rank in range(1, 37)
    ]


def test_playoff_resolution_feeds_round_of_16_draw():
    ranked_rows = _ranked_rows()

    playoff_pairings = (
        build_champions_league_playoff_ties(
            ranked_rows,
            Random(202627),
        )
    )

    generic_ties = []

    for pairing in playoff_pairings:
        generic_ties.append(
            Tie(
                team1=pairing.seeded_team,
                team2=pairing.unseeded_team,
                match_results=[
                    MatchResult(
                        team1=pairing.unseeded_team,
                        team2=pairing.seeded_team,
                        goals_team1=0,
                        goals_team2=1,
                    ),
                    MatchResult(
                        team1=pairing.seeded_team,
                        team2=pairing.unseeded_team,
                        goals_team1=2,
                        goals_team2=0,
                    ),
                ],
                metadata={
                    "band_id": pairing.band_id,
                },
            )
        )

    stage = Stage(
        name="Knockout Phase Playoffs",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=[
            row.team
            for row in ranked_rows[8:24]
        ],
        matches=generic_ties,
    )

    result = StageResolver().resolve(stage)

    assert len(result.qualifiers) == 8
    assert len(result.eliminated) == 8

    winners_by_band: dict[
        str,
        list[str],
    ] = {
        "I": [],
        "II": [],
        "III": [],
        "IV": [],
    }

    for tie_result in result.match_results:
        band_id = tie_result.metadata[
            "band_id"
        ]

        winners_by_band[band_id].append(
            tie_result.winner
        )

    normalized_winners = {
        band_id: tuple(winners)
        for band_id, winners
        in winners_by_band.items()
    }

    assert all(
        len(winners) == 2
        for winners
        in normalized_winners.values()
    )

    round_of_16_ties = (
        build_champions_league_round_of_16_ties(
            ranked_rows,
            normalized_winners,
            Random(202627),
        )
    )

    assert len(round_of_16_ties) == 8

    playoff_winners = set(
        result.qualifiers
    )

    round_of_16_playoff_teams = {
        tie.playoff_winner
        for tie in round_of_16_ties
    }

    assert (
        round_of_16_playoff_teams
        == playoff_winners
    )

    direct_qualifiers = {
        row.team
        for row in ranked_rows[:8]
    }

    round_of_16_seeded_teams = {
        tie.seeded_team
        for tie in round_of_16_ties
    }

    assert (
        round_of_16_seeded_teams
        == direct_qualifiers
    )

def test_round_of_16_resolution_feeds_quarterfinals_with_inherited_seed():
    ranked_rows = _ranked_rows()
    ranks = league_phase_rank_by_team(
        ranked_rows
    )

    playoff_winners_by_band = {
        "I": ("Team 09", "Team 10"),
        "II": ("Team 11", "Team 12"),
        "III": ("Team 13", "Team 14"),
        "IV": ("Team 15", "Team 16"),
    }

    round_of_16_pairings = (
        build_champions_league_round_of_16_ties(
            ranked_rows,
            playoff_winners_by_band,
            Random(202627),
        )
    )

    generic_ties = []

    for pairing in round_of_16_pairings:
        # Force one deliberate upset:
        # Team 15 defeats Team 01.
        upset = pairing.seeded_team == "Team 01"

        if upset:
            first_leg_seeded_goals = 0
            first_leg_playoff_goals = 1
            second_leg_seeded_goals = 1
            second_leg_playoff_goals = 1
        else:
            first_leg_seeded_goals = 1
            first_leg_playoff_goals = 0
            second_leg_seeded_goals = 2
            second_leg_playoff_goals = 0

        generic_ties.append(
            Tie(
                team1=pairing.seeded_team,
                team2=pairing.playoff_winner,
                match_results=[
                    MatchResult(
                        team1=pairing.playoff_winner,
                        team2=pairing.seeded_team,
                        goals_team1=(
                            first_leg_playoff_goals
                        ),
                        goals_team2=(
                            first_leg_seeded_goals
                        ),
                    ),
                    MatchResult(
                        team1=pairing.seeded_team,
                        team2=pairing.playoff_winner,
                        goals_team1=(
                            second_leg_seeded_goals
                        ),
                        goals_team2=(
                            second_leg_playoff_goals
                        ),
                    ),
                ],
                metadata={
                    "slot_id": pairing.slot_id,
                },
            )
        )

    stage = Stage(
        name="Round of 16",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=[
            team
            for pairing in round_of_16_pairings
            for team in (
                pairing.seeded_team,
                pairing.playoff_winner,
            )
        ],
        matches=generic_ties,
    )

    result = StageResolver().resolve(stage)

    assert len(result.qualifiers) == 8
    assert len(result.eliminated) == 8

    pairing_by_slot = {
        pairing.slot_id: pairing
        for pairing in round_of_16_pairings
    }

    winners_by_slot: dict[
        str,
        ChampionsLeagueBracketOccupant,
    ] = {}

    for tie_result in result.match_results:
        slot_id = tie_result.metadata["slot_id"]
        pairing = pairing_by_slot[slot_id]

        seeded = ChampionsLeagueBracketOccupant(
            team=pairing.seeded_team,
            source_rank=ranks[
                pairing.seeded_team
            ],
            inherited_seed_rank=ranks[
                pairing.seeded_team
            ],
            bracket_region=slot_id,
        )

        playoff = ChampionsLeagueBracketOccupant(
            team=pairing.playoff_winner,
            source_rank=ranks[
                pairing.playoff_winner
            ],
            inherited_seed_rank=ranks[
                pairing.playoff_winner
            ],
            bracket_region=slot_id,
        )

        winners_by_slot[slot_id] = (
            advance_bracket_occupant(
                winner=tie_result.winner,
                first=seeded,
                second=playoff,
            )
        )

    assert len(winners_by_slot) == 8
    assert set(winners_by_slot) == {
        f"R16-{index}"
        for index in range(1, 9)
    }

    quarterfinal_ties = (
        build_champions_league_quarterfinal_ties(
            winners_by_slot
        )
    )

    assert len(quarterfinal_ties) == 4

    assert [
        tie.quarterfinal_id
        for tie in quarterfinal_ties
    ] == ["1", "2", "3", "4"]

    rank_one_pairing = next(
        pairing
        for pairing in round_of_16_pairings
        if pairing.seeded_team == "Team 01"
    )

    upset_winner = winners_by_slot[
        rank_one_pairing.slot_id
    ]

    assert (
        upset_winner.team
        == rank_one_pairing.playoff_winner
    )

    assert upset_winner.source_rank == ranks[
        rank_one_pairing.playoff_winner
    ]

    assert upset_winner.inherited_seed_rank == 1

    upset_quarterfinal = next(
        tie
        for tie in quarterfinal_ties
        if upset_winner.team
        in {
            tie.team1.team,
            tie.team2.team,
        }
    )

    assert (
        upset_quarterfinal.second_leg_home
        == upset_winner.team
    )

def test_quarterfinals_progress_to_champions_league_champion():
    quarterfinals = [
        ChampionsLeagueQuarterfinalTie(
            quarterfinal_id="1",
            team1=ChampionsLeagueBracketOccupant(
                team="QF1 A",
                source_rank=15,
                inherited_seed_rank=1,
                bracket_region="R16-1",
            ),
            team2=ChampionsLeagueBracketOccupant(
                team="QF1 B",
                source_rank=7,
                inherited_seed_rank=7,
                bracket_region="R16-2",
            ),
            first_leg_home="QF1 B",
            second_leg_home="QF1 A",
        ),
        ChampionsLeagueQuarterfinalTie(
            quarterfinal_id="2",
            team1=ChampionsLeagueBracketOccupant(
                team="QF2 A",
                source_rank=3,
                inherited_seed_rank=3,
                bracket_region="R16-3",
            ),
            team2=ChampionsLeagueBracketOccupant(
                team="QF2 B",
                source_rank=5,
                inherited_seed_rank=5,
                bracket_region="R16-4",
            ),
            first_leg_home="QF2 B",
            second_leg_home="QF2 A",
        ),
        ChampionsLeagueQuarterfinalTie(
            quarterfinal_id="3",
            team1=ChampionsLeagueBracketOccupant(
                team="QF3 A",
                source_rank=4,
                inherited_seed_rank=4,
                bracket_region="R16-5",
            ),
            team2=ChampionsLeagueBracketOccupant(
                team="QF3 B",
                source_rank=6,
                inherited_seed_rank=6,
                bracket_region="R16-6",
            ),
            first_leg_home="QF3 B",
            second_leg_home="QF3 A",
        ),
        ChampionsLeagueQuarterfinalTie(
            quarterfinal_id="4",
            team1=ChampionsLeagueBracketOccupant(
                team="QF4 A",
                source_rank=2,
                inherited_seed_rank=2,
                bracket_region="R16-7",
            ),
            team2=ChampionsLeagueBracketOccupant(
                team="QF4 B",
                source_rank=8,
                inherited_seed_rank=8,
                bracket_region="R16-8",
            ),
            first_leg_home="QF4 B",
            second_leg_home="QF4 A",
        ),
    ]

    qf_generic_ties = []

    for quarterfinal in quarterfinals:
        qf_generic_ties.append(
            Tie(
                team1=quarterfinal.team1.team,
                team2=quarterfinal.team2.team,
                match_results=[
                    MatchResult(
                        team1=quarterfinal.first_leg_home,
                        team2=quarterfinal.second_leg_home,
                        goals_team1=0,
                        goals_team2=1,
                    ),
                    MatchResult(
                        team1=quarterfinal.second_leg_home,
                        team2=quarterfinal.first_leg_home,
                        goals_team1=2,
                        goals_team2=0,
                    ),
                ],
                metadata={
                    "quarterfinal_id": quarterfinal.quarterfinal_id,
                },
            )
        )

    qf_stage = Stage(
        name="Quarterfinals",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=[
            occupant.team
            for quarterfinal in quarterfinals
            for occupant in (
                quarterfinal.team1,
                quarterfinal.team2,
            )
        ],
        matches=qf_generic_ties,
    )

    qf_result = StageResolver().resolve(qf_stage)

    assert len(qf_result.qualifiers) == 4

    qf_by_id = {
        quarterfinal.quarterfinal_id: quarterfinal
        for quarterfinal in quarterfinals
    }

    qf_winners: dict[
        str,
        ChampionsLeagueBracketOccupant,
    ] = {}

    for tie_result in qf_result.match_results:
        quarterfinal_id = tie_result.metadata[
            "quarterfinal_id"
        ]
        quarterfinal = qf_by_id[quarterfinal_id]

        qf_winners[quarterfinal_id] = (
            advance_bracket_occupant(
                winner=tie_result.winner,
                first=quarterfinal.team1,
                second=quarterfinal.team2,
                destination_region=quarterfinal_id,
            )
        )

    assert set(qf_winners) == {"1", "2", "3", "4"}

    semifinal_1 = build_champions_league_semifinal_tie(
        {
            "1": qf_winners["1"],
            "2": qf_winners["2"],
        }
    )

    semifinal_2 = build_champions_league_semifinal_tie(
        {
            "1": qf_winners["3"],
            "2": qf_winners["4"],
        }
    )

    sf_generic_ties = [
        Tie(
            team1=semifinal_1.team1.team,
            team2=semifinal_1.team2.team,
            match_results=[
                MatchResult(
                    team1=semifinal_1.first_leg_home,
                    team2=semifinal_1.second_leg_home,
                    goals_team1=0,
                    goals_team2=1,
                ),
                MatchResult(
                    team1=semifinal_1.second_leg_home,
                    team2=semifinal_1.first_leg_home,
                    goals_team1=1,
                    goals_team2=0,
                ),
            ],
            metadata={"semifinal_id": "SF1"},
        ),
        Tie(
            team1=semifinal_2.team1.team,
            team2=semifinal_2.team2.team,
            match_results=[
                MatchResult(
                    team1=semifinal_2.first_leg_home,
                    team2=semifinal_2.second_leg_home,
                    goals_team1=0,
                    goals_team2=1,
                ),
                MatchResult(
                    team1=semifinal_2.second_leg_home,
                    team2=semifinal_2.first_leg_home,
                    goals_team1=2,
                    goals_team2=0,
                ),
            ],
            metadata={"semifinal_id": "SF2"},
        ),
    ]

    sf_stage = Stage(
        name="Semifinals",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=[
            semifinal_1.team1.team,
            semifinal_1.team2.team,
            semifinal_2.team1.team,
            semifinal_2.team2.team,
        ],
        matches=sf_generic_ties,
    )

    sf_result = StageResolver().resolve(sf_stage)

    assert len(sf_result.qualifiers) == 2

    sf_pairings = {
        "SF1": semifinal_1,
        "SF2": semifinal_2,
    }

    semifinal_winners: dict[
        str,
        ChampionsLeagueBracketOccupant,
    ] = {}

    for tie_result in sf_result.match_results:
        semifinal_id = tie_result.metadata[
            "semifinal_id"
        ]
        semifinal = sf_pairings[semifinal_id]

        semifinal_winners[semifinal_id] = (
            advance_bracket_occupant(
                winner=tie_result.winner,
                first=semifinal.team1,
                second=semifinal.team2,
                destination_region=semifinal_id,
            )
        )

    final = build_champions_league_final(
        semifinal_winners
    )

    assert final.neutral_site is True

    final_tie = Tie(
        team1=final.team1.team,
        team2=final.team2.team,
        match_results=[
            MatchResult(
                team1=final.team1.team,
                team2=final.team2.team,
                goals_team1=2,
                goals_team2=1,
            )
        ],
        metadata={
            "stage": "final",
        },
    )

    final_stage = Stage(
        name="Final",
        stage_type=StageType.FINAL,
        participants=[
            final.team1.team,
            final.team2.team,
        ],
        matches=[final_tie],
    )

    final_result = StageResolver().resolve(
        final_stage
    )

    assert final_result.winner == final.team1.team
    assert final_result.runner_up == final.team2.team
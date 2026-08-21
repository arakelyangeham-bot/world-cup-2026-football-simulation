#champions_league_bracket

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from simulation.competition.standings import StandingRow
from simulation.competition.match_result import MatchResult
from simulation.competition.tie import Tie


@dataclass(frozen=True)
class RankingPair:
    first: int
    second: int


@dataclass(frozen=True)
class PlayoffPairingBand:
    band_id: str
    seeded: RankingPair
    unseeded: RankingPair

@dataclass(frozen=True)
class ChampionsLeaguePlayoffTie:
    band_id: str
    seeded_team: str
    unseeded_team: str
    first_leg_home: str
    second_leg_home: str

@dataclass(frozen=True)
class ChampionsLeagueRoundOf16Region:
    region_id: str
    direct_seed_pair: RankingPair
    playoff_band_id: str

@dataclass(frozen=True)
class ChampionsLeagueQuarterfinalRegion:
    quarterfinal_id: str
    round_of_16_slots: tuple[str, str]

@dataclass(frozen=True)
class ChampionsLeagueSemifinalRegion:
    semifinal_id: str
    quarterfinal_regions: tuple[str, str]


CHAMPIONS_LEAGUE_QUARTERFINAL_REGIONS = (
    ChampionsLeagueQuarterfinalRegion(
        quarterfinal_id="1",
        round_of_16_slots=("R16-1", "R16-2"),
    ),
    ChampionsLeagueQuarterfinalRegion(
        quarterfinal_id="2",
        round_of_16_slots=("R16-3", "R16-4"),
    ),
    ChampionsLeagueQuarterfinalRegion(
        quarterfinal_id="3",
        round_of_16_slots=("R16-5", "R16-6"),
    ),
    ChampionsLeagueQuarterfinalRegion(
        quarterfinal_id="4",
        round_of_16_slots=("R16-7", "R16-8"),
    ),
)


@dataclass(frozen=True)
class ChampionsLeagueRoundOf16Tie:
    slot_id: str
    seeded_team: str
    playoff_winner: str
    first_leg_home: str
    second_leg_home: str

@dataclass(frozen=True)
class ChampionsLeagueBracketOccupant:
    team: str
    source_rank: int
    inherited_seed_rank: int
    bracket_region: str

@dataclass(frozen=True)
class ChampionsLeagueQuarterfinalTie:
    quarterfinal_id: str
    team1: ChampionsLeagueBracketOccupant
    team2: ChampionsLeagueBracketOccupant
    first_leg_home: str
    second_leg_home: str

@dataclass(frozen=True)
class ChampionsLeagueSemifinalTie:
    semifinal_id: str
    team1: ChampionsLeagueBracketOccupant
    team2: ChampionsLeagueBracketOccupant
    first_leg_home: str
    second_leg_home: str

@dataclass(frozen=True)
class ChampionsLeagueFinal:
    team1: ChampionsLeagueBracketOccupant
    team2: ChampionsLeagueBracketOccupant
    neutral_site: bool = True

@dataclass(frozen=True)
class ChampionsLeagueRoundOf16Slot:
    slot_id: str
    direct_seed_pair: RankingPair
    playoff_band_id: str
    quarterfinal_id: str

CHAMPIONS_LEAGUE_PLAYOFF_BANDS = (
    PlayoffPairingBand(
        band_id="I",
        seeded=RankingPair(9, 10),
        unseeded=RankingPair(23, 24),
    ),
    PlayoffPairingBand(
        band_id="II",
        seeded=RankingPair(11, 12),
        unseeded=RankingPair(21, 22),
    ),
    PlayoffPairingBand(
        band_id="III",
        seeded=RankingPair(13, 14),
        unseeded=RankingPair(19, 20),
    ),
    PlayoffPairingBand(
        band_id="IV",
        seeded=RankingPair(15, 16),
        unseeded=RankingPair(17, 18),
    ),
)

CHAMPIONS_LEAGUE_ROUND_OF_16_REGIONS = (
    ChampionsLeagueRoundOf16Region(
        region_id="A",
        direct_seed_pair=RankingPair(1, 2),
        playoff_band_id="IV",
    ),
    ChampionsLeagueRoundOf16Region(
        region_id="B",
        direct_seed_pair=RankingPair(3, 4),
        playoff_band_id="III",
    ),
    ChampionsLeagueRoundOf16Region(
        region_id="C",
        direct_seed_pair=RankingPair(5, 6),
        playoff_band_id="II",
    ),
    ChampionsLeagueRoundOf16Region(
        region_id="D",
        direct_seed_pair=RankingPair(7, 8),
        playoff_band_id="I",
    ),
)

CHAMPIONS_LEAGUE_ROUND_OF_16_SEEDED_PAIRS = (
    RankingPair(1, 2),
    RankingPair(3, 4),
    RankingPair(5, 6),
    RankingPair(7, 8),
)

CHAMPIONS_LEAGUE_ROUND_OF_16_SLOTS = (
    ChampionsLeagueRoundOf16Slot(
        slot_id="R16-1",
        direct_seed_pair=RankingPair(1, 2),
        playoff_band_id="IV",
        quarterfinal_id="1",
    ),
    ChampionsLeagueRoundOf16Slot(
        slot_id="R16-2",
        direct_seed_pair=RankingPair(8, 7),
        playoff_band_id="I",
        quarterfinal_id="1",
    ),
    ChampionsLeagueRoundOf16Slot(
        slot_id="R16-3",
        direct_seed_pair=RankingPair(5, 6),
        playoff_band_id="II",
        quarterfinal_id="2",
    ),
    ChampionsLeagueRoundOf16Slot(
        slot_id="R16-4",
        direct_seed_pair=RankingPair(4, 3),
        playoff_band_id="III",
        quarterfinal_id="2",
    ),
    ChampionsLeagueRoundOf16Slot(
        slot_id="R16-5",
        direct_seed_pair=RankingPair(3, 4),
        playoff_band_id="III",
        quarterfinal_id="3",
    ),
    ChampionsLeagueRoundOf16Slot(
        slot_id="R16-6",
        direct_seed_pair=RankingPair(6, 5),
        playoff_band_id="II",
        quarterfinal_id="3",
    ),
    ChampionsLeagueRoundOf16Slot(
        slot_id="R16-7",
        direct_seed_pair=RankingPair(7, 8),
        playoff_band_id="I",
        quarterfinal_id="4",
    ),
    ChampionsLeagueRoundOf16Slot(
        slot_id="R16-8",
        direct_seed_pair=RankingPair(2, 1),
        playoff_band_id="IV",
        quarterfinal_id="4",
    ),
)

CHAMPIONS_LEAGUE_SEMIFINAL_REGIONS = (
    ChampionsLeagueSemifinalRegion(
        semifinal_id="SF1",
        quarterfinal_regions=("1", "2"),
    ),
    ChampionsLeagueSemifinalRegion(
        semifinal_id="SF2",
        quarterfinal_regions=("3", "4"),
    ),
)

def build_champions_league_playoff_ties(
    ranked_rows: list[StandingRow],
    rng: Random,
) -> tuple[ChampionsLeaguePlayoffTie, ...]:
    if len(ranked_rows) != 36:
        raise ValueError(
            "Champions League playoff pairing requires "
            "exactly 36 ranked teams."
        )

    teams = [row.team for row in ranked_rows]

    if len(set(teams)) != 36:
        raise ValueError(
            "Champions League standings contain duplicate teams."
        )

    ties: list[ChampionsLeaguePlayoffTie] = []

    for band in CHAMPIONS_LEAGUE_PLAYOFF_BANDS:
        seeded = [
            teams[band.seeded.first - 1],
            teams[band.seeded.second - 1],
        ]
        unseeded = [
            teams[band.unseeded.first - 1],
            teams[band.unseeded.second - 1],
        ]

        rng.shuffle(unseeded)

        for seeded_team, unseeded_team in zip(
            seeded,
            unseeded,
            strict=True,
        ):
            ties.append(
                ChampionsLeaguePlayoffTie(
                    band_id=band.band_id,
                    seeded_team=seeded_team,
                    unseeded_team=unseeded_team,
                    first_leg_home=unseeded_team,
                    second_leg_home=seeded_team,
                )
            )

    return tuple(ties)

def build_champions_league_round_of_16_ties(
    ranked_rows: list[StandingRow],
    playoff_winners_by_band: dict[
        str,
        tuple[str, str],
    ],
    rng: Random,
) -> tuple[ChampionsLeagueRoundOf16Tie, ...]:
    if len(ranked_rows) != 36:
        raise ValueError(
            "Champions League round-of-16 pairing requires "
            "exactly 36 ranked teams."
        )

    teams = [row.team for row in ranked_rows]

    if len(set(teams)) != 36:
        raise ValueError(
            "Champions League standings contain duplicate teams."
        )

    expected_band_ids = {"I", "II", "III", "IV"}

    if set(playoff_winners_by_band) != expected_band_ids:
        raise ValueError(
            "Round-of-16 pairing requires playoff winners "
            "for bands I, II, III, and IV."
        )

    playoff_entrants = set(teams[8:24])

    supplied_winners = [
        team
        for winners in playoff_winners_by_band.values()
        for team in winners
    ]

    if len(supplied_winners) != 8:
        raise ValueError(
            "Round-of-16 pairing requires exactly "
            "eight playoff winners."
        )

    if len(set(supplied_winners)) != 8:
        raise ValueError(
            "Playoff winners contain duplicate teams."
        )

    if not set(supplied_winners).issubset(
        playoff_entrants
    ):
        raise ValueError(
            "Playoff winners must come from league-phase "
            "positions 9 through 24."
        )

    ties: list[ChampionsLeagueRoundOf16Tie] = []

    slots_by_band = {
        band_id: [
            slot
            for slot in CHAMPIONS_LEAGUE_ROUND_OF_16_SLOTS
            if slot.playoff_band_id == band_id
        ]
        for band_id in expected_band_ids
    }

    for band_id in ("I", "II", "III", "IV"):
        slots = slots_by_band[band_id]

        if len(slots) != 2:
            raise ValueError(
                "Each Champions League playoff band must "
                "feed exactly two round-of-16 slots."
            )

        playoff_winners = list(
            playoff_winners_by_band[band_id]
        )

        rng.shuffle(playoff_winners)

        first_slot = slots[0]

        expected_seed_ranks = {
            first_slot.direct_seed_pair.first,
            first_slot.direct_seed_pair.second,
        }

        for slot in slots[1:]:
            actual_seed_ranks = {
                slot.direct_seed_pair.first,
                slot.direct_seed_pair.second,
            }

            if actual_seed_ranks != expected_seed_ranks:
                raise ValueError(
                    "Round-of-16 slots within a playoff band "
                    "must share the same direct-seed pair."
                )

        seeded_teams = [
            teams[first_slot.direct_seed_pair.first - 1],
            teams[first_slot.direct_seed_pair.second - 1],
        ]

        rng.shuffle(seeded_teams)

        for slot, seeded_team, playoff_winner in zip(
            slots,
            seeded_teams,
            playoff_winners,
            strict=True,
        ):
            ties.append(
                ChampionsLeagueRoundOf16Tie(
                    slot_id=slot.slot_id,
                    seeded_team=seeded_team,
                    playoff_winner=playoff_winner,
                    first_leg_home=playoff_winner,
                    second_leg_home=seeded_team,
                )
            )

    return tuple(ties)

def _playoff_winners_by_band():
    return {
        "I": ("Team 09", "Team 10"),
        "II": ("Team 11", "Team 12"),
        "III": ("Team 13", "Team 14"),
        "IV": ("Team 15", "Team 16"),
    }

def advance_bracket_occupant(
    *,
    winner: str,
    first: ChampionsLeagueBracketOccupant,
    second: ChampionsLeagueBracketOccupant,
    destination_region: str | None = None,
) -> ChampionsLeagueBracketOccupant:
    if winner not in {first.team, second.team}:
        raise ValueError(
            "Winner must be one of the bracket occupants."
        )

    inherited_seed_rank = min(
        first.inherited_seed_rank,
        second.inherited_seed_rank,
    )

    source = first if winner == first.team else second

    return ChampionsLeagueBracketOccupant(
        team=winner,
        source_rank=source.source_rank,
        inherited_seed_rank=inherited_seed_rank,
        bracket_region=(
            destination_region
            if destination_region is not None
            else source.bracket_region
        ),
    )

def build_champions_league_quarterfinal_ties(
    round_of_16_winners: dict[
        str,
        ChampionsLeagueBracketOccupant,
    ],
) -> tuple[ChampionsLeagueQuarterfinalTie, ...]:
    expected_slots = {
        f"R16-{index}"
        for index in range(1, 9)
    }

    if set(round_of_16_winners) != expected_slots:
        raise ValueError(
            "Quarterfinal pairing requires winners from "
            "R16-1 through R16-8."
        )

    ties: list[ChampionsLeagueQuarterfinalTie] = []

    for quarterfinal in CHAMPIONS_LEAGUE_QUARTERFINAL_REGIONS:
        first_slot, second_slot = (
            quarterfinal.round_of_16_slots
        )

        team1 = round_of_16_winners[first_slot]
        team2 = round_of_16_winners[second_slot]

        if (
            team1.inherited_seed_rank
            < team2.inherited_seed_rank
        ):
            second_leg_home = team1.team
            first_leg_home = team2.team

        elif (
            team2.inherited_seed_rank
            < team1.inherited_seed_rank
        ):
            second_leg_home = team2.team
            first_leg_home = team1.team

        else:
            raise ValueError(
                "Quarterfinal occupants cannot have the same "
                "inherited seed rank."
            )

        ties.append(
            ChampionsLeagueQuarterfinalTie(
                quarterfinal_id=(
                    quarterfinal.quarterfinal_id
                ),
                team1=team1,
                team2=team2,
                first_leg_home=first_leg_home,
                second_leg_home=second_leg_home,
            )
        )

    return tuple(ties)

def _round_of_16_slot_winners():
    return {
        f"R16-{index}": ChampionsLeagueBracketOccupant(
            team=f"R16-{index} Winner",
            source_rank=index,
            inherited_seed_rank=index,
            bracket_region=f"R16-{index}",
        )
        for index in range(1, 9)
    }

def build_champions_league_semifinal_tie(
    quarterfinal_winners: dict[
        str,
        ChampionsLeagueBracketOccupant,
    ],
) -> ChampionsLeagueSemifinalTie:
    expected_quarterfinals = {"1", "2"}

    if set(quarterfinal_winners) != expected_quarterfinals:
        raise ValueError(
            "Semifinal pairing requires winners from "
            "quarterfinals 1 and 2."
        )

    team1 = quarterfinal_winners["1"]
    team2 = quarterfinal_winners["2"]

    if (
        team1.inherited_seed_rank
        < team2.inherited_seed_rank
    ):
        second_leg_home = team1.team
        first_leg_home = team2.team

    elif (
        team2.inherited_seed_rank
        < team1.inherited_seed_rank
    ):
        second_leg_home = team2.team
        first_leg_home = team1.team

    else:
        raise ValueError(
            "Semifinal occupants cannot have the same "
            "inherited seed rank."
        )

    return ChampionsLeagueSemifinalTie(
        semifinal_id="SF",
        team1=team1,
        team2=team2,
        first_leg_home=first_leg_home,
        second_leg_home=second_leg_home,
    )

def build_champions_league_final(
    semifinal_winners: dict[
        str,
        ChampionsLeagueBracketOccupant,
    ],
) -> ChampionsLeagueFinal:
    expected_semifinals = {"SF1", "SF2"}

    if set(semifinal_winners) != expected_semifinals:
        raise ValueError(
            "Final pairing requires winners from "
            "semifinals SF1 and SF2."
        )

    return ChampionsLeagueFinal(
        team1=semifinal_winners["SF1"],
        team2=semifinal_winners["SF2"],
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

def quarterfinal_to_tie(
    quarterfinal: ChampionsLeagueQuarterfinalTie,
    match_results: list[MatchResult],
) -> Tie:
    expected_teams = {
        quarterfinal.team1.team,
        quarterfinal.team2.team,
    }

    if len(match_results) != 2:
        raise ValueError(
            "Champions League quarterfinal tie requires "
            "exactly two match results."
        )

    for match in match_results:
        if {
            match.team1,
            match.team2,
        } != expected_teams:
            raise ValueError(
                "Quarterfinal match result contains "
                "unexpected participants."
            )

    return Tie(
        team1=quarterfinal.team1.team,
        team2=quarterfinal.team2.team,
        match_results=match_results,
        metadata={
            "quarterfinal_id": quarterfinal.quarterfinal_id,
            "first_leg_home": quarterfinal.first_leg_home,
            "second_leg_home": quarterfinal.second_leg_home,
        },
    )

def league_phase_rank_by_team(
    ranked_rows: list[StandingRow],
) -> dict[str, int]:
    teams = [row.team for row in ranked_rows]

    if len(teams) != 36:
        raise ValueError(
            "Champions League league-phase ranking requires "
            "exactly 36 teams."
        )

    if len(set(teams)) != 36:
        raise ValueError(
            "Champions League standings contain duplicate teams."
        )

    return {
        team: rank
        for rank, team in enumerate(
            teams,
            start=1,
        )
    }
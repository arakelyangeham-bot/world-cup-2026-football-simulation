#champions_league_structural_simulator

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from simulation.competition.match_result import MatchResult

from simulation.competition.champions_league_advancement import (
    resolve_champions_league_league_phase,
)
from simulation.competition.champions_league_league_phase import (
    build_synthetic_champions_league_league_phase_schedule,
)
from simulation.competition.stage import Stage, StageType
from simulation.competition.stage_resolver import StageResolver

from simulation.competition.champions_league_bracket import (
    ChampionsLeagueBracketOccupant,
    advance_bracket_occupant,
    build_champions_league_final,
    build_champions_league_playoff_ties,
    build_champions_league_quarterfinal_ties,
    build_champions_league_round_of_16_ties,
    build_champions_league_semifinal_tie,
    league_phase_rank_by_team,
)
from simulation.competition.tie import Tie

from collections.abc import Callable


MatchSimulator = Callable[
    [str, str, str],
    tuple[int, int],
]

@dataclass(frozen=True)
class ChampionsLeagueStructuralSimulationResult:
    champion: str
    runner_up: str
    league_phase_rankings: tuple[str, ...]
    direct_round_of_16: tuple[str, ...]
    knockout_playoff: tuple[str, ...]
    league_phase_eliminated: tuple[str, ...]
    playoff_winners: tuple[str, ...]
    round_of_16_winners: tuple[str, ...]
    quarterfinal_winners: tuple[str, ...]
    semifinal_winners: tuple[str, ...]

def _simulate_match(
    *,
    team1: str,
    team2: str,
    rng: Random,
    stage: str,
    allow_draw: bool,
) -> MatchResult:
    goals_team1 = rng.randint(0, 3)
    goals_team2 = rng.randint(0, 3)

    if not allow_draw:
        while goals_team1 == goals_team2:
            goals_team1 = rng.randint(0, 3)
            goals_team2 = rng.randint(0, 3)

    return MatchResult(
        team1=team1,
        team2=team2,
        goals_team1=goals_team1,
        goals_team2=goals_team2,
        stage=stage,
    )

def simulate_champions_league_structural(
    teams: list[str],
    seed: int = 202627,
    match_simulator: MatchSimulator | None = None,
) -> ChampionsLeagueStructuralSimulationResult:
    if len(teams) != 36:
        raise ValueError(
            "Champions League structural simulation requires "
            "exactly 36 teams."
        )

    if len(set(teams)) != 36:
        raise ValueError(
            "Champions League structural simulation contains "
            "duplicate teams."
        )

    rng = Random(seed)

    def simulate_match(
        *,
        home_team: str,
        away_team: str,
        stage: str,
        allow_draw: bool,
    ) -> MatchResult:
        if match_simulator is None:
            return _simulate_match(
                team1=home_team,
                team2=away_team,
                rng=rng,
                stage=stage,
                allow_draw=allow_draw,
            )

        home_goals, away_goals = match_simulator(
            home_team,
            away_team,
            stage,
        )

        if not allow_draw and home_goals == away_goals:
            raise ValueError(
                "Injected match simulator returned a draw "
                "for a stage requiring a winner."
            )

        return MatchResult(
            team1=home_team,
            team2=away_team,
            goals_team1=home_goals,
            goals_team2=away_goals,
            stage=stage,
        )

    fixtures = (
        build_synthetic_champions_league_league_phase_schedule(
            teams
        )
    )

    league_phase_results: list[MatchResult] = []

    for fixture in fixtures:
        league_phase_results.append(
            simulate_match(
                home_team=fixture.home_team,
                away_team=fixture.away_team,
                stage="League Phase",
                allow_draw=True,
            )
        )

    league_phase_stage = Stage(
        name="League Phase",
        stage_type=StageType.SWISS,
        participants=teams,
        matches=league_phase_results,
    )

    league_phase_result = (
        StageResolver().resolve(
            league_phase_stage
        )
    )

    if league_phase_result.standings is None:
        raise AssertionError(
            "League-phase simulation did not produce standings."
        )

    ranked_rows = (
        league_phase_result.standings.ranked_rows()
    )

    if len(ranked_rows) != 36:
        raise AssertionError(
            "League-phase simulation did not produce "
            "36 ranked teams."
        )

    advancement = (
        resolve_champions_league_league_phase(
            ranked_rows
        )
    )

    league_phase_rankings = tuple(
        row.team
        for row in ranked_rows
    )

    playoff_pairings = (
        build_champions_league_playoff_ties(
            ranked_rows,
            rng,
        )
    )

    playoff_ties: list[Tie] = []

    for pairing in playoff_pairings:
        first_leg = simulate_match(
            home_team=pairing.first_leg_home,
            away_team=pairing.second_leg_home,
            stage="Knockout Phase Playoff",
            allow_draw=True,
        )

        second_leg = simulate_match(
            home_team=pairing.second_leg_home,
            away_team=pairing.first_leg_home,
            stage="Knockout Phase Playoff",
            allow_draw=True,
        )

        # Structural simulator only:
        # if aggregate is level, nudge the second leg
        # so the generic knockout engine has a winner.
        aggregate_seeded = (
            (
                first_leg.goals_team2
                + second_leg.goals_team1
            )
        )
        aggregate_unseeded = (
            (
                first_leg.goals_team1
                + second_leg.goals_team2
            )
        )

        if aggregate_seeded == aggregate_unseeded:
            second_leg = MatchResult(
                team1=second_leg.team1,
                team2=second_leg.team2,
                goals_team1=second_leg.goals_team1 + 1,
                goals_team2=second_leg.goals_team2,
                stage=second_leg.stage,
            )

        playoff_ties.append(
            Tie(
                team1=pairing.seeded_team,
                team2=pairing.unseeded_team,
                match_results=[
                    first_leg,
                    second_leg,
                ],
                metadata={
                    "band_id": pairing.band_id,
                },
            )
        )

    playoff_stage = Stage(
        name="Knockout Phase Playoffs",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=list(
            advancement.knockout_playoff
        ),
        matches=playoff_ties,
    )

    playoff_result = (
        StageResolver().resolve(
            playoff_stage
        )
    )

    playoff_winners = tuple(
        playoff_result.qualifiers
    )

    winners_by_band: dict[
        str,
        list[str],
    ] = {
        "I": [],
        "II": [],
        "III": [],
        "IV": [],
    }

    for tie_result in playoff_result.match_results:
        band_id = tie_result.metadata["band_id"]

        winners_by_band[band_id].append(
            tie_result.winner
        )

    normalized_winners_by_band = {
        band_id: tuple(winners)
        for band_id, winners
        in winners_by_band.items()
    }

    round_of_16_pairings = (
        build_champions_league_round_of_16_ties(
            ranked_rows,
            normalized_winners_by_band,
            rng,
        )
    )

    round_of_16_ties: list[Tie] = []

    for pairing in round_of_16_pairings:
        first_leg = simulate_match(
            home_team=pairing.first_leg_home,
            away_team=pairing.second_leg_home,
            stage="Round of 16",
            allow_draw=True,
        )

        second_leg = simulate_match(
            home_team=pairing.second_leg_home,
            away_team=pairing.first_leg_home,
            stage="Round of 16",
            allow_draw=True,
        )

        aggregate_seeded = (
            first_leg.goals_team2
            + second_leg.goals_team1
        )

        aggregate_playoff = (
            first_leg.goals_team1
            + second_leg.goals_team2
        )

        if aggregate_seeded == aggregate_playoff:
            second_leg = MatchResult(
                team1=second_leg.team1,
                team2=second_leg.team2,
                goals_team1=second_leg.goals_team1 + 1,
                goals_team2=second_leg.goals_team2,
                stage=second_leg.stage,
            )

        round_of_16_ties.append(
            Tie(
                team1=pairing.seeded_team,
                team2=pairing.playoff_winner,
                match_results=[
                    first_leg,
                    second_leg,
                ],
                metadata={
                    "slot_id": pairing.slot_id,
                },
            )
        )

    round_of_16_stage = Stage(
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
        matches=round_of_16_ties,
    )

    round_of_16_result = (
        StageResolver().resolve(
            round_of_16_stage
        )
    )

    round_of_16_winners = tuple(
        round_of_16_result.qualifiers
    )

    ranks = league_phase_rank_by_team(
        ranked_rows
    )

    pairing_by_slot = {
        pairing.slot_id: pairing
        for pairing in round_of_16_pairings
    }

    winners_by_slot: dict[
        str,
        ChampionsLeagueBracketOccupant,
    ] = {}

    for tie_result in round_of_16_result.match_results:
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

    quarterfinal_pairings = (
        build_champions_league_quarterfinal_ties(
            winners_by_slot
        )
    )

    quarterfinal_ties: list[Tie] = []

    for pairing in quarterfinal_pairings:
        first_leg = simulate_match(
            home_team=pairing.first_leg_home,
            away_team=pairing.second_leg_home,
            stage="Quarterfinal",
            allow_draw=True,
        )

        second_leg = simulate_match(
            home_team=pairing.second_leg_home,
            away_team=pairing.first_leg_home,
            stage="Quarterfinal",
            allow_draw=True,
        )

        aggregate_team1 = 0
        aggregate_team2 = 0

        for match in (first_leg, second_leg):
            if match.team1 == pairing.team1.team:
                aggregate_team1 += match.goals_team1
                aggregate_team2 += match.goals_team2
            else:
                aggregate_team1 += match.goals_team2
                aggregate_team2 += match.goals_team1

        if aggregate_team1 == aggregate_team2:
            if second_leg.team1 == pairing.team1.team:
                second_leg = MatchResult(
                    team1=second_leg.team1,
                    team2=second_leg.team2,
                    goals_team1=second_leg.goals_team1 + 1,
                    goals_team2=second_leg.goals_team2,
                    stage=second_leg.stage,
                )
            else:
                second_leg = MatchResult(
                    team1=second_leg.team1,
                    team2=second_leg.team2,
                    goals_team1=second_leg.goals_team1,
                    goals_team2=second_leg.goals_team2 + 1,
                    stage=second_leg.stage,
                )

        quarterfinal_ties.append(
            Tie(
                team1=pairing.team1.team,
                team2=pairing.team2.team,
                match_results=[
                    first_leg,
                    second_leg,
                ],
                metadata={
                    "quarterfinal_id":
                        pairing.quarterfinal_id,
                },
            )
        )

    quarterfinal_stage = Stage(
        name="Quarterfinals",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=[
            occupant.team
            for pairing in quarterfinal_pairings
            for occupant in (
                pairing.team1,
                pairing.team2,
            )
        ],
        matches=quarterfinal_ties,
    )

    quarterfinal_result = (
        StageResolver().resolve(
            quarterfinal_stage
        )
    )

    quarterfinal_by_id = {
        pairing.quarterfinal_id: pairing
        for pairing in quarterfinal_pairings
    }

    quarterfinal_winner_occupants: dict[
        str,
        ChampionsLeagueBracketOccupant,
    ] = {}

    for tie_result in quarterfinal_result.match_results:
        quarterfinal_id = tie_result.metadata[
            "quarterfinal_id"
        ]

        pairing = quarterfinal_by_id[
            quarterfinal_id
        ]

        quarterfinal_winner_occupants[
            quarterfinal_id
        ] = advance_bracket_occupant(
            winner=tie_result.winner,
            first=pairing.team1,
            second=pairing.team2,
            destination_region=quarterfinal_id,
        )

    quarterfinal_winners = tuple(
        quarterfinal_result.qualifiers
    )

    semifinal_1 = build_champions_league_semifinal_tie(
        {
            "1": quarterfinal_winner_occupants["1"],
            "2": quarterfinal_winner_occupants["2"],
        }
    )

    semifinal_2 = build_champions_league_semifinal_tie(
        {
            "1": quarterfinal_winner_occupants["3"],
            "2": quarterfinal_winner_occupants["4"],
        }
    )

    semifinal_pairings = {
        "SF1": semifinal_1,
        "SF2": semifinal_2,
    }

    semifinal_ties: list[Tie] = []

    for semifinal_id, pairing in semifinal_pairings.items():
        first_leg = simulate_match(
            home_team=pairing.first_leg_home,
            away_team=pairing.second_leg_home,
            stage="Semifinal",
            allow_draw=True,
        )

        second_leg = simulate_match(
            home_team=pairing.second_leg_home,
            away_team=pairing.first_leg_home,
            stage="Semifinal",
            allow_draw=True,
        )

        aggregate_team1 = 0
        aggregate_team2 = 0

        for match in (first_leg, second_leg):
            if match.team1 == pairing.team1.team:
                aggregate_team1 += match.goals_team1
                aggregate_team2 += match.goals_team2
            else:
                aggregate_team1 += match.goals_team2
                aggregate_team2 += match.goals_team1

        if aggregate_team1 == aggregate_team2:
            if second_leg.team1 == pairing.team1.team:
                second_leg = MatchResult(
                    team1=second_leg.team1,
                    team2=second_leg.team2,
                    goals_team1=second_leg.goals_team1 + 1,
                    goals_team2=second_leg.goals_team2,
                    stage=second_leg.stage,
                )
            else:
                second_leg = MatchResult(
                    team1=second_leg.team1,
                    team2=second_leg.team2,
                    goals_team1=second_leg.goals_team1,
                    goals_team2=second_leg.goals_team2 + 1,
                    stage=second_leg.stage,
                )

        semifinal_ties.append(
            Tie(
                team1=pairing.team1.team,
                team2=pairing.team2.team,
                match_results=[
                    first_leg,
                    second_leg,
                ],
                metadata={
                    "semifinal_id": semifinal_id,
                },
            )
        )

    semifinal_stage = Stage(
        name="Semifinals",
        stage_type=StageType.TWO_LEG_KNOCKOUT,
        participants=[
            team
            for pairing in semifinal_pairings.values()
            for team in (
                pairing.team1.team,
                pairing.team2.team,
            )
        ],
        matches=semifinal_ties,
    )

    semifinal_result = (
        StageResolver().resolve(
            semifinal_stage
        )
    )

    semifinal_winners = tuple(
        semifinal_result.qualifiers
    )

    semifinal_winner_occupants: dict[
        str,
        ChampionsLeagueBracketOccupant,
    ] = {}

    for tie_result in semifinal_result.match_results:
        semifinal_id = tie_result.metadata[
            "semifinal_id"
        ]

        pairing = semifinal_pairings[
            semifinal_id
        ]

        semifinal_winner_occupants[
            semifinal_id
        ] = advance_bracket_occupant(
            winner=tie_result.winner,
            first=pairing.team1,
            second=pairing.team2,
            destination_region=semifinal_id,
        )

    final = build_champions_league_final(
        semifinal_winner_occupants
    )

    final_match = simulate_match(
        home_team=final.team1.team,
        away_team=final.team2.team,
        stage="Final",
        allow_draw=False,
    )

    final_tie = Tie(
        team1=final.team1.team,
        team2=final.team2.team,
        match_results=[final_match],
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

    if (
        final_result.winner is None
        or final_result.runner_up is None
    ):
        raise AssertionError(
            "Final did not produce champion and runner-up."
        )

    return ChampionsLeagueStructuralSimulationResult(
        champion=final_result.winner,
        runner_up=final_result.runner_up,
        league_phase_rankings=league_phase_rankings,
        direct_round_of_16=(
            advancement.direct_round_of_16
        ),
        knockout_playoff=(
            advancement.knockout_playoff
        ),
        league_phase_eliminated=(
            advancement.eliminated
        ),
        playoff_winners=playoff_winners,
        round_of_16_winners=round_of_16_winners,
        quarterfinal_winners=quarterfinal_winners,
        semifinal_winners=semifinal_winners,
    )
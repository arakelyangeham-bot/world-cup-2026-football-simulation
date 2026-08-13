#wc2026_tournament_simulator

from dataclasses import dataclass

from scripts.team_strength_loader import load_team_repository
from wc2026_group_stage import (
    extract_qualifiers_from_standings,
    simulate_group_stage_with_matches,
)
from wc2026_knockout_mapping import build_round_of_32
from wc2026_knockout_stage import (
    build_next_round,
    build_third_place_playoff,
    ROUND_OF_16_PAIRINGS,
    QUARTERFINAL_PAIRINGS,
    SEMIFINAL_PAIRINGS,
    FINAL_PAIRING,
)


@dataclass
class TournamentResult:
    champion: str
    runner_up: str
    third_place: str
    fourth_place: str
    group_stage_results: list
    round_of_32: list[str]
    round_of_16: list[str]
    quarterfinalists: list[str]
    semifinalists: list[str]
    finalists: list[str]

    standings: dict
    r32_results: list
    r16_results: list
    qf_results: list
    sf_results: list
    third_place_results: list
    final_results: list


def simulate_round(matches, strengths):
    return [
        match.simulate(strengths)
        for match in matches
    ]


def loser_of(result):
    if result.winner.team == result.team1.team:
        return result.team2
    return result.team1


def simulate_tournament(strengths=None) -> TournamentResult:
    if strengths is None:
        strengths = load_team_repository()

    standings, group_stage_results = simulate_group_stage_with_matches(
        strengths,
    )

    qualifiers = extract_qualifiers_from_standings(standings)

    round_of_32 = build_round_of_32(qualifiers)
    r32_results = simulate_round(round_of_32, strengths)

    round_of_16 = build_next_round(r32_results, ROUND_OF_16_PAIRINGS, 89)
    r16_results = simulate_round(round_of_16, strengths)

    quarterfinals = build_next_round(r16_results, QUARTERFINAL_PAIRINGS, 97)
    qf_results = simulate_round(quarterfinals, strengths)

    semifinals = build_next_round(qf_results, SEMIFINAL_PAIRINGS, 101)
    sf_results = simulate_round(semifinals, strengths)

    third_place_match = build_third_place_playoff(sf_results)
    third_place_results = simulate_round(third_place_match, strengths)
    third_place_result = third_place_results[0]

    final = build_next_round(sf_results, FINAL_PAIRING, 104)
    final_results = simulate_round(final, strengths)
    final_result = final_results[0]

    return TournamentResult(
        champion=final_result.winner.team,
        runner_up=loser_of(final_result).team,
        third_place=third_place_result.winner.team,
        fourth_place=loser_of(third_place_result).team,

        group_stage_results=group_stage_results,
        round_of_32=[q.team for q in qualifiers],
        round_of_16=[r.winner.team for r in r32_results],
        quarterfinalists=[r.winner.team for r in r16_results],
        semifinalists=[r.winner.team for r in qf_results],
        finalists=[r.winner.team for r in sf_results],

        standings=standings,
        r32_results=r32_results,
        r16_results=r16_results,
        qf_results=qf_results,
        sf_results=sf_results,
        third_place_results=third_place_results,
        final_results=final_results,
    )


if __name__ == "__main__":
    result = simulate_tournament()

    print("Champion:", result.champion)
    print("Runner-up:", result.runner_up)
    print("Third place:", result.third_place)
    print("Fourth place:", result.fourth_place)
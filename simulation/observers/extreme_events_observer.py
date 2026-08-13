#extreme_events_observer.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulation.observers.base_observer import TournamentObserver


@dataclass
class ExtremeEventRecord:
    event_name: str
    simulation_id: int
    value: float
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventLeaderboard:
    event_name: str
    keep_top_n: int = 10
    higher_is_more_extreme: bool = True
    records: list[ExtremeEventRecord] = field(default_factory=list)

    def consider(self, record: ExtremeEventRecord) -> None:
        self.records.append(record)

        self.records.sort(
            key=lambda item: item.value,
            reverse=self.higher_is_more_extreme,
        )

        self.records = self.records[: self.keep_top_n]


class ExtremeEventsObserver(TournamentObserver):
    """
    Observer that tracks extreme events across completed tournament simulations.

    This observer does not affect tournament simulation. It only reads completed
    TournamentResult objects and records the most extreme examples found so far.
    """

    def __init__(self, keep_top_n: int = 10) -> None:
        self.keep_top_n = keep_top_n
        self.leaderboards: dict[str, EventLeaderboard] = {}

    def observe(
        self,
        result: Any,
        simulation_id: int,
        team_repository: dict[str, dict] | None = None,
    ) -> None:
        self._observe_match_extremes(result, simulation_id, team_repository)
        self._observe_champion_extremes(result, simulation_id, team_repository)
        self._observe_progression_extremes(result, simulation_id, team_repository)
        self._observe_group_extremes(result, simulation_id, team_repository)

    def finalize(self) -> dict[str, Any]:
        rows = []

        for leaderboard in self.leaderboards.values():
            for rank, record in enumerate(leaderboard.records, start=1):
                rows.append(
                    self._record_to_row(
                        record=record,
                        rank=rank,
                    )
                )

        return {
            "leaderboards": self.leaderboards,
            "rows": rows,
        }

    def _consider_larger_is_extreme(
        self,
        event_name: str,
        simulation_id: int,
        value: float,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._consider_record(
            event_name=event_name,
            simulation_id=simulation_id,
            value=value,
            description=description,
            metadata=metadata,
            higher_is_more_extreme=True,
        )


    def _consider_smaller_is_extreme(
        self,
        event_name: str,
        simulation_id: int,
        value: float,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._consider_record(
            event_name=event_name,
            simulation_id=simulation_id,
            value=value,
            description=description,
            metadata=metadata,
            higher_is_more_extreme=False,
        )


    def _consider_record(
        self,
        event_name: str,
        simulation_id: int,
        value: float,
        description: str,
        metadata: dict[str, Any] | None,
        higher_is_more_extreme: bool,
    ) -> None:
        leaderboard = self.leaderboards.get(event_name)

        if leaderboard is None:
            leaderboard = EventLeaderboard(
                event_name=event_name,
                keep_top_n=self.keep_top_n,
                higher_is_more_extreme=higher_is_more_extreme,
            )
            self.leaderboards[event_name] = leaderboard

        record = ExtremeEventRecord(
            event_name=event_name,
            simulation_id=simulation_id,
            value=value,
            description=description,
            metadata=metadata or {},
        )

        leaderboard.consider(record)

    def _record_to_row(
        self,
        record: ExtremeEventRecord,
        rank: int,
    ) -> dict[str, Any]:
        row = {
            "event_name": record.event_name,
            "rank": rank,
            "simulation_id": record.simulation_id,
            "value": record.value,
            "description": record.description,
        }

        for key, value in record.metadata.items():
            row[f"metadata_{key}"] = value

        return row

    def _observe_match_extremes(
        self,
        result: Any,
        simulation_id: int,
        team_repository: dict[str, dict] | None,
    ) -> None:
        for match in iter_all_matches(result):
            team1 = get_team_name(match.team1)
            team2 = get_team_name(match.team2)

            goals1 = match.goals_team1
            goals2 = match.goals_team2

            total_goals = goals1 + goals2
            goal_margin = abs(goals1 - goals2)

            if goals1 > goals2:
                winner = team1
                loser = team2
                winner_goals = goals1
                loser_goals = goals2
            elif goals2 > goals1:
                winner = team2
                loser = team1
                winner_goals = goals2
                loser_goals = goals1
            else:
                winner = None
                loser = None
                winner_goals = None
                loser_goals = None

            description = f"{team1} {goals1}-{goals2} {team2}"

            self._consider_larger_is_extreme(
                event_name="biggest_blowout",
                simulation_id=simulation_id,
                value=goal_margin,
                description=description,
                metadata={
                    "team1": team1,
                    "team2": team2,
                    "goals_team1": goals1,
                    "goals_team2": goals2,
                    "stage": get_match_stage(match),
                },
            )

            self._consider_larger_is_extreme(
                event_name="highest_scoring_match",
                simulation_id=simulation_id,
                value=total_goals,
                description=description,
                metadata={
                    "team1": team1,
                    "team2": team2,
                    "goals_team1": goals1,
                    "goals_team2": goals2,
                    "stage": get_match_stage(match),
                },
            )

            if goals1 == goals2:
                self._consider_larger_is_extreme(
                    event_name="highest_scoring_draw",
                    simulation_id=simulation_id,
                    value=total_goals,
                    description=description,
                    metadata={
                        "team1": team1,
                        "team2": team2,
                        "goals_team1": goals1,
                        "goals_team2": goals2,
                        "stage": get_match_stage(match),
                    },
                )

            if team_repository is not None and winner is not None:
                winner_strength = get_team_strength(winner, team_repository)
                loser_strength = get_team_strength(loser, team_repository)

                if winner_strength is not None and loser_strength is not None:
                    upset_gap = loser_strength - winner_strength

                    if upset_gap > 0:
                        upset_description = (
                            f"{winner} defeated {loser} "
                            f"{winner_goals}-{loser_goals}"
                        )

                        self._consider_larger_is_extreme(
                            event_name="biggest_underdog_win",
                            simulation_id=simulation_id,
                            value=upset_gap,
                            description=upset_description,
                            metadata={
                                "winner": winner,
                                "loser": loser,
                                "winner_strength": winner_strength,
                                "loser_strength": loser_strength,
                                "upset_gap": upset_gap,
                                "stage": get_match_stage(match),
                            },
                        )

    def _observe_champion_extremes(
        self,
        result: Any,
        simulation_id: int,
        team_repository: dict[str, dict] | None,
    ) -> None:
        champion = result.champion
        goals_for, goals_against = get_team_tournament_goals(result, champion)

        self._consider_larger_is_extreme(
            event_name="most_goals_by_champion",
            simulation_id=simulation_id,
            value=goals_for,
            description=f"{champion} won the tournament with {goals_for} goals scored.",
            metadata={
                "team": champion,
                "goals_for": goals_for,
                "goals_against": goals_against,
            },
        )

        self._consider_smaller_is_extreme(
            event_name="fewest_goals_conceded_by_champion",
            simulation_id=simulation_id,
            value=goals_against,
            description=(
                f"{champion} won the tournament while conceding "
                f"{goals_against} goals."
            ),
            metadata={
                "team": champion,
                "goals_for": goals_for,
                "goals_against": goals_against,
            },
        )

        dominance_score = goals_for - goals_against

        self._consider_larger_is_extreme(
            event_name="most_dominant_champion",
            simulation_id=simulation_id,
            value=dominance_score,
            description=(
                f"{champion} won the tournament with a "
                f"{goals_for}-{goals_against} goal record."
            ),
            metadata={
                "team": champion,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goal_difference": dominance_score,
            },
        )

        if team_repository is not None:
            champion_strength = get_team_strength(champion, team_repository)

            if champion_strength is not None:
                self._consider_smaller_is_extreme(
                    event_name="weakest_champion",
                    simulation_id=simulation_id,
                    value=champion_strength,
                    description=(
                        f"{champion} won the tournament with pre-tournament "
                        f"strength {champion_strength:.3f}."
                    ),
                    metadata={
                        "team": champion,
                        "team_strength": champion_strength,
                    },
                )
    
    def _observe_progression_extremes(
        self,
        result: Any,
        simulation_id: int,
        team_repository: dict[str, dict] | None,
    ) -> None:
        if team_repository is None:
            return

        group_stage_teams = {
            row.team
            for group_rows in result.standings.values()
            for row in group_rows
        }

        round_of_32_teams = set(result.round_of_32)
        eliminated_in_groups = group_stage_teams - round_of_32_teams

        for team in eliminated_in_groups:
            strength = get_team_strength(team, team_repository)

            if strength is not None:
                self.self._consider_larger_is_extreme(
                    event_name="strongest_group_stage_elimination",
                    simulation_id=simulation_id,
                    value=strength,
                    description=(
                        f"{team} was eliminated in the group stage "
                        f"with pre-tournament strength {strength:.3f}."
                    ),
                    metadata={
                        "team": team,
                        "team_strength": strength,
                    },
                )

        for team in result.semifinalists:
            strength = get_team_strength(team, team_repository)

            if strength is not None:
                self._consider_smaller_is_extreme(
                    event_name="weakest_semifinalist",
                    simulation_id=simulation_id,
                    value=strength,
                    description=(
                        f"{team} reached the semifinals with pre-tournament "
                        f"strength {strength:.3f}."
                    ),
                    metadata={
                        "team": team,
                        "team_strength": strength,
                    },
                )

        for team in result.finalists:
            strength = get_team_strength(team, team_repository)

            if strength is not None:
                self._consider_smaller_is_extreme(
                    event_name="weakest_finalist",
                    simulation_id=simulation_id,
                    value=strength,
                    description=(
                        f"{team} reached the final with pre-tournament "
                        f"strength {strength:.3f}."
                    ),
                    metadata={
                        "team": team,
                        "team_strength": strength,
                    },
                )

    def _observe_group_extremes(
        self,
        result: Any,
        simulation_id: int,
        team_repository: dict[str, dict] | None,
    ) -> None:
        qualified_teams = set(result.round_of_32)

        for group_name, group_rows in result.standings.items():
            sorted_rows = sorted(
                group_rows,
                key=lambda row: (
                    row.points,
                    row.goal_difference,
                    row.goals_for,
                ),
                reverse=True,
            )

            if not sorted_rows:
                continue

            qualifiers = [
                row for row in sorted_rows
                if row.team in qualified_teams
            ]

            eliminated = [
                row for row in sorted_rows
                if row.team not in qualified_teams
            ]

            point_spread = sorted_rows[0].points - sorted_rows[-1].points
            group_goal_total = sum(row.goals_for for row in sorted_rows)

            chaos_score = group_goal_total - point_spread

            self._consider_larger_is_extreme(
                event_name="most_chaotic_group",
                simulation_id=simulation_id,
                value=chaos_score,
                description=(
                    f"{group_name} produced {group_goal_total} total goals "
                    f"with a {point_spread}-point spread."
                ),
                metadata={
                    "group": group_name,
                    "total_goals": group_goal_total,
                    "point_spread": point_spread,
                    "chaos_score": chaos_score,
                },
            )

            for row in qualifiers:
                self._consider_smaller_is_extreme(
                    event_name="lowest_points_qualifier",
                    simulation_id=simulation_id,
                    value=row.points,
                    description=(
                        f"{row.team} qualified from {group_name} "
                        f"with only {row.points} points."
                    ),
                    metadata={
                        "team": row.team,
                        "group": group_name,
                        "points": row.points,
                        "goal_difference": row.goal_difference,
                        "goals_for": row.goals_for,
                    },
                )

            for row in eliminated:
                self._consider_larger_is_extreme(
                    event_name="highest_points_eliminated_team",
                    simulation_id=simulation_id,
                    value=row.points,
                    description=(
                        f"{row.team} was eliminated from {group_name} "
                        f"despite earning {row.points} points."
                    ),
                    metadata={
                        "team": row.team,
                        "group": group_name,
                        "points": row.points,
                        "goal_difference": row.goal_difference,
                        "goals_for": row.goals_for,
                    },
                )


def iter_group_matches(result: Any):
    yield from result.group_stage_results


def iter_knockout_matches(result: Any):
    knockout_stages = (
        result.r32_results,
        result.r16_results,
        result.qf_results,
        result.sf_results,
        result.third_place_results,
        result.final_results,
    )

    for stage in knockout_stages:
        yield from stage


def iter_all_matches(result: Any):
    yield from iter_group_matches(result)
    yield from iter_knockout_matches(result)


def get_match_stage(match: Any) -> str:
    return getattr(match, "stage", "unknown")


def get_team_strength(
    team: str,
    team_repository: dict[str, dict],
) -> float | None:
    team_data = team_repository.get(team)

    if team_data is None:
        return None

    for key in (
        "overall_strength",
        "team_strength",
        "strength",
        "rating",
        "elo",
    ):
        value = team_data.get(key)

        if value is not None:
            return float(value)

    return None


def get_team_tournament_goals(result: Any, team: str) -> tuple[int, int]:
    goals_for = 0
    goals_against = 0

    for match in iter_all_matches(result):
        team1 = get_team_name(match.team1)
        team2 = get_team_name(match.team2)

        if team1 == team:
            goals_for += match.goals_team1
            goals_against += match.goals_team2
        elif team2 == team:
            goals_for += match.goals_team2
            goals_against += match.goals_team1

    return goals_for, goals_against

def get_team_name(team_or_entry: Any) -> str:
    if hasattr(team_or_entry, "team"):
        return team_or_entry.team

    return str(team_or_entry)
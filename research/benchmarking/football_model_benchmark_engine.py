#football_model_benchmark_engine

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from competition_catalog import (
    CompetitionBuilder,
    CompetitionDefinition,
    StageDefinition,
)
from fixture_generation import (
    RoundRobinFixtureGenerator,
)
from simulation.competition import (
    CompetitionEngine,
)
from simulation.league_match_simulator import (
    LeagueMatchSimulator,
)

from research.studies.study_042_cross_league_opta_prior_calibration.build_league_fingerprints import (
    build_league_fingerprint,
)


@dataclass(frozen=True)
class FootballModelBenchmarkSpec:
    """
    Runtime specification for one football model participating
    in a repeated league benchmark.
    """

    model_id: str
    display_name: str
    football_model: Any
    runtime_names: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError(
                "Benchmark model ID must not be empty."
            )

        if not self.display_name.strip():
            raise ValueError(
                "Benchmark model display name must not be empty."
            )

        if not self.runtime_names:
            raise ValueError(
                "Benchmark model must define runtime club names."
            )


@dataclass(frozen=True)
class FootballModelBenchmarkConfig:
    """
    Shared benchmark configuration.

    Every football model receives the same canonical clubs,
    fixtures, dates, competition rules, season count, and
    season-seed sequence.
    """

    benchmark_id: str
    benchmark_name: str

    canonical_clubs: tuple[str, ...]

    simulation_count: int
    base_seed: int

    season_start_date: date
    days_between_matchdays: int = 7

    double_round_robin: bool = True

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError(
                "Benchmark ID must not be empty."
            )

        if not self.benchmark_name.strip():
            raise ValueError(
                "Benchmark name must not be empty."
            )

        if len(self.canonical_clubs) < 2:
            raise ValueError(
                "Benchmark requires at least two clubs."
            )

        if len(self.canonical_clubs) != len(
            set(self.canonical_clubs)
        ):
            raise ValueError(
                "Canonical benchmark clubs must be unique."
            )

        if self.simulation_count < 1:
            raise ValueError(
                "simulation_count must be positive."
            )

        if self.base_seed < 0:
            raise ValueError(
                "base_seed must not be negative."
            )

        if self.days_between_matchdays < 1:
            raise ValueError(
                "days_between_matchdays must be positive."
            )

class CanonicalFootballModel:
    """
    Translate canonical benchmark names into model-specific
    runtime names.

    This wrapper deliberately exposes only simulate_match().
    The benchmark engine remains ignorant of repositories,
    ClubElo, expected goals, and scoreline-generation details.
    """

    def __init__(
        self,
        *,
        model_id: str,
        football_model: Any,
        runtime_names: Mapping[str, str],
    ) -> None:
        self.model_id = model_id
        self.football_model = football_model
        self.runtime_names = dict(
            runtime_names
        )

    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        prediction_date=None,
    ) -> tuple[int, int]:
        try:
            runtime_home = self.runtime_names[
                home_team
            ]

            runtime_away = self.runtime_names[
                away_team
            ]

        except KeyError as error:
            raise KeyError(
                "No runtime alias exists for canonical club "
                f"{error.args[0]!r} in model "
                f"{self.model_id!r}."
            ) from error

        return self.football_model.simulate_match(
            runtime_home,
            runtime_away,
            prediction_date=prediction_date,
        )

@dataclass(frozen=True)
class FootballModelBenchmarkResult:
    """
    Raw repeated-season outputs for all benchmark models.
    """

    season_statistics: pd.DataFrame
    club_season_statistics: pd.DataFrame
    match_statistics: pd.DataFrame

    def validate(self) -> None:
        if self.season_statistics.empty:
            raise ValueError(
                "Season statistics are empty."
            )

        if self.club_season_statistics.empty:
            raise ValueError(
                "Club-season statistics are empty."
            )

        if self.match_statistics.empty:
            raise ValueError(
                "Match statistics are empty."
            )

class FootballModelBenchmarkEngine:
    """
    Run repeated, matched league seasons for one or more
    football models.

    Responsibilities
    ----------------
    - Build one shared canonical fixture population.
    - Execute the same season-seed sequence for each model.
    - Resolve complete league standings.
    - Return raw season, club-season, and match outputs.
    - Reuse the established Study 042 fingerprint definition.

    Non-responsibilities
    --------------------
    - Construct football models.
    - Load repositories.
    - Fit prediction models.
    - Decide which model is superior.
    - Compare against historical reality.
    - Write files or reports.
    """

    def __init__(
        self,
        config: FootballModelBenchmarkConfig,
    ) -> None:
        self.config = config

    def run(
        self,
        model_specs: Sequence[
            FootballModelBenchmarkSpec
        ],
    ) -> FootballModelBenchmarkResult:
        specs = tuple(
            model_specs
        )

        self._validate_model_specs(
            specs
        )

        fixtures = self._build_fixtures()

        season_rows: list[
            dict[str, object]
        ] = []

        club_rows: list[
            dict[str, object]
        ] = []

        match_rows: list[
            dict[str, object]
        ] = []

        for spec in specs:
            canonical_model = (
                CanonicalFootballModel(
                    model_id=spec.model_id,
                    football_model=(
                        spec.football_model
                    ),
                    runtime_names=(
                        spec.runtime_names
                    ),
                )
            )

            for season_number in range(
                1,
                self.config.simulation_count + 1,
            ):
                season_seed = (
                    self.config.base_seed
                    + season_number
                    - 1
                )

                (
                    season_row,
                    season_club_rows,
                    season_match_rows,
                ) = self._simulate_season(
                    spec=spec,
                    canonical_model=(
                        canonical_model
                    ),
                    fixtures=fixtures,
                    season_number=(
                        season_number
                    ),
                    season_seed=season_seed,
                )

                season_rows.append(
                    season_row
                )

                club_rows.extend(
                    season_club_rows
                )

                match_rows.extend(
                    season_match_rows
                )

        result = FootballModelBenchmarkResult(
            season_statistics=pd.DataFrame(
                season_rows
            ),
            club_season_statistics=pd.DataFrame(
                club_rows
            ),
            match_statistics=pd.DataFrame(
                match_rows
            ),
        )

        result.validate()

        self._validate_result(
            result=result,
            model_specs=specs,
            fixture_count=len(fixtures),
        )

        return result

    def _validate_model_specs(
        self,
        model_specs: tuple[
            FootballModelBenchmarkSpec,
            ...,
        ],
    ) -> None:
        if not model_specs:
            raise ValueError(
                "At least one football model is required."
            )

        model_ids = [
            spec.model_id
            for spec in model_specs
        ]

        if len(model_ids) != len(
            set(model_ids)
        ):
            raise ValueError(
                "Benchmark model IDs must be unique."
            )

        canonical_population = set(
            self.config.canonical_clubs
        )

        for spec in model_specs:
            alias_population = set(
                spec.runtime_names
            )

            if alias_population != (
                canonical_population
            ):
                missing = sorted(
                    canonical_population
                    - alias_population
                )

                extra = sorted(
                    alias_population
                    - canonical_population
                )

                raise ValueError(
                    "Runtime-name mapping does not match the "
                    "canonical club population for model "
                    f"{spec.model_id!r}. "
                    f"Missing={missing}, extra={extra}."
                )

    def _build_fixtures(
        self,
    ):
        generator = (
            RoundRobinFixtureGenerator()
        )

        fixtures = generator.generate(
            participants=list(
                self.config.canonical_clubs
            ),
            double_round_robin=(
                self.config
                .double_round_robin
            ),
            competition_id=(
                self.config.benchmark_id
            ),
            start_date=(
                self.config
                .season_start_date
            ),
            days_between_matchdays=(
                self.config
                .days_between_matchdays
            ),
        )

        club_count = len(
            self.config.canonical_clubs
        )

        if self.config.double_round_robin:
            expected_fixture_count = (
                club_count
                * (club_count - 1)
            )
        else:
            expected_fixture_count = (
                club_count
                * (club_count - 1)
                // 2
            )

        if len(fixtures) != (
            expected_fixture_count
        ):
            raise AssertionError(
                "Unexpected benchmark fixture count: "
                f"{len(fixtures)} vs "
                f"{expected_fixture_count}."
            )

        if any(
            fixture.match_date is None
            for fixture in fixtures
        ):
            raise AssertionError(
                "Benchmark fixtures must all have dates."
            )

        return fixtures

    def _build_competition_definition(
        self,
    ) -> CompetitionDefinition:
        club_count = len(
            self.config.canonical_clubs
        )

        matches_per_team = (
            2 * (club_count - 1)
            if self.config.double_round_robin
            else club_count - 1
        )

        return CompetitionDefinition(
            name=(
                self.config.benchmark_name
            ),
            competition_type=(
                "football_model_benchmark"
            ),
            region=None,
            governing_body=None,
            participant_count=club_count,
            stages=[
                StageDefinition(
                    name="League Season",
                    stage_type="league",
                    participant_count=(
                        club_count
                    ),
                    competition_format=(
                        "double_round_robin"
                        if self.config
                        .double_round_robin
                        else "single_round_robin"
                    ),
                    metadata={
                        "matches_per_team":
                            matches_per_team,
                        "points_system":
                            "3-1-0",
                        "benchmark_id":
                            self.config
                            .benchmark_id,
                    },
                )
            ],
            metadata={
                "benchmark_id":
                    self.config.benchmark_id,
                "simulation_count":
                    self.config
                    .simulation_count,
            },
        )

    def _simulate_season(
        self,
        *,
        spec: FootballModelBenchmarkSpec,
        canonical_model: CanonicalFootballModel,
        fixtures,
        season_number: int,
        season_seed: int,
    ) -> tuple[
        dict[str, object],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        random.seed(
            season_seed
        )

        np.random.seed(
            season_seed
        )

        definition = (
            self._build_competition_definition()
        )

        competition = (
            CompetitionBuilder().build(
                definition=definition,
                participants=list(
                    self.config
                    .canonical_clubs
                ),
            )
        )

        league_stage = (
            competition.stages[0]
        )

        matches = (
            LeagueMatchSimulator()
            .simulate_fixtures(
                fixtures=fixtures,
                football_model=(
                    canonical_model
                ),
                stage_name=(
                    league_stage.name
                ),
                competition_name=(
                    competition.name
                ),
            )
        )

        if len(matches) != len(
            fixtures
        ):
            raise AssertionError(
                "Season simulation did not preserve the "
                "fixture population."
            )

        league_stage.matches = matches

        competition_result = (
            CompetitionEngine().resolve(
                competition
            )
        )

        if not (
            competition_result.stage_results
        ):
            raise AssertionError(
                "Competition produced no stage results."
            )

        stage_result = (
            competition_result
            .stage_results[0]
        )

        if stage_result.standings is None:
            raise AssertionError(
                "League season produced no standings."
            )

        standings = (
            stage_result.standings
            .as_rows()
        )

        match_dataframe = (
            self._build_match_dataframe(
                spec=spec,
                matches=matches,
                season_number=season_number,
                season_seed=season_seed,
            )
        )

        fingerprint = (
            build_league_fingerprint(
                dataframe=match_dataframe,
                competition_key=(
                    "premier_league"
                ),
                season_start_year=(
                    self.config
                    .season_start_date
                    .year
                ),
            )
        )

        season_row = (
            self._build_season_row(
                spec=spec,
                standings=standings,
                fingerprint=fingerprint,
                season_number=season_number,
                season_seed=season_seed,
            )
        )

        club_rows = (
            self._build_club_rows(
                spec=spec,
                standings=standings,
                season_number=season_number,
                season_seed=season_seed,
            )
        )

        match_rows = (
            match_dataframe
            .to_dict(
                orient="records"
            )
        )

        return (
            season_row,
            club_rows,
            match_rows,
        )

    def _build_match_dataframe(
        self,
        *,
        spec: FootballModelBenchmarkSpec,
        matches,
        season_number: int,
        season_seed: int,
    ) -> pd.DataFrame:
        rows: list[
            dict[str, object]
        ] = []

        for match in matches:
            metadata = (
                match.metadata or {}
            )

            home_score = int(
                match.goals_team1
            )

            away_score = int(
                match.goals_team2
            )

            if home_score > away_score:
                outcome = "home_win"

            elif home_score < away_score:
                outcome = "away_win"

            else:
                outcome = "draw"

            rows.append(
                {
                    "model_id":
                        spec.model_id,
                    "model_name":
                        spec.display_name,
                    "season_number":
                        season_number,
                    "season_seed":
                        season_seed,
                    "competition_key":
                        "premier_league",
                    "season_start_year":
                        self.config
                        .season_start_date
                        .year,
                    "event_id":
                        (
                            f"{spec.model_id}_"
                            f"s{season_number}_"
                            f"{match.match_id}"
                        ),
                    "match_id":
                        match.match_id,
                    "matchday":
                        metadata.get(
                            "matchday"
                        ),
                    "match_date":
                        metadata.get(
                            "match_date"
                        ),
                    "home_team":
                        match.team1,
                    "away_team":
                        match.team2,
                    "home_score":
                        home_score,
                    "away_score":
                        away_score,
                    "goal_difference":
                        (
                            home_score
                            - away_score
                        ),
                    "total_goals":
                        (
                            home_score
                            + away_score
                        ),
                    "outcome":
                        outcome,
                    "completed":
                        True,
                }
            )

        dataframe = pd.DataFrame(
            rows
        )

        required_columns = {
            "competition_key",
            "season_start_year",
            "event_id",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "goal_difference",
            "total_goals",
            "outcome",
            "completed",
        }

        missing = (
            required_columns
            - set(dataframe.columns)
        )

        if missing:
            raise AssertionError(
                "Benchmark match DataFrame is missing "
                f"fingerprint columns: {sorted(missing)}"
            )

        return dataframe

    def _build_season_row(
        self,
        *,
        spec: FootballModelBenchmarkSpec,
        standings: list[
            dict[str, object]
        ],
        fingerprint: dict[
            str,
            object
        ],
        season_number: int,
        season_seed: int,
    ) -> dict[str, object]:
        ordered = sorted(
            standings,
            key=lambda row: int(
                row["rank"]
            ),
        )

        champion = ordered[0]
        bottom = ordered[-1]

        points = [
            int(row["points"])
            for row in ordered
        ]

        goal_differences = [
            int(
                row["goal_difference"]
            )
            for row in ordered
        ]

        return {
            "model_id":
                spec.model_id,
            "model_name":
                spec.display_name,
            "season_number":
                season_number,
            "season_seed":
                season_seed,
            "champion":
                champion["team"],
            "champion_points":
                int(
                    champion["points"]
                ),
            "bottom_club":
                bottom["team"],
            "bottom_points":
                int(
                    bottom["points"]
                ),
            "points_spread":
                max(points)
                - min(points),
            "goal_difference_spread":
                max(goal_differences)
                - min(goal_differences),
            **{
                key: value
                for key, value
                in fingerprint.items()
                if key not in {
                    "competition_key",
                    "competition_name",
                    "season_start_year",
                }
            },
        }

    def _build_club_rows(
        self,
        *,
        spec: FootballModelBenchmarkSpec,
        standings: list[
            dict[str, object]
        ],
        season_number: int,
        season_seed: int,
    ) -> list[dict[str, object]]:
        rows: list[
            dict[str, object]
        ] = []

        for standing in standings:
            rank = int(
                standing["rank"]
            )

            rows.append(
                {
                    "model_id":
                        spec.model_id,
                    "model_name":
                        spec.display_name,
                    "season_number":
                        season_number,
                    "season_seed":
                        season_seed,
                    "club":
                        standing["team"],
                    "position":
                        rank,
                    "matches_played":
                        int(
                            standing[
                                "matches_played"
                            ]
                        ),
                    "wins":
                        int(
                            standing["wins"]
                        ),
                    "draws":
                        int(
                            standing["draws"]
                        ),
                    "losses":
                        int(
                            standing["losses"]
                        ),
                    "goals_for":
                        int(
                            standing[
                                "goals_for"
                            ]
                        ),
                    "goals_against":
                        int(
                            standing[
                                "goals_against"
                            ]
                        ),
                    "goal_difference":
                        int(
                            standing[
                                "goal_difference"
                            ]
                        ),
                    "points":
                        int(
                            standing["points"]
                        ),
                    "is_champion":
                        rank == 1,
                    "is_top_four":
                        rank <= 4,
                    "is_bottom_three":
                        rank > (
                            len(standings) - 3
                        ),
                }
            )

        return rows

    def _validate_result(
        self,
        *,
        result: FootballModelBenchmarkResult,
        model_specs: tuple[
            FootballModelBenchmarkSpec,
            ...,
        ],
        fixture_count: int,
    ) -> None:
        model_count = len(
            model_specs
        )

        season_count = (
            self.config.simulation_count
        )

        club_count = len(
            self.config.canonical_clubs
        )

        expected_season_rows = (
            model_count
            * season_count
        )

        expected_club_rows = (
            model_count
            * season_count
            * club_count
        )

        expected_match_rows = (
            model_count
            * season_count
            * fixture_count
        )

        if len(
            result.season_statistics
        ) != expected_season_rows:
            raise AssertionError(
                "Unexpected season-statistics row count."
            )

        if len(
            result.club_season_statistics
        ) != expected_club_rows:
            raise AssertionError(
                "Unexpected club-season row count."
            )

        if len(
            result.match_statistics
        ) != expected_match_rows:
            raise AssertionError(
                "Unexpected match-statistics row count."
            )

        expected_matches_per_club = (
            2 * (club_count - 1)
            if self.config.double_round_robin
            else club_count - 1
        )

        if not result.club_season_statistics[
            "matches_played"
        ].eq(
            expected_matches_per_club
        ).all():
            raise AssertionError(
                "At least one club played an unexpected "
                "number of matches."
            )

        grouped_club_counts = (
            result.club_season_statistics
            .groupby(
                [
                    "model_id",
                    "season_number",
                ]
            )[
                "club"
            ]
            .nunique()
        )

        if not grouped_club_counts.eq(
            club_count
        ).all():
            raise AssertionError(
                "At least one model-season does not contain "
                "the complete club population."
            )

        champion_counts = (
            result.club_season_statistics
            .groupby(
                [
                    "model_id",
                    "season_number",
                ]
            )[
                "is_champion"
            ]
            .sum()
        )

        if not champion_counts.eq(
            1
        ).all():
            raise AssertionError(
                "At least one model-season does not contain "
                "exactly one champion."
            )

        goals = result.match_statistics[
            [
                "home_score",
                "away_score",
                "total_goals",
            ]
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            goals
        ).all():
            raise AssertionError(
                "Benchmark match outputs contain "
                "non-finite goals."
            )

        if (
            goals < 0
        ).any():
            raise AssertionError(
                "Benchmark match outputs contain "
                "negative goals."
            )
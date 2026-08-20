#domestic_league_membership_bootstrap

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


MEMBERSHIP_COLUMNS = [
    "competition",
    "competition_type",
    "competition_id",
    "season_id",
    "season_year",
    "player_id",
    "player",
    "player_slug",
    "team_id",
    "team",
    "team_slug",
]


@dataclass(frozen=True)
class DomesticLeagueMembershipBootstrapConfig:
    target_competition: str
    target_competition_id: int
    target_season_id: int
    target_season_year: str

    previous_top_flight_competition: str
    previous_top_flight_season_year: str

    promoted_source_competition: str
    promoted_source_season_year: str

    target_participants_path: Path
    previous_top_flight_memberships_path: Path
    promoted_source_memberships_path: Path

    output_path: Path


@dataclass(frozen=True)
class DomesticLeagueMembershipBootstrapResult:
    target_club_count: int
    returning_club_count: int
    promoted_club_count: int
    departed_club_count: int
    membership_rows: int
    unique_players: int

    returning_clubs: tuple[str, ...]
    promoted_clubs: tuple[str, ...]
    departed_clubs: tuple[str, ...]


def _read_memberships(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Membership file does not exist: {path}"
        )

    frame = pd.read_csv(
        path,
        dtype={
            "player_id": str,
            "team_id": str,
            "season_year": str,
        },
        low_memory=False,
    )

    missing = (
        set(MEMBERSHIP_COLUMNS)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{path} is missing membership columns: "
            f"{sorted(missing)}"
        )

    return frame


def build_domestic_league_membership_candidate(
    config: DomesticLeagueMembershipBootstrapConfig,
) -> DomesticLeagueMembershipBootstrapResult:
    participants = pd.read_csv(
        config.target_participants_path,
        dtype={"team_id": str},
    )

    required_participant_columns = {
        "team_id",
        "team",
    }

    missing_participant_columns = (
        required_participant_columns
        - set(participants.columns)
    )

    if missing_participant_columns:
        raise ValueError(
            "Target participant file is missing columns: "
            f"{sorted(missing_participant_columns)}"
        )

    if participants.empty:
        raise ValueError(
            "Target participant file is empty."
        )

    if participants["team_id"].duplicated().any():
        raise ValueError(
            "Target participant file contains duplicate "
            "team IDs."
        )

    target_teams = set(
        participants["team"]
        .dropna()
        .astype(str)
    )

    previous = _read_memberships(
        config.previous_top_flight_memberships_path
    )

    promoted_source = _read_memberships(
        config.promoted_source_memberships_path
    )

    previous = previous.loc[
        previous["competition"].eq(
            config.previous_top_flight_competition
        )
        & previous["season_year"].astype(str).eq(
            config.previous_top_flight_season_year
        )
    ].copy()

    promoted_source = promoted_source.loc[
        promoted_source["competition"].eq(
            config.promoted_source_competition
        )
        & promoted_source["season_year"].astype(str).eq(
            config.promoted_source_season_year
        )
    ].copy()

    if previous.empty:
        raise ValueError(
            "No previous top-flight memberships matched "
            f"{config.previous_top_flight_competition} "
            f"{config.previous_top_flight_season_year}."
        )

    if promoted_source.empty:
        raise ValueError(
            "No promoted-source memberships matched "
            f"{config.promoted_source_competition} "
            f"{config.promoted_source_season_year}."
        )

    previous_teams = set(
        previous["team"]
        .dropna()
        .astype(str)
    )

    promoted_source_teams = set(
        promoted_source["team"]
        .dropna()
        .astype(str)
    )

    returning_clubs = (
        target_teams
        & previous_teams
    )

    promoted_clubs = (
        target_teams
        - previous_teams
    )

    departed_clubs = (
        previous_teams
        - target_teams
    )

    missing_promoted_clubs = (
        promoted_clubs
        - promoted_source_teams
    )

    if missing_promoted_clubs:
        raise ValueError(
            "Promoted-source membership does not cover "
            "all newly entering target clubs: "
            f"{sorted(missing_promoted_clubs)}"
        )

    returning_rows = previous.loc[
        previous["team"].isin(
            returning_clubs
        )
    ].copy()

    promoted_rows = promoted_source.loc[
        promoted_source["team"].isin(
            promoted_clubs
        )
    ].copy()

    candidate = pd.concat(
        [
            returning_rows,
            promoted_rows,
        ],
        ignore_index=True,
        sort=False,
    )

    #
    # Resolve target team IDs from the authoritative
    # target-season participant artifact.
    #
    target_team_ids = (
        participants[
            [
                "team",
                "team_id",
            ]
        ]
        .drop_duplicates()
        .set_index("team")["team_id"]
        .astype(str)
        .to_dict()
    )

    candidate["team_id"] = (
        candidate["team"]
        .map(target_team_ids)
    )

    if candidate["team_id"].isna().any():
        missing_teams = sorted(
            candidate.loc[
                candidate["team_id"].isna(),
                "team",
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        raise ValueError(
            "Could not resolve target-season team IDs for: "
            f"{missing_teams}"
        )

    #
    # Relabel the carry-forward membership into the target
    # competition-season while preserving player identity.
    #
    candidate["competition"] = (
        config.target_competition
    )

    candidate["competition_type"] = (
        "club_league"
    )

    candidate["competition_id"] = (
        config.target_competition_id
    )

    candidate["season_id"] = (
        config.target_season_id
    )

    candidate["season_year"] = (
        config.target_season_year
    )

    candidate = candidate[
        MEMBERSHIP_COLUMNS
    ].copy()

    candidate = candidate.drop_duplicates(
        subset=[
            "competition_id",
            "season_id",
            "player_id",
            "team_id",
        ],
        keep="last",
    )

    observed_teams = set(
        candidate["team"]
        .dropna()
        .astype(str)
    )

    if observed_teams != target_teams:
        missing = sorted(
            target_teams - observed_teams
        )

        unexpected = sorted(
            observed_teams - target_teams
        )

        raise AssertionError(
            "Target membership population mismatch. "
            f"Missing={missing}; "
            f"unexpected={unexpected}"
        )

    if candidate.empty:
        raise AssertionError(
            "Membership candidate is empty."
        )

    config.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate = candidate.sort_values(
        [
            "team",
            "player",
            "player_id",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    candidate.to_csv(
        config.output_path,
        index=False,
    )

    return DomesticLeagueMembershipBootstrapResult(
        target_club_count=len(target_teams),
        returning_club_count=len(
            returning_clubs
        ),
        promoted_club_count=len(
            promoted_clubs
        ),
        departed_club_count=len(
            departed_clubs
        ),
        membership_rows=len(candidate),
        unique_players=(
            candidate["player_id"].nunique()
        ),
        returning_clubs=tuple(
            sorted(returning_clubs)
        ),
        promoted_clubs=tuple(
            sorted(promoted_clubs)
        ),
        departed_clubs=tuple(
            sorted(departed_clubs)
        ),
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a domestic-league target-season membership "
            "candidate from previous top-flight and promoted-club "
            "membership evidence."
        )
    )

    parser.add_argument(
        "--target-competition",
        required=True,
    )

    parser.add_argument(
        "--target-competition-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--target-season-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--target-season-year",
        required=True,
    )

    parser.add_argument(
        "--previous-top-flight-competition",
        required=True,
    )

    parser.add_argument(
        "--previous-top-flight-season-year",
        required=True,
    )

    parser.add_argument(
        "--promoted-source-competition",
        required=True,
    )

    parser.add_argument(
        "--promoted-source-season-year",
        required=True,
    )

    parser.add_argument(
        "--target-participants-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--previous-top-flight-memberships-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--promoted-source-memberships-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--resolved-output-path",
        type=Path,
        default=None,
        help=(
            "Optional resolved-membership output path. "
            "When the candidate contains no duplicate player IDs, "
            "the validated candidate is promoted unchanged to this path."
        ),
    )

    return parser.parse_args()

def promote_unambiguous_membership(
    *,
    candidate_path: Path,
    resolved_output_path: Path,
) -> int:
    frame = pd.read_csv(
        candidate_path,
        low_memory=False,
    )

    duplicate_players = int(
        frame["player_id"]
        .astype(str)
        .duplicated()
        .sum()
    )

    if duplicate_players:
        raise ValueError(
            "Membership candidate still contains "
            f"{duplicate_players} duplicate player IDs; "
            "profile-based ambiguity resolution is required."
        )

    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        resolved_output_path,
        index=False,
    )

    return len(frame)

def main() -> None:
    arguments = parse_arguments()

    config = DomesticLeagueMembershipBootstrapConfig(
        target_competition=arguments.target_competition,
        target_competition_id=arguments.target_competition_id,
        target_season_id=arguments.target_season_id,
        target_season_year=arguments.target_season_year,

        previous_top_flight_competition=(
            arguments.previous_top_flight_competition
        ),
        previous_top_flight_season_year=(
            arguments.previous_top_flight_season_year
        ),

        promoted_source_competition=(
            arguments.promoted_source_competition
        ),
        promoted_source_season_year=(
            arguments.promoted_source_season_year
        ),

        target_participants_path=(
            arguments.target_participants_path
        ),
        previous_top_flight_memberships_path=(
            arguments.previous_top_flight_memberships_path
        ),
        promoted_source_memberships_path=(
            arguments.promoted_source_memberships_path
        ),

        output_path=arguments.output_path,
    )

    result = build_domestic_league_membership_candidate(
        config
    )

    print()
    print("Domestic League Membership Bootstrap")
    print("=" * 72)
    print(
        f"Target clubs: {result.target_club_count}"
    )
    print(
        f"Returning clubs: {result.returning_club_count}"
    )
    print(
        f"Promoted clubs: {result.promoted_club_count}"
    )
    print(
        f"Departed clubs: {result.departed_club_count}"
    )
    print(
        f"Membership rows: {result.membership_rows}"
    )
    print(
        f"Unique players: {result.unique_players}"
    )
    print(
        f"Output: {arguments.output_path}"
    )

    if arguments.resolved_output_path is not None:
        resolved_rows = promote_unambiguous_membership(
            candidate_path=arguments.output_path,
            resolved_output_path=(
                arguments.resolved_output_path
            ),
        )

        print(
            "Resolution: no ambiguous player IDs detected; "
            "candidate promoted unchanged."
        )
        print(
            f"Resolved rows: {resolved_rows}"
        )
        print(
            "Resolved output: "
            f"{arguments.resolved_output_path}"
        )

if __name__ == "__main__":
    main()
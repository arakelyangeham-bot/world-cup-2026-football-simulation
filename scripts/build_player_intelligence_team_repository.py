from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from functools import partial

from research.player_intelligence.starting_xi_builder import (
    StartingXIBuilder,
)
from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
    build_team_representation_from_squad,
    build_team_representation_from_starting_xi,
    build_team_representation_from_starting_xi_contributions,
)

from research.player_intelligence.player_evidence_repository import (
    PlayerEvidenceRepository,
)
from research.player_intelligence.player_repository import (
    PlayerRepository,
)
from research.player_intelligence.player_schema import (
    Squad,
)
from research.player_intelligence.roster_builder import (
    RosterBuilder,
)
from research.player_intelligence.team_representation_builder import (
    TeamRepresentation,
    build_team_representation_from_squad,
)
from research.player_intelligence.team_repository_builder import (
    project_representation_to_repository_entry,
)
from shared.national_team_priors import (
    load_fifa_points,
)
from shared.team_name_normalizer import (
    normalize_team_name,
)


RepresentationBuilder = Callable[
    [Squad],
    TeamRepresentation,
]


DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "player_intelligence"
)

DEFAULT_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIRECTORY
    / "player_intelligence_team_repository.csv"
)

FORMATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "formation_manifest.csv"
)

DEFAULT_FORMATION = "4-3-3"

DEFAULT_REPRESENTATION_POLICY = "full_squad_legacy"

REPRESENTATION_POLICY_NAMES = (
    "full_squad_legacy",
    "expected_xi_legacy",
    "expected_xi_contribution_zero",
)

REPRESENTATION_BUILDERS: dict[
    str,
    RepresentationBuilder,
] = {
    "full_squad_legacy":
        build_team_representation_from_squad,
}


REPOSITORY_COLUMNS = (
    "nation",
    "att_composite",
    "mid_composite",
    "def_composite",
    "gk_composite",
    "poisson_attack_adj",
    "poisson_defense_adj",
    "representation_type",
    "aggregation_profile",
    "player_count",
    "available_player_count",
    "squad_quality",
    "evidence_score",
    "attack_depth",
    "midfield_depth",
    "defense_depth",
    "fifa_points",
)



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a player-intelligence national-team "
            "repository through an injected representation policy."
        )
    )

    parser.add_argument(
        "--representation-policy",
        default=DEFAULT_REPRESENTATION_POLICY,
        choices=REPRESENTATION_POLICY_NAMES,
        help=(
            "Registered policy used to convert each Squad "
            "into a TeamRepresentation."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "CSV path receiving the generated repository."
        ),
    )

    return parser.parse_args()

def build_expected_xi_legacy_representation(
    squad: Squad,
    *,
    formation_df: pd.DataFrame,
    formation: str = DEFAULT_FORMATION,
) -> TeamRepresentation:
    lineup_builder = StartingXIBuilder(
        formation=formation
    )

    starting_xi = (
        lineup_builder.build_for_squad(
            squad=squad,
            formation_df=formation_df,
        )
    )

    return build_team_representation_from_starting_xi(
        starting_xi
    )

def load_formation(
    *,
    formation_path: Path = FORMATION_PATH,
    formation: str = DEFAULT_FORMATION,
) -> pd.DataFrame:
    manifest = pd.read_csv(
        formation_path
    )

    required_columns = {
        "slot",
        "role",
    }

    missing = (
        required_columns
        - set(manifest.columns)
    )

    if missing:
        raise ValueError(
            "Formation manifest is missing required "
            f"columns: {sorted(missing)}"
        )

    if "formation" in manifest.columns:
        selected = manifest.loc[
            manifest["formation"]
            .astype(str)
            .eq(formation)
        ].copy()
    else:
        selected = manifest.copy()

    if selected.empty:
        raise ValueError(
            f"No formation rows found for {formation!r}."
        )

    if selected["slot"].duplicated().any():
        raise ValueError(
            f"Formation {formation!r} contains duplicate slots."
        )

    return selected.reset_index(
        drop=True
    )

def build_expected_xi_contribution_representation(
    squad: Squad,
    *,
    formation_df: pd.DataFrame,
    formation: str = DEFAULT_FORMATION,
) -> TeamRepresentation:
    lineup_builder = StartingXIBuilder(
        formation=formation
    )

    starting_xi = (
        lineup_builder.build_for_squad(
            squad=squad,
            formation_df=formation_df,
        )
    )

    return (
        build_team_representation_from_starting_xi_contributions(
            starting_xi
        )
    )

def resolve_representation_builder(
    policy: str,
) -> RepresentationBuilder:
    if policy == "full_squad_legacy":
        return build_team_representation_from_squad

    formation_df = load_formation()

    if policy == "expected_xi_legacy":
        return partial(
            build_expected_xi_legacy_representation,
            formation_df=formation_df,
        )

    if policy == "expected_xi_contribution_zero":
        return partial(
            build_expected_xi_contribution_representation,
            formation_df=formation_df,
        )

    raise ValueError(
        "Unknown representation policy: "
        f"{policy!r}. Available policies: "
        "['expected_xi_contribution_zero', "
        "'expected_xi_legacy', "
        "'full_squad_legacy']"
    )


def validate_representation(
    *,
    source_team: str,
    representation: TeamRepresentation,
) -> None:
    canonical_source = normalize_team_name(
        source_team
    )

    canonical_representation = (
        normalize_team_name(
            representation.national_team
        )
    )

    if canonical_representation != canonical_source:
        raise ValueError(
            "Representation national team does not match "
            "the source squad. "
            f"Source={canonical_source!r}, "
            f"representation="
            f"{canonical_representation!r}."
        )

    numeric_values = (
        representation.attack,
        representation.midfield,
        representation.defense,
        representation.goalkeeper,
        representation.attack_depth,
        representation.midfield_depth,
        representation.defense_depth,
        representation.squad_quality,
        representation.evidence_score,
    )

    if not all(
        pd.notna(value)
        for value in numeric_values
    ):
        raise ValueError(
            f"{canonical_source}: representation contains "
            "missing numeric values."
        )

    if representation.player_count < 0:
        raise ValueError(
            f"{canonical_source}: player count cannot "
            "be negative."
        )

    if representation.available_player_count < 0:
        raise ValueError(
            f"{canonical_source}: available-player count "
            "cannot be negative."
        )

    if (
        representation.available_player_count
        > representation.player_count
    ):
        raise ValueError(
            f"{canonical_source}: available-player count "
            "exceeds player count."
        )


def representation_to_repository_row(
    *,
    canonical_team: str,
    representation: TeamRepresentation,
    fifa_points: float,
) -> dict[str, object]:
    entry = (
        project_representation_to_repository_entry(
            representation=representation,
            fifa_points=fifa_points,
        )
    )

    return {
        "nation": canonical_team,

        "att_composite": entry.attack,
        "mid_composite": entry.midfield,
        "def_composite": entry.defense,
        "gk_composite": entry.gk,

        "poisson_attack_adj":
            entry.poisson_attack,
        "poisson_defense_adj":
            entry.poisson_defense,

        "representation_type":
            entry.representation_type,
        "aggregation_profile":
            entry.aggregation_profile,
        "player_count":
            entry.player_count,
        "available_player_count":
            entry.available_player_count,

        "squad_quality":
            representation.squad_quality,
        "evidence_score":
            representation.evidence_score,
        "attack_depth":
            representation.attack_depth,
        "midfield_depth":
            representation.midfield_depth,
        "defense_depth":
            representation.defense_depth,

        "fifa_points": float(
            fifa_points
        ),
    }


def build_repository_dataframe(
    *,
    roster_builder: RosterBuilder,
    representation_builder:
        RepresentationBuilder,
    fifa_lookup: dict[str, float],
    included_teams: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for team in roster_builder.list_teams():
        canonical_team = normalize_team_name(
            team
        )

        if (
            included_teams is not None
            and canonical_team not in included_teams
        ):
            continue

        squad = roster_builder.get_squad(
            team
        )

        if not squad.players:
            continue

        fifa_points = fifa_lookup.get(
            canonical_team
        )

        if fifa_points is None or pd.isna(
            fifa_points
        ):
            print(
                f"Skipping {canonical_team}: "
                "missing FIFA points."
            )
            continue

        representation = (
            representation_builder(
                squad
            )
        )

        validate_representation(
            source_team=canonical_team,
            representation=representation,
        )

        rows.append(
            representation_to_repository_row(
                canonical_team=canonical_team,
                representation=representation,
                fifa_points=float(
                    fifa_points
                ),
            )
        )

    repository = pd.DataFrame(
        rows,
        columns=REPOSITORY_COLUMNS,
    )

    if repository.empty:
        raise ValueError(
            "Repository generation produced no teams."
        )

    if repository[
        "nation"
    ].duplicated().any():
        duplicates = (
            repository.loc[
                repository[
                    "nation"
                ].duplicated(
                    keep=False
                ),
                "nation",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Repository contains duplicate teams: "
            f"{duplicates[:20]}"
        )

    return (
        repository
        .sort_values(
            "nation"
        )
        .reset_index(drop=True)
    )


def create_default_roster_builder() -> (
    RosterBuilder
):
    evidence_repository = (
        PlayerEvidenceRepository()
    )

    player_repository = PlayerRepository(
        evidence_repository=(
            evidence_repository
        ),
    )

    return RosterBuilder(
        repository=player_repository
    )


def write_repository(
    repository: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    repository.to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    arguments = parse_arguments()

    representation_builder = (
        resolve_representation_builder(
            arguments.representation_policy
        )
    )

    roster_builder = (
        create_default_roster_builder()
    )

    fifa_lookup = load_fifa_points()

    repository = (
        build_repository_dataframe(
            roster_builder=roster_builder,
            representation_builder=(
                representation_builder
            ),
            fifa_lookup=fifa_lookup,
        )
    )

    write_repository(
        repository=repository,
        output_path=arguments.output_path,
    )

    print(
        "Player Intelligence Team Repository Build"
    )
    print("-" * 41)
    print(
        "Representation policy: "
        f"{arguments.representation_policy}"
    )
    print(
        f"Teams: {len(repository)}"
    )
    print(
        repository
        .head(20)
        .round(4)
        .to_string(index=False)
    )
    print()
    print(
        f"Wrote -> {arguments.output_path}"
    )


if __name__ == "__main__":
    main()
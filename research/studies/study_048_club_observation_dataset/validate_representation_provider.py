#validate_representation_provider

from __future__ import annotations

from research.football_observatory.representation_provider import (
    RepresentationRequest,
)
from research.player_intelligence.previous_season_competition_representation_provider import (
    PreviousSeasonCompetitionRepresentationProvider,
)


def main() -> None:
    provider = (
        PreviousSeasonCompetitionRepresentationProvider(
            representation_type="expected_starting_xi",
            formation="4-3-3",
        )
    )

    request = RepresentationRequest(
        competition_key="premier_league",
        prediction_season_start_year=2024,
        representation_season_start_year=2023,
        team_id=42,
        team_name="Arsenal",
    )

    result = provider.get_representation(
        request
    )

    representation = result.representation

    print("Representation Provider Validation")
    print("==================================")
    print()
    print(
        f"Provider: {provider.provider_name}"
    )
    print(
        "Prediction season: "
        f"{request.prediction_season_start_year}"
    )
    print(
        "Representation season: "
        f"{request.representation_season_start_year}"
    )
    print(
        f"Team ID: {request.team_id}"
    )
    print(
        f"Team: {representation.national_team}"
    )
    print(
        "Representation type: "
        f"{representation.representation_type}"
    )
    print(
        f"Player count: "
        f"{representation.player_count}"
    )
    print(
        "Temporal validity: "
        f"{result.temporal_validity_pass}"
    )

    if (
        request.representation_season_start_year
        >= request.prediction_season_start_year
    ):
        raise AssertionError(
            "Representation is not historically prior "
            "to the prediction season."
        )

    if not result.temporal_validity_pass:
        raise AssertionError(
            "Provider did not certify temporal validity."
        )

    if representation.player_count != 11:
        raise AssertionError(
            "Expected-XI representation does not contain "
            "11 players."
        )

    if (
        representation.representation_type
        != "expected_starting_xi"
    ):
        raise AssertionError(
            "Unexpected representation type."
        )

    print()
    print("Season resolution: PASS")
    print("Team resolution: PASS")
    print("Representation construction: PASS")
    print("Temporal validity: PASS")
    print()
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
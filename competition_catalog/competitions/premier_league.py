#premier_league.py

from competition_catalog import CompetitionDefinition, StageDefinition


PREMIER_LEAGUE = CompetitionDefinition(
    name="Premier League",
    competition_type="domestic_league",
    region="England",
    governing_body="The Football Association",
    participant_count=20,
    stages=[
        StageDefinition(
            name="League Season",
            stage_type="league",
            participant_count=20,
            competition_format="double_round_robin",
            metadata={
                "matches_per_team": 38,
                "points_system": "3-1-0",
            },
        )
    ],
    metadata={
        "season": "generic",
    },
)
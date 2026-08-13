#world_cup

from competition_catalog import CompetitionDefinition, StageDefinition


WORLD_CUP_2026 = CompetitionDefinition(
    name="FIFA World Cup 2026",
    competition_type="international_tournament",
    region="World",
    governing_body="FIFA",
    participant_count=48,
    stages=[
        StageDefinition(
            name="Group Stage",
            stage_type="group_stage",
            participant_count=48,
            competition_format="12_groups_of_4",
            metadata={
                "group_count": 12,
                "teams_per_group": 4,
                "matches_per_team": 3,
                "total_matches": 72,
                "advancement_rule": (
                    "Top two teams from each group plus the "
                    "eight best third-placed teams"
                ),
                "qualifier_count": 32,
            },
        ),
        StageDefinition(
            name="Round of 32",
            stage_type="knockout",
            participant_count=32,
            competition_format="single_elimination",
            metadata={
                "match_count": 16,
            },
        ),
        StageDefinition(
            name="Round of 16",
            stage_type="knockout",
            participant_count=16,
            competition_format="single_elimination",
            metadata={
                "match_count": 8,
            },
        ),
        StageDefinition(
            name="Quarterfinals",
            stage_type="knockout",
            participant_count=8,
            competition_format="single_elimination",
            metadata={
                "match_count": 4,
            },
        ),
        StageDefinition(
            name="Semifinals",
            stage_type="knockout",
            participant_count=4,
            competition_format="single_elimination",
            metadata={
                "match_count": 2,
            },
        ),
        StageDefinition(
            name="Third-Place Playoff",
            stage_type="placement_match",
            participant_count=2,
            competition_format="single_match",
            metadata={
                "participants": "semifinal_losers",
                "match_count": 1,
            },
        ),
        StageDefinition(
            name="Final",
            stage_type="final",
            participant_count=2,
            competition_format="single_match",
            metadata={
                "participants": "semifinal_winners",
                "match_count": 1,
            },
        ),
    ],
    metadata={
        "edition": "2026",
        "hosts": [
            "Canada",
            "Mexico",
            "United States",
        ],
        "total_matches": 104,
        "group_stage_matches": 72,
        "knockout_matches": 32,
    },
)
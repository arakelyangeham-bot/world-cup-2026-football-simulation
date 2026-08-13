#validate_round_robin_fixture_generator

from __future__ import annotations

from collections import Counter, defaultdict

from fixture_generation import RoundRobinFixtureGenerator


PREMIER_LEAGUE_TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]


def main() -> None:
    generator = RoundRobinFixtureGenerator()

    fixtures = generator.generate(
        participants=PREMIER_LEAGUE_TEAMS,
        double_round_robin=True,
        competition_id="premier_league",
    )

    matchdays: dict[int, list] = defaultdict(list)

    for fixture in fixtures:
        matchdays[fixture.matchday].append(fixture)

    appearance_counts = Counter()
    home_counts = Counter()
    away_counts = Counter()
    directional_pair_counts = Counter()
    unordered_pair_counts = Counter()

    for fixture in fixtures:
        appearance_counts[fixture.home_team] += 1
        appearance_counts[fixture.away_team] += 1

        home_counts[fixture.home_team] += 1
        away_counts[fixture.away_team] += 1

        directional_pair_counts[
            (fixture.home_team, fixture.away_team)
        ] += 1

        unordered_pair_counts[
            frozenset((fixture.home_team, fixture.away_team))
        ] += 1

    expected_pair_count = (
        len(PREMIER_LEAGUE_TEAMS)
        * (len(PREMIER_LEAGUE_TEAMS) - 1)
        // 2
    )

    assert len(matchdays) == 38
    assert len(fixtures) == 380
    assert all(len(day_fixtures) == 10 for day_fixtures in matchdays.values())
    assert all(count == 38 for count in appearance_counts.values())
    assert all(count == 19 for count in home_counts.values())
    assert all(count == 19 for count in away_counts.values())
    assert len(unordered_pair_counts) == expected_pair_count
    assert all(count == 2 for count in unordered_pair_counts.values())

    for matchday, day_fixtures in matchdays.items():
        teams_on_matchday = [
            team
            for fixture in day_fixtures
            for team in fixture.teams
        ]

        assert len(teams_on_matchday) == len(set(teams_on_matchday)), (
            f"A team appears more than once on matchday {matchday}."
        )

    for team1 in PREMIER_LEAGUE_TEAMS:
        for team2 in PREMIER_LEAGUE_TEAMS:
            if team1 == team2:
                continue

            assert directional_pair_counts[(team1, team2)] == 1

    print("Double Round Robin Validation")
    print("=============================")
    print(f"Teams: {len(PREMIER_LEAGUE_TEAMS)}")
    print(f"Matchdays: {len(matchdays)}")
    print(f"Fixtures: {len(fixtures)}")
    print(f"Fixtures per matchday: {len(matchdays[1])}")
    print(f"Matches per team: {appearance_counts['Arsenal']}")
    print(f"Home matches per team: {home_counts['Arsenal']}")
    print(f"Away matches per team: {away_counts['Arsenal']}")
    print()
    print("All structural checks passed.")
    print()

    print("Matchday 1")
    print("----------")

    for fixture in matchdays[1]:
        print(f"{fixture.home_team} vs {fixture.away_team}")

    print()
    print("Matchday 20")
    print("-----------")

    for fixture in matchdays[20]:
        print(f"{fixture.home_team} vs {fixture.away_team}")


if __name__ == "__main__":
    main()
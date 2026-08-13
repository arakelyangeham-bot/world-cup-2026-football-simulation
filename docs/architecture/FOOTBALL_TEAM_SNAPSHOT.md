FOOTBALL_TEAM_SNAPSHOT

# Football Team Snapshot

## Status

Design specification — Version 1

## Purpose

A Football Team Snapshot is the canonical representation of one football
team at a specific pre-match point in time.

It provides a common boundary between:

1. upstream football-intelligence systems that construct team information;
2. downstream prediction and simulation systems that consume team information.

The snapshot may represent either:

- a national team; or
- a club team.

Downstream models should not need separate interfaces for clubs and national
teams unless the underlying football problem genuinely requires one.

---

# Core principle

A snapshot must contain only information that would have been available before
the match being predicted.

The snapshot represents:

> What the project believed about this team immediately before kickoff.

It must not contain information derived from:

- the match result;
- events occurring during the match;
- final season statistics unavailable at prediction time;
- future player performances;
- future lineup confirmations;
- retrospective ratings that include the target match.

---

# Snapshot identity

Every snapshot must include:

| Field | Type | Description |
|---|---|---|
| `snapshot_id` | string | Unique identifier for this team snapshot |
| `team_id` | string | Canonical team identifier |
| `team_name` | string | Canonical display name |
| `team_type` | string | `national_team` or `club` |
| `snapshot_timestamp` | datetime | Time at which the snapshot is valid |
| `competition_key` | string or null | Competition in which the match occurs |
| `season_start_year` | integer or null | Domestic season identity when applicable |

A recommended snapshot identifier is:

```text
<team_id>_<snapshot_timestamp>

or, when snapshots are match-specific:

<match_id>_<team_id>
Strength representation

Version 1 should expose four football dimensions:

Field	Description
attack_strength	Expected attacking quality
midfield_strength	Expected midfield quality
defense_strength	Expected defensive quality
goalkeeper_strength	Expected goalkeeper quality

It may also expose:

Field	Description
overall_strength	Aggregated total team quality
poisson_attack	Attack value used by expected-goals models
poisson_defense	Defense value used by expected-goals models
rating_prior	External team-strength prior

Version 1 does not prescribe one universal formula for these quantities.

Different snapshot builders may derive them from different evidence:

national-team repository values;
player ratings;
expected starting lineups;
dynamic club ratings;
external ranking systems.

The snapshot interface standardizes the output, not the upstream estimation
method.

Lineup information

A snapshot may include an expected lineup.

Recommended fields:

Field	Description
expected_formation	Expected tactical formation
expected_starting_xi	Ordered or structured list of expected starters
expected_substitutes	Expected bench or depth options
lineup_confidence	Confidence in the predicted starting lineup
availability_confidence	Confidence in injury, suspension, and availability information

Lineup information is optional in Version 1.

A snapshot must remain valid when only aggregate team-strength information is
available.

Environment information

Environmental variables are optional contextual features.

Examples include:

Field	Description
environment_pc1	First learned football-environment coordinate
environment_pc2	Second learned football-environment coordinate
home_advantage_context	Competition- or venue-specific home effect
competition_scoring_baseline	Expected scoring level for the competition

Environmental values must be leakage-safe.

A league-season coordinate calculated using all matches from the same season
must not be used to predict matches within that season unless it was generated
using only information available before each target match.

Version 1 therefore permits environmental values but does not require them.

Uncertainty

Strength estimates should eventually be accompanied by uncertainty.

Candidate fields include:

Field	Description
strength_confidence	Overall confidence in the snapshot
lineup_confidence	Confidence in the expected XI
data_coverage	Proportion of required source information available
missing_player_count	Number of expected players without valid ratings
snapshot_quality_flag	Human-readable quality category

These fields may initially be optional, but downstream models should not assume
that all snapshots are equally reliable.

Metadata and provenance

Every snapshot should retain enough provenance to explain how it was created.

Recommended metadata:

Field	Description
builder_name	Snapshot builder implementation
builder_version	Builder or schema version
source_dataset_ids	Input datasets used
feature_timestamp	Latest timestamp represented in the source features
created_at	Snapshot creation time
schema_version	Football Team Snapshot schema version

This allows predictions to be audited and reproduced.

Version 1 required fields

The smallest valid Version 1 snapshot contains:

snapshot_id
team_id
team_name
team_type
snapshot_timestamp
attack_strength
midfield_strength
defense_strength
goalkeeper_strength
poisson_attack
poisson_defense
rating_prior
schema_version

Everything else is optional.

Conceptual Python representation
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class FootballTeamSnapshot:
    snapshot_id: str
    team_id: str
    team_name: str
    team_type: str
    snapshot_timestamp: datetime

    attack_strength: float
    midfield_strength: float
    defense_strength: float
    goalkeeper_strength: float

    poisson_attack: float
    poisson_defense: float
    rating_prior: float

    schema_version: str = "1.0"

    competition_key: str | None = None
    season_start_year: int | None = None
    overall_strength: float | None = None

    expected_formation: str | None = None
    expected_starting_xi: tuple[str, ...] | None = None
    expected_substitutes: tuple[str, ...] | None = None

    lineup_confidence: float | None = None
    availability_confidence: float | None = None
    strength_confidence: float | None = None
    data_coverage: float | None = None

    environment_pc1: float | None = None
    environment_pc2: float | None = None

    metadata: dict[str, Any] | None = None

This representation is conceptual only.

No production class should be implemented until the existing national-team
repository has been mapped against this specification.

Builder responsibilities

A snapshot builder is responsible for:

collecting source information;
enforcing temporal validity;
estimating team-strength dimensions;
recording uncertainty and missing information;
returning a valid Football Team Snapshot.

Potential builders include:

NationalTeamSnapshotBuilder
ClubTeamSnapshotBuilder
HistoricalClubSnapshotBuilder
ExpectedLineupSnapshotBuilder

Each builder may use different source data while producing the same public
interface.

Consumer responsibilities

Downstream consumers may use snapshot fields but must not reconstruct upstream
player or repository logic.

Potential consumers include:

ExpectedGoalsModel
ScorelineFirstMatchEngine
MatchProbabilityModel
TournamentSimulator
ClubMatchPredictor

For example:

Home Football Team Snapshot
+
Away Football Team Snapshot
        ↓
Expected Goals Model
        ↓
lambda_home, lambda_away
Compatibility with the current project

The existing national-team repository already contains close equivalents of:

poisson_attack
poisson_defense
rating_prior

The first implementation step should therefore be an adapter:

National Team Repository Entry
        ↓
Football Team Snapshot Adapter
        ↓
Football Team Snapshot

The existing repository should not be rewritten immediately.

The adapter allows the new interface to be tested without destabilizing the
production simulator.

Club implementation path

A future club snapshot builder should follow:

Historical player information
        ↓
Player ratings and attribute scores
        ↓
Expected starting XI
        ↓
Dimension aggregation
        ↓
Dynamic club-strength prior
        ↓
Football Team Snapshot

League-environment features should be added only after a leakage-safe club
prediction baseline exists.

Non-goals for Version 1

Version 1 does not attempt to:

define the best club-strength algorithm;
force national and club data into one training population;
require lineup data for every team;
require environmental variables;
replace the current national-team repository;
redesign the scoreline-first match engine;
prescribe one universal feature scale.

Its purpose is only to establish a stable interface.

Acceptance criteria

The specification is ready for implementation when:

every required field has an unambiguous meaning;
no required field depends on future information;
an existing national-team repository entry can be adapted into the schema;
a future club builder can produce the same schema;
downstream expected-goals code can consume the schema without knowing which
builder produced it.
Immediate implementation milestone

The first implementation should be:

NationalTeamRepositorySnapshotAdapter

It should convert one existing national-team repository entry into a
Football Team Snapshot without changing the repository or production match
engine.

Only after that adapter is validated should the project implement:

ClubTeamSnapshotBuilder
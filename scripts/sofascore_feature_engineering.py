from pathlib import Path
import pandas as pd

import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]


#df = pd.read_csv(PROJECT_ROOT / "data" / "raw" /"sofascore" / "single_player_competition_stats.csv")

IN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical_player_evidence.csv"
)

OUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical_player_features.csv"
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-90 player features while preserving "
            "the input evidence grain."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=IN_FILE,
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUT_FILE,
    )

    return parser.parse_args()

arguments = parse_arguments()

input_file = arguments.input_file
output_file = arguments.output_file

df = pd.read_csv(
    input_file
)

input_rows = len(df)

TOTAL_STATS = [
    "goals",
    "assists",
    "bigChancesCreated",
    "bigChancesMissed",
    "goalsAssistsSum",
    "expectedAssists",
    "accuratePasses",
    "inaccuratePasses",
    "totalPasses",
    "accurateOwnHalfPasses",
    "accurateOppositionHalfPasses",
    "accurateFinalThirdPasses",
    "totalShots",
    "groundDuelsWon",
    "keyPasses",
    "successfulDribbles",
    "tackles",
    "interceptions",
    "yellowCards",
    "directRedCards",
    "redCards",
    "accurateCrosses",
    "shotsOnTarget",
    "shotsOffTarget",
    "aerialDuelsWon",
    "totalDuelsWon",
    "penaltiesTaken",
    "penaltyGoals",
    "penaltyWon",
    "penaltyConceded",
    "shotFromSetPiece",
    "freeKickGoal",
    "goalsFromInsideTheBox",
    "goalsFromOutsideTheBox",
    "shotsFromInsideTheBox",
    "shotsFromOutsideTheBox",
    "headedGoals",
    "leftFootGoals",
    "rightFootGoals",
    "accurateLongBalls",
    "clearances",
    "errorLeadToGoal",
    "errorLeadToShot",
    "dispossessed",
    "possessionLost",
    "possessionWonAttThird",
    "totalChippedPasses",
    "accurateChippedPasses",
    "touches",
    "wasFouled",
    "fouls",
    "hitWoodwork",
    "ownGoals",
    "dribbledPast",
    "offsides",
    "blockedShots",
    "passToAssist",
    "saves",
    "cleanSheet",
    "penaltyFaced",
    "penaltySave",
    "savedShotsFromInsideTheBox",
    "savedShotsFromOutsideTheBox",
    "goalsConcededInsideTheBox",
    "goalsConcededOutsideTheBox",
    "punches",
    "runsOut",
    "successfulRunsOut",
    "highClaims",
    "crossesNotClaimed",
    "totalAttemptAssist",
    "totalContest",
    "totalCross",
    "duelLost",
    "aerialLost",
    "attemptPenaltyMiss",
    "attemptPenaltyPost",
    "attemptPenaltyTarget",
    "totalLongBalls",
    "goalsConceded",
    "tacklesWon",
    "yellowRedCards",
    "savesCaught",
    "savesParried",
    "totalOwnHalfPasses",
    "totalOppositionHalfPasses",
    "goalKicks",
    "ballRecovery",
    "outfielderBlocks",
    "expectedGoals",
    "goalsPrevented",
]

PERCENTAGE_STATS = [
    "accuratePassesPercentage",
    "successfulDribblesPercentage",
    "groundDuelsWonPercentage",
    "accurateCrossesPercentage",
    "aerialDuelsWonPercentage",
    "totalDuelsWonPercentage",
    "goalConversionPercentage",
    "accurateLongBallsPercentage",
    "tacklesWonPercentage",
    "penaltyConversion",
    "setPieceConversion",
]

AVERAGE_STATS = [
    "rating",
    "ScoringFrequency"
]

METADATA = [
    "totalRating",
    "countRating",
    "appearances",
    "matchesStarted",
    "minutesPlayed",
    "totwAppearances"
]

minutes = (
    pd.to_numeric(
        df["minutesPlayed"],
        errors="coerce",
    )
    .replace(0, pd.NA)
)

for stat in TOTAL_STATS:
    #
    # Preserve a stable engineered schema across
    # competitions. Some Sofascore competitions do not
    # expose every production source statistic.
    #
    # Missing source evidence is represented explicitly
    # as NA rather than by omitting the feature column.
    #
    if stat not in df.columns:
        df[stat] = pd.NA

    df[stat] = pd.to_numeric(
        df[stat],
        errors="coerce",
    )

    df[f"{stat}_per90"] = (
        df[stat] * 90 / minutes
    )

output_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    output_file,
    index=False,
)
df.filter(regex="_per90$").describe()

if "position" in df.columns:
    print(df.groupby("position").mean(numeric_only=True))

print(
    f"Evidence rows preserved: {len(df):,}"
)

if len(df) != input_rows:
    raise AssertionError(
        "Feature engineering changed the evidence-row "
        "population."
    )

print(f"Wrote: {output_file}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
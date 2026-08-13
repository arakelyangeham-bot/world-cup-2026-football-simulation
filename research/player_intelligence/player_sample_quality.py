#player_sample_quality.py

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PlayerSampleQualityReport:
    total_players: int
    observed_count: int
    limited_minutes_count: int
    unobserved_or_zero_rating_count: int


def classify_sample_quality(row: pd.Series) -> str:
    minutes = pd.to_numeric(row.get("minutesPlayed"), errors="coerce")
    rating = pd.to_numeric(row.get("rating"), errors="coerce")
    appearances = pd.to_numeric(row.get("appearances"), errors="coerce")
    starts = pd.to_numeric(row.get("matchesStarted"), errors="coerce")

    minutes = 0 if pd.isna(minutes) else float(minutes)
    rating = 0 if pd.isna(rating) else float(rating)
    appearances = 0 if pd.isna(appearances) else float(appearances)
    starts = 0 if pd.isna(starts) else float(starts)

    if minutes <= 0 or rating <= 0:
        return "unobserved_or_zero_rating"

    if minutes < 90 or appearances < 2:
        return "limited_minutes"

    return "observed"


def add_sample_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sample_quality"] = df.apply(classify_sample_quality, axis=1)
    return df


def build_sample_quality_report(df: pd.DataFrame) -> PlayerSampleQualityReport:
    counts = df["sample_quality"].value_counts()

    return PlayerSampleQualityReport(
        total_players=len(df),
        observed_count=int(counts.get("observed", 0)),
        limited_minutes_count=int(counts.get("limited_minutes", 0)),
        unobserved_or_zero_rating_count=int(
            counts.get("unobserved_or_zero_rating", 0)
        ),
    )


def sample_quality_report_to_dataframe(
    report: PlayerSampleQualityReport,
) -> pd.DataFrame:
    return pd.DataFrame([report.__dict__])


def sample_quality_by_team(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["nation", "sample_quality"])
        .size()
        .reset_index(name="players")
    )

    pivot = grouped.pivot_table(
        index="nation",
        columns="sample_quality",
        values="players",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    for column in [
        "observed",
        "limited_minutes",
        "unobserved_or_zero_rating",
    ]:
        if column not in pivot.columns:
            pivot[column] = 0

    pivot["total_players"] = (
        pivot["observed"]
        + pivot["limited_minutes"]
        + pivot["unobserved_or_zero_rating"]
    )

    pivot["observed_share"] = (
        pivot["observed"] / pivot["total_players"]
    )

    return pivot.sort_values(
        ["observed_share", "observed"],
        ascending=[True, True],
    )
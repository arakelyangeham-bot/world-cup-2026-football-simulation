#player_evidence_repository.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PLAYER_STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sofascore"
    / "sofascore_player_stats.csv"
)


@dataclass(frozen=True)
class PlayerEvidenceEntry:
    player_id: str
    player_name: str | None

    competition: str | None
    competition_type: str | None
    competition_id: int | None
    season_id: int | None
    season_year: str | None

    team: str | None
    team_id: int | None

    minutes_played: float | None
    rating: float | None
    appearances: float | None
    matches_started: float | None


@dataclass(frozen=True)
class PlayerEvidenceHistory:
    player_id: str
    entries: tuple[PlayerEvidenceEntry, ...]

    @property
    def total_minutes(self) -> float:
        return sum(
            entry.minutes_played or 0.0
            for entry in self.entries
        )

    @property
    def competition_count(self) -> int:
        return len(
            {
                entry.competition_id
                for entry in self.entries
                if entry.competition_id is not None
            }
        )

    @property
    def season_count(self) -> int:
        return len(
            {
                entry.season_id
                for entry in self.entries
                if entry.season_id is not None
            }
        )


class PlayerEvidenceRepository:
    def __init__(
        self,
        player_stats_path: Path = DEFAULT_PLAYER_STATS_PATH,
    ) -> None:
        self.player_stats_path = player_stats_path
        self._histories: dict[str, PlayerEvidenceHistory] | None = None

    def load_histories(self) -> dict[str, PlayerEvidenceHistory]:
        if self._histories is not None:
            return self._histories

        df = pd.read_csv(
            self.player_stats_path,
            dtype={"season_year": str},
            low_memory=False,
        )

        entries = [
            evidence_entry_from_row(row)
            for _, row in df.iterrows()
        ]

        histories: dict[str, PlayerEvidenceHistory] = {}

        for player_id, group_entries in _group_entries_by_player(entries).items():
            histories[player_id] = PlayerEvidenceHistory(
                player_id=player_id,
                entries=tuple(group_entries),
            )

        self._histories = histories
        return histories

    def get_history(
        self,
        player_id: str,
    ) -> PlayerEvidenceHistory | None:
        return self.load_histories().get(str(player_id))


def _row_value(row: pd.Series, column: str, default=None):
    if column not in row or pd.isna(row[column]):
        return default

    return row[column]


def _row_int(row: pd.Series, column: str) -> int | None:
    value = _row_value(row, column, None)

    if value is None:
        return None

    return int(value)


def _row_float(row: pd.Series, column: str) -> float | None:
    value = _row_value(row, column, None)

    if value is None:
        return None

    return float(value)


def evidence_entry_from_row(row: pd.Series) -> PlayerEvidenceEntry:
    return PlayerEvidenceEntry(
        player_id=str(_row_value(row, "player_id")),
        player_name=_row_value(row, "player", None),

        competition=_row_value(row, "competition", None),
        competition_type=_row_value(row, "competition_type", None),
        competition_id=_row_int(row, "competition_id"),
        season_id=_row_int(row, "season_id"),
        season_year=str(_row_value(row, "season_year", "")),

        team=_row_value(row, "team", None),
        team_id=_row_int(row, "team_id"),

        minutes_played=_row_float(row, "minutesPlayed"),
        rating=_row_float(row, "rating"),
        appearances=_row_float(row, "appearances"),
        matches_started=_row_float(row, "matchesStarted"),
    )


def _group_entries_by_player(
    entries: list[PlayerEvidenceEntry],
) -> dict[str, list[PlayerEvidenceEntry]]:
    grouped: dict[str, list[PlayerEvidenceEntry]] = {}

    for entry in entries:
        grouped.setdefault(entry.player_id, []).append(entry)

    return grouped
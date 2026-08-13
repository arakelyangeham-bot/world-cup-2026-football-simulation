#bracket.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulation.competition.tie import Tie


@dataclass
class Bracket:
    """
    Generic knockout bracket.

    A bracket contains the ties for a knockout stage.
    """

    name: str
    ties: list[Tie] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_stage_matches(self) -> list[Tie]:
        return self.ties


class BracketBuilder:
    """
    Builds simple knockout brackets from ordered team lists.

    Version 1 supports high-seed vs low-seed pairing.
    """

    def build_high_low_bracket(
        self,
        teams: list[str],
        name: str = "Knockout Bracket",
    ) -> Bracket:
        if len(teams) % 2 != 0:
            raise ValueError("BracketBuilder requires an even number of teams.")

        ties: list[Tie] = []

        left = 0
        right = len(teams) - 1

        while left < right:
            ties.append(
                Tie(
                    team1=teams[left],
                    team2=teams[right],
                    metadata={
                        "seed_team1": left + 1,
                        "seed_team2": right + 1,
                        "pairing_type": "high_low",
                    },
                )
            )

            left += 1
            right -= 1

        return Bracket(
            name=name,
            ties=ties,
            metadata={
                "builder": "BracketBuilder",
                "pairing_type": "high_low",
                "team_count": len(teams),
                "tie_count": len(ties),
            },
        )
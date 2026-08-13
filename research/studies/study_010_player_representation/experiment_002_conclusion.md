experiment_002_conclusion.md

# Experiment 002 Conclusion — Competition-Aware Player Representation

## Summary

Competition-aware evidence weighting produced meaningful differences between strategies.

## Main Findings

- Competition count and season count currently add little discriminative value.
- Weighted evidence is the strongest current competition-aware signal.
- Combined competition evidence is more stable but less expressive than weighted evidence alone.
- The useful distinction is not simply how many competitions a player appears in, but how much weighted evidence supports the player.

## Interpretation

The current Player Representation benefits more from quantitative evidence quality than from simple metadata counts.

## Recommendation

Use `total_weighted_evidence` as the primary competition-aware evidence field for now.

Treat `competition_count` and `season_count` as metadata, not primary weighting signals.

## Next Experiment

Experiment 003 — Recency-Aware Player Representation

Research question:

Should recent player evidence matter more than older evidence?
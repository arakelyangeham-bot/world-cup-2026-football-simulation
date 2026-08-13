# Representation Research Summary

## Status

**Representation Infrastructure and Matched Benchmark Phase: Complete**

**Current leading production candidate:** `robust_zscore`

**Production promotion status:** Not yet approved

---

## 1. Purpose

This research phase investigated whether changing the mathematical transformation applied to player-level features could improve the quality of downstream football representations and, ultimately, predictive performance.

The central question was:

> Does changing the player-feature transformation improve football prediction when fixtures, targets, ClubElo priors, model specification, train/test splits, regularization, and benchmark implementation are held constant?

The work deliberately separated two concerns:

1. **Representation construction** — how raw player features are transformed into player attributes, player ratings, and team representations.
2. **Predictive evaluation** — whether those representations improve held-out football predictions.

---

## 2. Representations evaluated

Three matched representation branches were constructed:

- `global_zscore`
- `percentile_normal`
- `robust_zscore`

The global Z-score branch preserved the historical production behavior and served as the control.

The percentile-normal and robust Z-score branches changed only the feature-transformation layer. All downstream logic remained shared.

---

## 3. Architectural outcome

The representation pipeline is now modular across the following chain:

```text
Raw player features
        ↓
FeatureTransformationStrategy
        ↓
Player attribute construction
        ↓
Player role ratings
        ↓
Competition player repository
        ↓
Competition roster builder
        ↓
Competition team repository
        ↓
Production-format club repository
        ↓
Observation projection
        ↓
ClubElo enrichment
        ↓
Goal-model benchmark
```

The key architectural result is that the feature transformation can now be changed without rewriting the rest of the pipeline.

The following existing abstractions were reused rather than replaced:

- `PlayerRepository`
- `CompetitionPlayerRepository`
- `CompetitionRosterBuilder`
- `CompetitionTeamRepository`
- `ProductionClubRepository`
- `ClubEloRepository`
- `GoalModelBenchmarkConfig`
- `run_goal_model_benchmark()`
- `write_goal_model_benchmark_outputs()`

This made it possible to isolate the representation as the independent variable.

---

## 4. Production-equivalence safeguards

Before generating alternative branches, the historical global Z-score behavior was frozen through exact equivalence checks.

### Player attribute builder

The configurable player-attribute builder reproduced the pre-refactor artifact with:

- identical row count;
- identical columns;
- identical player IDs;
- identical missing-value patterns;
- maximum numeric difference of `0.0`.

### Player rating builder

The configurable player-rating builder reproduced the pre-refactor artifact with:

- `9,699` rows in both versions;
- identical columns;
- identical player IDs;
- identical missing-value patterns;
- maximum numeric difference of `0.0`.

These checks established the following contract:

```text
No arguments
→ global_zscore
→ canonical output paths
→ exact historical artifacts
```

---

## 5. Multi-representation artifact generation

Three complete player-representation branches were generated:

```text
global_zscore
    ↓
player attributes
    ↓
player ratings

percentile_normal
    ↓
player attributes
    ↓
player ratings

robust_zscore
    ↓
player attributes
    ↓
player ratings
```

The artifact audit confirmed:

- identical `9,699`-player populations;
- identical schemas;
- identical identity fields;
- identical evidence and provenance fields;
- zero duplicate players;
- zero non-finite numeric values;
- genuinely different attribute values;
- genuinely different rating values;
- no overwrite of canonical production artifacts.

---

## 6. Transformation-specific club repositories

Three Bundesliga production-format club repositories were generated:

- `bundesliga_club_repository_global_zscore.csv`
- `bundesliga_club_repository_percentile_normal.csv`
- `bundesliga_club_repository_robust_zscore.csv`

The repository audit confirmed:

- `18` clubs in every branch;
- identical club populations;
- zero missing values;
- zero non-finite values;
- successful runtime reload through `ProductionClubRepository`;
- alternative representation values detected;
- canonical Study 078 output unchanged.

---

## 7. Bundesliga fixture-date ClubElo priors

A dedicated Bundesliga ClubElo prior artifact was generated for the complete 2024–25 fixture population.

The artifact contained:

- `306` fixture rows;
- `306` unique event IDs;
- `18` clubs;
- complete rating coverage;
- fixture-date-valid home and away rating intervals;
- validated `rating_prior_diff = home_rating_prior - away_rating_prior`;
- `clubelo` as the sole rating-prior source.

This prior dataset was representation-invariant and reused identically across all three branches.

---

## 8. Matched observation datasets

Each transformation-specific repository was projected onto the same Bundesliga 2024–25 fixture population.

The final enriched datasets contained:

- `306` fixtures each;
- `306` unique event IDs each;
- the same home and away club populations;
- the same match targets;
- the same fixture-date-valid ClubElo priors;
- identical schemas;
- zero duplicate events;
- zero missing representation values;
- zero non-finite representation values;
- different player-derived representation values only.

The experimental control was therefore:

```text
same fixtures
same targets
same ClubElo priors
same feature specification
same split policy
same regularization
same benchmark engine
only representation changes
```

---

## 9. Study 092C2B — Primary matched benchmark

### Configuration

- Competition: Bundesliga 2024–25
- Observation count: `306`
- Training matches: `229`
- Test matches: `77`
- Train fraction: `0.75`
- Alpha: `0.0`
- Feature specification: `attack_defense_attack_depth_rating_prior`
- Shared chronological split: yes
- Shared ClubElo priors: yes
- Standard benchmark engine reused unchanged: yes

### Result

`robust_zscore` was the single leader.

It won `8` of the `10` comparison metrics and achieved a mean metric rank of `1.4`.

### Primary predictive winners

`robust_zscore` produced the lowest:

- combined Poisson deviance: `1.12859400`;
- combined goal MAE: `0.980189`;
- total-goal MAE: `1.46304`;
- goal-difference MAE: `1.23978`;
- outcome log loss: `1.00123851`;
- outcome Brier score: `0.600260`;
- exact-score log loss: `3.06320748`;
- absolute away-goal mean error: `0.16795`.

`global_zscore` won:

- absolute draw-rate error;
- absolute home-goal mean error.

`percentile_normal` won none of the ten comparison metrics.

### Interpretation

The first matched benchmark provided direct evidence that robust scaling produced a more useful downstream football representation than the historical global Z-score baseline in this experiment.

---

## 10. Study 092C2C — Robustness benchmark

### Configuration grid

The primary result was tested across:

- train fractions: `0.60`, `0.70`, `0.75`, `0.80`;
- alpha values: `0.0`, `0.01`, `0.1`, `1.0`.

This produced:

```text
3 representations
× 4 train fractions
× 4 alpha values
= 48 benchmark runs
```

There were `16` matched split-alpha configurations.

### Result

- Preferred representation: `robust_zscore`
- Evidence classification: **STRONG**
- Configuration win rate: `87.5%`
- Primary metric win rate: `92.0%`
- Primary metric wins: `103 / 112`

### Robust primary-metric win rates

| Metric | Robust win rate |
|---|---:|
| Combined Poisson deviance | 87.5% |
| Combined goal MAE | 87.5% |
| Total-goal MAE | 81.25% |
| Goal-difference MAE | 100% |
| Outcome log loss | 100% |
| Outcome Brier score | 100% |
| Exact-score log loss | 87.5% |

### Interpretation

The robust advantage persisted across reasonable changes to chronological split and Poisson regularization.

This substantially reduced the likelihood that the Study 092C2B result was caused by one arbitrary train/test split or one alpha value.

---

## 11. Current research conclusion

Within the scope of the completed experiments:

> `robust_zscore` is the strongest player-feature transformation evaluated and is the current leading production representation candidate.

The evidence is stronger than a single benchmark result because it includes:

- exact production-equivalence checks for the control branch;
- matched player populations;
- matched club populations;
- matched fixture populations;
- representation-invariant ClubElo priors;
- a shared benchmark engine;
- a shared chronological split policy;
- robustness across multiple train fractions;
- robustness across multiple regularization values.

The burden of proof has therefore shifted. Future candidate representations should now be compared against `robust_zscore`, not only against the historical global Z-score baseline.

---

## 12. Methodological boundary

These studies do **not** establish universal superiority.

The current evidence is limited by the following conditions:

- one league;
- one season;
- retrospective evaluation;
- static season-level player representations;
- player evidence not independently frozen before every fixture date;
- no cross-league validation;
- no multi-season validation;
- no World Cup tournament-level validation yet.

ClubElo priors were fixture-date valid, but the player-derived representation was not a fully prediction-date-frozen historical representation.

Accordingly, the correct conclusion is:

> Robust Z-score is the leading production candidate under the controlled Bundesliga experiments completed so far.

The incorrect conclusion would be:

> Robust Z-score has been proven universally optimal.

---

## 13. Production recommendation

Do not replace the canonical production baseline immediately.

The next phase should be **Production Candidate Validation**.

### Study 093A — Paired production artifact construction

Build two compatible production goal-model artifacts:

```text
global observations
→ global fitted goal-model artifact

robust observations
→ robust fitted goal-model artifact
```

Both artifacts must use:

- the same feature specification;
- the same training population;
- the same chronological cutoff;
- the same alpha;
- the same artifact contract.

### Study 093B — Paired production replay

Run the unchanged production pipeline with compatible repository-model pairs:

```text
global repository
+ global goal-model artifact
```

versus:

```text
robust repository
+ robust goal-model artifact
```

The existing production factory already accepts independent paths for:

- the club repository;
- the ClubElo cache;
- the goal-model artifact.

### Study 093C — Replay comparison

Compare:

- runtime stability;
- expected-goal predictions;
- goal errors;
- outcome log loss;
- Brier score;
- exact-score log loss;
- draw calibration;
- home- and away-goal calibration;
- scoreline distributions;
- fixture-level prediction deltas.

Only compatible representation-model pairs should be compared. A Robust repository must not be paired with a Global-trained goal-model artifact.

---

## 14. Longer-term validation roadmap

After Study 093:

1. **Cross-league validation**
   - Premier League
   - Serie A
   - La Liga
   - other leagues with sufficiently complete data

2. **Multi-season validation**
   - repeat matched benchmarks across additional seasons;
   - test temporal stability.

3. **Tournament simulation sensitivity**
   - compare Global and Robust World Cup simulation outputs;
   - champion probabilities;
   - advancement probabilities;
   - draw rates;
   - upset rates;
   - scoreline distribution;
   - knockout volatility.

4. **Production decision**
   - promote Robust only if replay and broader validation remain favorable.

---

## 15. Final phase decision

### Representation Infrastructure Phase

**Complete**

### Representation Benchmark Phase

**Complete**

### Current production candidate

`robust_zscore`

### Production baseline

`global_zscore` remains canonical until Production Candidate Validation is completed.

### Next study

**Study 093A — Paired Production Goal-Model Artifact Builder**

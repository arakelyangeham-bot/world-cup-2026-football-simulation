# Research Protocol 004

## One-Goal Football

### An Empirical Investigation of Competitive Football

---

# Protocol Status

**Status:** Approved for implementation

This protocol defines the objectives, methodology, observables, analyses, and interpretation criteria for Study 004.

The purpose of this document is to specify the study before analysis begins.

---

# 1. Motivation

One-goal matches are among the most common outcomes in football.

They also represent one of the largest remaining discrepancies identified by the Football Gap Report.

Rather than attempting to directly modify the match generator, this study seeks to understand one-goal football as an observable football phenomenon.

The study is descriptive rather than predictive.

---

# 2. Primary Research Question

> What measurable properties characterize one-goal football?

---

# 3. Secondary Research Questions

### RQ1

How common are one-goal matches?

---

### RQ2

How does one-goal frequency vary with competitive balance?

---

### RQ3

How does one-goal frequency vary across different pre-match representations?

Representations include:

- FIFA points gap
- Attack gap
- Midfield gap
- Defense gap
- Goalkeeper gap
- Poisson attack gap
- Poisson defense gap

---

### RQ4

How often do one-goal matches involve:

- clean sheets
- both teams scoring
- high-scoring matches

---

### RQ5

Do different representations explain one-goal football differently?

---

### RQ6

Does the current production match generator reproduce one-goal football realistically?

---

# 4. Dataset

Version 1 uses:

Historical international matches.

Current pre-match observations.

Final scorelines.

No event-level information is required.

---

# 5. Football Observables

Primary observable:

```text
OneGoalMatch

Secondary observables:

CleanSheet

BothTeamsScored

HighScoring

Draw

# 6. Independent Variables

Current study will analyze:

Competitive Balance (FIFA gap)

Attack gap

Midfield gap

Defense gap

Goalkeeper gap

Poisson attack gap

Poisson defense gap

# 7. Observatory Methodology

For every representation:

Representation

↓

OneGoalMatch

Generate:

response curve
confidence intervals
sample sizes

Repeat for secondary observables.

# 8. Planned Outputs

The study should produce:

Observable summary

Overall one-goal frequency.

Conditional response curves

One-goal frequency vs.

each representation.

Representation comparison

Compare response ranges across all representations.

Football interpretation

Separate:

Observation

Evidence

Hypothesis

Conclusion

# 9. Interpretation Rules

Every reported finding must be classified as one of:

Observation

Evidence

Hypothesis

Conclusion

No hypothesis should be presented as an established football law.

# 10. Success Criteria

Study 004 succeeds if it:

characterizes one-goal football descriptively,
identifies promising football relationships,
generates at least one testable hypothesis,
suggests at least one future improvement to the Football Match Generator.

Prediction accuracy is not a success criterion.

# 11. Limitations

Current study cannot investigate:

match tempo
game states
substitutions
tactical changes
possession
shot quality

These require event-level datasets.

# 12. Expected Contribution

Study 004 should establish the first empirical characterization of one-goal football within the Football Observatory framework.

The expected contribution is methodological and descriptive rather than predictive.

# 13. Next Studies

Potential follow-up studies include:

Study 005

Draw Dynamics

Study 006

Clean Sheets

Study 007

Both Teams Scored

Study 008

High-Scoring Football

These studies will reuse the methodology introduced here.

Guiding Principle

The objective is not to prove preconceived football ideas.

The objective is to allow the historical data to reveal how one-goal football behaves.

# 6. Results

## Observation 6.1 — One-goal football is common

### Observation

Approximately half of the historical matches in the dataset were decided by exactly one goal.

### Evidence

The Football Profile measured a one-goal match frequency of approximately 50% across the historical dataset.

### Interpretation

One-goal matches are not a niche phenomenon.

They represent one of the dominant forms of football and therefore deserve explicit study rather than being treated as isolated scorelines.

---

## Observation 6.2 — One-goal football is not simply low-scoring football

### Observation

One-goal matches produced only a modest reduction in average total goals compared with the historical baseline.

### Evidence

Average total goals decreased slightly relative to the overall football profile rather than collapsing into exclusively low-scoring matches.

### Interpretation

This suggests that one-goal football includes both:

- defensive results such as 1–0
- open results such as 2–1 and 3–2

Consequently, one-goal football should not be equated with defensive football.

---

## Observation 6.3 — One-goal football contains multiple football styles

### Observation

One-goal matches include:

- clean-sheet victories,
- both-teams-scored matches,
- low-scoring matches,
- relatively high-scoring matches.

### Evidence

The Football Profile demonstrates meaningful variation across these observables.

### Interpretation

One-goal football appears to be a family of football outcomes rather than a single homogeneous category.

Future research should therefore distinguish between different subclasses of one-goal football.

---

# 7. Discussion

The principal contribution of this study is conceptual rather than predictive.

Instead of viewing scorelines such as:

```text
1–0
2–1
1–2

as unrelated events, the Observatory suggests treating them as members of a broader football phenomenon:

One-goal football.

This perspective provides a more natural scientific unit for future investigation than individual scorelines.

It also aligns more closely with how football is discussed by analysts and coaches, who often refer to "tight matches" or "close games" rather than isolated scorelines.

The present study therefore proposes that one-goal football should become a first-class observable within the Football Observatory.

8. Engineering Decision
Production Recommendation

No immediate changes to the production match generator are recommended.

The current evidence is descriptive rather than prescriptive.

Research Recommendation

Future work should investigate:

which pre-match representations best explain one-goal football,
whether one-goal football exhibits distinct conditional response curves,
whether one-goal football is primarily determined before kickoff or emerges from match dynamics.

No new generator component should be implemented until these questions have been investigated.

9. What We Learned

This study established several important observations.

One-goal football represents one of the dominant observable patterns in football.
One-goal football is not synonymous with defensive football.
One-goal football contains multiple distinct scoreline families.
Football populations can be characterized using reusable Football Profiles.
Football Profiles complement response curves by describing football populations rather than conditional relationships.
10. What We Still Do Not Know

Several important questions remain unanswered.

Why do some balanced matches become one-goal matches while others become draws?
Which pre-match representation best explains one-goal football?
Does one-goal football arise primarily from pre-match balance or from in-match dynamics?
How does event-level football differ inside one-goal matches?
Can one-goal football be reproduced more accurately without harming other football observables?

These questions motivate the next stage of Computational Football research.

Conclusion

Study 004 represents the first empirical characterization of one-goal football within the Football Observatory.

Rather than proposing changes to the production simulator, the study establishes one-goal football as an observable football population worthy of independent investigation.

More broadly, the study demonstrates that Football Profiles provide a complementary research instrument alongside Football Relationships and Football Response Curves.

Together these instruments allow the Observatory to move beyond individual scorelines toward the systematic study of football phenomena.
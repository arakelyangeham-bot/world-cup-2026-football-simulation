representation_assumption_audit

# Study 088A — Team Representation Assumption Audit

## 1. Purpose

This document audits the mathematical and football assumptions encoded
by the current team-representation aggregation method.

The goal is not to reject the existing representation.

The goal is to make its implicit assumptions explicit before proposing
alternative aggregation architectures.

The current representation has been valuable because it is:

- deterministic;
- interpretable;
- stable;
- computationally inexpensive;
- compatible with multiple competitions;
- independent of the downstream goal model;
- easy to validate and serialize.

Study 087C nevertheless showed that changing one or two players in a
projected starting XI produced only very small changes in the resulting
team representation.

That finding raises a foundational question:

> Is this stability desirable robustness, excessive smoothing, or some
> combination of both?

Study 088A begins by examining the current representation on its own
terms.

---

## 2. Current Representation Pipeline

The current conceptual pipeline is:

```text
Player role ratings
        ↓
Player dimensional projections
        ↓
Team-level aggregation
        ↓
TeamRepresentation

Each player is projected into football dimensions such as:

attack
midfield
defense
goalkeeper

The team is then summarized approximately as:

attack
=
mean of the five highest player attack projections

midfield
=
mean of the five highest player midfield projections

defense
=
mean of the five highest player defense projections

goalkeeper
=
maximum goalkeeper projection

attack_depth
=
mean attack projection across the player population

midfield_depth
=
mean midfield projection across the player population

defense_depth
=
mean defense projection across the player population

For starting-XI representations, the player population contains eleven
selected players.

For full-squad representations, the player population contains the
eligible squad.

3. What the Current Representation Does Well

Before identifying limitations, it is important to record the
properties the current method intentionally preserves.

3.1 Stability

Small changes in the player population usually produce small changes in
the representation.

This reduces sensitivity to:

uncertain lineup projections;
noisy player ratings;
minor roster changes;
rotation between similarly rated players;
incomplete player evidence.

Study 087C confirmed this behavior empirically.

3.2 Interpretability

Every output feature has a direct explanation.

For example:

attack
=
average of the five strongest attacking projections

No hidden nonlinear transformation is required to understand the
feature.

3.3 Domain portability

The same aggregation can be used for:

domestic clubs;
national teams;
tournament squads;
projected starting lineups;
full squads.

The method does not depend on one league, formation, or data provider.

3.4 Computational simplicity

The representation requires:

sorting;
arithmetic means;
a maximum operation.

It introduces no fitted parameters and no additional model artifact.

3.5 Separation of concerns

The representation is built upstream of the goal model.

The goal model consumes numerical team features without needing to know:

which players were selected;
how the lineup was generated;
what formation was used;
how player roles were projected.

This architectural boundary should be preserved.

4. Assumption 1 — Additive Player Contribution
Mathematical assumption

Each player contributes independently to a team dimension.

Conceptually:

team attack
=
aggregate of individual attacking projections

There are no player-player interaction terms.

Football interpretation

The contribution of one attacker is assumed not to depend on:

the identity of the striker beside them;
the creativity of the midfield;
the overlapping ability of the fullback;
the team's formation;
tactical role compatibility;
manager instructions.
Strength

This assumption makes the representation:

general;
stable;
easy to estimate;
resistant to sparse interaction data.
Weakness

Football systems are not fully additive.

Examples include:

Winger quality
×
Fullback overlap
Striker movement
×
Attacking-midfielder creativity
Center-back quality
×
Defensive-midfielder protection

A team containing individually strong players may perform below the sum
of their isolated abilities if their roles conflict.

Conversely, a coherent unit may outperform the apparent sum of its
players.

Current assessment

The additive assumption is reasonable as a production baseline.

It should not be treated as a complete description of team strength.

Interaction-aware representations should remain future research rather
than the first Version 2B modification.

5. Assumption 2 — Arithmetic Averaging
Mathematical assumption

Within each dimension, selected player projections contribute equally
to the top-five mean.

For attack:

attack
=
(a1 + a2 + a3 + a4 + a5) / 5
Football interpretation

The strongest attacker and the fifth-strongest attacker receive equal
weight inside the aggregate.

Example

These two attacking populations have the same arithmetic mean:

Team A:
0.95, 0.90, 0.85, 0.80, 0.75

Team B:
0.85, 0.85, 0.85, 0.85, 0.85

Both produce:

0.85

Yet they describe different teams.

Team A is top-heavy.

Team B is balanced.

Strength

Arithmetic averaging rewards broad quality and avoids excessive
dependence on one player.

Weakness

It erases information about the shape of the quality distribution.

It cannot distinguish:

one superstar plus good support;
five equally strong contributors;
a steep quality drop after the first player;
an unusually balanced unit.
Current assessment

The arithmetic mean is a strong stability-oriented baseline.

Alternative representations should test whether distribution shape adds
incremental information without discarding the mean.

6. Assumption 3 — Top-Five Sufficiency
Mathematical assumption

The five strongest projections contain enough information to describe a
team's attack, midfield, or defense.

Players ranked below fifth do not affect the primary strength feature.

They only influence depth features.

Football interpretation

A team dimension is determined by a small group of its strongest
contributors.

Strength

The rule avoids diluting team strength with:

reserves;
youth players;
marginal squad members;
players who rarely contribute in that dimension.

This is especially useful for full-squad representations.

Weakness

The number five is a structural choice rather than a universal football
law.

Different dimensions may require different effective player counts.

For example:

attack:
perhaps three or four primary contributors

midfield:
perhaps four or five contributors

defense:
perhaps five or six contributors

The correct count may also depend on formation.

Current assessment

Top-five aggregation should remain the control condition.

Study 088 should test whether dimension-specific contributor counts are
more informative than one shared value of five.

7. Assumption 4 — Ranking Within Dimensions Is Sufficient
Mathematical assumption

The attack feature uses the players with the highest attack
projections, regardless of their formation slots.

The same applies independently to midfield and defense.

Football interpretation

A player's dimensional quality matters more than their specific place
in the lineup.

Strength

This avoids making the representation dependent on one formation
taxonomy.

A player may legitimately contribute to multiple dimensions.

Weakness

The representation can ignore structural balance.

For example, a nominal starting XI might contain:

several strong attacking projections;
too few natural defenders;
no credible defensive midfielder.

The top-five method may still produce strong attack, midfield, and
defense values if multifunctional players score well across dimensions.

It does not explicitly confirm that each tactical responsibility is
covered by an appropriate player.

Current assessment

Role balance and dimensional quality should be treated as separate
information.

The current dimensional features should not necessarily be replaced,
but they may be supplemented by structural-balance features.

8. Assumption 5 — Goalkeeper as a Maximum
Mathematical assumption

Goalkeeper strength is the maximum goalkeeper projection in the player
population.

Football interpretation

Only the best available goalkeeper matters.

Strength

For a fixed starting XI, this is usually appropriate because only one
goalkeeper starts.

For a squad, it identifies the likely first-choice goalkeeper without
diluting the feature with reserves.

Weakness

For prediction-date representations, the best-rated goalkeeper may be:

injured;
suspended;
recently transferred;
not selected;
unavailable for another reason.

The maximum is only valid if the candidate population is itself valid.

It also ignores goalkeeper depth and uncertainty.

Current assessment

The maximum remains a sensible starting-goalkeeper feature.

A separate goalkeeper-depth or goalkeeper-availability feature may be
needed rather than changing the maximum operator itself.

9. Assumption 6 — Depth as an Arithmetic Mean
Mathematical assumption

Depth is represented by the mean projection across the entire player
population.

Football interpretation

Every additional player contributes proportionally to squad depth.

Strength

This provides a simple distinction between:

strong first team with weak reserves;
strong first team with strong reserves.
Weakness

A whole-squad mean is sensitive to roster size and marginal players.

Two clubs may have equally useful benches but different numbers of
registered youth or fringe players.

The mean also treats:

the twelfth-best player;
the twenty-fifth-best player;

as equally relevant observations.

Current assessment

Depth should probably focus on the most plausible replacement players,
not every roster member equally.

Candidate alternatives include:

mean of ranks 6–10
mean of the best two replacements per dimension
drop-off from starter group to replacement group
10. Assumption 7 — Linear Scale Meaning
Mathematical assumption

A difference of:

0.05

has the same meaning at every point on the representation scale.

For example:

0.60 → 0.65

is treated like:

0.85 → 0.90
Football interpretation

Player and team quality improve linearly.

Strength

Linear scales are easy to interpret and work naturally with regression
models.

Weakness

Elite quality may have nonlinear effects.

A movement from very good to world-class may matter more than an equal
numerical movement among average players.

Alternatively, diminishing returns may apply when several elite players
occupy the same dimension.

Current assessment

Nonlinear transformations should be tested carefully.

They must not be adopted merely because elite teams are currently
underpredicted.

Any nonlinear representation must prove that it improves held-out
performance rather than simply inflating strong clubs.

11. Assumption 8 — Representation Invariance to Opponent
Mathematical assumption

A team has one intrinsic representation independent of its opponent.

Football interpretation

The same team strength is used against:

an aggressive pressing team;
a low defensive block;
a possession-dominant opponent;
a counterattacking opponent.
Strength

Opponent-independent representations are reusable and easy to cache.

They preserve a clean distinction between:

team identity

and:

match interaction
Weakness

Some qualities are matchup dependent.

Examples include:

aerial strength against a weak aerial defense;
pace against a high defensive line;
midfield press resistance;
vulnerability to transitions.
Current assessment

Opponent interaction should probably remain downstream of the core team
representation.

The base representation should describe the team.

A later matchup layer may describe how two representations interact.

12. Assumption 9 — Representation Invariance to Time
Mathematical assumption

A stored team representation remains valid until explicitly rebuilt.

Football interpretation

The represented squad quality is sufficiently stable across the
evaluation period.

Strength

Static repositories are efficient and reproducible.

Weakness

They cannot capture:

transfers;
injuries;
suspensions;
changing lineups;
managerial changes;
player development;
declining form;
tactical evolution.

Studies 085A and 085B found temporal residual structure consistent with
this limitation.

Current assessment

Prediction-date validity is a separate problem from aggregation design.

A better aggregation method cannot compensate for an invalid player
population.

Version 2B should therefore distinguish:

Who is represented?

from:

How are they aggregated?
13. Assumption 10 — Stability Is Preferable to Responsiveness
Mathematical assumption

The aggregation intentionally suppresses small player-level changes.

Football interpretation

Replacing one similarly rated player should not substantially change
team strength.

Strength

This protects the model from noisy lineup estimates.

It also reflects the reality that many rotations have limited impact.

Weakness

The same smoothing may suppress genuinely important changes involving:

a superstar;
the only natural striker;
the first-choice goalkeeper;
the primary creator;
a critical defensive midfielder.
Current assessment

The representation should be stable for ordinary substitutions but
responsive to high-leverage substitutions.

This may be the most important design objective for the next
aggregation architecture.

14. Summary of Current Assumptions
Assumption	Main benefit	Main limitation
Additive player contribution	Simplicity and portability	No chemistry or interaction
Arithmetic averaging	Stability	Erases distribution shape
Top-five sufficiency	Avoids squad dilution	Fixed contributor count
Dimension-only ranking	Formation independence	Weak structural balance
Goalkeeper maximum	Correct first-choice emphasis	Depends on availability validity
Whole-population depth mean	Simple squad-depth measure	Sensitive to roster size and fringe players
Linear scale	Interpretability	May miss elite nonlinearities
Opponent invariance	Reusable team identity	No matchup specificity
Time invariance	Reproducibility	Static and potentially stale
Stability preference	Noise resistance	Can suppress important absences
15. Desired Properties of a Future Representation

A future representation should preserve the strongest properties of the
current system while adding targeted sensitivity.

It should be:

Interpretable

Every feature should have a football explanation.

Stable

Minor substitutions between similar players should not create large
changes.

Responsive

The loss or addition of a high-impact player should be visible.

Role-aware

The representation should preserve positional and structural balance.

Distribution-aware

It should distinguish elite top-heavy units from uniformly good units.

Depth-aware

It should measure realistic replacement quality rather than roster size.

Prediction-date compatible

The aggregation should accept a date-valid player population without
requiring redesign.

Domain-general

The same representation framework should remain usable for clubs and
national teams.

Model-independent

The representation should not be designed solely to exploit one fitted
goal-model specification.

Reproducible

All aggregation parameters and choices should be explicit and frozen.

16. Provisional Conclusion

The current representation is not poorly designed.

It is designed primarily for:

stability
interpretability
portability

Study 087C showed that it succeeds at those goals.

Its principal limitation is not that it averages players.

Its limitation is that it does not distinguish enough between:

ordinary player replacement

and:

high-leverage player replacement

The next representation architecture should therefore not discard the
current top-five and depth features.

A stronger direction is likely:

current stable features
+
distribution-shape features
+
star influence
+
role balance
+
replacement drop-off

This preserves the validated baseline while testing whether carefully
chosen additional information improves football understanding and
downstream prediction.

17. Study Boundary

This document does not recommend a production change.

It does not select an alternative aggregator.

It establishes the assumptions and desired properties that the remaining
Study 088 documents will use to evaluate candidate aggregation families.
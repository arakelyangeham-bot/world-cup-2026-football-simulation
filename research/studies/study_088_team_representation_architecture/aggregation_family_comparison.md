aggregation_family_comparison.md

# Study 088A — Aggregation Family Comparison

## 1. Purpose

This document compares candidate mathematical families for converting
player-level football intelligence into team-level representations.

The comparison is guided by the design principles established in:

`representation_assumption_audit.md`

The purpose is not to select a production aggregator immediately.

The purpose is to define a controlled research space in which each
candidate family:

- has an explicit mathematical form;
- encodes a clear football hypothesis;
- can be benchmarked against the current representation;
- preserves architectural separation from the downstream goal model;
- can be validated without relying on intuition alone.

The current top-five mean and depth-mean representation remains the
control condition.

---

## 2. Evaluation Criteria

Each aggregation family will be evaluated conceptually against the
following properties.

### Interpretability

Can the output be explained in football terms?

### Stability

Does the representation remain reasonably unchanged under minor
substitutions between similarly rated players?

### Responsiveness

Does the representation react when a high-impact player is added,
removed, or replaced?

### Role awareness

Does the method preserve positional and structural information?

### Distribution awareness

Can it distinguish a top-heavy unit from a balanced unit with the same
mean quality?

### Depth awareness

Does it describe realistic replacement quality rather than raw roster
size?

### Prediction-date compatibility

Can it consume a date-valid player population without changing the
aggregation architecture?

### Domain portability

Can the method be used for clubs and national teams?

### Parameter burden

How many manually chosen or fitted parameters are required?

### Overfitting risk

How easily could the method encode noise or league-specific quirks?

### Computational cost

Can the representation be constructed efficiently at runtime?

### Compatibility

Can the method coexist with the current `TeamRepresentation` and
production observation contracts?

---

## 3. Family 0 — Current Top-Five Mean Baseline

### Mathematical form

For one team dimension with ordered player projections:

```text
x₁ ≥ x₂ ≥ ... ≥ xₙ

the current primary strength feature is:

S_top5
=
(x₁ + x₂ + x₃ + x₄ + x₅) / 5

The current depth feature is:

D_mean
=
(x₁ + x₂ + ... + xₙ) / n

Goalkeeper strength is:

G
=
max(g₁, g₂, ..., gₙ)
Football hypothesis

A team dimension is described by its strongest five contributors,
while the wider player population describes depth.

Strengths
highly interpretable;
stable;
parameter free;
resistant to squad dilution in the primary feature;
compatible with both full squads and starting XIs;
easy to serialize and reproduce;
already validated throughout Version 2A.
Weaknesses
equal weight within the top five;
fixed contributor count across dimensions;
no distribution-shape information;
no explicit role balance;
limited responsiveness to one-player changes;
whole-population depth mean may be affected by roster size.
Research role

This is the mandatory control for every Study 088 benchmark.

No alternative should be considered useful merely because it produces
larger variation.

It must show better football behavior or predictive value.

4. Family 1 — Rank-Weighted Top-K Aggregation
Mathematical form

For the top k player projections:

x₁ ≥ x₂ ≥ ... ≥ xₖ

define:

S_weighted
=
w₁x₁ + w₂x₂ + ... + wₖxₖ

subject to:

wᵢ ≥ 0

and:

w₁ + w₂ + ... + wₖ = 1

Example:

w = (0.30, 0.25, 0.20, 0.15, 0.10)
Football hypothesis

The strongest contributors matter more than the weakest members of the
primary group.

Expected behavior

Compared with the arithmetic mean, this method should:

increase sensitivity to elite players;
preserve broad contribution from several players;
remain more stable than a maximum;
distinguish some top-heavy teams from balanced teams.
Strengths
easy to interpret;
minor extension of the current method;
compatible with existing dimensional projections;
computationally trivial;
explicit control over star influence;
can use the same k as the baseline.
Weaknesses
weights are structural choices;
rankings may be unstable when player values are close;
the same weights may not suit attack, midfield, and defense;
still ignores role-slot structure;
may exaggerate elite-player influence if weights are too steep.
Candidate variants
Mild weighting
(0.24, 0.22, 0.20, 0.18, 0.16)
Moderate weighting
(0.30, 0.25, 0.20, 0.15, 0.10)
Strong weighting
(0.40, 0.25, 0.15, 0.12, 0.08)
Initial assessment

This is the strongest first alternative because it modifies the
existing representation conservatively.

5. Family 2 — Dimension-Specific Top-K Aggregation
Mathematical form

Use different contributor counts by dimension:

Attack:
top k_A players

Midfield:
top k_M players

Defense:
top k_D players

For example:

k_A = 4
k_M = 5
k_D = 6
Football hypothesis

Different football dimensions depend on different numbers of primary
contributors.

Attack may be concentrated among fewer players.

Defense may depend on a broader unit.

Strengths
intuitive football rationale;
preserves the current averaging philosophy;
low parameter burden;
may improve dimensional specificity;
compatible with current schemas.
Weaknesses
contributor counts are manually chosen;
formation dependence may remain hidden;
the identities of top projected contributors may not align with
actual tactical roles;
changes in k can create abrupt threshold effects.
Candidate configurations
A: (4, 5, 6)
B: (3, 5, 5)
C: (4, 4, 5)
D: (5, 5, 5)  baseline
Initial assessment

This is a suitable low-risk benchmark family.

It should be tested before introducing more complex nonlinear methods.

6. Family 3 — Distribution-Shape Augmentation
Mathematical form

Retain the current top-five mean but add features describing the
distribution of player quality.

For one dimension:

mean_top5
maximum
minimum_top5
standard_deviation_top5
range_top5

Possible features include:

star_gap
=
x₁ - mean(x₂, x₃, x₄, x₅)
primary_spread
=
x₁ - x₅
top5_standard_deviation
quality_concentration
=
x₁ / (x₁ + ... + x₅)
Football hypothesis

Two teams with the same mean may behave differently depending on
whether quality is:

concentrated in one superstar;
evenly distributed;
sharply declining after the first few players.
Strengths
preserves the validated current features;
adds information rather than replacing information;
highly interpretable;
directly addresses a known weakness of arithmetic averaging;
allows the goal model to decide whether distribution shape matters.
Weaknesses
increases feature dimensionality;
shape features may be correlated;
small rating noise can affect ranks and spreads;
requires careful feature selection to avoid redundancy;
downstream model must be refitted.
Initial assessment

This is one of the most promising families.

It provides richer information without forcing one new scalar to replace
the current team-strength feature.

7. Family 4 — Star-Influence Augmentation
Mathematical form

Retain the baseline mean and add an explicit star term.

Example:

S_star
=
mean_top5
+
α(maximum - mean_top5)

where:

α ≥ 0

Equivalent form:

S_star
=
(1 - α)mean_top5
+
α maximum

when 0≤α≤1.

Football hypothesis

The best player in a unit may influence match outcomes more than the
arithmetic mean captures.

Candidate values
α ∈ {0.10, 0.20, 0.30}
Strengths
simple;
interpretable;
explicitly responsive to elite players;
only one additional parameter per dimension or globally;
compatible with current aggregation outputs.
Weaknesses
may overstate one-player influence;
maximum ratings can be noisy;
does not describe broader distribution shape;
may inflate already strong teams;
could appear to solve elite-team compression in-sample without
generalizing.
Initial assessment

This family is useful as a controlled test of the elite-player
hypothesis.

It should not be promoted without held-out validation.

8. Family 5 — Power-Mean Aggregation
Mathematical form

For positive player projections:

M_p
=
[
(1/k) Σ xᵢ^p
]^(1/p)

Special cases:

p = 1

gives the arithmetic mean.

p > 1

gives greater influence to stronger players.

As p becomes large:

M_p → maximum

Candidate values:

p ∈ {1.25, 1.50, 2.00}
Football hypothesis

Player influence may increase nonlinearly with quality.

Strengths
mathematically coherent;
smooth transition from mean toward maximum;
one interpretable sensitivity parameter;
avoids manually choosing rank weights;
remains permutation invariant.
Weaknesses
less intuitive to nontechnical readers;
sensitive to feature scale;
requires positive and consistently normalized inputs;
may duplicate the effect of star-influence features;
still ignores role structure.
Initial assessment

This is a useful nonlinear benchmark, but not the first production
candidate.

It should be treated as a diagnostic family.

9. Family 6 — Softmax-Weighted Aggregation
Mathematical form

Define data-dependent weights:

wᵢ
=
exp(βxᵢ)
/
Σ exp(βxⱼ)

Then:

S_softmax
=
Σ wᵢxᵢ

Parameter:

β ≥ 0

When:

β = 0

all players receive equal weight.

As β increases, higher-rated players receive more weight.

Football hypothesis

The influence of strong players increases smoothly according to their
quality rather than fixed rank.

Strengths
smooth;
differentiable;
data-dependent;
avoids abrupt rank-weight boundaries;
explicit control over concentration.
Weaknesses
sensitive to rating scale;
less interpretable than fixed weights;
may be unstable if inputs are not normalized consistently;
parameter tuning may overfit;
highest-rated players can dominate quickly.
Initial assessment

This is conceptually attractive but carries greater calibration and
interpretability risk than fixed rank weighting.

It belongs in a later diagnostic benchmark.

10. Family 7 — Role-Slot Aggregation
Mathematical form

Aggregate players according to explicit tactical roles rather than
dimension rankings.

For a 4-3-3 attack example:

Attack
=
w_ST × striker_quality
+
w_WL × left_winger_quality
+
w_WR × right_winger_quality
+
w_AM × attacking_midfield_support

Defense might include:

CB1
CB2
FB1
FB2
DM
GK

with explicit weights.

Football hypothesis

Team strength arises from occupied tactical responsibilities rather
than only from the highest dimensional projections.

Strengths
football-specific;
structurally interpretable;
sensitive to missing specialists;
prevents a team from appearing balanced merely because multifunctional
players score well across dimensions;
naturally compatible with starting-XI representations.
Weaknesses
formation dependent;
requires consistent slot definitions;
difficult to apply to full squads;
role-assignment errors directly affect representation;
large manual parameter burden;
cross-competition formations vary;
may reduce portability.
Initial assessment

This family has strong long-term potential but is too foundational for
the first aggregation benchmark.

It should initially be implemented as supplementary structural features,
not a replacement for the dimension-based baseline.

11. Family 8 — Structural-Balance Features
Mathematical form

Keep the current strength features and add diagnostics describing
whether essential role groups are adequately covered.

Examples:

natural_center_back_count
natural_fullback_count
natural_defensive_midfielder_count
natural_winger_count
natural_striker_count

Possible normalized balance features:

defensive_structure_score
attacking_role_coverage
formation_fit_score
fallback_role_count
Football hypothesis

A collection of high-quality players can still be structurally
imbalanced.

Strengths
highly interpretable;
directly uses information already generated in Study 087B;
supplements rather than replaces current strength;
allows the downstream model to estimate whether imbalance matters;
useful for availability-adjusted lineups.
Weaknesses
depends on formation and role taxonomy;
role ratings may not perfectly describe tactical suitability;
feature definitions can become arbitrary;
requires careful normalization across formations.
Initial assessment

This is a high-priority augmentation family.

The existing exact_role_match and fallback_role_count data provide a
natural starting point.

12. Family 9 — Replacement-Drop-Off Depth
Mathematical form

Instead of the mean of the entire squad, measure the difference between
primary contributors and their best replacements.

Example:

starter_attack
=
mean of top 5 attack projections
replacement_attack
=
mean of ranks 6–10
attack_dropoff
=
starter_attack - replacement_attack

Alternative:

best_replacement_mean
=
mean of next 3 contributors
Football hypothesis

Depth is not the average quality of every registered player.

Depth is the quality of the players most likely to replace starters.

Strengths
robust to roster size;
directly interpretable;
responsive to bench quality;
useful for injury and rotation scenarios;
retains distinction between first team and depth.
Weaknesses
rank thresholds remain structural choices;
full-squad data are required;
starting-XI-only representations cannot estimate broader depth;
replacement roles may not align with simple dimensional ranks.
Initial assessment

This is the strongest candidate for replacing the whole-squad mean depth
features.

It should be included in the first implementation benchmark.

13. Family 10 — Quantile-Based Depth
Mathematical form

Represent player-quality distributions using quantiles.

Examples:

90th percentile
75th percentile
median
25th percentile

Or:

top_quartile_mean
second_quartile_mean
Football hypothesis

Squad quality is better described by distribution levels than by one
mean.

Strengths
less sensitive to roster size than a full mean;
preserves distribution shape;
avoids fixed player ranks when squad sizes differ;
general across competitions.
Weaknesses
less directly football-specific;
small squads produce unstable quantiles;
quantile definitions may be redundant with order statistics;
starting-XI populations are too small for rich quantile summaries.
Initial assessment

Useful for full-squad depth research, but lower priority than explicit
replacement-drop-off features.

14. Family 11 — Availability-Weighted Aggregation
Mathematical form

Each player contribution is weighted by estimated availability or start
probability:

S_available
=
Σ pᵢxᵢ
/
Σ pᵢ

where:

pᵢ
=
probability player is available or starts
Football hypothesis

A player's contribution should depend on the probability that they
actually participate.

Strengths
prediction-date oriented;
smooth treatment of lineup uncertainty;
better than treating an uncertain expected XI as certain;
allows multiple plausible lineups to be represented implicitly.
Weaknesses
requires prediction-date-valid probabilities;
unavailable in the current historical data;
errors in availability estimation propagate into representation;
may blur role feasibility.
Initial assessment

This is a major long-term target.

It cannot be properly benchmarked using only the current season-level
usage aggregates.

15. Family 12 — Interaction-Augmented Representation
Mathematical form

Add pairwise or subsystem interaction terms.

Examples:

winger_fullback_synergy
center_back_pair_quality
striker_creator_interaction
midfield_balance

A generic interaction form might be:

I_ab
=
f(x_a, x_b)
Football hypothesis

The value of one player depends partly on the players and roles around
them.

Strengths
closest to football as a structured system;
can represent chemistry and complementary roles;
may explain why equally rated squads perform differently.
Weaknesses
substantial data requirements;
combinatorial feature growth;
high overfitting risk;
difficult to validate;
difficult to transfer across teams and competitions;
requires stable tactical and role data.
Initial assessment

This is outside the immediate Version 2B implementation scope.

It should remain a research roadmap item.

16. Family 13 — Learned Permutation-Invariant Aggregation
Mathematical form

Use a learned function over a set of player vectors:

TeamRepresentation
=
ρ(
Σ φ(player_i)
)

This resembles a Deep Sets architecture.

Other possibilities include:

attention pooling;
set transformers;
graph neural networks.
Football hypothesis

The optimal aggregation may be learned from data rather than manually
specified.

Strengths
highly flexible;
can learn nonlinear player importance;
can include interactions;
naturally supports variable squad size.
Weaknesses
low interpretability;
high data requirements;
high overfitting risk;
difficult to debug;
creates a new fitted artifact;
complicates production architecture;
may entangle representation learning with goal prediction.
Initial assessment

This is not appropriate for the next implementation phase.

The project should exhaust interpretable aggregation families first.

17. Comparative Matrix
Family	Interpretability	Stability	Responsiveness	Role awareness	Distribution awareness	Parameter burden	Near-term priority
Current top-five mean	High	High	Low	Low	Low	None	Control
Rank-weighted top-k	High	Medium-high	Medium	Low	Medium	Low	High
Dimension-specific top-k	High	High	Medium	Low	Low	Low	High
Distribution-shape augmentation	High	High	Medium	Low	High	Low-medium	Very high
Star-influence augmentation	High	Medium	High	Low	Medium	Low	High
Power mean	Medium	Medium	Medium-high	Low	Medium	Low	Medium
Softmax weighting	Medium	Medium	High	Low	Medium	Medium	Low-medium
Role-slot aggregation	High	Medium	High	High	Medium	High	Later
Structural-balance features	High	High	Medium	High	Low	Medium	Very high
Replacement-drop-off depth	High	High	Medium	Medium	Medium	Low	Very high
Quantile-based depth	Medium	High	Medium	Low	High	Low	Medium
Availability-weighted	High	Medium	High	Medium	Medium	High data burden	Long-term
Interaction-augmented	Medium	Variable	High	High	High	High	Future
Learned set aggregation	Low	Variable	High	Potentially high	High	Very high	Future
18. Recommended First Benchmark Families

The first implementation benchmark should remain deliberately small.

Recommended families:

Control
Current top-five mean
+
current depth mean
Candidate A
Mild rank-weighted top-five
Candidate B
Moderate rank-weighted top-five
Candidate C
Dimension-specific top-k
Candidate D
Current mean
+
distribution-shape features
Candidate E
Current mean
+
star-influence features
Candidate F
Replacement-drop-off depth
Candidate G
Structural-balance features

These families cover the most important hypotheses while preserving
interpretability and limiting parameter growth.

19. Recommended Non-Goals for the First Benchmark

The first aggregation benchmark should not yet include:

learned neural aggregation;
player-pair chemistry;
opponent-specific representation;
manager-specific tactics;
availability probabilities;
complex formation embeddings;
large hyperparameter searches;
production promotion.

These are legitimate long-term directions, but they would obscure the
first scientific question:

Does a modestly richer, still-interpretable aggregation contain useful
information beyond the current top-five means?

20. Benchmarking Philosophy

Candidate aggregators should not be judged only by how much they change
team representations.

A larger representation delta is not automatically better.

Each candidate should be evaluated across four layers.

Layer 1 — Mathematical behavior

Test controlled synthetic player populations.

Examples:

balanced versus top-heavy teams;
one superstar removed;
one reserve replaced;
roster size increased with weak players;
goalkeeper unavailable;
structural role missing.
Layer 2 — Empirical sensitivity

Apply aggregators to existing Bundesliga player populations.

Measure:

club rankings;
feature ranges;
elite versus non-elite separation;
sensitivity to lineup changes;
sensitivity to fringe players;
full-squad versus XI differences.
Layer 3 — Incremental information

Measure overlap with:

current team representations;
ClubElo;
dynamic form;
observed goals.

A new feature that merely duplicates ClubElo may have limited value.

Layer 4 — Predictive validation

Only after the first three layers should candidates enter a held-out
goal-model benchmark.

21. Provisional Recommendation

The current representation should remain intact as the baseline.

The first Version 2B aggregation implementation should emphasize
augmentation rather than replacement:

Current stable strength features
+
star influence
+
quality distribution
+
structural balance
+
replacement drop-off

This approach has several advantages:

preserves backward compatibility;
keeps validated production features;
allows ablation testing;
avoids betting the project on one new scalar;
lets the downstream model determine which new information is useful;
creates a clean path for feature-by-feature promotion.
22. Study Boundary

This document does not select a final aggregator.

It defines the candidate families and narrows the first implementation
benchmark to interpretable, architecture-compatible alternatives.

The next document should define the non-negotiable principles that every
future representation must satisfy:

representation_design_principles.md
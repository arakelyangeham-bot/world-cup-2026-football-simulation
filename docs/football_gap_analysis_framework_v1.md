# Football Gap Analysis Framework v1.0

## Purpose

The Football Gap Analysis Framework defines how the project will compare generated football against real football.

Its goal is not to improve the simulator directly.

Its goal is to identify where the current match generator differs from historical football, so that future research projects are driven by evidence rather than intuition.

## Core Research Loop

```text
Current Match Generator
  ↓
Generated Football
  ↓
Football Gap Analyzer
  ↓
Largest Remaining Discrepancy
  ↓
Football Hypothesis
  ↓
Prototype Model
  ↓
Benchmark
  ↓
Promotion or Rejection
```

## Primary Question

What measurable stochastic properties of real football are not yet captured by the current match generator?

## Inputs

The framework compares two datasets:

```text
Historical Football
Generated Football
```

Historical football comes from the historical training dataset.

Generated football comes from the current production or research match generator.

## Required Analyzer Sections

### 1. Scoreline Distribution

Measures whether the generated scoreline frequencies match real football.

Metrics:

```text
scoreline total variation distance
largest overproduced scorelines
largest underproduced scorelines
0-0 frequency
1-1 frequency
1-0 frequency
2-1 frequency
other high-frequency scorelines
```

### 2. Goal Distribution

Measures whether generated goal totals resemble real football.

Metrics:

```text
mean total goals
home goals mean
away goals mean
total goal variance
home goal variance
away goal variance
goal difference variance
five-plus-goal rate
six-plus-goal rate
```

### 3. Draw Behavior

Measures whether draws are generated realistically.

Metrics:

```text
overall draw rate
0-0 share of draws
1-1 share of draws
2-2 share of draws
draw rate by expected-goal balance
draw rate by team-strength gap
```

### 4. Low-Score Dependence

Measures whether low-score outcomes behave realistically.

Metrics:

```text
0-0 frequency
1-0 frequency
0-1 frequency
1-1 frequency
2-1 frequency
1-2 frequency
low-score TVD
Dixon-Coles-sensitive scorelines
```

### 5. Blowout and Tail Behavior

Measures whether the generator handles extreme results realistically.

Metrics:

```text
three-plus-goal margin rate
four-plus-goal margin rate
five-plus total goals
six-plus total goals
other scoreline bucket
```

### 6. Home / Team1 Advantage

Measures whether the first-listed team advantage is realistic in the historical dataset.

Metrics:

```text
home/team1 win rate
away/team2 win rate
home/team1 goals
away/team2 goals
home/team1 goal difference
```

### 7. Calibration and Outcome Realism

Measures whether the scoreline generator produces outcome probabilities consistent with real results.

Metrics:

```text
multiclass Brier score
multiclass log loss
expected calibration error
home/draw/away reliability
predicted vs actual outcome distribution
```

### 8. Tournament Stability

Measures whether improvements in match generation produce reasonable tournament-level behavior.

Metrics:

```text
champion probability stability
runner-up probability stability
semifinal probability stability
quarterfinal probability stability
round-of-16 probability stability
group qualification stability
```

## Overall Realism Score

The framework may compute an overall realism score, but this score should not hide the component metrics.

A model may improve the overall score while worsening a specific football phenomenon.

Therefore, every report must include both:

```text
aggregate realism score
component-level diagnostics
```

## Gap Prioritization

Each identified gap should be classified by:

```text
size
football importance
modeling feasibility
data availability
production risk
```

Suggested priority labels:

```text
High
Medium
Low
Needs more data
Do not model yet
```

## Research Recommendation Section

Every Football Gap Report should end with:

```text
Largest remaining discrepancy
Evidence
Candidate football explanation
Recommended next research project
Expected benchmark impact
```

## Promotion Rule

A new match-generation component may be promoted only if it:

```text
1. Reduces at least one high-priority football gap.
2. Does not materially worsen other major realism metrics.
3. Preserves the simulate_match_score boundary.
4. Passes scoreline realism benchmarks.
5. Passes outcome calibration benchmarks.
6. Does not destabilize tournament probabilities without explanation.
```

## Non-Goals

The Football Gap Analyzer should not:

```text
choose tournament winners
change production simulation behavior
train predictive models
encode subjective football opinions
replace benchmark scripts
```

It should measure gaps and guide research.

## Architectural Role

The Football Gap Analyzer sits beside the match generator.

```text
Historical Football
        │
        ▼
Football Gap Analyzer
        ▲
        │
Generated Football
```

It is a feedback system for research.

## Guiding Principle

The historical data should tell us what to study next.

The project should not add complexity because an idea sounds plausible.

It should add complexity only when a measurable football gap justifies it.

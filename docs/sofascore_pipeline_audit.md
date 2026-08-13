sofascore_pipeline_audit.md

# SofaScore Pipeline Audit

## Purpose

Map the existing SofaScore scraping and feature-engineering scripts onto the new canonical Football Data Pipeline architecture.

The goal is to refactor and promote existing working code rather than rebuild the pipeline from scratch.

---

# Existing Pipeline

```text
sofascore_discover_league_seasons.py
    ↓
sofascore_build_competitions_config.py
    ↓
sofascore_build_competition_manifest.py
    ↓
ingest_teams.py
    ↓
ingest_players.py
    ↓
ingest_player_stats.py
    ↓
build_stat_manifest.py
    ↓
aggregate_player_history.py
    ↓
sofascore_feature_engineering.py

Architecture Mapping
Existing Script	Canonical Pipeline Layer	Status
sofascore_discover_league_seasons.py	Competition discovery	Existing
sofascore_build_competitions_config.py	Competition config	Existing
sofascore_build_competition_manifest.py	Competition manifest / scrape plan	Existing
ingest_teams.py	Team discovery	Existing
ingest_players.py	Automated player / roster discovery	Existing
ingest_player_stats.py	Player statistics ingestion	Existing
build_stat_manifest.py	Stat schema / aggregation rules	Existing
aggregate_player_history.py	Multi-season player evidence aggregation	Existing
sofascore_feature_engineering.py	Player feature engineering	Existing
Key Finding

The project already contains the core of an automated data pipeline.

Most importantly, ingest_players.py discovers players from competition-season data rather than relying on a manually constructed roster.

This means the long-term pipeline should evolve from the current SofaScore scripts rather than be rebuilt.

Refactor Goal

Promote the current pipeline into a source-specific adapter structure:

data_sources/
    sofascore/
        discover_seasons.py
        build_competitions.py
        build_manifest.py
        ingest_teams.py
        ingest_players.py
        ingest_player_stats.py

The downstream Player Intelligence layer should consume canonical outputs, not raw provider-specific files.

Canonical Outputs

The promoted pipeline should produce:

competitions.csv
teams.csv
players.csv
player_stats.csv
stat_manifest.csv
player_features.csv

These outputs should eventually feed:

Player Repository
    ↓
Squad / Starting XI
    ↓
Team Strength Aggregation
    ↓
Team Repository
Immediate Decision

Do not build a new scraper.

Refactor the existing SofaScore scraping pipeline.

Next Engineering Step

Create a pipeline runner or manifest-driven orchestrator that executes the existing stages in order.

Potential file:

scripts/run_sofascore_pipeline.py

or, after refactor:

data_sources/sofascore/run_pipeline.py
Open Questions
Should source-specific scripts remain under scripts/ or move under data_sources/?
Should World Cup and club pipelines share the same manifest format?
Should competition_manifest.csv become the canonical scrape plan?
How should failed tasks be tracked across teams, players, and stats?
Which outputs should be considered raw, processed, or canonical?
Guiding Principle

Refactor working scraper logic into the new architecture.

Do not rewrite working code unless the interface requires it.
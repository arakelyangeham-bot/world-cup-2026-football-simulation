PREMIER_LEAGUE_2026_27_PRODUCTION_NOTES

# Premier League 2026-27 Production Notes

## Rating-prior cutoff

The 2026-27 Premier League preseason simulation freezes ClubElo-based
rating priors at the forecast cutoff date:

- Rating-prior date: `2026-08-15`

Fixture dates remain the real Premier League calendar dates. The fixed
rating-prior date prevents future ClubElo information from leaking into
a preseason season forecast.

## Leeds United ClubElo provenance

The normal ClubElo history endpoint was unavailable during the
2026-27 production bootstrap.

For Leeds United, the local production cache therefore uses a
web-verified ClubElo preseason value of:

- Club: Leeds United
- ClubElo: `1770`
- Effective source date: `2026-05-24`
- Frozen simulation coverage: through `2027-05-30`
- Cache provenance label:
  `clubelo_web_verified_frozen_preseason`

The generated cache file is local processed data and is not committed
to the repository.

This exception should be replaced by the normal ClubElo history source
when reliable historical access becomes available.

## Interpretation

Production Monte Carlo results may be interpreted as preseason model
forecasts only when:

- `model_mode` is `production`
- `forecast_interpretation_allowed` is `true`
- the production repository and goal-model artifact are the intended
  2026-27 inputs
- rating priors use the documented preseason cutoff
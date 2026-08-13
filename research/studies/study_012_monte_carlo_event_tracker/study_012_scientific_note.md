## Scientific Note: Scoreline Tail Behavior

Study 012 successfully introduced a Monte Carlo observation layer capable of surfacing extreme simulated tournament events. The event leaderboards revealed several unusually high-scoring matches in the 1,000-tournament sample, including scorelines far beyond realistic World Cup expectations.

This does not invalidate the observer framework. On the contrary, it demonstrates its value: the observer surfaced tail behavior that aggregate probability tables may hide.

A likely future investigation is whether extreme scorelines are caused by the scoreline sampler itself, by the distribution of team-strength values, or by interaction effects between the sampler and the active `TEAM_REPOSITORY_SOURCE`. At the time of this study, the active source was `legacy`; future calibration should compare tail behavior across `legacy`, `dimension-specific`, `top-11 mean`, `top-5 mean`, `star-weighted`, and `starter-plus-depth`.

This issue is noted for future scoreline-tail calibration, but it is outside the scope of Study 012. Study 012 remains focused on building narrative Monte Carlo event leaderboards and validating the observer architecture.
# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gittoshigit/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gittoshigit/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts the earn rate of every Grand Theft Auto VI PS5 trophy on Exophase at the
case target, as five quantiles per trophy. The model works in earner counts, not
rates: it segments the case history into Exophase refresh states, separates the
launch cohort from later arrivals, and projects players-tracked growth and
per-trophy progression to the target, so the dilution that lands at each refresh
is forecast instead of extrapolated away. Python 3.13, standard library only,
deterministic. The design is documented in DESIGN.md.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## v2 (2026-11-20)

Practice grade of v1 (loss 0.95 vs house persistence 1.51, coverage_90 0.18) exposed three defects, fixed in v2 (see DESIGN_V2.md, DIAGNOSIS_V2.md, IMPL_V2_REPORT.md):

- R0 refresh clock now prices state age (equilibrium renewal + selection term), and the no-refresh model is a negative-binomial count model, so the median sits in the jump region and the lower tail reaches the zero-new-earners floor.
- R2 calibration backtest gained a third, launch-window origin; s_cal mixes the pooled optimum with the worst origin and is floored at 1.0.
- The forecast pipeline is staged (S1-S4): calibration and the second kernel are no longer silently disabled above 40 states, and a slow stage degrades to the last completed stage instead of the persistence fallback.

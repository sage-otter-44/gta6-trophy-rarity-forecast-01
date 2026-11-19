[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sage-otter-44/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sage-otter-44/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

A Python 3.13 standard-library program that forecasts week-one earn rates for all 55 Grand Theft Auto VI PlayStation 5 trophies, as five quantiles (p10, p25, p50, p75, p90) per trophy. Built for the [make.fun gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/).

## What it does

`main.py` reads the trophy list and the calibration data, fits a per-trophy earn-rate model, and writes the forecast (55 trophies × 5 quantiles) to stdout as JSON. No third-party packages, no network access, fully deterministic — the same input always produces byte-identical output.

## Method

- **Per-trophy earners/players fit on log-time.** Each trophy's earn rate is modelled as a function of log elapsed time since launch, fitted from the observed earners and players counts, so early-week growth is captured on the scale where it is roughly linear.
- **Tier ceilings.** Every trophy is capped by its grade tier (platinum, gold, silver, bronze): no forecast quantile may exceed the ceiling its tier allows, which keeps rare trophies rare and prevents the fit from extrapolating past what the tier can support in week one.
- **Asymmetric logit spreads.** Uncertainty is expressed on the logit scale with separate upper and lower spreads, so the quantile band is wider on the side the data is less certain about and the median stays honest instead of being pulled toward the middle.
- **Pinball loss on the logit scale.** The fit is scored and tuned with pinball loss evaluated on logit-transformed rates, so each quantile is trained against its own loss and the five quantiles stay ordered and calibrated.
- **Deterministic.** Fixed seeds, sorted iteration, no timestamps and no randomness anywhere in the pipeline: re-running the program reproduces the forecast exactly.

## Run it

```sh
python3 main.py
```

Requires Python 3.13 or later. Standard library only — nothing to install.

## License

MIT. See [LICENSE](LICENSE).

# Pricing American Put Options with CRR and Longstaff–Schwartz Monte Carlo

## Overview

This project benchmarks three approaches for pricing SPY American put options against live market mid-prices:

- Black–Scholes European put benchmark
- Cox–Ross–Rubinstein (CRR) binomial tree with early exercise
- Longstaff–Schwartz least-squares Monte Carlo (LSM) with optimal exercise estimation

The analysis compares model prices to market mid-prices using root mean squared error (RMSE), then examines the sensitivity of CRR to the number of tree steps and LSM to the number of simulated paths.

## Key findings

For the reported SPY option-chain sample, the Black–Scholes benchmark produced the lowest RMSE at **0.424516**, despite its European-exercise assumption. The report finds that CRR and LSM were close but slightly less accurate in that sample, illustrating that greater model complexity does not automatically lead to a closer fit to observed market quotes.

The numerical sensitivity analysis found:

- CRR RMSE was stable across 50 to 1,000 tree steps (approximately 0.5154 to 0.5132).
- LSM RMSE improved sharply when paths increased from 100 to 2,000 (from 1.2305 to approximately 0.4874), with residual variability caused by simulation and continuation-regression noise.

## Methodology

1. Retrieve a live SPY option chain using `yfinance`.
2. Select the first expiration at least 21 days away.
3. Build market mid-prices from bid and ask quotes and extract contract implied volatilities.
4. Price puts with Black–Scholes, a 200-step CRR tree, and LSM using 20,000 simulated paths and 50 exercise dates.
5. Calculate price RMSE for each approach against market mid-prices.
6. Run step-count and path-count sensitivity tests.

## Repository structure

```text
src/american_option_pricing.py  # Original project script; uploaded separately
report.pdf                      # Project report; uploaded separately
requirements.txt                # Dependencies
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/american_option_pricing.py
```

Because the script retrieves live data, the selected expiry, option sample, prices, and resulting metrics may change between runs.

## Limitations

- Market mid-prices are proxies for executable prices and do not fully capture bid–ask spreads, liquidity, and transaction costs.
- Black–Scholes is used only as a European-option benchmark; it does not model early exercise.
- CRR accuracy depends on time-step discretisation.
- LSM results depend on path count, exercise dates, basis functions, and random sampling.
- This repository is a quantitative-finance portfolio project, not investment advice or production trading infrastructure.

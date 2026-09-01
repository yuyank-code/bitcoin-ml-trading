# Research 11 — Nested OOS and Trial Accounting (2026-09-02)

## Audit result

The current `research_v2.py` has a materially useful leakage control: for a forward label with horizon H, each walk-forward training set ends at `start - H`. This prevents label endpoints from crossing the first test timestamp.

However, the resulting `research_v2_results.csv` is not present in the public repository, so the numerical search cannot currently be independently reproduced from checked-in artifacts. No performance number from an uncommitted local output should be promoted as evidence.

## New overfitting issue

The current runner evaluates 3 model families × 5 feature sets × 5 probability thresholds = 75 strategy configurations on the same OOS prediction stream, then sorts the resulting table to find the winner. Those predictions are OOS with respect to model fitting, but choosing the best configuration on that same OOS sample makes it a model-selection sample rather than a pristine final holdout.

This is not label leakage; it is selection bias / OOS overfitting.

## Required protocol

1. Generate purged walk-forward predictions once.
2. Split the prediction period into a development-selection segment and an untouched final holdout, chronologically.
3. Use only the development segment to select model, feature set, threshold and execution policy.
4. Freeze the complete policy, including costs and sizing rules.
5. Evaluate exactly once on the final holdout.
6. Report every trial, not only the winner.
7. Apply search-adjusted inference (DSR and, where sample structure permits, CSCV/PBO).
8. Require fold-level consistency and a predeclared transaction-cost/slippage grid.

## Hypotheses

- H30: the selected configuration retains positive net expectancy on the untouched final holdout.
- H31: performance remains positive under conservative fee/slippage assumptions.
- H32: the winner is not concentrated in a small number of walk-forward folds.
- H33: cost-aware abstention survives final holdout evaluation with predictions frozen.
- H34: the selected result remains credible after accounting for all 75 tested configurations (or the actual complete trial registry if additional experiments are run).

## Literature basis

Bailey & Lopez de Prado (2014), *The Deflated Sharpe Ratio*, explicitly addresses selection bias from multiple backtest trials and non-normal returns. Bailey, Borwein, Lopez de Prado & Zhu (2017), *The Probability of Backtest Overfitting*, proposes CSCV/PBO because ordinary holdouts can be unreliable for investment backtests.

## Promotion rule

Until the frozen final-holdout artifacts and complete trial registry are checked in, the project has **no promotable performance claim**. New model families or thresholds should not be added merely to improve the current ranking.

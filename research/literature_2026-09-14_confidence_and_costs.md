# Research update — 2026-09-14

## High-quality literature

### Allena (2026), Review of Financial Studies
**Confident Risk Premiums and Investments Using Machine Learning Uncertainties**

The paper derives ex-ante confidence intervals for ML risk-premium forecasts and reports that selecting observations with more precise forecasts can improve out-of-sample investment performance across model classes. The result is from cross-sectional equities, so it is a hypothesis for BTC rather than evidence of BTC profitability.

**Project implication:** the existing ensemble already exposes `model_disagreement`. Test whether low disagreement adds incremental economic value beyond `prob_up`, but only after executable horizon/event accounting is corrected. Register confidence-filter variants before outer OOS evaluation and include them in PBO/DSR trial accounting.

### Jensen, Kelly, Malamud & Pedersen (2026), Review of Financial Studies
**Machine Learning and the Implementable Efficient Frontier**

The paper argues that ML strategies should be evaluated on net-of-trading-cost risk/return opportunities. Cost-agnostic prediction can overemphasize fleeting signals with high turnover; incorporating trading costs changes which predictive information is economically useful.

**Project implication:** keep adaptive cost-aware policy selection separate from frozen-policy cost repricing. A model should not be promoted because of gross predictive quality if the executable net-of-cost frontier deteriorates under realistic friction.

### Saly-Kaufmann et al. (2026), large-scale financial time-series benchmark
**Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance**

The benchmark evaluates statistical significance, downside/tail risk, breakeven transaction costs, random-seed robustness and computational efficiency in addition to Sharpe. This supports treating seed sensitivity and cost-break-even as first-class robustness diagnostics rather than optional reporting.

### Bysik & Ślepaczuk (2026), BTC walk-forward study
**Machine Learning-Based Bitcoin Trading Under Transaction Costs**

Using roughly 70,000 hourly BTC-USDT observations and 27 walk-forward folds, the study reports that naïve sign-based ML trading can fail under 10-bps costs, while a cost-aware threshold can reduce turnover and recover selected configurations. Bootstrap evidence does not establish formal model-family dominance.

**Project implication:** the primary research question is execution-quality and economic robustness, not which architecture wins a frictionless prediction contest.

## Testable hypothesis H96

The ensemble's `model_disagreement` contains incremental information about forecast reliability.

Protocol:
1. Preserve the current unfiltered probability policy as benchmark.
2. Use only a small predeclared confidence-rule family.
3. Fit thresholds inside development folds only.
4. Evaluate predictive calibration/discrimination separately from net executable P&L.
5. Use corrected H80/H71/H90/H94 execution/event protocol and the existing realistic cost grid.
6. Register every trial for multiplicity/PBO/DSR accounting.
7. Require stable benefit across development paths and costs before touching the final holdout.

**Status:** hypothesis registered as GitHub Issue #46; deliberately not run yet because H80/H90/H94 remain execution-validity prerequisites.

## Current hard blockers

- H80: target/entry/six-bar holding-period alignment.
- H71: overlapping-position/exposure accounting.
- H94: immutable executable exit timestamp.
- H90: full OOS equity-grid reconstruction for frozen cost repricing.
- H72: point-in-time availability/vintage provenance for external data.
- H70/H77: dependence-aware economic/predictive inference.

No performance claim is made from this research note.

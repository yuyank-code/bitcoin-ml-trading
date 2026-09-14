# Research audit — 2026-09-14

## New literature evidence

1. **Bysik & Ślepaczuk (2026), hourly BTC-USDT under transaction costs.** Their 27-fold walk-forward study of roughly 70,000 hourly observations reports that naive sign strategies fail at 10 bps, while cost-aware filtering can materially reduce turnover and recover selected configurations. They explicitly report no formal statistical dominance of the candidate model families. Source: https://arxiv.org/abs/2606.00060

2. **Kim (2026), VALID framework for crypto ML validation.** Across 340 strategy variants, the paper emphasizes the statistical-to-economic disconnect and transaction-cost omission, and advocates a fixed-order validation protocol with economic gates rather than accuracy-only selection. Source: https://doi.org/10.2139/ssrn.6508779

3. **Zhai (2026), live-trading negative evidence.** A sequentially developed crypto ML system reports a walk-forward AUC of 0.85 but only about $0.12/day live PnL, highlighting execution/prediction mismatch, regime non-stationarity and complexity escalation as failure modes. Source: https://doi.org/10.2139/ssrn.6566940

4. **Research-design sensitivity.** Lalwani/Jindal et al. (2025/2026), *European Financial Management*, evaluate 5,376 ML portfolios and find large return dispersion caused by research-design choices; this supports treating execution, window, filtering and portfolio-construction choices as registered research degrees of freedom. Source: https://doi.org/10.1111/eufm.70033

5. **Uncertainty-adjusted sorting (2026).** Liu et al. find that uncertainty-adjusted ML predictions can improve portfolio construction, with gains primarily through lower volatility. This motivates uncertainty/disagreement tests, but does not justify assuming a direction for disagreement in BTC. Source: https://arxiv.org/abs/2601.00593

## New pipeline gate: H101

The current production engine is not yet economically valid despite having a six-bar predictive label. `production_research.py` labels `Close[t+6]/Close[t]-1`, enters at `t+1 Open`, but a no-barrier trade exits on the entry bar's close; it also persists `entry_time` without an `exit_time`. Frozen-cost repricing reconstructs the equity path using entry timestamps. These are event-time mismatches that can distort drawdown, Sharpe and other path-dependent statistics.

GitHub Issue #49 records the acceptance tests: exact horizon exit, explicit barrier/ambiguous-bar policy, immutable entry/exit timestamps, event-time frozen-cost replay on the full OOS grid, and future-row perturbation tests.

## Testable next step

Do not run candidate-model economic ranking until H101 passes. The next executable experiment should be a deterministic synthetic execution suite first. Only after it passes should H100 selective no-trade deployment and H96/H99 disagreement experiments be run on the real frozen OOS dataset.

## Interpretation

The fresh literature converges on the same failure mode: predictive metrics can remain strong while executable net returns disappear. Therefore the highest-value improvement is currently execution/event-time correctness, not additional architecture search.

No performance claim is made in this note.
